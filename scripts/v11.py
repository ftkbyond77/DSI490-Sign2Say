"""v11 data roles and sealed split (prompt/phase3/prompt-opt-v11.md, TPH 1 §1A.3 / §1B).

  python main.py v11 seal          # BLIND-v11 (seeded, existing data only) → artifacts/eval/blind_v11_manifest.json
  python main.py v11 tiers         # Core / Extended / Tail + effective signers → cloud_s3/data_prep/manifest/tiers.json
  python main.py v11 forms         # word → forms → clips (labels + embedding merge) → cloud_s3/data_prep/manifest/forms.json
  python main.py v11 sheets --words อะไร,กิน   # contact sheets (hold frames) of the forms of focus words → artifacts/reports/v11_sheets/

Definitions (prompt C6): effective signers of a word = 1 / Σ p_s² over its TRAIN clips (p_s = signer s's share).
Core = conversation words (models/model-v10/vocab_conversation.json, variants folded) with ≥ 5 effective signers ·
Extended = the other conversation words (2–4 effective signers; words below 2 are kept in Extended and flagged "thin" so the
vocabulary does not shrink) · Tail = every other concept: never a label of the v11 decision, used as hard negatives and as
"known unknown" examples for the open-set head.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from modules.utils import ART, PREP, get_logger, read_json, write_json

log = get_logger("v11")
SHARDS = PREP / "shards"
MAN = PREP / "manifest"
V10 = ROOT / "models" / "model-v10"
FOCUS = ["กิน", "คุณ", "ยาย", "ไป", "อะไร", "เหงา", "น้อง", "น้า", "เที่ยว", "ข้าว", "อา", "ย่า"]
GROUPS = [["ยาย", "ญาติ", "ย่า", "ปู่", "ตา", "ป้า"], ["น้า", "อา", "น้ำ"], ["ไป", "กลับ", "ส่ง", "มา"], ["อะไร", "ที่ไหน", "โสด", "หนึ่ง"],
          ["กิน", "ดู", "ฉัน", "ข้าว", "สวัสดี"], ["คุณ", "ฉัน", "แฟน", "ต้อง"], ["น้อง", "ลูก", "ปลา", "น้องชาย"], ["เหงา", "หมู", "ภูมิใจ"],
          ["พี่", "สูง"]]


def eff_signers(signers):
    p = pd.Series(signers).value_counts(normalize=True).values
    return float(1.0 / (p ** 2).sum()) if len(p) else 0.0


def vocab():
    vc = read_json(V10 / "vocab_conversation.json")
    return vc["concepts"], vc.get("variants", {})


# ----------------------------------------------------------------------------- seal
def cmd_seal(a):
    from modules.sealed import MANIFEST
    if MANIFEST.exists() and not a.force:
        log.info("already sealed: %s (use --force only before any v11 training)", MANIFEST)
        return
    rng = np.random.RandomState(a.seed)
    c = pd.read_parquet(SHARDS / "clips.parquet")
    iso = c[(c.kind == "isolated")]
    pick = {}
    # people from the current TEST roles (never seen by v10's training or tuning): TSL-ONE-S 2 · th_sl 1 · TTRS 1 · YouTube 3
    for ds, k in (("tslone", 2), ("th_sl", 1), ("ttrs", 1), ("agent", 3)):
        cand = sorted(iso[(iso.dataset == ds) & (iso.role == "test")].signer.dropna().unique())
        if ds == "agent":           # YouTube people listed in signers.csv with ≥ 2 clips (a person, not a single upload)
            sg = pd.read_csv(ROOT / "cloud_s3" / "agent_data" / "youtube_words" / "signers.csv")
            ok = set(sg[sg.n_clips >= 2].signer)
            cand = [s for s in cand if s in ok] or cand
        if ds == "ttrs":            # TTRS roles are per WORD: every TTRS test person also has 671–2,359 TRAIN clips. The unseen TTRS person
            n = iso[iso.dataset == "ttrs"].groupby("signer").size()      # with the smallest training footprint (≥ 50 clips) is sealed instead
            cand = [n[n >= 50].index[np.argmin([int(((iso.signer == s) & (iso.role == "train")).sum()) for s in n[n >= 50].index])]]
        pick[ds] = sorted(rng.choice(cand, min(k, len(cand)), replace=False).tolist())
    signers = sorted(s for v in pick.values() for s in v)
    clips = sorted(c[c.signer.isin(signers)].clip_id.astype(str))
    A = read_json(ART / "runs" / "seq-v7" / "tsl51_alignment.json")
    tmpl = sorted({tuple(v["glosses"]) for v in A.values() if v["role"] == "sentence_test"})
    n_t = int(round(0.4 * len(tmpl)))
    sel = [tmpl[i] for i in sorted(rng.choice(len(tmpl), n_t, replace=False))]
    sel_set = set(sel)
    sents = sorted(k for k, v in A.items() if v["role"] == "sentence_test" and tuple(v["glosses"]) in sel_set)
    # TSL51 sentence clips also live in the pose store as "t51s:<name>" (and an RTMW twin "#rtmw")
    m = dict(version="blind-v11", seed=a.seed, created="2026-11-13", rule="prompt-opt-v11 C4", people=pick, signers=signers,
             clips=clips, templates=[list(t) for t in sel], t51_sentences=sents, n_templates_total=len(tmpl),
             note="opened once in TPH 4 (THAISLM_OPEN_BLIND=1); every opening is logged in 'openings'", openings=[])
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("sealed %d people %s, %d clips, %d / %d TSL51 test templates (%d sentence clips) → %s", len(signers), pick, len(clips), n_t,
             len(tmpl), len(sents), MANIFEST)


# ----------------------------------------------------------------------------- tiers
def cmd_tiers(a):
    from modules.sealed import sealed_signers
    c = pd.read_parquet(SHARDS / "clips.parquet")
    tr = c[(c.kind == "isolated") & (c.view == "primary") & (c.role == "train") & c.concept.notna() & ~c.signer.isin(sealed_signers())]
    assert not set(tr.signer) & sealed_signers(), "sealed people in TRAIN"
    words, variants = vocab()
    fold = lambda x: variants.get(x, x)                                              # noqa: E731
    tr = tr.assign(word=tr.concept.map(fold))
    eff = tr.groupby("word").signer.apply(lambda s: eff_signers(s.values))
    nsg = tr.groupby("word").signer.nunique()
    ncl = tr.groupby("word").size()
    out = dict(doc=__doc__.split("Definitions")[1].strip(), core=[], extended=[], thin=[], tail_n=0, words={})
    for w in words:
        e = float(eff.get(w, 0.0))
        tier = "core" if e >= 5 else "extended"
        out[tier].append(w)
        if e < 2:
            out["thin"].append(w)
        out["words"][w] = dict(tier=tier, eff_signers=round(e, 2), signers=int(nsg.get(w, 0)), train_clips=int(ncl.get(w, 0)))
    allc = set(c[c.kind == "isolated"].concept.dropna().map(fold))
    tail = sorted(allc - set(words))
    out["tail_n"] = len(tail)
    # label space of encoder run R1: Core + Extended + every concept with ≥ 2 TRAIN signers (others: negatives only)
    sig_all = tr.groupby("concept").signer.nunique()
    out["label_concepts_ge2"] = sorted(set(sig_all[sig_all >= 2].index))
    write_json(MAN / "tiers.json", out)
    log.info("tiers: core %d · extended %d (thin < 2 eff. signers: %d) · tail %d · concepts with ≥ 2 train signers %d → %s",
             len(out["core"]), len(out["extended"]), len(out["thin"]), len(tail), len(out["label_concepts_ge2"]), MAN / "tiers.json")


# ----------------------------------------------------------------------------- forms
def _form_label(ds, variant, concept, base):
    if concept != base:                                         # politeness / school variant concept folded into its word
        return "variant:" + concept[len(base):].strip(" ()") if concept.startswith(base) else "variant:" + concept
    if variant is None or (isinstance(variant, float) and np.isnan(variant)) or variant == "":
        return "unlabelled"
    v = str(variant)
    if v.startswith("ท่าที่"):
        return "thsl:" + v.replace("ท่าที่", "")
    if v.startswith("var_"):
        return "t51:" + v
    if v.startswith("L0"):
        return "ttrs:" + v
    return "variant:" + v


def cmd_forms(a):
    """forms.json = {word: {forms: {form_id: {clips, signers, sources, labels}}, merged_from, rule}}.
    1. raw form label per clip (manifest sign_variant / variant concepts);
    2. per raw label a prototype = mean of per-signer means of the v10 bank embeddings (Uni-Sign, all people of the production bank
       except BLIND); two labels of one word are the SAME form when cos(prototypes) ≥ the 10th percentile of same-label cross-signer
       cosines over all words (an automatic, data-calibrated rule; the focus words are then checked by eye on contact sheets);
    3. "unlabelled" clips join the nearest form when above that threshold, else form "unlabelled"."""
    from modules.sealed import sealed_signers
    words, variants = vocab()
    fold = lambda x: variants.get(x, x)                                              # noqa: E731
    man = pd.read_parquet(MAN / "clips.parquet", columns=["clip_id", "sign_variant", "dataset"])
    sv = dict(zip(man.clip_id, man.sign_variant))
    c = pd.read_parquet(SHARDS / "clips.parquet")
    c = c[(c.kind == "isolated") & (c.view == "primary") & c.concept.notna() & ~c.signer.isin(sealed_signers())].copy()
    c["signer"] = c.signer.fillna(c.dataset + ":unknown")
    c["word"] = c.concept.map(fold)
    c = c[c.word.isin(words)]
    c["raw_form"] = [_form_label(d, sv.get(i), cc, w) for d, i, cc, w in zip(c.dataset, c.clip_id, c.concept, c.word)]
    b = np.load(V10 / "bank.npz")
    Zb = {str(k): v for k, v in zip(b["clip"], b["Z"])}
    def proto(g):
        zs = []
        for _, gs in g.groupby("signer"):
            z = [Zb[x].astype(np.float32) for x in gs.clip_id if x in Zb]
            if z:
                m = np.mean(z, 0); zs.append(m / (np.linalg.norm(m) + 1e-8))
        if not zs:
            return None, []
        p = np.mean(zs, 0)
        return p / (np.linalg.norm(p) + 1e-8), zs
    # threshold: same raw label, different signers
    same = []
    for (w, f), g in c[c.raw_form != "unlabelled"].groupby(["word", "raw_form"]):
        _, zs = proto(g)
        for i in range(len(zs)):
            for j in range(i + 1, len(zs)):
                same.append(float(zs[i] @ zs[j]))
    thr = float(np.percentile(same, 50)) if same else 0.8
    out = dict(rule=cmd_forms.__doc__ + "\n    (revised: labels documented by the SAME source are distinct forms and never merge; a label of another source joins "
               "its most similar form when cos ≥ the MEDIAN same-label cross-signer cosine — the 10th percentile merged documented forms)",
               threshold=round(thr, 4), n_same_pairs=len(same), words={})
    n_multi = 0
    src = lambda f: f.split(":")[0] + (":" + f.split("#")[1] if "#" in f else "")       # noqa: E731  (a school variant is its own source)
    for w, g in c.groupby("word"):
        labs = {f: proto(gg)[0] for f, gg in g.groupby("raw_form")}
        named = [f for f in labs if f != "unlabelled" and labs[f] is not None]
        named.sort(key=lambda f: -int((g.raw_form == f).sum()))
        parent = {f: f for f in named}
        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        for i, f2 in enumerate(named):          # biggest labels first; each later label may join one earlier form of OTHER sources
            best, bs = None, thr
            for f1 in named[:i]:
                root = find(f1)
                if any(src(m) == src(f2) for m in named if find(m) == root):
                    continue
                s_ = float(labs[f1] @ labs[f2])
                if s_ >= bs:
                    best, bs = root, s_
            if best is not None:
                parent[f2] = best
        groups = {}
        for f in named:
            groups.setdefault(find(f), []).append(f)
        assign = {f: find(f) for f in named}
        if "unlabelled" in labs:
            if labs["unlabelled"] is not None and named:
                best = max(named, key=lambda f: float(labs["unlabelled"] @ labs[f]))
                if float(labs["unlabelled"] @ labs[best]) >= thr:
                    assign["unlabelled"] = find(best)
                else:
                    assign["unlabelled"] = "unlabelled"; groups["unlabelled"] = ["unlabelled"]
            else:
                assign["unlabelled"] = "unlabelled"; groups["unlabelled"] = ["unlabelled"]
        forms = {}
        for k, (root, members) in enumerate(sorted(groups.items(), key=lambda kv: -int(g.raw_form.isin(kv[1]).sum()))):
            gg = g[g.raw_form.map(assign) == root]
            forms[f"f{k}"] = dict(labels=sorted(members), clips=sorted(gg.clip_id.astype(str)), signers=sorted(gg.signer.unique().tolist()),
                                  sources=sorted(gg.dataset.unique().tolist()), eff_signers=round(eff_signers(gg.signer.values), 2))
        out["words"][w] = dict(forms=forms, raw_labels=sorted(labs))
        n_multi += len(forms) >= 2
    # clip → (word, form) lookup for the training sampler
    out["clip_form"] = {cl: [w, f] for w, d in out["words"].items() for f, fd in d["forms"].items() for cl in fd["clips"]}
    write_json(MAN / "forms.json", out)
    log.info("forms: %d words, %d with ≥ 2 forms; merge threshold cos ≥ %.3f (10th pct of %d same-label cross-signer pairs) → %s",
             len(out["words"]), n_multi, thr, len(same), MAN / "forms.json")
    for w in FOCUS:
        d = out["words"].get(w)
        if d:
            log.info("  %s: %s", w, {f: (x["labels"], len(x["clips"]), x["eff_signers"]) for f, x in d["forms"].items()})


# ----------------------------------------------------------------------------- contact sheets
def hold_frame(kp, sc, f0, f1):
    """the hold of a sign = the frame of minimum dominant-wrist speed in the middle 60 % of the span"""
    a, b = f0 + int(0.2 * (f1 - f0)), max(f0 + int(0.8 * (f1 - f0)), f0 + int(0.2 * (f1 - f0)) + 1)
    w = kp[a:b, 10]
    v = np.r_[np.inf, np.linalg.norm(np.diff(w, axis=0), axis=1)]
    return a + int(np.argmin(v))


def draw_pose(ax, kp, sc, title=""):
    from modules.schema import LH, RH
    ok = sc > 0.3
    body = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12)]
    for i, j in body:
        if ok[i] and ok[j]:
            ax.plot(kp[[i, j], 0], -kp[[i, j], 1], "k-", lw=1)
    fingers = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
    for hand, col in ((LH, "tab:blue"), (RH, "tab:red")):
        h = np.asarray(hand)
        if ok[h].mean() > 0.5:
            for f in fingers:
                ax.plot(kp[h[f], 0], -kp[h[f], 1], "-", color=col, lw=1)
    face = list(range(23, 91))
    fk = [k for k in face if ok[k]]
    if fk:
        ax.plot(kp[fk, 0], -kp[fk, 1], ".", color="gray", ms=1)
    ax.set_title(title, fontsize=6); ax.set_aspect("equal"); ax.axis("off")


def cmd_sheets(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("Tahoma", "Leelawadee UI", "TH Sarabun New"):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f; break
    from scripts.train_encoder import PoseStoreMM, ensure_mm
    from modules.parts import to_iso
    from modules.schema import canonical_hands
    ensure_mm(); store = PoseStoreMM()
    F = read_json(MAN / "forms.json")
    c = pd.read_parquet(SHARDS / "clips.parquet").set_index("clip_id")
    out = ART / "reports" / "v11_sheets"; out.mkdir(parents=True, exist_ok=True)
    for w in (a.words.split(",") if a.words else FOCUS):
        d = F["words"].get(w)
        if not d:
            continue
        rows = [(f, cl) for f, x in d["forms"].items() for cl in x["clips"][:: max(1, len(x["clips"]) // 6)][:6]]
        n = len(rows); cols = 6; r_ = int(np.ceil(n / cols))
        fig, axes = plt.subplots(r_, cols, figsize=(cols * 1.6, r_ * 1.9))
        axes = np.atleast_2d(axes)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (f, cl) in zip(axes.flat, rows):
            if cl not in store.row:
                continue
            r = c.loc[cl]
            kp, sc, hw = store.get(store.row[cl])
            kp, sc, _ = canonical_hands(kp, sc, hw)
            kp = to_iso(kp, hw)
            f0, f1 = int(r.span_f0), int(r.span_f1)
            t = hold_frame(kp, sc, f0, max(f1, f0 + 2))
            draw_pose(ax, kp[t], sc[t], f"{f} {d['forms'][f]['labels'][0][:14]}\n{r.signer[:12]}")
        fig.suptitle(w); fig.tight_layout()
        fig.savefig(out / f"forms_{w}.png", dpi=110); plt.close(fig)
        log.info("sheet %s", out / f"forms_{w}.png")


def cmd_phonology(a):
    """pseudo-labels of every non-sealed isolated clip (sign span) → manifest/phonology_labels.npz + audit sheets (30 random clips per
    sub-space: the hold frame with the label written on it) → artifacts/reports/v11_sheets/phon_<space>.png"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from concurrent.futures import ProcessPoolExecutor
    from modules.phonology import LOC_NAMES, label_all
    from modules.sealed import sealed_signers
    c = pd.read_parquet(SHARDS / "clips.parquet")
    c = c[(c.kind == "isolated") & (c.view == "primary") & c.has_span & ~c.signer.isin(sealed_signers())].reset_index(drop=True)
    t0 = __import__("time").time()
    with ProcessPoolExecutor(a.workers) as ex:
        items = list(ex.map(_phon_one, zip(c.clip_id, c.span_f0, c.span_f1), chunksize=64))
    log.info("phonology measurements for %d clips (%.0f s), %d unmeasurable", len(items), __import__("time").time() - t0, sum(r is None for r in items))
    lab, km = label_all(items, (c.role == "train").values, k=50)
    np.savez(MAN / "phonology_labels.npz", clip=c.clip_id.values.astype(str), labels=lab, **{f"km_{k}": v for k, v in km.items()})
    for k, name in enumerate(["handshape", "orientation", "location", "movement"]):
        v = lab[:, k][lab[:, k] >= 0]
        log.info("  %s: %d classes used, largest share %.3f", name, len(np.unique(v)), np.bincount(v).max() / len(v))
    # audit sheets
    from scripts.train_encoder import PoseStoreMM, ensure_mm
    from modules.parts import to_iso
    from modules.schema import canonical_hands
    from modules.phonology import hold_index
    ensure_mm(); store = PoseStoreMM()
    out = ART / "reports" / "v11_sheets"; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    pick = rng.choice(np.flatnonzero(lab[:, 0] >= 0), 30, replace=False)
    for k, name in enumerate(["handshape", "orientation", "location", "movement"]):
        fig, axes = plt.subplots(5, 6, figsize=(11, 10))
        for ax, i in zip(axes.flat, pick):
            r = c.iloc[i]
            kp, sc, hw = store.get(store.row[r.clip_id]); kp, sc, _ = canonical_hands(kp, sc, hw); kp = to_iso(kp, hw)
            f0, f1 = int(r.span_f0), int(r.span_f1)
            t = hold_index(kp, sc, f0, max(f1, f0 + 4)) or (f0 + f1) // 2
            v = lab[i, k]
            txt = (LOC_NAMES[v] if v >= 0 else "-") if k == 2 else f"hs{v % 50}{'+2h' if v >= 50 else ''}" if k == 0 else                 f"palm{v // 6} dir{v % 6}" if k == 1 else f"mov{v // 3} int{v % 3}"
            draw_pose(ax, kp[t], sc[t], f"{txt} {r.concept[:10]}")
            if k == 3:
                w = kp[f0:f1, 10]; ax.plot(w[:, 0], -w[:, 1], "g-", lw=0.8)
        fig.suptitle(f"phonology audit: {name}"); fig.tight_layout(); fig.savefig(out / f"phon_{name}.png", dpi=90); plt.close(fig)
    log.info("audit sheets → %s", out)


def _phon_one(args):
    global _STORE
    from modules.parts import to_iso
    from modules.phonology import raw_features
    from modules.schema import canonical_hands
    if "_STORE" not in globals():
        from scripts.train_encoder import PoseStoreMM
        _STORE = PoseStoreMM()
    cid, f0, f1 = args
    if cid not in _STORE.row:
        return None
    kp, sc, hw = _STORE.get(_STORE.row[cid])
    kp, sc, _ = canonical_hands(kp, sc, hw)
    try:
        return raw_features(to_iso(kp, hw), sc, f0, f1)
    except Exception:  # noqa: BLE001
        return None


def cmd_model_build(a):
    """models/model-v11 from models/model-v10 + the shipped candidate (artifacts/v11/<cand>): v11 landmark encoder (onnx/lmenc.onnx) and its
    banks, spotting / calibration (bank v11, decision layer v11 block, verifier flag), verifier/ (ONNX + NumPy), bank_forms.npy; v10's
    encoder kept as onnx/lmenc_v10.onnx + lm_bank_v10.npz for THAISLM_V11=0 (every v11 track off = v10 exactly)."""
    import shutil
    src, cand, out = V10, ART / "v11" / a.cand, ROOT / "models" / "model-v11"
    if out.exists():
        bk = ROOT / "cache" / "model-v11-backup"
        if bk.exists():
            shutil.rmtree(bk)
        shutil.move(str(out), str(bk))
    shutil.copytree(src, out)
    shutil.copyfile(out / "onnx" / "lmenc.onnx", out / "onnx" / "lmenc_v10.onnx")
    shutil.copyfile(out / "lm_bank.npz", out / "lm_bank_v10.npz")
    for f in ("onnx/lmenc.onnx", "onnx/parity_lmenc.json", "lm_bank.npz", "spotting.json", "segment_calibration.json", "class_prior.npz"):
        shutil.copyfile(cand / f, out / f)
    shutil.copytree(cand / "verifier", out / "verifier")
    F = read_json(MAN / "forms.json")["clip_form"]
    b = np.load(out / "bank.npz")
    form = np.array([int(F.get(str(c).split("@")[0].split("#")[0], [None, "f0"])[1][1:]) for c in b["clip"]], np.int64)
    np.save(out / "bank_forms.npy", form)
    cfg = read_json(out / "config.json"); cfg.update(version="v11", v11=dict(candidate=a.cand, encoder="artifacts/runs/v11-final (3-seed ensemble of R3)",
                                                                           path="A", off_switch="THAISLM_V11=0"))
    write_json(out / "config.json", cfg)
    import os
    mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2 ** 20
    log.info("models/model-v11 built from %s (+ %s); %d bank rows with forms; %.0f MB on disk", src, cand, len(form), mb)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seal"); s.add_argument("--seed", type=int, default=11); s.add_argument("--force", action="store_true")
    sub.add_parser("tiers"); sub.add_parser("forms")
    mbp = sub.add_parser("model-build"); mbp.add_argument("--cand", default="v11-S0")
    ph = sub.add_parser("phonology"); ph.add_argument("--workers", type=int, default=6); ph.add_argument("--sheets_only", action="store_true")
    sh = sub.add_parser("sheets"); sh.add_argument("--words", default="")
    a = ap.parse_args()
    dict(seal=cmd_seal, tiers=cmd_tiers, forms=cmd_forms, sheets=cmd_sheets, phonology=cmd_phonology,
         **{"model-build": cmd_model_build})[a.cmd](a)


if __name__ == "__main__":
    main()
