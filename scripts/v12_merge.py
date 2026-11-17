"""v12 merge experiment: keep what v11 does better (unknown signs / non-signing not shown as words, fluent reading) without what it does
worse (fewer right words shown, TSL51). Selection on TUNE only (BLIND is spent); dev-test gate + real web test afterwards.

  python scripts/v12_merge.py prepare                       # candidate folders artifacts/v11/v11-m-* (ship only what differs from v10)
  python scripts/v12_merge.py ablate --tags v10,v11-m-r,v11-m-rb,v11      # which part causes what (TUNE + Tail + idle)
  python scripts/v12_merge.py fit --tag v11-m-r --out v11-m-r2            # two-stage decision: v10 calibration marks structure,
                                                                           # a P(correct) fitted for this reader decides acceptance
Reports → artifacts/reports/v12_merge_*.json
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from decision import (ART, REP, TUNE_SETS, TEST_SETS, V10, V11, Harness, _labels, _p, _raw, aggregate, cache_path, class_prior, cv_logloss,
                      fit_lr, load_set, score, _isotonic_cv, ece)
from modules.decode import FEATS, FEATS_V8, FEATS_V81, FEATS_V9, candidate_probs, class_features, seg_features, wer
from modules.utils import read_json, write_json

EXTRA = ["tail_tune", "idle_tune"]
WEB_TUNE = ["w_t51_tune", "w_iso_tune"]
WEB_EXTRA = ["w_tail_tune", "w_idle_tune"]


def prepare(a):
    """v11-m-r  = v10 + the v11 landmark encoder (reader) and its banks, v10 calibration
       v11-m-rb = v11-m-r + v11 form-prototype bank scoring for the reader (spotting.json bank_v11, λ 0.08)"""
    for name, bank_v11 in (("v11-m-r", False), ("v11-m-rb", True)):
        d = ART / "v11" / name
        (d / "onnx").mkdir(parents=True, exist_ok=True)
        shutil.copy2(V11 / "onnx" / "lmenc.onnx", d / "onnx" / "lmenc.onnx")
        shutil.copy2(V11 / "lm_bank.npz", d / "lm_bank.npz")
        ev = ART / "eval" / name; ev.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ART / "eval" / "v11" / "lm_bank_eval.npz", ev / "lm_bank_eval.npz")
        if bank_v11:
            sp = read_json(V10 / "spotting.json"); sp["bank_v11"] = read_json(V11 / "spotting.json")["bank_v11"]
            write_json(d / "spotting.json", sp)
            shutil.copy2(V11 / "bank_forms.npy", d / "bank_forms.npy")
        write_json(d / "config.json", dict(version=name, note="v12 merge candidate (scripts/v12_merge.py); files absent here = model-v10"))
        _p("prepared %s", d)


WEB_FPS = 13.0          # what the browser's MediaPipe Holistic reaches on this laptop in the real web test (result_v11.md §8.1)


def _video_of(r, S3):
    if isinstance(getattr(r, "source_file", None), str) and (S3 / r.source_file).is_file():
        return S3 / r.source_file
    if str(r.clip_id).startswith("agent:yt:"):
        p = S3 / "agent_data/youtube_words/videos" / f"{r.clip_id[9:]}.mp4"
        return p if p.exists() else None
    return None


def web_items(name):
    """Web-domain sets (prompt: calibrate for the domain the product runs in, without touching BLIND / data_test): the SOURCE VIDEOS of
    the TUNE (val) or dev-test (test) people re-extracted with MediaPipe Holistic — the model file the browser runs — at the browser's
    frame rate (13 fps). No gold spans (real-video scoring: multiset).
      w_t51_<tune|test>   TSL51 sentence videos of the tune / test templates (sealed BLIND templates excluded)
      w_iso_<tune|test>   isolated conversation-word clips of held-out people with a video (≤ 2 per person × word, ≤ 260)
      w_tail_<tune|test>  isolated clips of words OUTSIDE the vocabulary (known unknowns, ≤ 120): right answer = nothing accepted
      w_idle_<tune|test>  TSL51 hands-up-but-not-signing videos (null_*)"""
    import pandas as pd
    from modules.mediapipe_pose import extract, load_raw, save_raw
    from modules.sealed import is_sealed, sealed_signers
    S3 = ROOT / "cloud_s3"
    _, kind, split = name.split("_")
    role = "val" if split == "tune" else "test"
    cache = ROOT / "cache" / "v12_web"; cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(12)

    def raw_of(video, key):
        f = cache / f"{key}.npz"
        if not f.exists():
            r = extract(video, fps=WEB_FPS, normalize=False)
            if r is None:
                return None
            save_raw(f, r)
        return load_raw(f)

    out = []
    if kind == "t51":
        A = read_json(ART / "runs" / "seq-v7" / "tsl51_alignment.json")
        for cid, a in A.items():
            if not cid.startswith("t51s:") or ("tune" if a["role"] == "tagger_train" else "test") != split or is_sealed(cid):
                continue
            v = S3 / "raw_data/TSL51/videos/user_sentence" / f"{cid.split(':', 1)[1]}.mp4"
            if v.exists():
                out.append(dict(id=f"w:{cid}", raw=raw_of(v, cid.replace(":", "_")), ref=list(a["glosses"]), gold=None, bank="eval", signer="S_51_user"))
        return [o for o in out if o["raw"] is not None]
    C = pd.read_parquet(S3 / "data_prep/shards/clips.parquet")
    M = pd.read_parquet(S3 / "data_prep/manifest/clips.parquet")[["clip_id", "source_file"]]
    if kind == "idle":       # TSL51 user hands-up-not-signing videos (the expert null clips have no video); the 8 played in the v11 web
        pool = ROOT / "artifacts/reports/webtest_v11_pool.json"          # test are left out, the rest alternate tune / test
        used = {p["src"].replace("\\", "/").split("/")[-1] for p in read_json(pool) if p["cat"] == "idle"} if pool.exists() else set()
        vids = [v for v in sorted((S3 / "raw_data/TSL51/videos/user_sign").glob("null_*.mp4")) if v.name not in used]
        for i, v in enumerate(vids):
            if (i % 2 == 0) == (split == "tune"):
                out.append(dict(id=f"w:t51u_null:{v.stem}", raw=raw_of(v, "t51u_" + v.stem), ref=[], gold=None, bank="eval", signer="S_51_user"))
        return [o for o in out if o["raw"] is not None]
    V = set(read_json(V10 / "vocab_conversation.json")["concepts"])
    q = C[(C.kind == "isolated") & (C.view == "primary") & (C.role == role) & C.concept.notna() & ~C.signer.isin(sealed_signers())]
    q = q[~q.clip_id.map(is_sealed)].merge(M, on="clip_id", how="left")
    q = q.assign(video=[_video_of(r, S3) for r in q.itertuples()])
    q = q[q.video.notna()]
    if kind == "iso":
        q = q[q.concept.isin(V)].groupby(["signer", "concept"], group_keys=False)[q.columns.tolist()].apply(lambda g: g.head(2))
        q = q.sample(min(260, len(q)), random_state=12)
    else:
        q = q[~q.concept.isin(V)].drop_duplicates(["signer", "concept"]).sample(frac=1.0, random_state=12).head(120)
    for r in q.itertuples():
        out.append(dict(id=f"w:{r.clip_id}", raw=raw_of(r.video, r.clip_id.replace(":", "_").replace("/", "_")),
                        ref=[r.concept] if kind == "iso" else [], gold=None, bank="eval", signer=r.signer, dataset=r.dataset))
    return [o for o in out if o["raw"] is not None]


def displayed(segs, ref):
    return wer(ref, [s["nearest"] for s in segs if s["status"] == "accepted"])


def eval_raw(h, raw):
    """decide() on copies of the decoded segments → per set: aggregate + displayed WER; Tail: share of clips with ≥ 1 accepted word;
    idle: accepted words per 10 s"""
    out = {}
    for name, (rows, raws) in raw.items():
        dec = [h.pipe.decide(copy.deepcopy(r)) for r in raws]
        if "tail" in name:
            out[name] = dict(n=len(rows), tail_false_accept=float(np.mean([any(s["status"] == "accepted" for s in d) for d in dec])))
            continue
        if "idle" in name:
            sec = sum(r["T"] for r in rows) / 25.0
            out[name] = dict(n=len(rows), idle_words_per_10s=10 * sum(sum(s["status"] == "accepted" for s in d) for d in dec) / max(sec, 1e-6))
            continue
        sc = [score(d, row, h.vocab) for d, row in zip(dec, rows)]
        m = aggregate(sc)
        m["displayed_WER"] = float(np.mean([displayed(d, row["ref"]) for d, row in zip(dec, rows)]))
        m["_wer"] = [x["wer"] for x in sc]
        m["_dwer"] = [displayed(d, row["ref"]) for d, row in zip(dec, rows)]
        out[name] = m
    return out


def summary(res):
    keys = ("WER", "displayed_WER", "acc_cover", "acc_prec", "false_accept_rate")
    s = {n: {k: round(v[k], 4) for k in keys} for n, v in res.items() if "WER" in v}
    s.update({n: {k: round(x, 4) for k, x in v.items() if k != "n"} for n, v in res.items() if "WER" not in v})
    tune = [n for n in TUNE_SETS if n in res]
    if tune:
        s["mean"] = {k: round(float(np.mean([res[n][k] for n in tune])), 4) for k in ("WER", "displayed_WER", "acc_cover")}
    web = [n for n in res if n.startswith("w_") and "WER" in res[n]]
    if web:
        s["web_mean"] = {k: round(float(np.mean([res[n][k] for n in web])), 4) for k in ("WER", "displayed_WER", "acc_cover")}
    return s


def load_rows(names):
    return {n: load_set(n) for n in names if cache_path(n).exists()}


def ablate(a):
    h = Harness()
    rows = load_rows(a.sets.split(",") if a.sets else TUNE_SETS + EXTRA + WEB_TUNE + WEB_EXTRA)
    out = {}
    for tag in a.tags.split(","):
        t = time.time()
        h.use(tag)
        raw = _raw(h, list(rows), rows)
        res = eval_raw(h, raw)
        out[tag] = summary(res)
        _p("%-10s %s (%.0f s)", tag, json.dumps({k: out[tag].get(k) for k in ("mean", "web_mean")}), time.time() - t)
        for n in rows:
            _p("    %-14s %s", n, out[tag][n])
    write_json(REP / f"v12_merge_ablate{a.suffix}.json", out)


def fit(a):
    """Two-stage decision for --tag: features → LR (feature set by CV log-loss, set-balanced weights, isotonic if CV ECE drops) on the
    TUNE segments decoded with this reader; τ_accept = the lowest τ ≥ 0.80 whose accepted precision and false-accept rate on EVERY
    TUNE set are at least as good as v10's (± 0.005) — v10's own release rule; τ_uncertain = largest τ whose rejected words are
    ≥ 85 % wrong (≥ 80 % per set). Writes artifacts/v11/<out>/ (= the tag's folder + segment_calibration.json with an "accept" block)."""
    h = Harness()
    FS = a.sets.split(",")                                  # sets the accept stage is fitted and its τ checked on
    rows = load_rows(FS + EXTRA + WEB_EXTRA)
    FS = [n for n in FS if n in rows]
    h.use("v10")
    ref = eval_raw(h, _raw(h, list(rows), rows))
    b = h.use(a.tag)
    raw = _raw(h, list(rows), rows)
    base = eval_raw(h, raw)
    allowed = {h.pipe.concepts.index(c) for c in h.vocab}
    prior = class_prior(h.bank_files[0], h.pipe.concepts, allowed, h.fold)
    T = h.pipe.cal.get("temperature")
    S, Y, W, G = [], [], [], []
    for n in FS:
        segs, y, _ = _labels(h, *copy.deepcopy(raw[n]))
        if T:
            candidate_probs(segs, T)
        class_features(segs, h.pipe.concepts, prior)
        S += segs; Y.append(y); W.append(np.full(len(y), 1.0 / max(len(y), 1)))
        G += [f"{n}|{i}" for i, r in enumerate(raw[n][1]) for _ in r]
    Y, W, G = np.concatenate(Y), np.concatenate(W), np.array(G)
    setname = np.array([g.split("|")[0] for g in G])
    fsets = {"v7": FEATS, "v8": FEATS_V8, "v8.1": FEATS_V81, "v9": FEATS_V9}
    cvr = {k: cv_logloss(seg_features(S, f), Y, W, G)[:2] for k, f in fsets.items()}
    best = min(cvr, key=lambda k: cvr[k][0])
    lr = fit_lr(seg_features(S, fsets[best]), Y, W)
    _, e_lr, P_oof = cv_logloss(seg_features(S, fsets[best]), Y, W, G)
    iso_c = _isotonic_cv(P_oof, Y, G)
    iso = iso_c["table"] if iso_c["report"]["cv_ECE"] < e_lr and iso_c["report"]["cv_logloss"] <= cvr[best][0] + 0.005 else None
    _p("[fit] %d segments · features %s (CV logloss %s) · isotonic %s", len(Y), best, {k: round(v[0], 4) for k, v in cvr.items()}, bool(iso))
    P_old = h.pipe.lr.predict(seg_features(S, getattr(h.pipe.lr, "feats", FEATS)))
    model = lr.to_dict(fsets[best])
    P = np.interp(lr.predict(seg_features(S, fsets[best])), iso["x"], iso["y"]) if iso else lr.predict(seg_features(S, fsets[best]))
    rep = dict(tag=a.tag, features=best, cv=cvr, isotonic=bool(iso), ECE_shipped_cal=ece(P_old, Y), ECE_new_in_sample=ece(P, Y),
               ECE_new_oof=e_lr, n=int(len(Y)), taus=[])
    # τ_uncertain on the fitted P (rejected words must really be wrong)
    tau_rej = 0.05
    for t in np.round(np.arange(0.05, 0.5, 0.01), 2):
        rej = P < t
        if rej.sum() and (1 - Y[rej].mean()) >= 0.85 and all((1 - Y[rej & (setname == n)].mean()) >= 0.8 for n in FS
                                                               if (rej & (setname == n)).sum() >= 10):
            tau_rej = float(t)
    cal0 = h.pipe.cal
    chosen = None
    for tau in np.round(np.arange(0.80, 0.98, 0.01), 2):
        h.configure(cal=dict(cal0, accept=dict(model=model, isotonic=iso, tau_accept=float(tau), tau_uncertain=tau_rej)))
        r = eval_raw(h, {n: raw[n] for n in FS + [x for x in rows if "tail" in x]})
        fine = all(r[n]["false_accept_rate"] <= ref[n]["false_accept_rate"] + 0.005 and r[n]["acc_prec"] >= ref[n]["acc_prec"] - 0.005 for n in FS)             and all(r[n]["tail_false_accept"] <= ref[n]["tail_false_accept"] + 0.005 for n in r if "tail" in n)
        sm = summary(r)
        rep["taus"].append(dict(tau=float(tau), ok=fine, **{k: sm.get(k) for k in ("mean", "web_mean")}, tail={n: sm[n] for n in sm if "tail" in n}))
        _p("[tau %.2f] %s %s", tau, "OK" if fine else "—", {k: sm.get(k) for k in ("mean", "web_mean")} | {n: sm[n] for n in sm if "tail" in n})
        if fine:
            chosen = float(tau); break
    if chosen is None:
        _p("[fit] no τ keeps v10's precision / false-accept on every TUNE set → no accept stage"); rep["chosen"] = None
        write_json(REP / f"v12_merge_fit_{a.out}.json", rep); return
    acc = dict(model=model, isotonic=iso, tau_accept=chosen, tau_uncertain=tau_rej, fitted_on=FS, rule="v10 release rule vs v10 (±0.005) + Tail FA ≤ v10",
               reader=a.tag)
    h.configure(cal=dict(cal0, accept=acc))
    fin = eval_raw(h, raw)
    rep.update(chosen=chosen, tau_uncertain=tau_rej, reference_v10=summary(ref), before=summary(base), after=summary(fin))
    src = b["dir"]; d = ART / "v11" / a.out
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(src, d)
    if (ART / "eval" / a.tag / "lm_bank_eval.npz").exists():
        (ART / "eval" / a.out).mkdir(parents=True, exist_ok=True)
        shutil.copy2(ART / "eval" / a.tag / "lm_bank_eval.npz", ART / "eval" / a.out / "lm_bank_eval.npz")
    full = read_json(src / "segment_calibration.json") if (src / "segment_calibration.json").exists() else read_json(V10 / "segment_calibration.json")
    full["conversation"]["accept"] = acc
    write_json(d / "segment_calibration.json", full)
    write_json(REP / f"v12_merge_fit_{a.out}.json", rep)
    for nm, x in (("v10", ref), (a.tag + " before", base), (a.tag + " after", fin)):
        sm = summary(x); _p("[fit] %-22s %s", nm, {k: sm.get(k) for k in ("mean", "web_mean")} | {n: sm[n] for n in sm if "tail" in n or "idle" in n})
    _p("[fit] → %s", d)


def attach_lm2(h, d):
    """Harness support for the v12 dual reader: the second landmark encoder + its banks (eval bank: artifacts/eval/v11, BLIND rows out)."""
    from modules.lmenc import OnnxLM
    from modules.pipeline import lm_bank
    o, prod, ev = d / "onnx" / "lmenc2.onnx", d / "lm2_bank.npz", ART / "eval" / "v11" / "lm_bank_eval.npz"
    if not o.exists():
        h.pipe.lm2 = None; return
    if getattr(h, "_lm2_dir", None) != d:
        h.pipe.lm2 = OnnxLM(o, device="cpu"); h._lm2_dir = d
    cidx = {c: i for i, c in enumerate(h.pipe.concepts)}
    allowed = {cidx[w] for w in h.vocab_words if w in cidx}
    for name, f in (("eval", ev), ("prod", prod)):
        h.banks[name].lm2 = lm_bank(f, len(h.pipe.concepts), allowed, h.fold, keep=h.banks[name].keep)
    h.pipe.bank = h.banks["eval"]


def dual(a):
    """v11-m-d = v10 + the v11 encoder as a SECOND landmark reader (lmenc2) — grid of (v10 lm_w, v11 lm2_w) on offline + web TUNE,
    v10 calibration and everything else unchanged."""
    d = ART / "v11" / "v11-m-d"; (d / "onnx").mkdir(parents=True, exist_ok=True)
    shutil.copy2(V11 / "onnx" / "lmenc.onnx", d / "onnx" / "lmenc2.onnx"); shutil.copy2(V11 / "lm_bank.npz", d / "lm2_bank.npz")
    write_json(d / "config.json", dict(version="v11-m-d", note="v12 dual reader candidate: v10 + v11 landmark encoder as lmenc2"))
    h = Harness()
    rows = load_rows(TUNE_SETS + EXTRA + WEB_TUNE + WEB_EXTRA)
    h.use("v11-m-d"); attach_lm2(h, d)
    spot0 = copy.deepcopy(h.pipe.spot)
    out = {}
    for lw, lw2 in [(float(x), float(y)) for x, y in (p.split(":") for p in a.grid.split(","))]:
        t = time.time()
        h.configure(spot=dict(spot0, read=dict(spot0.get("read", {}), lm_w=lw, lm2_w=lw2)))
        res = eval_raw(h, _raw(h, list(rows), rows))
        sm = summary(res); out[f"{lw}:{lw2}"] = sm
        _p("[dual lm_w %.2f lm2_w %.2f] %s (%.0f s)", lw, lw2, {k: sm.get(k) for k in ("mean", "web_mean")} |
           {n: sm[n] for n in sm if "tail" in n or "idle" in n}, time.time() - t)
        _p("      %s", {n: (sm[n]["WER"], sm[n]["displayed_WER"]) for n in sm if isinstance(sm[n], dict) and "WER" in sm[n]})
    write_json(REP / "v12_merge_dual.json", out)


def dual_tau(a):
    """For each dual-reader mix: keep v10's calibration model, re-choose only τ_accept = the lowest τ ≥ 0.80 such that on EVERY word set
    (offline + web TUNE) accepted precision ≥ v10 − 0.005 and false-accept ≤ v10 + 0.005, Tail false accept ≤ v10 + 0.005 (both
    domains) and idle words ≤ v10. Reports the chosen τ and its metrics → v12_merge_dual_tau.json"""
    d = ART / "v11" / "v11-m-d"
    h = Harness()
    rows = load_rows(TUNE_SETS + EXTRA + WEB_TUNE + WEB_EXTRA)
    WS = [n for n in TUNE_SETS + WEB_TUNE if n in rows]
    h.use("v10"); ref = eval_raw(h, _raw(h, list(rows), rows))
    h.use("v11-m-d"); attach_lm2(h, d)
    spot0, cal0 = copy.deepcopy(h.pipe.spot), copy.deepcopy(h.pipe.cal)
    out = dict(reference_v10=summary(ref), mixes={})
    for lw, lw2 in [(float(x), float(y)) for x, y in (p.split(":") for p in a.grid.split(","))]:
        h.configure(spot=dict(spot0, read=dict(spot0.get("read", {}), lm_w=lw, lm2_w=lw2)), cal=cal0)
        raw = _raw(h, list(rows), rows)
        rec = []
        for tau in np.round(np.arange(0.80, 0.97, 0.01), 2):
            h.configure(cal=dict(cal0, tau_accept=float(tau)))
            r = eval_raw(h, raw)
            ok = all(r[n]["acc_prec"] >= ref[n]["acc_prec"] - 0.005 and r[n]["false_accept_rate"] <= ref[n]["false_accept_rate"] + 0.005 for n in WS)                 and all(r[n]["tail_false_accept"] <= ref[n]["tail_false_accept"] + 0.005 for n in r if "tail" in n)                 and all(r[n]["idle_words_per_10s"] <= ref[n]["idle_words_per_10s"] + 1e-9 for n in r if "idle" in n)
            sm = summary(r)
            rec.append(dict(tau=float(tau), ok=ok, **{k: sm.get(k) for k in ("mean", "web_mean")}, neg={n: sm[n] for n in sm if "tail" in n or "idle" in n},
                            per_set={n: sm[n] for n in WS}))
            if ok:
                break
        ch = rec[-1] if rec[-1]["ok"] else None
        out["mixes"][f"{lw}:{lw2}"] = dict(chosen=ch, scanned=rec)
        _p("[dual %.2f:%.2f] τ %s → %s", lw, lw2, ch and ch["tau"], ch and ({k: ch.get(k) for k in ("mean", "web_mean")} | ch["neg"]))
        if ch:
            _p("      per set %s", {n: (v["WER"], v["displayed_WER"]) for n, v in ch["per_set"].items()})
    _p("v10 %s", {k: summary(ref).get(k) for k in ("mean", "web_mean")})
    write_json(REP / "v12_merge_dual_tau.json", out)


def curves(a):
    """Precision–coverage curves (pooled over the word sets) for v10 and candidates, τ_accept swept on v10's calibration model:
    a candidate is genuinely better only if, at a τ where its pooled accepted precision and Tail / idle false accepts are no worse than
    v10 AS SHIPPED, it shows more right words (coverage) — per domain (offline TUNE, web TUNE)."""
    d = ART / "v11" / "v11-m-d"
    h = Harness()
    rows = load_rows(TUNE_SETS + EXTRA + WEB_TUNE + WEB_EXTRA)
    doms = {"offline": [n for n in TUNE_SETS if n in rows], "web": [n for n in WEB_TUNE if n in rows]}
    negs = {"offline": ["tail_tune", "idle_tune"], "web": ["w_tail_tune", "w_idle_tune"]}
    cands = [("v10", "v10", None)] + [(f"dual {m}", "v11-m-d", tuple(float(x) for x in m.split(":"))) for m in a.mixes.split(",")] + [("v11-m-r", "v11-m-r", None)]
    out = {}
    for name, tag, mix in cands:
        h.use(tag)
        if tag == "v11-m-d":
            attach_lm2(h, d)
        spot0, cal0 = copy.deepcopy(h.pipe.spot), copy.deepcopy(h.pipe.cal)
        if mix:
            h.configure(spot=dict(spot0, read=dict(spot0.get("read", {}), lm_w=mix[0], lm2_w=mix[1])))
        raw = _raw(h, list(rows), rows)
        pts = []
        for tau in np.round(np.arange(0.70, 0.97, 0.01), 2):
            h.configure(cal=dict(cal0, tau_accept=float(tau)))
            r = eval_raw(h, raw)
            pt = dict(tau=float(tau))
            for dn, sets in doms.items():
                acc = sum(r[n]["acc"] for n in sets); ok = sum(r[n]["acc_ok"] for n in sets); ref_n = sum(r[n]["words_ref"] for n in sets)
                pt[dn] = dict(prec=ok / max(acc, 1), cover=ok / max(ref_n, 1), wer=float(np.mean([r[n]["WER"] for n in sets])),
                              dwer=float(np.mean([r[n]["displayed_WER"] for n in sets])),
                              tail=r[negs[dn][0]]["tail_false_accept"], idle=r[negs[dn][1]]["idle_words_per_10s"])
            pts.append(pt)
        out[name] = pts
        _p("%s done", name)
    v10 = next(p for p in out["v10"] if abs(p["tau"] - 0.80) < 1e-9)
    res = {}
    for name, pts in out.items():
        best = {}
        for dn in doms:
            feas = [p for p in pts if p[dn]["prec"] >= v10[dn]["prec"] - 1e-9 and p[dn]["tail"] <= v10[dn]["tail"] + 1e-9 and p[dn]["idle"] <= v10[dn]["idle"] + 1e-9]
            best[dn] = max(feas, key=lambda p: p[dn]["cover"])[dn] | {"tau": max(feas, key=lambda p: p[dn]["cover"])["tau"]} if feas else None
        res[name] = best
        _p("%-14s offline %s | web %s", name, best["offline"] and {k: round(v, 4) for k, v in best["offline"].items()},
           best["web"] and {k: round(v, 4) for k, v in best["web"].items()})
    _p("v10 shipped (τ 0.80): offline %s | web %s", {k: round(v, 4) for k, v in v10["offline"].items()}, {k: round(v, 4) for k, v in v10["web"].items()})
    write_json(REP / "v12_merge_curves.json", dict(points=out, matched=res, v10_shipped=v10))


def check(a):
    """Report only (no selection): v10 vs the TUNE-chosen merge candidates on dev-test (offline + web-domain test sets), shipped τ,
    with per-utterance paired bootstrap of top-1 / displayed WER per set → v12_merge_check.json"""
    d = ART / "v11" / "v11-m-d"
    h = Harness()
    sets = TEST_SETS + ["tail_test", "idle_test", "w_t51_test", "w_iso_test", "w_tail_test", "w_idle_test"]
    rows = load_rows(sets)
    res = {}
    for name, tag, mix in [("v10", "v10", None), ("dual 0.3:0.15", "v11-m-d", (0.3, 0.15)), ("v11-m-r", "v11-m-r", None)]:
        h.use(tag)
        if tag == "v11-m-d":
            attach_lm2(h, d)
            sp = copy.deepcopy(h.pipe.spot); h.configure(spot=dict(sp, read=dict(sp.get("read", {}), lm_w=mix[0], lm2_w=mix[1])))
        res[name] = eval_raw(h, _raw(h, list(rows), rows))
        _p("%-14s %s", name, {n: (round(v["WER"], 4), round(v["displayed_WER"], 4)) if "WER" in v else {k: round(x, 4) for k, x in v.items() if k != "n"}
                              for n, v in res[name].items()})
    rng = np.random.RandomState(0); paired = {}
    for name in res:
        if name == "v10":
            continue
        paired[name] = {}
        for n, v in res[name].items():
            if "_wer" not in v:
                continue
            for k in ("_wer", "_dwer"):
                dd = np.asarray(v[k]) - np.asarray(res["v10"][n][k])
                bs = [dd[rng.randint(0, len(dd), len(dd))].mean() for _ in range(2000)]
                paired[name].setdefault(n, {})[k[1:]] = dict(diff=round(float(dd.mean()), 4), ci=[round(float(np.percentile(bs, q)), 4) for q in (2.5, 97.5)])
        _p("paired %s %s", name, paired[name])
    write_json(REP / "v12_merge_check.json", dict(results={k: summary(v) for k, v in res.items()}, paired=paired))


def guard(a):
    """Tail guard on top of a reader mix: accepted words with tail_gap < g → "[?]". Grid of g on offline + web TUNE at v10's τ; chosen =
    the g with the most right words shown (coverage, both domains summed) whose Tail false accepts (both domains) are ≤ v10's and whose
    pooled accepted precision (both domains) is ≥ v10's. → v12_merge_guard.json"""
    d = ART / "v11" / "v11-m-d"
    h = Harness()
    rows = load_rows(TUNE_SETS + EXTRA + WEB_TUNE + WEB_EXTRA)
    doms = {"offline": [n for n in TUNE_SETS if n in rows], "web": [n for n in WEB_TUNE if n in rows]}
    negs = {"offline": "tail_tune", "web": "w_tail_tune"}

    def pooled(r):
        o = {}
        for dn, sets in doms.items():
            acc = sum(r[n]["acc"] for n in sets); ok = sum(r[n]["acc_ok"] for n in sets); nr = sum(r[n]["words_ref"] for n in sets)
            o[dn] = dict(prec=ok / max(acc, 1), cover=ok / max(nr, 1), wer=float(np.mean([r[n]["WER"] for n in sets])),
                         dwer=float(np.mean([r[n]["displayed_WER"] for n in sets])), tail=r[negs[dn]]["tail_false_accept"])
        return o
    h.use("v10"); cal_v10 = copy.deepcopy(h.pipe.cal)
    raw10 = _raw(h, list(rows), rows)
    out = dict(v10={}, mixes={})
    for g in [None] + [float(x) for x in a.gaps.split(",")]:
        h.configure(cal=dict(cal_v10, **({"tail_guard": dict(min_gap=g)} if g is not None else {})))
        out["v10"][str(g)] = pooled(eval_raw(h, raw10))
    ref = out["v10"]["None"]
    _p("v10 %s", {dn: {k: round(v, 4) for k, v in x.items()} for dn, x in ref.items()})
    for m in a.mixes.split(","):
        lw, lw2 = (float(x) for x in m.split(":"))
        h.use("v11-m-d"); attach_lm2(h, d)
        sp = copy.deepcopy(h.pipe.spot); cal0 = copy.deepcopy(h.pipe.cal)
        h.configure(spot=dict(sp, read=dict(sp.get("read", {}), lm_w=lw, lm2_w=lw2)))
        raw = _raw(h, list(rows), rows)
        rec = {}
        for g in [None] + [float(x) for x in a.gaps.split(",")]:
            h.configure(cal=dict(cal0, **({"tail_guard": dict(min_gap=g)} if g is not None else {})))
            rec[str(g)] = pooled(eval_raw(h, raw))
            ok = all(rec[str(g)][dn]["tail"] <= ref[dn]["tail"] + 1e-9 and rec[str(g)][dn]["prec"] >= ref[dn]["prec"] - 1e-9 for dn in doms)
            _p("[%s g=%s] %s %s", m, g, "OK" if ok else "—", {dn: {k: round(v, 4) for k, v in x.items()} for dn, x in rec[str(g)].items()})
        out["mixes"][m] = rec
    for g, r in out["v10"].items():
        _p("[v10 g=%s] %s", g, {dn: {k: round(v, 4) for k, v in x.items()} for dn, x in r.items()})
    write_json(REP / "v12_merge_guard.json", out)


def build(a):
    """The TUNE-chosen merge → a candidate folder (harness: artifacts/v11/<name>) and, with --full, a complete model folder for the web
    stack (models/model-v10 + lmenc2 + lm2_bank + read.lm2_w + τ_accept)."""
    sp = read_json(V10 / "spotting.json"); sp["conversation"]["read"] = dict(sp["conversation"].get("read", {}), lm2_w=a.lm2_w)
    cal = read_json(V10 / "segment_calibration.json"); cal["conversation"]["tau_accept"] = a.tau
    cal["conversation"]["v12"] = dict(rule="TUNE (offline + web-domain): most right words shown with pooled accepted precision ≥ v10 − 0.005 and "
                                           "Tail false accept ≤ v10 + 0.005 in both domains (scripts/v12_merge.py curves)", tau_accept_v10=read_json(V10 / "segment_calibration.json")["conversation"]["tau_accept"])
    for d in [ART / "v11" / a.name] + ([Path(a.full)] if a.full else []):
        if a.full and d == Path(a.full):
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(V10, d)
        (d / "onnx").mkdir(parents=True, exist_ok=True)
        shutil.copy2(V11 / "onnx" / "lmenc.onnx", d / "onnx" / "lmenc2.onnx"); shutil.copy2(V11 / "lm_bank.npz", d / "lm2_bank.npz")
        write_json(d / "spotting.json", sp); write_json(d / "segment_calibration.json", cal)
        cfg = read_json(V10 / "config.json"); cfg.update(version="v12", based_on="model-v10", second_reader="v11-final landmark encoder (lmenc2)",
                                                          note="v12 merge: v10 + the v11 encoder as a second reader (read.lm2_w) + τ_accept re-chosen")
        write_json(d / "config.json", cfg)
        _p("built %s", d)


def check2(a):
    """Report: v10 vs candidate folders on dev-test (offline + web-domain), paired bootstrap per set → v12_merge_check2.json"""
    h = Harness()
    sets = TEST_SETS + ["tail_test", "idle_test", "w_t51_test", "w_iso_test", "w_tail_test", "w_idle_test"]
    rows = load_rows(sets)
    res = {}
    for tag in a.cands.split(","):
        h.use(tag)
        attach_lm2(h, ART / "v11" / tag) if tag != "v10" else attach_lm2(h, ART / "v11" / "__none__")
        res[tag] = eval_raw(h, _raw(h, list(rows), rows))
    base = a.cands.split(",")[0]
    rng = np.random.RandomState(0); paired = {}
    for tag in res:
        if tag == base:
            continue
        for n, v in res[tag].items():
            if "_wer" not in v:
                continue
            for k in ("_wer", "_dwer"):
                dd = np.asarray(v[k]) - np.asarray(res[base][n][k])
                bs = [dd[rng.randint(0, len(dd), len(dd))].mean() for _ in range(2000)]
                paired.setdefault(tag, {}).setdefault(n, {})[k[1:]] = dict(diff=round(float(dd.mean()), 4), ci=[round(float(np.percentile(bs, q)), 4) for q in (2.5, 97.5)])
    for tag, r in res.items():
        _p("%-12s %s", tag, {n: {k: round(v[k], 4) for k in ("WER", "displayed_WER", "acc_cover", "acc_prec", "false_accept_rate")} if "WER" in v else
                              {k: round(x, 4) for k, x in v.items() if k != "n"} for n, v in r.items()})
    for tag, pp in paired.items():
        for n, x in pp.items():
            _p("paired %s %-14s top1 %+.4f %s | displayed %+.4f %s", tag, n, x["wer"]["diff"], x["wer"]["ci"], x["dwer"]["diff"], x["dwer"]["ci"])
    write_json(REP / "v12_merge_check2.json", dict(results={k: summary(v) for k, v in res.items()}, paired=paired))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    ab = sub.add_parser("ablate"); ab.add_argument("--tags", default="v10,v11-m-r,v11-m-rb,v11"); ab.add_argument("--suffix", default="")
    ab.add_argument("--sets", default="")
    f = sub.add_parser("fit"); f.add_argument("--tag", required=True); f.add_argument("--out", required=True)
    f.add_argument("--sets", default=",".join(TUNE_SETS + WEB_TUNE))
    du = sub.add_parser("dual"); du.add_argument("--grid", default="0.3:0,0:0.3,0.15:0.15,0.3:0.15,0.2:0.2,0.3:0.3,0.15:0.3")
    dt = sub.add_parser("dual_tau"); dt.add_argument("--grid", default="0.3:0.15,0.2:0.2,0.3:0.3,0.15:0.3,0.0:0.3")
    cu = sub.add_parser("curves"); cu.add_argument("--mixes", default="0.3:0.15,0.2:0.2,0.15:0.3")
    sub.add_parser("check")
    bu = sub.add_parser("build"); bu.add_argument("--name", default="v11-m-d89"); bu.add_argument("--lm2_w", type=float, default=0.15)
    bu.add_argument("--tau", type=float, default=0.89); bu.add_argument("--full", default="")
    c2 = sub.add_parser("check2"); c2.add_argument("--cands", default="v10,v11-m-d89")
    gu = sub.add_parser("guard"); gu.add_argument("--mixes", default="0.3:0.15,0.2:0.2"); gu.add_argument("--gaps", default="-0.05,0.0,0.02,0.04,0.06,0.08,0.1")
    a = ap.parse_args()
    dict(prepare=prepare, ablate=ablate, fit=fit, dual=dual, dual_tau=dual_tau, curves=curves, check=check, guard=guard, build=build, check2=check2)[a.cmd](a)


if __name__ == "__main__":
    main()
