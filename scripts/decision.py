"""Forward decision layer (v8 → v8.1 → v9): reader (hand-shape cue, second recogniser) · segment merge / split · calibration ·
Accept / Uncertain / Reject · fingerspelling settings, all tuned on TUNE data only, starting from a frozen version
(artifacts/baselines/<v8|v8.1>/). v9 writes models/model-v9/ (V9).

  python main.py decision cache [--sets a,b]  # front pass (encoder + tagger) of every tune / test / real utterance → cache/decision/*.pkl
  python main.py decision hands               # hand-shape descriptors of every bank clip + metric (training signers) → hand_bank.npz
  python main.py decision lmbank [--out dir]  # v9 second recogniser: landmark encoder → <out>/onnx/lmenc.onnx + lm_bank.npz (+ eval bank)
  python main.py decision tune --start v8.1 [--grammar off] [--spot_from spotting.json] [--write]
                                              # reader · segmentation · calibration (+ isotonic) · thresholds · spelling on the TUNE sets →
                                              #   models/model-v9/{spotting.json, segment_calibration.json, class_prior.npz} + decision_tune_v9.json
  python main.py decision report [--versions v8.1,v9]   # TEST sets + data_test + iPhone + team-drive (+ enrolled signer, per-performance /
                                              #   in-vocabulary WER, bootstrap, latency) → artifacts/reports/decision_<tag>.json
  python main.py decision figures [--tag v8.1]  # result_reporting/figures_<tag>/*.png

Sets (bank = training signers only, artifacts/eval/bank_eval.npz, except the real videos which use models/bank.npz):
  t51   TSL51 sentences (one researcher signer) — tune = tagger_train templates, test = sentence_test templates; speeds 1× / 1.5× / 2×
  iso   whole isolated clips (rest → sign → rest) of held-out signers, conversation words — tune = val signers, test = test signers.
        One clip = one sign: extra segments are the "แมว → กิน แมว" over-segmentation.
  flu   fluent synthetic sentences (pitty/scripts/fluent_synth.py, trimmed cores + blended seams) — tune = val, test = test signers
  data_test (5 videos) · iphone_word (9) · iphone_sentence (5) · team (18, data_test/team-drive, 3 people): real videos, report only,
  never used to choose anything.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pitty" / "scripts"))

from modules.decode import wer  # noqa: E402
from modules.utils import ART, MODEL, PREP, get_logger, read_json, write_json  # noqa: E402

log = get_logger("decision")


def _p(fmt_, *args):
    print(fmt_ % args, flush=True)
CACHE = ROOT / "cache" / "decision"
REP = ART / "reports"
SPEEDS = (1.0, 1.5, 2.0)
TUNE_SETS = ["t51_tune_x1.0", "t51_tune_x1.5", "t51_tune_x2.0", "iso_tune", "flu_tune"]
TEST_SETS = ["t51_test_x1.0", "t51_test_x1.5", "t51_test_x2.0", "iso_test", "flu_test"]
REAL_SETS = ["data_test", "iphone_word", "iphone_sentence", "team"]


# ----------------------------------------------------------------------------- sets
def _t51_items(split, speed):
    from modules.segment import resample
    A = read_json(ART / "runs" / "seq-v7" / "tsl51_alignment.json")
    out = []
    for cid, a in A.items():
        if not cid.startswith("t51s:") or ("tune" if a["role"] == "tagger_train" else "test") != split:
            continue
        z = np.load(PREP / "pose_mp" / "tsl51_user_sentence" / f"{cid.split(':', 1)[1]}.npz")
        kp, sc, hw = z["kp"].astype(np.float32), z["sc"].astype(np.float32), tuple(int(v) for v in z["hw"])
        n = len(kp)
        gold = [(s["f0"], s["f1"]) for s in a["segments"]]
        if speed != 1.0:
            kp, sc = resample(kp, sc, speed)
            r = (len(kp) - 1) / max(n - 1, 1)
            gold = [(int(round(f0 * r)), int(round(f1 * r))) for f0, f1 in gold]
        out.append(dict(id=f"{cid}@{speed}", kp=kp, sc=sc, hw=hw, ref=list(a["glosses"]), gold=gold, bank="eval"))
    return out


def _heldout(role):
    from fluent_synth import heldout_pool
    store, q = heldout_pool()
    return store, q[q.role == role]


def _iso_items(role, n_max=700, seed=0):
    """tune (val signers, few of them) takes up to 3 clips per signer × word, test 1 (more distinct signers)."""
    store, q = _heldout(role)
    q = q[q.dataset != "tsl51"]                      # (no val role there anyway) — keep the held-out-signer sources
    q = q.groupby(["signer", "concept"], group_keys=False)[q.columns.tolist()].apply(lambda g: g.head(3 if role == "val" else 1))
    q = q.sample(min(n_max, len(q)), random_state=seed)
    out = []
    for r in q.itertuples():
        kp, sc, hw = store.get(store.row[r.clip_id])
        out.append(dict(id=r.clip_id, kp=kp, sc=sc, hw=hw, ref=[r.concept], gold=[(int(r.span_f0), int(r.span_f1))], bank="eval",
                        signer=r.signer, dataset=r.dataset))
    return out


def _flu_items(role, n=120, seed=0):
    from fluent_synth import FluentSynth
    from modules.segment import resample
    store, q = _heldout(role)
    groups = [g.drop_duplicates("concept") for _, g in q.groupby("signer")]
    groups = [g for g in groups if len(g) >= 4]
    rng = np.random.RandomState(7 + seed)
    fs = FluentSynth(store, q, seed=11 + seed)
    out = []
    for i in range(n):
        g = groups[i % len(groups)]
        picks = g.sample(min(len(g), rng.randint(3, 6)), random_state=rng.randint(1e9))
        kp, sc, bounds, words = fs.sentence(picks, mode="fluent+blend", trim=rng.uniform(0.15, 0.3), overlap=(0, 4))
        n0 = len(kp)
        kp, sc = resample(kp, sc, rng.uniform(0.9, 1.3))
        r = (len(kp) - 1) / max(n0 - 1, 1)
        hw = tuple(int(x) for x in store.hw[store.row[picks.clip_id.iloc[0]]])
        out.append(dict(id=f"flu_{role}_{i:03d}", kp=kp, sc=sc, hw=hw, ref=list(words), gold=[(int(a * r), int(b * r)) for a, b in bounds],
                        bank="eval", signer=picks.signer.iloc[0]))
    return out


def _real_items(name):
    from modules.lexicon import DATA_TEST_REFS, IPHONE_TEST_REFS
    from modules.mediapipe_pose import extract, load_raw, save_raw
    out = []
    if name == "data_test":
        canon = _canon()
        for stem, ref in DATA_TEST_REFS.items():
            f = ROOT / "cache" / "data_test_mp" / f"{stem}.npz"
            if not f.exists():
                save_raw(f, extract(ROOT / "data_test" / f"{stem}.mp4"))
            out.append(dict(id=stem, raw=load_raw(f), ref=[canon(w) for w in ref], gold=None, bank="prod", video=str(ROOT / "data_test" / f"{stem}.mp4")))
        return out
    if name == "team":          # data_test/team-drive (18 videos, 3 people; landmarks of scripts/modeling_compare.py extract)
        mc = ROOT / "cache" / "modeling_compare"
        items = read_json(mc / "team_items.json")
        for it in items:
            if not Path(it["mp"]).exists():
                save_raw(it["mp"], extract(it["video"], fps=25.0, normalize=False))
            out.append(dict(id=it["id"], raw=load_raw(it["mp"]), ref=list(it["ref"]), gold=None, bank="prod", video=it["video"],
                            signer=it["person"]))
        return out
    kind = name.split("_")[1]
    for stem, ref in IPHONE_TEST_REFS[kind].items():
        video = ROOT / "cache" / "iphone_test_mp4" / f"{kind}_{stem}.mp4"
        if not video.exists():
            _iphone_mp4(stem, kind, video)
        f = ROOT / "cache" / "iphone_test_mp" / f"{kind}_{stem}.npz"
        if not f.exists():
            save_raw(f, extract(video))
        out.append(dict(id=f"{kind}/{stem}", raw=load_raw(f), ref=list(ref), gold=None, bank="prod", video=str(video)))
    return out


def _iphone_mp4(stem, kind, out):
    """data_test/iphone_test/**/<stem>.MOV → 25 fps H.264 (ffmpeg applies the iPhone rotation), same as lab2.ipynb."""
    import subprocess
    src = next(p for p in (ROOT / "data_test" / "iphone_test").rglob("*") if p.stem == stem and p.suffix.lower() == ".mov")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", "fps=25", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)


def _canon():
    from modules.lexicon import TEST_SURFACE
    cm = read_json(PREP / "manifest" / "concepts.json")["concept"]
    return lambda w: cm.get(TEST_SURFACE.get(w, w), TEST_SURFACE.get(w, w))


def build_items(name):
    if name.startswith("t51_"):
        _, split, sp = name.split("_")
        return _t51_items(split, float(sp[1:]))
    if name.startswith("iso_"):
        return _iso_items("val" if name.endswith("tune") else "test")
    if name.startswith("flu_"):
        return _flu_items("val" if name.endswith("tune") else "test", n=240 if name.endswith("tune") else 120, seed=0 if name.endswith("tune") else 1)
    return _real_items(name)


# ----------------------------------------------------------------------------- pipeline + cache
class Harness:
    def __init__(self):
        import logging
        logging.disable(logging.WARNING)
        from modules.pipeline import SignPipeline, lm_bank
        from modules.runtime import Bank
        self.pipe = SignPipeline(model_dir=V9, backend="onnx", device="cpu", vocab="conversation", use_llm=False, use_face=False,
                                 bank_file=ART / "eval" / "bank_eval.npz")
        b = np.load(V9 / "bank.npz")
        allowed = {self.pipe.concepts.index(c) for c in read_json(V9 / "vocab_conversation.json")["concepts"] if c in self.pipe.concepts}
        self.fold = self.pipe.fold if self.pipe.fold is not None else np.arange(len(self.pipe.concepts))
        self.banks = {"eval": self.pipe.bank, "prod": Bank(b["Z"], self.fold[b["c"]], len(self.pipe.concepts), allowed)}
        from modules.handshape import HandBank
        hf = V9 / "hand_bank.npz"
        self.banks["prod"].hand = HandBank.load(hf, len(self.pipe.concepts), allowed, self.fold) if hf.exists() else None
        self.banks["prod"].lm = lm_bank(V9 / "lm_bank.npz", len(self.pipe.concepts), allowed, self.fold) if (V9 / "lm_bank.npz").exists() else None
        self.vocab = {self.pipe.concepts[c] for c in self.pipe.bank.concepts_present}
        self.vocab_words = sorted(self.vocab)

    def set_vocab(self, words, variants=None):
        """Both banks (+ hand-cue banks) restricted to another vocabulary, sign variants folded into their word (each version is
        evaluated with its own)."""
        from modules.handshape import HandBank
        from modules.pipeline import lm_bank
        from modules.runtime import Bank, variant_lut
        key = (sorted(words), sorted((variants or {}).items()))
        if key == getattr(self, "_vkey", None):
            return
        cidx = {c: i for i, c in enumerate(self.pipe.concepts)}
        allowed = {cidx[w] for w in words if w in cidx}
        C = len(self.pipe.concepts)
        self.fold = variant_lut(self.pipe.concepts, variants or {})
        for name, bf, hf, lf in (("eval", ART / "eval" / "bank_eval.npz", ART / "eval" / "hand_bank_eval.npz", ART / "eval" / "lm_bank_eval.npz"),
                                 ("prod", V9 / "bank.npz", V9 / "hand_bank.npz", V9 / "lm_bank.npz")):
            z = np.load(bf)
            self.banks[name] = Bank(z["Z"], self.fold[z["c"]], C, allowed)
            self.banks[name].hand = HandBank.load(hf, C, allowed, self.fold) if hf.exists() else None
            self.banks[name].lm = lm_bank(lf, C, allowed, self.fold) if lf.exists() else None
        self.pipe.bank = self.banks["eval"]
        self.vocab = {self.pipe.concepts[c] for c in self.banks["eval"].concepts_present}
        self.vocab_words = sorted(words)
        self._vkey = key

    def standard(self, it):
        from modules.schema import SH_L, SH_R, canonical_hands
        if "raw" in it:
            kp, sc, hw, present, _ = self.pipe.standardise(it["raw"])
            return kp, sc, hw, present
        kp, sc, hw = it["kp"], it["sc"], it["hw"]
        present = sc[:, [0, SH_L, SH_R]].mean(1) > 0.5
        sc = sc.copy(); sc[~present] = 0.0; kp = kp.copy(); kp[~present] = 0.0
        kp, sc, _ = canonical_hands(kp, sc, hw)
        return kp, sc, hw, present

    def front(self, it):
        st = self.pipe.front(*self.standard(it))
        return {k: (v.astype(np.float16) if k in ("H", "H2") and v is not None else v) for k, v in st.items()}

    def back(self, st, bank, raw=False):
        st = {k: (v.astype(np.float32) if k in ("H", "H2") and v is not None else v) for k, v in st.items()}
        segs, _, _ = self.pipe.back(st, self.banks[bank], raw=raw)
        if not raw and self.alphabet != "off" and st["H"] is not None:      # v8.1: fingerspelling runs on every utterance
            segs = self.pipe._spell(st["kpi"], st["sc"], st["H"], segs, self.alphabet)
        return segs

    alphabet = "off"

    def use(self, version):
        """Everything a version ships with: vocabulary, decoder settings, calibration, class prior, spelling."""
        b = baseline(version)
        self.set_vocab(b["vocab"], b["variants"])
        allowed = {self.pipe.concepts.index(c) for c in self.vocab}
        prior = class_prior(ART / "eval" / "bank_eval.npz", self.pipe.concepts, allowed, self.fold) if b["prior"] else None
        self.configure(spot=b["spot"], cal=b["cal"], prior=prior)
        self.prod_prior = dict(np.load(b["prior"])) if b["prior"] else None
        self.pipe.spell_cfg = dict(self.pipe.spell_cfg, **(b["spell"] or {}))
        self.alphabet = b["alphabet"]
        return b

    def configure(self, spot=None, cal=None, prior="keep"):
        """Swap decoder settings / calibration / class prior (models/ files are only written by `tune --write`)."""
        from modules.decode import LogReg
        if spot is not None:
            self.pipe.spot = spot
        if cal is not None:
            self.pipe.cal = dict(cal, tau_accept=max(float(cal.get("tau_accept", 0.0)), float(self.pipe.spot_all.get("accept_min", 0.8))))
            self.pipe.lr = LogReg.from_dict(cal["model"])
        if prior != "keep":
            self.pipe.class_prior = prior


def cache_path(name):
    return CACHE / f"{name}.pkl"


def load_set(name):
    with open(cache_path(name), "rb") as f:
        return pickle.load(f)


def cmd_cache(a):
    h = Harness()
    CACHE.mkdir(parents=True, exist_ok=True)
    names = a.sets.split(",") if a.sets else TUNE_SETS + TEST_SETS + REAL_SETS
    for name in names:
        if cache_path(name).exists() and not a.force:
            _p("%s cached", name); continue
        t = time.time()
        items = build_items(name)
        rows = []
        for it in items:
            st = h.front(it)
            rows.append(dict({k: v for k, v in it.items() if k not in ("kp", "sc", "raw")}, st=st, T=len(st["prob"])))
        with open(cache_path(name), "wb") as f:
            pickle.dump(rows, f)
        _p("%s: %d utterances, %.0f s", name, len(rows), time.time() - t)


# ----------------------------------------------------------------------------- scoring
def _iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(max(a[1], b[1]) - min(a[0], b[0]), 1)


def score(segs, row, vocab):
    """One utterance → per-segment correctness + utterance metrics. With gold boundaries: IoU ≥ 0.3 matching (greedy, best first);
    without (real videos): a segment is correct when its word is still unused in the reference (multiset)."""
    live = [s for s in segs if s["status"] != "transition"]
    ref, gold = row["ref"], row.get("gold")
    hyp = [s["nearest"] for s in live]
    correct, hit1, hit5 = [False] * len(live), set(), set()
    true = [None] * len(live)
    seg_tp = 0
    if gold:
        pairs = sorted(((_iou((s["f0"], s["f1"]), g), i, j) for i, s in enumerate(live) for j, g in enumerate(gold)), reverse=True)
        used_i, used_j, match = set(), set(), {}
        for v, i, j in pairs:
            if v < 0.3 or i in used_i or j in used_j:
                continue
            used_i.add(i); used_j.add(j); match[i] = j
        seg_tp = len(match)
        for i, j in match.items():
            cw = [c["word"] for c in live[i]["candidates"][:5]]
            correct[i] = live[i]["nearest"] == ref[j]
            true[i] = ref[j]
            if correct[i]:
                hit1.add(j)
            if ref[j] in cw:
                hit5.add(j)
    else:
        left = list(ref)
        for i, s in enumerate(live):
            if s["nearest"] in left:
                correct[i] = True; left.remove(s["nearest"])
        for j, w in enumerate(ref):
            if w in hyp:
                hit1.add(j)
            if any(w in [c["word"] for c in s["candidates"][:5]] for s in live):
                hit5.add(j)
    inv = [j for j, w in enumerate(ref) if vocab is None or w in vocab]
    st = [s["status"] for s in live]
    return dict(n_pred=len(live), n_ref=len(ref), seg_tp=seg_tp, wer=wer(ref, hyp), n_inv=len(inv),
                hit1=sum(j in hit1 for j in inv), hit5=sum(j in hit5 for j in inv), correct=correct, status=st,
                n_acc=st.count("accepted"), n_acc_ok=sum(c for c, x in zip(correct, st) if x == "accepted"),
                n_unc=st.count("uncertain"), n_unc_ok=sum(c for c, x in zip(correct, st) if x == "uncertain"), n_rej=st.count("unknown"),
                hyp=hyp, ref=ref, has_gold=bool(gold), true=true)


def aggregate(scores):
    S = scores
    n_pred, n_ref, n_inv = sum(s["n_pred"] for s in S), sum(s["n_ref"] for s in S), sum(s["n_inv"] for s in S)
    out = dict(n=len(S), WER=float(np.mean([s["wer"] for s in S])), R1=sum(s["hit1"] for s in S) / max(n_inv, 1),
               R5=sum(s["hit5"] for s in S) / max(n_inv, 1), count_err=float(np.mean([abs(s["n_pred"] - s["n_ref"]) for s in S])),
               exact_count=float(np.mean([s["n_pred"] == s["n_ref"] for s in S])), words_pred=n_pred, words_ref=n_ref,
               acc=sum(s["n_acc"] for s in S), acc_ok=sum(s["n_acc_ok"] for s in S), unc=sum(s["n_unc"] for s in S),
               unc_ok=sum(s["n_unc_ok"] for s in S), rej=sum(s["n_rej"] for s in S), top1_prec=sum(sum(s["correct"]) for s in S) / max(n_pred, 1))
    if all(s["has_gold"] for s in S):
        tp = sum(s["seg_tp"] for s in S)
        p, r = tp / max(n_pred, 1), tp / max(n_ref, 1)
        out["seg_F1"] = 2 * p * r / max(p + r, 1e-9)
        pairs = [(t, h_) for x in S for t, h_ in zip(x["true"], x["hyp"]) if t is not None]
        f1 = []
        for w in {t for t, _ in pairs}:
            tp = sum(t == w and h_ == w for t, h_ in pairs); fp = sum(t != w and h_ == w for t, h_ in pairs); fn = sum(t == w and h_ != w for t, h_ in pairs)
            f1.append(2 * tp / max(2 * tp + fp + fn, 1))
        out["macro_F1"] = float(np.mean(f1)) if f1 else None
    n_ok = sum(sum(x["correct"]) for x in S); n_bad = n_pred - n_ok
    out["false_accept_rate"] = (out["acc"] - out["acc_ok"]) / max(n_bad, 1)               # wrong words that were accepted
    out["correct_not_accepted"] = (n_ok - out["acc_ok"]) / max(n_ok, 1)                  # right words shown only as "อาจหมายถึง" / rejected
    out["correct_rejected"] = sum(sum(c and st_ == "unknown" for c, st_ in zip(x["correct"], x["status"])) for x in S) / max(n_ok, 1)
    out["acc_prec"] = out["acc_ok"] / max(out["acc"], 1)
    out["acc_cover"] = out["acc_ok"] / max(n_ref, 1)
    return out


def evaluate(h, rows, keep_segs=False):
    out, segs_all = [], []
    for r in rows:
        segs = h.back(r["st"], r["bank"])
        out.append(score(segs, r, h.vocab))
        if keep_segs:
            segs_all.append(segs)
    return (out, segs_all) if keep_segs else out


def fmt(m):
    keys = ["WER", "seg_F1", "R1", "R5", "count_err", "exact_count", "acc_prec", "acc_cover", "top1_prec"]
    s = " ".join(f"{k}={m[k]:.3f}" for k in keys if k in m)
    return f"{s} acc={m['acc_ok']}/{m['acc']} unc={m['unc_ok']}/{m['unc']} rej={m['rej']}"


def cmd_baseline(a):
    h = Harness()
    names = a.sets.split(",") if a.sets else TUNE_SETS + TEST_SETS + REAL_SETS
    res = {}
    for name in names:
        t = time.time()
        rows = load_set(name)
        m = aggregate(evaluate(h, rows))
        res[name] = m
        _p("%-18s %s (%.1f s)", name, fmt(m), time.time() - t)
    if a.out:
        write_json(Path(a.out), res)


# ----------------------------------------------------------------------------- per-word reliability inside the bank
def class_prior(bank_npz, concepts, allowed, fold=None, alpha=3.0):
    """Leave-one-signer-out top-1 accuracy of every vocabulary word inside the bank (top-2-mean rule of runtime.Bank), smoothed
    towards the global accuracy, + signers per word. Uses only the bank (no tune / test labels)."""
    z = np.load(bank_npz)
    Z, c, sg = z["Z"].astype(np.float32), z["c"].astype(np.int64), np.asarray(z["signer"])
    c = c if fold is None else fold[c]
    keep = np.isin(c, np.asarray(sorted(allowed)))
    Z, c, sg = Z[keep], c[keep], sg[keep]
    o = np.argsort(c, kind="stable"); Z, c, sg = Z[o], c[o], sg[o]
    present, starts = np.unique(c, return_index=True)
    seg = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(c)]))
    hit = np.zeros(len(c), bool)
    for a in range(0, len(c), 512):
        S = Z[a:a + 512] @ Z.T
        S[sg[a:a + 512, None] == sg[None, :]] = -2.0
        t1 = np.maximum.reduceat(S, starts, axis=1)
        S2 = np.where(S >= t1[:, seg] - 1e-7, -2.0, S)
        t2 = np.maximum.reduceat(S2, starts, axis=1)
        top = np.where(t2 > -1.5, (t1 + t2) / 2, t1)
        hit[a:a + 512] = present[top.argmax(1)] == c[a:a + 512]
    g = hit.mean()
    rel = np.full(len(concepts), g, np.float32); depth = np.zeros(len(concepts), np.float32)
    df = pd.DataFrame(dict(c=c, hit=hit, sg=sg))
    for ci, d in df.groupby("c"):
        rel[ci] = (d.hit.sum() + alpha * g) / (len(d) + alpha)
        depth[ci] = d.sg.nunique()
    return dict(rel=rel, depth=depth)


# ----------------------------------------------------------------------------- v8.1 hand-shape cue: descriptors of every bank clip
def _npz_memmap(path):
    """Arrays of an uncompressed .npz memory-mapped (the 1.2 GB pose store never has to fit in RAM)."""
    import zipfile
    out = {}
    with zipfile.ZipFile(path) as z, open(path, "rb") as f:
        for info in z.infolist():
            f.seek(info.header_offset)
            h = f.read(30)
            start = info.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little")
            f.seek(start)
            ver = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format._read_array_header(f, ver)
            if info.compress_type:
                out[info.filename[:-4]] = np.load(path, allow_pickle=True)[info.filename[:-4]]
            elif dtype.hasobject:
                f.seek(start); out[info.filename[:-4]] = np.lib.format.read_array(f, allow_pickle=True)
            else:
                out[info.filename[:-4]] = np.memmap(path, dtype=dtype, mode="r", offset=f.tell(), shape=shape, order="F" if fortran else "C")
    return out


def bank_clip_loader():
    """clip id of a bank (models/bank.npz, artifacts/eval/bank_eval.npz) → (kp iso [T,133,2] canonical right-dominant, sc, f0, f1)
    from the memory-mapped pose store (sign span from clips.parquet; TSL51 sentence-aligned words "<sentence>@<f0>" from the
    alignment), or None when the clip is not in the store."""
    from modules.encoder import SLOTS
    from modules.parts import to_iso
    from modules.schema import canonical_hands
    S = _npz_memmap(PREP / "shards" / "pose_store.npz")
    row = {str(c): i for i, c in enumerate(S["clip_id"])}
    clips = pd.read_parquet(PREP / "shards" / "clips.parquet", columns=["clip_id", "has_span", "span_f0", "span_f1"])
    span = {r.clip_id: (r.span_f0, r.span_f1) for r in clips.itertuples() if r.has_span}
    align = read_json(ART / "runs" / "seq-v7" / "tsl51_alignment.json")

    def load(clip):
        cid, f0, f1 = clip, None, None
        if "@" in clip:                                     # TSL51 sentence-aligned word: "<sentence clip>@<f0>"
            cid, f0 = clip.rsplit("@", 1); f0 = int(f0)
            f1 = next((s["f1"] for s in align.get(cid, {}).get("segments", []) if s["f0"] == f0), None)
        else:
            f0, f1 = span.get(clip, (None, None))
        i = row.get(cid)
        if i is None or ("@" in clip and f1 is None):
            return None
        s, e = int(S["off"][i]), int(S["off"][i + 1])
        kp = np.zeros((e - s, 133, 2), np.float32); sc = np.zeros((e - s, 133), np.float32)
        kp[:, SLOTS] = S["KP"][s:e]; sc[:, SLOTS] = S["SC"][s:e]
        hw = tuple(int(x) for x in S["hw"][i])
        kp, sc, _ = canonical_hands(kp, sc, hw)
        return to_iso(kp, hw), sc, f0, f1
    return load


def cmd_hands(a):
    """Hand-shape descriptor of every clip of models/bank.npz and artifacts/eval/bank_eval.npz (sign span from clips.parquet;
    TSL51 sentence-aligned words from the alignment) → metric fitted on TRAINING signers only (eval bank) → models/hand_bank.npz,
    artifacts/eval/hand_bank_eval.npz."""
    from modules.handshape import DIM, HandBank, descriptor
    load = bank_clip_loader()
    cache = {}

    def desc(clip):
        if clip not in cache:
            x = load(clip)
            cache[clip] = None if x is None else descriptor(*x)
        return cache[clip]

    banks = {}
    for name, path in (("eval", ART / "eval" / "bank_eval.npz"), ("prod", MODEL / "bank.npz")):
        z = np.load(path)
        t = time.time()
        D = np.zeros((len(z["clip"]), DIM), np.float32)
        for k, clip in enumerate(z["clip"]):
            d = desc(str(clip))
            if d is not None:
                D[k] = d
        banks[name] = (D, z["c"].astype(np.int64), np.asarray(z["signer"]))
        _p("%s bank: %d clips, %.1f%% described (%.0f s)", name, len(D), 100 * (np.abs(D).sum(1) > 0).mean(), time.time() - t)
    D, c, sg = banks["eval"]
    m = np.abs(D).sum(1) > 0
    tf = HandBank.fit_transform(D[m], c[m], sg[m], shrink=a.shrink, dim=a.dim)
    for name, out in (("eval", ART / "eval" / "hand_bank_eval.npz"), ("prod", MODEL / "hand_bank.npz")):
        D, c, _ = banks[name]
        P = np.where((np.abs(D).sum(1) > 0)[:, None], HandBank.project(D, tf), 0.0)
        np.savez(out, P=P.astype(np.float16), c=c.astype(np.int32), **tf)
        _p("written %s", out)


def cmd_lmbank(a):
    """(v9) Second opinion: the v8 Thai-only landmark encoder ensemble (artifacts/runs/enc-v8, modules/lmenc.py) → ONNX
    (<out>/onnx/lmenc.onnx) and its embedding of every bank clip's sign span → <out>/lm_bank.npz (production bank) and
    artifacts/eval/lm_bank_eval.npz (training signers). Same feature path as the runtime (lmenc.prep on canonical iso landmarks)."""
    from modules import lmenc
    out = Path(a.out)
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    onnx = out / "onnx" / "lmenc.onnx"
    err = lmenc.export_onnx(ART / "runs" / "enc-v8" / "encoder_best.pt", onnx)
    _p("lmenc.onnx parity max|Δ| = %.2e", err)
    lm = lmenc.OnnxLM(onnx)
    load = bank_clip_loader()
    for name, path, dst in (("eval", ART / "eval" / "bank_eval.npz", ART / "eval" / "lm_bank_eval.npz"), ("prod", Path(a.bank), out / "lm_bank.npz")):
        z = np.load(path)
        t = time.time()
        F, ok = [], np.zeros(len(z["clip"]), bool)
        for k, clip in enumerate(z["clip"]):
            x = load(str(clip))
            if x is None:
                continue
            kp, sc, f0, f1 = x
            f0 = 0 if f0 is None else int(f0); f1 = len(kp) if f1 is None else int(f1)
            if f1 - f0 < 4:
                f0, f1 = 0, len(kp)
            F.append(lmenc.prep(kp[f0:f1], sc[f0:f1])); ok[k] = True
        E = lm.embed(F)
        Z = np.zeros((len(ok), E.shape[1]), np.float16); Z[ok] = E
        np.savez(dst, Z=Z, c=z["c"].astype(np.int32), ok=ok)
        _p("%s: %d / %d clips embedded (%.0f s) → %s", name, ok.sum(), len(ok), time.time() - t, dst)


# ----------------------------------------------------------------------------- tuning
def _decide_all(h, raw_by_set):
    """raw (decoded) segments → decide() with the current calibration → scores per set."""
    import copy
    res = {}
    for name, (rows, raws) in raw_by_set.items():
        res[name] = aggregate([score(h.pipe.decide(copy.deepcopy(r)), row, h.vocab) for row, r in zip(rows, raws)])
    return res


def _raw(h, names, rows_by_set):
    return {n: (rows_by_set[n], [h.back(r["st"], r["bank"], raw=True) for r in rows_by_set[n]]) for n in names}


def _objective(res, sets):
    return -float(np.mean([res[n]["WER"] for n in sets]))


def _labels(h, rows, raws):
    """Every decoded segment of the tune sets (before calibration) with its correctness (IoU-matched gold word) and the rank of
    the true word among its candidates (-1 = not among them)."""
    segs, y, gold_rank = [], [], []
    for row, r in zip(rows, raws):
        for s in r:
            s["status"] = "uncertain"
        sc = score(r, row, h.vocab)
        segs += r; y += sc["correct"]
        gold = row.get("gold") or []
        for s in r:
            best = max(((_iou((s["f0"], s["f1"]), g), j) for j, g in enumerate(gold)), default=(0, None))
            w = row["ref"][best[1]] if best[0] >= 0.3 else None
            cw = [c["word"] for c in s["candidates"]]
            gold_rank.append(cw.index(w) if w in cw else -1)
    return segs, np.array(y, float), np.array(gold_rank)


def fit_temperature(segs, gold_rank):
    from modules.decode import candidate_probs
    best = None
    for T in np.exp(np.linspace(np.log(0.003), np.log(0.3), 60)):
        candidate_probs(segs, T)
        nll = -np.mean([np.log(max(s["candidates"][k]["prob"], 1e-6)) for s, k in zip(segs, gold_rank) if k >= 0])
        if best is None or nll < best[0]:
            best = (nll, float(T))
    candidate_probs(segs, best[1])
    return best[1], best[0]


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    e = 0.0
    for a in range(bins):
        m = (p >= a / bins) & (p < (a + 1) / bins + (a == bins - 1))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def reliability(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    out = []
    for a in range(bins):
        m = (p >= a / bins) & (p < (a + 1) / bins + (a == bins - 1))
        out.append(dict(lo=a / bins, n=int(m.sum()), conf=float(p[m].mean()) if m.any() else None, acc=float(y[m].mean()) if m.any() else None))
    return out


def fit_lr(X, y, w_set):
    """Logistic calibration with every tune set weighted equally (TSL51 speeds, isolated clips, fluent sentences)."""
    from modules.decode import LogReg
    reps = np.maximum(np.round(w_set / w_set.min()).astype(int), 1)          # integer re-weighting by repetition (LogReg has no weights)
    reps = np.maximum((reps / reps.max() * 20).round().astype(int), 1)
    return LogReg().fit(np.repeat(X, reps, 0), np.repeat(y, reps))


def cv_logloss(X, y, w_set, groups, k=5, seed=0):
    """k-fold by utterance: held-out log-loss / ECE of the calibration (so a richer feature set has to earn its place)."""
    rng = np.random.RandomState(seed)
    ug = np.unique(groups); rng.shuffle(ug)
    fold = {g: i % k for i, g in enumerate(ug)}
    f = np.array([fold[g] for g in groups])
    P = np.zeros(len(y))
    for i in range(k):
        m = f == i
        P[m] = fit_lr(X[~m], y[~m], w_set[~m]).predict(X[m])
    P = np.clip(P, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(P) + (1 - y) * np.log(1 - P))), ece(P, y), P


def _isotonic_fit(p, y, n_knots=20):
    """Pool-adjacent-violators on (p, y) → a monotone table on quantile knots (x, y) for np.interp."""
    o = np.argsort(p); p, y = p[o], y[o].astype(float)
    blocks = []
    for v, x in zip(y, p):
        blocks.append([v, 1.0, [x]])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            b2, b1 = blocks.pop(), blocks.pop()
            ww = b1[1] + b2[1]
            blocks.append([(b1[0] * b1[1] + b2[0] * b2[1]) / ww, ww, b1[2] + b2[2]])
    fx = np.array([np.mean(b[2]) for b in blocks]); fy = np.array([b[0] for b in blocks])
    q = np.unique(np.quantile(p, np.linspace(0, 1, n_knots)))
    return dict(x=[0.0] + q.tolist() + [1.0], y=[float(np.interp(0.0, fx, fy))] + np.interp(q, fx, fy).tolist() + [float(np.interp(1.0, fx, fy))])


def _isotonic_cv(P, Y, groups, k=5, seed=0):
    """Isotonic re-mapping evaluated out-of-fold (folds by utterance) on the LR's own out-of-fold P → CV ECE / log-loss + table."""
    ug = np.unique(groups); rng = np.random.RandomState(seed); rng.shuffle(ug)
    fold = {g: i % k for i, g in enumerate(ug)}
    f = np.array([fold[g] for g in groups])
    Q = np.zeros_like(P)
    for i in range(k):
        t = _isotonic_fit(P[f != i], Y[f != i])
        Q[f == i] = np.interp(P[f == i], t["x"], t["y"])
    Qc = np.clip(Q, 1e-4, 1 - 1e-4)
    return dict(table=_isotonic_fit(P, Y), report=dict(cv_ECE=ece(Q, Y), cv_logloss=float(-np.mean(Y * np.log(Qc) + (1 - Y) * np.log(1 - Qc)))))


def merge_pairs(h, spot, rows):
    """Every pair of neighbouring decoded segments of the tune sets with its features (decode.pair_features) and label: 1 = both
    parts overlap the same gold sign most, 0 = different gold signs; pairs touching no gold sign are left out."""
    h.configure(spot=dict(spot, merge=dict(on=True, max_gap=6, max_len=72)))
    X, y, g = [], [], []
    for n in TUNE_SETS:
        for k, r in enumerate(rows[n]):
            col = []
            h.pipe.collect_pairs = col
            h.back(r["st"], r["bank"], raw=True)
            h.pipe.collect_pairs = None
            f = r["st"]["f"]
            gold = r.get("gold") or []
            for e in col:
                ga, gb = (_gold_of((x0 / f, x1 / f), gold) for x0, x1 in (e["a"], e["b"]))
                if ga is None and gb is None or (ga is None) != (gb is None):
                    continue
                X.append(e["x"]); y.append(float(ga == gb)); g.append(f"{n}|{k}")
    h.configure(spot=spot)
    return np.array(X, np.float32), np.array(y, float), np.array(g)


def split_cuts(h, spot, rows):
    """Every candidate cut of the split test on the tune sets (decode.split_long collect mode) with its label: 1 = the two parts lie
    on different gold signs, 0 = the same gold sign; cuts whose parts touch no gold sign are left out."""
    h.configure(spot=dict(spot, split=dict(on=True, min_part=8, min_len=24)))
    X, y, g = [], [], []
    for n in TUNE_SETS:
        for k, r in enumerate(rows[n]):
            col = []
            h.pipe.collect_splits = col
            h.back(r["st"], r["bank"], raw=True)
            h.pipe.collect_splits = None
            f = r["st"]["f"]
            gold = r.get("gold") or []
            for e in col:
                ga, gb = (_gold_of((x0 / f, x1 / f), gold) for x0, x1 in (e["a"], e["b"]))
                if ga is None or gb is None:
                    continue
                X.append(e["x"]); y.append(float(ga != gb)); g.append(f"{n}|{k}")
    h.configure(spot=spot)
    return np.array(X, np.float32), np.array(y, float), np.array(g)


def _gold_of(seg, gold):
    best = max(((min(seg[1], g[1]) - max(seg[0], g[0]), j) for j, g in enumerate(gold)), default=(0, None))
    return best[1] if best[0] > 0.3 * max(seg[1] - seg[0], 1) else None


def _cv_auc(X, y, groups, k=5):
    from modules.decode import LogReg
    ug = np.unique(groups); rng = np.random.RandomState(0); rng.shuffle(ug)
    fold = {gg: i % k for i, gg in enumerate(ug)}
    f = np.array([fold[gg] for gg in groups]); P = np.zeros(len(y))
    for i in range(k):
        m = f == i
        P[m] = LogReg().fit(X[~m], y[~m]).predict(X[m])
    pos, neg = P[y == 1], P[y == 0]
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()) if len(pos) and len(neg) else float("nan")
    return auc, P


def fit_rerank(S, GR, groups, prior, concepts, l2=1e-2, iters=600, lr=0.5):
    """Conditional logit over each segment's top-5 (decode.candidate_matrix): learn w so the true word scores highest. Only
    segments whose true word is among the candidates. → (w, CV top-1 of the re-ranker, top-1 without it)."""
    from modules.decode import candidate_matrix
    data = [(candidate_matrix(s, concepts, prior), int(k), g) for s, k, g in zip(S, GR, groups) if k >= 0 and len(s.get("candidates") or []) >= 2]
    F = data[0][0].shape[1]

    def train(items):
        Xs = np.zeros((len(items), 5, F), np.float32); M = np.zeros((len(items), 5), bool); yk = np.array([k for _, k, _ in items])
        for i, (X, _, _) in enumerate(items):
            Xs[i, :len(X)] = X; M[i, :len(X)] = True
        mu = Xs[M].mean(0); sd = Xs[M].std(0) + 1e-6
        Xn = (Xs - mu) / sd
        w = np.zeros(F)
        w[0] = 1.0                                           # start at "keep the visual order"
        for _ in range(iters):
            u = np.where(M, Xn @ w, -1e9); u -= u.max(1, keepdims=True)
            p = np.exp(u); p /= p.sum(1, keepdims=True)
            onehot = np.zeros_like(p); onehot[np.arange(len(yk)), yk] = 1
            g = ((p - onehot)[:, :, None] * Xn).sum((0, 1)) / len(yk) + l2 * w
            w -= lr * g
        return w / sd, -(mu / sd) @ w                        # raw-feature weights (the bias is irrelevant for an argmax)
    def top1(items, w):
        return float(np.mean([int(np.argmax(X @ w)) == k for X, k, _ in items]))
    ug = np.unique([g for _, _, g in data]); rng = np.random.RandomState(0); rng.shuffle(ug)
    fold = {gg: i % 5 for i, gg in enumerate(ug)}
    hits = []
    for f in range(5):
        tr = [d for d in data if fold[d[2]] != f]; te = [d for d in data if fold[d[2]] == f]
        w, _ = train(tr)
        hits += [int(np.argmax(X @ w)) == k for X, k, _ in te]
    w, _ = train(data)
    return w, float(np.mean(hits)), float(np.mean([k == 0 for _, k, _ in data])), len(data)


def _relabel(S, raw):
    """correctness + true-word rank again after the re-ranker changed segment words (same order as _labels over TUNE_SETS)."""
    Y, GR, k = [], [], 0
    for n in TUNE_SETS:
        rows, raws = raw[n]
        for row, r in zip(rows, raws):
            sc = score(r, row, None)
            Y += sc["correct"]
            gold = row.get("gold") or []
            for s in r:
                best = max(((_iou((s["f0"], s["f1"]), g), j) for j, g in enumerate(gold)), default=(0, None))
                w = row["ref"][best[1]] if best[0] >= 0.3 else None
                cw = [c["word"] for c in s["candidates"]]
                GR.append(cw.index(w) if w in cw else -1)
    return np.array(Y, float), np.array(GR)


def cmd_tune(a):
    import copy
    from modules.decode import FEATS, FEATS_V8, FEATS_V81, FEATS_V9, class_features, seg_features
    h = Harness()
    out_dir = Path(a.out)
    vc = read_json(out_dir / "vocab_conversation.json")
    h.set_vocab(vc["concepts"], vc.get("variants", {}))
    rows = {n: load_set(n) for n in TUNE_SETS}
    allowed = {h.pipe.concepts.index(c) for c in h.vocab}
    prior_eval = class_prior(ART / "eval" / "bank_eval.npz", h.pipe.concepts, allowed, h.fold)
    v8 = baseline(a.start)                    # always start from a frozen version's settings (re-runs are idempotent): v8.1 for v9
    spot0 = v8["spot"]
    h.configure(spot=spot0, cal=v8["cal"], prior=prior_eval)
    rep = dict(baseline={}, segmentation=[], calibration={}, policy=[])

    # ---------------- 1. recognition + segmentation (v8 calibration, so only the readings / segments change)
    def seg_eval(spot, tag):
        h.configure(spot=spot)
        t = time.time()
        res = _decide_all(h, _raw(h, TUNE_SETS, rows))
        obj = _objective(res, TUNE_SETS)
        rep["segmentation"].append(dict(tag=tag, spot={k: spot.get(k) for k in ("hand_w", "merge", "grammar_rerank_weight")}, objective=obj,
                                        **{n: {k: res[n][k] for k in ("WER", "count_err", "R1", "seg_F1") if k in res[n]} for n in TUNE_SETS}))
        _p("[seg] %-44s obj %.4f | %s (%.0f s)", tag, obj,
           " ".join(f"{n.replace('_tune', '')}:{res[n]['WER']:.3f}/{res[n]['R1']:.3f}" for n in TUNE_SETS), time.time() - t)
        return obj, res

    base_obj, base_res = seg_eval(spot0, f"baseline ({a.start})")
    rep["baseline"] = base_res
    if a.spot_from:        # re-run of the decision stages only: the segmentation chosen by an earlier run (its spotting.json)
        spot = dict(read_json(Path(a.spot_from))["conversation"])

    def ok(res):          # no tune set may get worse than the baseline (0.003 WER = noise); a mean gain bought on one set is not enough
        return all(res[n]["WER"] <= base_res[n]["WER"] + 0.003 for n in TUNE_SETS)

    # hand-shape cue weight (0 = v8: the decoder's window-ensemble reading); a stronger weight must win by > 0.002
    best = (base_obj, spot0)
    if a.spot_from:
        best = (seg_eval(spot, "segmentation from " + Path(a.spot_from).parent.name)[0], spot)
    elif a.start == "v8":
        grid = [dict(hand_w=w) for w in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)]
    else:                 # v9 reader: boundary-robust span reading (crops) + the landmark encoder's opinion (lm_w), hand weight re-checked
        grid = [dict(read=dict(crops=c, lm_w=lw), hand_w=hw) for c in (0, 1) for lw in (0.0, 0.2, 0.3, 0.4) for hw in (1.0, 1.2)
                if (c, lw, hw) != (0, 0.0, spot0.get("hand_w"))]
    for g in ([] if a.spot_from else grid):
        sp = dict(spot0, **g)
        o, r = seg_eval(sp, ("reader " if "read" in g else "") + " ".join(f"{k}={v}" for k, v in g.items()))
        if ok(r) and o > best[0] + 0.002:
            best = (o, sp)
    base_spot = spot if a.spot_from else best[1]
    # merge model: features of every neighbouring pair on the tune sets, label = both parts lie on the same gold sign
    Xp, yp, gp = merge_pairs(h, base_spot, rows) if not a.spot_from else (np.zeros((2, 13)), np.array([0, 1]), np.array(["a", "b"]))
    from modules.decode import LogReg, MERGE_FEATS
    auc, Pcv = _cv_auc(Xp, yp, gp)
    mlr = LogReg().fit(Xp, yp)
    rep["merge_model"] = dict(pairs=int(len(yp)), positives=int(yp.sum()), cv_auc=auc,
                              weights=dict(zip(MERGE_FEATS, np.round(mlr.w / mlr.sd, 3).tolist())))
    _p("[merge] %d pairs (%d same-sign) · CV AUC %.3f · weights %s", len(yp), yp.sum(), auc, rep["merge_model"]["weights"])
    mbest = best
    for thr in () if a.spot_from else (0.7, 0.6, 0.5, 0.4, 0.3):              # conservative first: a more aggressive merge must win by > 0.002
        sp = dict(base_spot, merge=dict(on=True, max_gap=6, max_len=72, thr=thr, model=mlr.to_dict(MERGE_FEATS)))
        o, r = seg_eval(sp, f"merge model thr={thr}")
        if ok(r) and o > mbest[0] + 0.002:
            mbest = (o, sp)
    spot = mbest[1]
    # v8.1 split test (two fluent signs read as one): same procedure as the merge model, on the merged segmentation
    from modules.decode import SPLIT_FEATS
    Xs, ys, gs = split_cuts(h, spot, rows) if not a.spot_from else (np.zeros((2, 11)), np.array([0, 1]), np.array(["a", "b"]))
    auc_s, _ = _cv_auc(Xs, ys, gs)
    slr = LogReg().fit(Xs, ys)
    rep["split_model"] = dict(cuts=int(len(ys)), positives=int(ys.sum()), cv_auc=auc_s, weights=dict(zip(SPLIT_FEATS, np.round(slr.w / slr.sd, 3).tolist())))
    _p("[split] %d candidate cuts (%d between two signs) · CV AUC %.3f · weights %s", len(ys), ys.sum(), auc_s, rep["split_model"]["weights"])
    for thr in () if a.spot_from else (0.8, 0.7, 0.6, 0.5):                    # conservative first; only inside utterances of ≥ 2 signs
        sp = dict(spot, split=dict(on=True, min_part=8, min_len=24, min_segments=2, thr=thr, model=slr.to_dict(SPLIT_FEATS)))
        o, r = seg_eval(sp, f"split model thr={thr} (≥ 2 signs)")
        if ok(r) and o > mbest[0] + 0.002:
            mbest = (o, sp)
    spot = mbest[1]
    # TSL word-order re-rank (on since v7). v8.1: it may only act on utterances of ≥ 2 signs, measured on the ranking's own
    # quantity, within a visual gap chosen here. Grammar (any variant) is kept only if it lowers tune WER by > 0.005 vs off (v7's
    # own rule) — the tune sentences are TSL51 templates, so a word-order prior has a head start there that real users do not give it
    w_g = spot.get("grammar_rerank_weight") or 0.5
    o_off, r_off = seg_eval(dict(spot, grammar_rerank_weight=0.0), "grammar re-rank off") if not a.spot_from else (0.0, base_res)
    gbest = (o_off, dict(spot, grammar_rerank_weight=0.0)) if (ok(r_off) or a.grammar == "off") else (mbest[0] - 1.0, spot)
    # v9 (prompt_13): recognition first, no rule-based word order inside the recognised words — grammar is the optional LLM layer
    for gap in (0.005, 0.01, 0.02, 0.03) if a.grammar == "tune" else ():
        sp = dict(spot, grammar_rerank_weight=w_g, grammar=dict(min_segments=2, key="support", min_gap=gap))
        o, r = seg_eval(sp, f"grammar guarded gap={gap}")
        if ok(r) and o > max(gbest[0], o_off + 0.005):
            gbest = (o, sp)
    spot = gbest[1]
    h.configure(spot=spot)
    _p("[seg] chosen: hand_w=%s merge thr=%s grammar=%s", spot.get("hand_w"), (spot.get("merge") or {}).get("thr"), spot.get("grammar_rerank_weight"))

    # ---------------- 2. calibration on the chosen segmentation
    raw = _raw(h, TUNE_SETS, rows)
    raw_cal = copy.deepcopy(raw)                      # calibration works on a copy (the re-ranker edits segments in place)
    S, Y, G, W, GR = [], [], [], [], []
    for n in TUNE_SETS:
        segs, y, gr = _labels(h, *raw_cal[n])
        S += segs; Y.append(y); GR.append(gr)
        W.append(np.full(len(y), 1.0 / max(len(y), 1)))
        G += [f"{n}|{i}" for i, r in enumerate(raw_cal[n][1]) for _ in r]
    Y, GR, W = np.concatenate(Y), np.concatenate(GR), np.concatenate(W)
    G = np.array(G)
    setname = np.array([g.split("|")[0] for g in G])
    # learned top-5 re-ranker: kept only when its cross-validated top-1 beats the visual order by ≥ 1 point on tune
    w_rr, cv_rr, base_rr, n_rr = fit_rerank(S, GR, G, prior_eval, h.pipe.concepts)
    use_rr = cv_rr >= base_rr + 0.01
    if use_rr:              # v9: a pooled top-1 gain is not enough — tune WER with the re-ranker may not get worse on any set
        c0 = h.pipe.cal
        h.configure(cal={k: v for k, v in c0.items() if k != "rerank"}); r_off = _decide_all(h, raw)
        h.configure(cal=dict(c0, rerank=dict(w=w_rr.tolist()))); r_on = _decide_all(h, raw)
        h.configure(cal=c0)
        guard = all(r_on[n]["WER"] <= r_off[n]["WER"] + 0.003 for n in TUNE_SETS) and _objective(r_on, TUNE_SETS) > _objective(r_off, TUNE_SETS)
        _p("[rerank] tune WER without / with: %s → %s", " ".join(f"{n.replace('_tune', '')}:{r_off[n]['WER']:.3f}/{r_on[n]['WER']:.3f}" for n in TUNE_SETS),
           "kept" if guard else "dropped (a tune set gets worse)")
        use_rr = guard
    rep["rerank"] = dict(segments=n_rr, cv_top1=cv_rr, visual_top1=base_rr, used=bool(use_rr), w=w_rr.tolist())
    _p("[rerank] segments with the true word in the top-5: %d · top-1 visual %.3f -> re-ranked (CV) %.3f · %s", n_rr, base_rr, cv_rr,
       "USED" if use_rr else "not used")
    if use_rr:
        from modules.decode import rerank_candidates
        rerank_candidates(S, w_rr, h.pipe.concepts, prior_eval)
        Y, GR = _relabel(S, raw_cal)
    T, nll = fit_temperature(S, GR)
    class_features(S, h.pipe.concepts, prior_eval)
    cal_rep = dict(temperature=T, nll=nll, n_segments=int(len(Y)), base_rate=float(Y.mean()))
    feats_sets = {"v7 features (8)": FEATS, "v8 features (+prob, margin, word reliability, depth, merged)": FEATS_V8,
                  "v8 without word reliability / depth": [f for f in FEATS_V8 if f not in ("cls_rel", "log_depth")],
                  "v8.1 features (+hand cue z, hand cue agrees)": FEATS_V81, "v9 features (+landmark encoder agrees)": FEATS_V9}
    old_lr = h.pipe.lr
    P_old = np.clip(old_lr.predict(seg_features(S, getattr(old_lr, "feats", FEATS))), 1e-4, 1 - 1e-4)
    cal_rep["shipped v8 calibration"] = dict(logloss=float(-np.mean(Y * np.log(P_old) + (1 - Y) * np.log(1 - P_old))),
                                               ECE=ece(P_old, Y), reliability=reliability(P_old, Y))
    _p("[cal] T=%.4f (NLL %.3f) · %d segments, %.1f%% correct", T, nll, len(Y), 100 * Y.mean())
    _p("[cal] %-62s logloss %.4f  ECE %.4f", "shipped v8 LR (refit for the new readings below)", cal_rep["shipped v8 calibration"]["logloss"],
       cal_rep["shipped v8 calibration"]["ECE"])
    for name, fs in feats_sets.items():
        ll, e, P = cv_logloss(seg_features(S, fs), Y, W, G)
        cal_rep[name] = dict(cv_logloss=ll, cv_ECE=e, reliability=reliability(P, Y))
        _p("[cal] %-62s CV logloss %.4f  ECE %.4f", name, ll, e)
    best_fs = min(feats_sets, key=lambda k: cal_rep[k]["cv_logloss"])
    lr = fit_lr(seg_features(S, feats_sets[best_fs]), Y, W)
    cal_rep["chosen_features"] = best_fs
    # v9: isotonic re-mapping of the LR output, fitted on out-of-fold predictions; kept only if CV ECE drops and log-loss does not rise
    iso = None
    _, e_lr, P_oof = cv_logloss(seg_features(S, feats_sets[best_fs]), Y, W, G)
    iso_c = _isotonic_cv(P_oof, Y, G)
    cal_rep["isotonic"] = dict(cv_ECE_lr=e_lr, **iso_c["report"])
    if iso_c["report"]["cv_ECE"] < e_lr - 0.005 and iso_c["report"]["cv_logloss"] <= cal_rep[best_fs]["cv_logloss"] + 0.002:
        iso = iso_c["table"]
    _p("[cal] isotonic: CV ECE %.4f -> %.4f, logloss %.4f -> %.4f · %s", e_lr, iso_c["report"]["cv_ECE"], cal_rep[best_fs]["cv_logloss"],
       iso_c["report"]["cv_logloss"], "USED" if iso else "not used")
    rep["calibration"] = cal_rep

    # ---------------- 3. decision policy: thresholds + warning flags, chosen on tune (accepted precision ≥ target on EVERY tune set)
    P = lr.predict(seg_features(S, feats_sets[best_fs]))
    if iso:
        P = np.interp(P, iso["x"], iso["y"])
    seglen = np.array([s["f1"] - s["f0"] for s in S]); pm = np.array([s.get("prob_margin", 1.0) for s in S])
    target = a.accept_precision

    def policy_eval(tau, short, mm, tau_rej):
        acc_all = (P >= tau) & ~((short > 0) & (seglen < short)) & ~((mm > 0) & (pm < mm))
        out = {}
        for n in TUNE_SETS:
            m = setname == n
            acc, rej = acc_all[m], P[m] < tau_rej
            out[n] = dict(acc=int(acc.sum()), acc_ok=int(Y[m][acc].sum()), prec=float(Y[m][acc].mean()) if acc.any() else 1.0,
                          rej=int(rej.sum()), rej_wrong=float(1 - Y[m][rej].mean()) if rej.any() else 1.0, n=int(m.sum()), n_ok=int(Y[m].sum()))
        return out

    floor = float(h.pipe.spot_all.get("accept_min", 0.8))
    cands = []
    for short in (0, 10, 14):
        for mm in (0.0, 0.05, 0.1, 0.2):
            for tau in np.round(np.arange(0.5, 0.97, 0.01), 2):
                r = policy_eval(tau, short, mm, 0.0)
                cands.append(dict(tau=float(tau), short=short, margin=mm, min_prec=min(v["prec"] for v in r.values()),
                                  acc_ok=sum(v["acc_ok"] for v in r.values()), acc=sum(v["acc"] for v in r.values()), per_set=r))
    feas = [c for c in cands if c["min_prec"] >= target and c["tau"] >= floor - 1e-9]
    for label, filt in (("global threshold", lambda c: c["short"] == 0 and c["margin"] == 0),
                        ("threshold + top-1/top-2 margin", lambda c: c["short"] == 0),
                        ("threshold + margin + segment length", lambda c: True)):
        f = [c for c in feas if filt(c)]
        if f:
            b = max(f, key=lambda c: (c["acc_ok"], -c["short"], -c["margin"], -c["tau"]))
            rep["policy"].append(dict(policy=label, **{k: b[k] for k in ("tau", "short", "margin", "min_prec", "acc_ok", "acc")}))
            _p("[policy] %-38s tau=%.2f short<%d margin<%.2f -> accepted %d/%d (min set precision %.3f)", label, b["tau"], b["short"],
               b["margin"], b["acc_ok"], b["acc"], b["min_prec"])
    chosen = max(feas, key=lambda c: (c["acc_ok"], -c["short"], -c["margin"], -c["tau"])) if feas else dict(tau=0.95, short=0, margin=0.0)
    # reject: the largest τ_reject whose rejected segments are ≥ 85 % wrong over the tune sets and ≥ 80 % wrong in each set with ≥ 10
    # of them ("no word is reliable here" must really mean that); everything between τ_reject and τ_accept is Uncertain
    tau_rej = 0.05
    for t in np.round(np.arange(0.05, 0.5, 0.01), 2):
        r = policy_eval(chosen["tau"], chosen["short"], chosen["margin"], t)
        n_rej = sum(v["rej"] for v in r.values())
        pooled = sum(v["rej_wrong"] * v["rej"] for v in r.values()) / max(n_rej, 1)
        if pooled >= 0.85 and all(v["rej_wrong"] >= 0.8 for v in r.values() if v["rej"] >= 10):
            tau_rej = float(t)
    cal = dict(model=lr.to_dict(feats_sets[best_fs]), temperature=T, **({"rerank": dict(w=w_rr.tolist())} if use_rr else {}), tau_accept=float(chosen["tau"]), tau_uncertain=tau_rej,
               decision=dict(short_frames=int(chosen["short"]), margin_min=float(chosen["margin"])), fitted_on=TUNE_SETS,
               accept_precision_target=target, version=a.version, **({"isotonic": iso} if iso else {}))
    _p("[policy] chosen tau_accept=%.2f short<%d margin<%.2f tau_reject=%.2f", cal["tau_accept"], chosen["short"], chosen["margin"], tau_rej)
    rep["chosen"] = dict(spot={k: spot.get(k) for k in ("hand_w", "merge", "grammar_rerank_weight")}, calibration={k: v for k, v in cal.items() if k != "model"})

    # ---------------- 4. repeat guard (v7 "hub" rule): leave an occurrence alone when its own P(correct) ≥ repeat_keep_p. Chosen on
    # tune: no set's WER worse than the old rule (+0.003), then the fewest correct words pushed to Reject
    h.configure(spot=spot, cal=cal, prior=prior_eval)
    rep["repeat_guard"] = []
    best_rg = None
    for kp in (2.0, 0.8, 0.6, 0.5):            # ties → the more permissive rule (a repeated word is natural in a sentence)
        sp = dict(spot, post=dict(spot.get("post", {}), repeat_keep_p=kp))
        h.configure(spot=sp)
        res = _decide_all(h, raw)
        rej_ok = _correct_rejected(h, raw)
        rep["repeat_guard"].append(dict(repeat_keep_p=kp, correct_rejected=rej_ok, **{n: res[n]["WER"] for n in TUNE_SETS}))
        _p("[repeat] keep_p=%.2f correct words rejected %d | %s", kp, rej_ok, " ".join(f"{n.replace('_tune', '')}:{res[n]['WER']:.3f}" for n in TUNE_SETS))
        if best_rg is None:
            ref_res = res
        if all(res[n]["WER"] <= ref_res[n]["WER"] + 0.003 for n in TUNE_SETS) and (best_rg is None or rej_ok <= best_rg[0]):
            best_rg = (rej_ok, sp)
    spot = best_rg[1]

    # ---------------- 5. tune results with everything new
    h.configure(spot=spot, cal=cal, prior=prior_eval)
    rep["final_tune"] = _decide_all(h, raw)
    for n in TUNE_SETS:
        _p("[final] %-16s %s", n, fmt(rep["final_tune"][n]))

    # ---------------- 6. automatic fingerspelling (English letters, always on): settings chosen on tune — no letter may be inserted
    # into any tune utterance (none contains spelling), then the most letters read in the alphabet lessons of the training channels
    spell = tune_spelling(h, rows, rep)
    write_json(REP / f"decision_tune_{a.version}.json", rep)
    if a.write:
        spot_all = read_json(out_dir / "spotting.json"); spot_all["conversation"] = spot
        spot_all["spell"] = spell
        spot_all["alphabet"] = "en"
        write_json(out_dir / "spotting.json", spot_all)
        cal_all = read_json(out_dir / "segment_calibration.json"); cal_all["conversation"] = cal
        write_json(out_dir / "segment_calibration.json", cal_all)
        prod_allowed = {h.pipe.concepts.index(c) for c in read_json(out_dir / "vocab_conversation.json")["concepts"] if c in h.pipe.concepts}
        pr = class_prior(out_dir / "bank.npz", h.pipe.concepts, prod_allowed, h.fold)
        np.savez(out_dir / "class_prior.npz", rel=pr["rel"], depth=pr["depth"])
        _p("written: %s/{spotting.json, segment_calibration.json, class_prior.npz}", out_dir)


def _correct_rejected(h, raw):
    import copy
    n = 0
    for name, (rows, raws) in raw.items():
        for row, r in zip(rows, raws):
            segs = h.pipe.decide(copy.deepcopy(r))
            sc = score(segs, row, h.vocab)
            n += sum(c and s == "unknown" for c, s in zip(sc["correct"], sc["status"]))
    return n


def _letter_lessons():
    """Positives for spelling: 4 consecutive letter holds of an English alphabet lesson (continuous video between them)."""
    from train_fingerspell import std_raw
    from modules.utils import DATA
    it = pd.read_csv(DATA / "agent_data" / "en_alpha" / "items.csv").sort_values(["video", "f0"])
    pos = {"train": [], "test": []}
    for vid, g in it.groupby("video"):
        g = g.reset_index(drop=True)
        kpi, sc = std_raw(DATA / "agent_data" / "en_alpha" / "pose_mp_raw" / f"{vid}.npz")
        for i in range(0, len(g) - 3, 4):
            w = g.iloc[i:i + 4]
            a, b = max(0, int(w.f0.min()) - 8), min(len(kpi), int(w.f1.max()) + 8)
            if b - a <= 400:
                pos[w.role.iloc[0]].append(dict(kpi=kpi[a:b], sc=sc[a:b], truth=list(w.letter)))
        # v9: lone letters (one hold, as "ชื่อ J") — every 5th hold of the lesson, cut at the midpoints to its neighbours
        for i in range(2, len(g) - 1, 5):
            r = g.iloc[i]
            a = (int(g.f1.iloc[i - 1]) + int(r.f0)) // 2; b = (int(r.f1) + int(g.f0.iloc[i + 1])) // 2
            if 8 <= b - a <= 150:
                pos.setdefault(f"single_{r.role}", []).append(dict(kpi=kpi[a:b], sc=sc[a:b], truth=[r.letter]))
    return pos


def tune_spelling(h, rows, rep):
    """alphabet "en" runs on every utterance (v8.1: no switch to press). Negatives = every tune utterance (words only), positives =
    alphabet lessons of the training channels. Rule: among the settings that insert no letter into any tune utterance, the one that
    reads the most letters (edit distance); held-out lesson channels are reported."""
    import copy
    import itertools
    sys.path.insert(0, str(ROOT / "scripts"))
    p = h.pipe
    neg = []
    for n in TUNE_SETS:
        for r in rows[n]:
            st = {k: (v.astype(np.float32) if k in ("H", "H2") and v is not None else v) for k, v in r["st"].items()}
            if st["H"] is not None:
                neg.append(dict(kpi=st["kpi"], sc=st["sc"], H=st["H"], segs=h.back(r["st"], r["bank"])))
    pos = _letter_lessons()
    for split in pos.values():
        for u in split:
            present = u["sc"][:, [0, 5, 6]].mean(1) > 0.5          # lesson landmarks are already isotropic (std_raw)
            prob, H = p._tag(u["kpi"], u["sc"], present)
            u.update(H=H, segs=p.back(dict(kpi=u["kpi"], sc=u["sc"], present=present, prob=prob, H=H, f=1.0))[0])
    base = dict(p.spell_cfg)

    def run(cfg, items):
        p.spell_cfg = cfg
        return [p._spell(u["kpi"], u["sc"], u["H"], copy.deepcopy(u["segs"]), "en") for u in items]

    def evaluate(cfg, lessons):
        false = sum(sum(s.get("kind") == "letter" for s in out) for out in run(cfg, neg))
        ok = n = 0
        for u, out in zip(lessons, run(cfg, lessons)):
            got = [s["nearest"] for s in out if s.get("kind") == "letter"]
            n += len(u["truth"]); ok += len(u["truth"]) - round(wer(u["truth"], got) * len(u["truth"]))
        return dict(false_letters=false, letters_read=round(ok / max(n, 1), 3))
    grid = [dict(base, thr=t, single_min=s1, run_min=rm, run_gap=rg) for t, s1, rm, rg in
            itertools.product([0.6, 0.75, 0.85], [0.85, 0.9, 0.95, 0.98, 1.01], [0.7, 0.8, 0.9], [30])]
    res = [(cfg, evaluate(cfg, pos["train"])) for cfg in grid]
    clean = [r for r in res if r[1]["false_letters"] == 0] or sorted(res, key=lambda r: r[1]["false_letters"])[:1]
    # v9: among clean settings, letter runs first, then LONE letters of the training lessons (v8.1 never measured them: "J" alone)
    for r in clean:
        r[1]["single_read"] = evaluate(r[0], pos.get("single_train", []))["letters_read"]
    cfg, tr = max(clean, key=lambda r: (r[1]["letters_read"], r[1]["single_read"], r[0]["single_min"], r[0]["thr"]))
    # a lone letter right after a name sign: the lowest bar that still inserts no letter into any tune utterance
    cfg = dict(cfg, single_after_name=2.0)
    for v in (0.95, 0.9, 0.85, 0.8, 0.7):
        c2 = dict(cfg, single_after_name=v)
        if evaluate(c2, [])["false_letters"] == 0:
            cfg = c2
    te = evaluate(cfg, pos["test"])
    te["single_read"] = evaluate(cfg, pos.get("single_test", []))["letters_read"]
    p.spell_cfg = cfg
    chosen = {k: cfg[k] for k in ("thr", "accept", "single_min", "run_gap", "w_bank", "run_min", "single_after_name")}
    rep["spelling"] = dict(chosen=chosen, tune=tr, held_out_channels=te, negatives=len(neg), lessons={k: len(v) for k, v in pos.items()})
    _p("[spell] chosen %s → tune %s · held-out lesson channels %s", chosen, tr, te)
    return chosen


# ----------------------------------------------------------------------------- report (test sets + real videos, old vs new)
def _git_json(path):
    import subprocess
    out = subprocess.run(["git", "show", f"{a_ref()}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
    return json.loads(out.decode("utf-8"))


def a_ref():
    return "f08a56d"            # the last forward model before v8 (pitty's tagger + min_frames 10, merged into main)


VERSIONS = ("v7.2", "v8", "v8.1", "v9")
V9 = ROOT / "models" / "model-v9"          # v9 model folder (prompt_13); v8.1 = models/archive/Jack-model, frozen in artifacts/baselines/v8.1


def baseline(tag):
    """What a forward version ships: v7.2 = git f08a56d (pitty's tagger, before the decision layer), v8 = artifacts/baselines/v8/
    (prompt_11 decision layer; also the rollback copy), v8.1 = models/ now. Models (encoder, tagger, span head, bank) are shared."""
    if tag == "v7.2":
        sp = _git_json("models/spotting.json")
        return dict(spot=sp["conversation"], cal=_git_json("models/segment_calibration.json")["conversation"], prior=None,
                    vocab=_git_json("models/vocab_conversation.json")["concepts"], variants={}, spell=sp.get("spell"), alphabet="off")
    d = V9 if tag == "v9" else ART / "baselines" / tag
    sp, vc = read_json(d / "spotting.json"), read_json(d / "vocab_conversation.json")
    return dict(spot=sp["conversation"], cal=read_json(d / "segment_calibration.json")["conversation"], prior=d / "class_prior.npz",
                vocab=vc["concepts"], variants=vc.get("variants", {}), spell=sp.get("spell"), alphabet=sp.get("alphabet", "off"))


def _seg_view(s):
    c = s.get("candidates") or []
    return dict(t0=s.get("t0"), t1=s.get("t1"), status=s["status"], reason=s.get("reason"), p=round(s.get("p_correct", 0), 3),
                word=s["nearest"], top5=[(x["word"], round(x["score"], 3), round(x.get("prob", 0), 3)) for x in c[:5]],
                prob1=round(s.get("prob1", 0), 3), prob2=round(s.get("prob2", 0), 3), merged=s.get("merged", 1), frames=s["f1"] - s["f0"])


def performances(raw, gap_s=1.0, min_s=0.6):
    """How many times a clip's content is performed: runs of raised hands (modules/face.hands_up) separated by ≥ gap_s of hands
    down. team-drive: Boom's clips sign the sentence 2–3 times (checked by eye), every other clip once."""
    from modules.face import hands_up
    up = hands_up(raw); t = raw["t"]
    idx = np.flatnonzero(up)
    if not len(idx):
        return 1
    n, s, p = 0, idx[0], idx[0]
    for i in list(idx[1:]) + [None]:
        if i is None or t[i] - t[p] > gap_s:
            n += (t[p] - t[s]) > min_s
            if i is not None:
                s = i
        if i is not None:
            p = i
    return max(1, n)


def enrolled_bank(h):
    """(scenario) the TSL51 signer enrolled: their own isolated word recordings (t51u clips of the production bank, never the test
    sentences) added to the training-signer bank — what the web's personal sign memory does for a regular user."""
    from modules.handshape import HandBank
    from modules.runtime import Bank
    z = np.load(V9 / "bank.npz"); clip = np.asarray(z["clip"]).astype(str)
    m = np.array([c.startswith("t51u:") for c in clip])
    ev = np.load(ART / "eval" / "bank_eval.npz")
    C = len(h.pipe.concepts); allowed = set(h.banks["eval"].concepts_present.tolist())
    eb = Bank(np.r_[ev["Z"], z["Z"][m]], h.fold[np.r_[ev["c"], z["c"][m]]], C, allowed)
    hp, he = np.load(V9 / "hand_bank.npz"), np.load(ART / "eval" / "hand_bank_eval.npz")
    eb.hand = HandBank(np.r_[he["P"], hp["P"][m]], h.fold[np.r_[he["c"], hp["c"][m]]], C, dict(mu=he["mu"], sd=he["sd"], A=he["A"]), allowed)
    lp, le = np.load(V9 / "lm_bank.npz"), np.load(ART / "eval" / "lm_bank_eval.npz")
    okp, oke = lp["ok"].astype(bool), le["ok"].astype(bool)
    eb.lm = Bank(np.r_[le["Z"][oke], lp["Z"][m & okp]].astype(np.float32), h.fold[np.r_[le["c"][oke], lp["c"][m & okp]]], C, allowed)
    eb.personal = True
    return eb


def cmd_report(a):
    h = Harness()
    per_utt = {}
    out = dict(sets={}, calibration={}, utterances={}, latency={}, per_signer={}, configs={}, enrolled={}, real_extra={})
    versions = a.versions.split(",")
    for tag in versions:
        b = baseline(tag)
        out["configs"][tag] = dict(spot={x: b["spot"].get(x) for x in ("hand_w", "merge", "grammar_rerank_weight", "post")}, alphabet=b["alphabet"],
                                   vocab=len(b["vocab"]), cal={x: y for x, y in b["cal"].items() if x != "model"})
    for name in TEST_SETS + REAL_SETS:
        rows = load_set(name)
        out["sets"][name] = {}
        for tag in versions:
            h.use(tag)
            if rows[0]["bank"] == "prod":
                h.configure(prior=h.prod_prior)
            t = time.time()
            sc, segs = evaluate(h, rows, keep_segs=True)
            per_utt.setdefault(name, {})[tag] = np.array([x["wer"] for x in sc])
            m = aggregate(sc)
            m["back_ms_per_utt"] = (time.time() - t) / max(len(rows), 1) * 1000
            # P(correct) quality on this set (matched correctness; real videos: multiset correctness)
            P = np.array([s.get("p_correct", 0) for ss in segs for s in ss if s["status"] != "transition"], float)
            Yc = np.array([c_ for x in sc for c_ in x["correct"]], float)
            if len(P):
                Pc = np.clip(P, 1e-4, 1 - 1e-4)
                m.update(brier=float(np.mean((P - Yc) ** 2)), logloss=float(-np.mean(Yc * np.log(Pc) + (1 - Yc) * np.log(1 - Pc))))
            out["sets"][name][tag] = m
            out["calibration"].setdefault(name, {})[tag] = dict(ECE=ece(P, Yc) if len(P) else None, reliability=reliability(P, Yc) if len(P) else [],
                                                              mean_p_correct_when_right=float(P[Yc == 1].mean()) if (Yc == 1).any() else None,
                                                              mean_p_correct_when_wrong=float(P[Yc == 0].mean()) if (Yc == 0).any() else None)
            if "signer" in rows[0]:
                out["per_signer"].setdefault(name, {})[tag] = {str(k): dict(n=len(g), WER=float(np.mean([x["wer"] for x in g])))
                                                              for k, g in pd.Series(sc).groupby([r.get("signer") for r in rows])}
            if name in REAL_SETS:
                out["utterances"].setdefault(name, {})[tag] = [dict(id=r["id"], ref=r["ref"], hyp=x["hyp"], wer=x["wer"], status=x["status"],
                                                                   segments=[_seg_view(s) for s in ss if s["status"] != "transition"])
                                                              for r, x, ss in zip(rows, sc, segs)]
                # v9: WER against the in-vocabulary reference words (an out-of-vocabulary sign can only be "[?]") and, for clips
                # that perform the sentence k times, against the reference repeated k times; precision of the words shown
                inv = [wer([w for w in r["ref"] if w in h.vocab], x["hyp"]) for r, x in zip(rows, sc) if any(w in h.vocab for w in r["ref"])]
                reps = [performances(_reload_raw(r)) for r in rows]
                perf = [wer(list(r["ref"]) * k, x["hyp"]) for r, x, k in zip(rows, sc, reps)]
                shown = [(st_, c_) for x in sc for st_, c_ in zip(x["status"], x["correct"]) if st_ in ("accepted", "uncertain")]
                out["real_extra"].setdefault(name, {})[tag] = dict(WER_in_vocab=float(np.mean(inv)) if inv else None, WER_per_performance=float(np.mean(perf)),
                                                                   performances=reps, shown_precision=float(np.mean([c_ for _, c_ in shown])) if shown else None,
                                                                   oov_ref_words=int(sum(w not in h.vocab for r in rows for w in r["ref"])),
                                                                   ref_words=int(sum(len(r["ref"]) for r in rows)))
                per_utt.setdefault(name + "|perf", {})[tag] = np.array(perf)
            _p("%-16s %-5s %s", name, tag, fmt(m))
            if name.startswith("t51_test") and tag in ("v8.1", "v9"):        # the enrolled-signer scenario (separate, clearly labelled)
                keep, h.banks["eval"] = h.banks["eval"], enrolled_bank(h)
                sce = evaluate(h, rows)
                h.banks["eval"] = keep
                me = aggregate(sce)
                out["enrolled"].setdefault(name, {})[tag] = me
                per_utt.setdefault(name + "|enrolled", {})[tag] = np.array([x["wer"] for x in sce])
                _p("%-16s %-5s ENROLLED %s", name, tag, fmt(me))
    # paired bootstrap (10,000 resamples of utterances) of the WER difference on identical samples
    rng = np.random.RandomState(0)
    out["bootstrap"] = {}
    pairs = [(n_, o_) for n_, o_ in (("v9", "v8.1"), ("v8.1", "v8"), ("v8.1", "v7.2"), ("v8", "v7.2")) if n_ in versions and o_ in versions]
    for name, W in per_utt.items():
        for new, old in pairs:
            if new not in W or old not in W:
                continue
            d = W[new] - W[old]
            bs = d[rng.randint(0, len(d), (10000, len(d)))].mean(1)
            out["bootstrap"].setdefault(name, {})[f"{new}-{old}"] = dict(diff=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                                                                         n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()))
        k = f"{pairs[0][0]}-{pairs[0][1]}"
        if k in out["bootstrap"].get(name, {}):
            x = out["bootstrap"][name][k]
            _p("%-22s %s dWER %+.3f [%+.3f, %+.3f] better %d worse %d", name, k, x["diff"], x["lo"], x["hi"], x["better"], x["worse"])
    # latency + memory: full pipeline per real video (front + back + spelling; MediaPipe measured separately on two videos)
    import os
    import psutil
    from modules.mediapipe_pose import extract
    proc = psutil.Process(os.getpid())
    vids = [r for n in REAL_SETS for r in load_set(n)]
    for tag in versions:
        h.use(tag); h.configure(prior=h.prod_prior)
        ts, secs = [], []
        for r in vids:
            kp, sc_, hw, present = h.standard(dict(raw=_reload_raw(r)))
            t = time.perf_counter()
            st = h.pipe.front(kp, sc_, hw, present)
            segs, _, _ = h.pipe.back(st, h.banks["prod"])
            if h.alphabet != "off" and st["H"] is not None:
                h.pipe._spell(st["kpi"], st["sc"], st["H"], segs, h.alphabet)
            ts.append((time.perf_counter() - t) * 1000); secs.append(len(kp) / 25.0)
        out["latency"][tag] = dict(model_ms_mean=float(np.mean(ts)), model_ms_per_video_second=float(np.sum(ts) / np.sum(secs)), videos=len(ts),
                                   rss_mb=proc.memory_info().rss / 2 ** 20)
    t = time.perf_counter(); n = 0
    for r in vids[:2]:
        raw = extract(r["video"]); n += len(raw["t"])
    out["latency"]["mediapipe_ms_per_frame"] = (time.perf_counter() - t) * 1000 / max(n, 1)
    _p("latency %s", out["latency"])
    write_json(REP / f"decision_{a.tag}.json", out)


def _reload_raw(row):
    from modules.mediapipe_pose import load_raw
    v = Path(row["video"])
    if v.parent.name == "data_test":
        return load_raw(ROOT / "cache" / "data_test_mp" / f"{v.stem}.npz")
    if "team-drive" in v.parts:
        return load_raw(ROOT / "cache" / "modeling_compare" / "mp" / f"{v.parent.name}__{v.stem}.npz")
    return load_raw(ROOT / "cache" / "iphone_test_mp" / f"{v.stem}.npz")


# ----------------------------------------------------------------------------- figures for result_reporting/result_<tag>.md
def cmd_figures(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Leelawadee UI", "Tahoma", "DejaVu Sans"], "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})
    out = ROOT / "result_reporting" / f"figures_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    rep = read_json(REP / f"decision_{a.tag}.json")
    tune = read_json(REP / "decision_tune.json")
    COL = {"v7.2": "#cbd5e1", "v8": "#94a3b8", "v8.1": "#4f46e5"}
    sets = TEST_SETS + REAL_SETS
    lab = {"t51_test_x1.0": "TSL51 1×", "t51_test_x1.5": "TSL51 1.5×", "t51_test_x2.0": "TSL51 2×", "iso_test": "word clips\n(held-out)",
           "flu_test": "fluent synth", "data_test": "data_test", "iphone_word": "iPhone word", "iphone_sentence": "iPhone sentence"}
    # 1. WER / Recall@1 / count error / accepted precision per set, three versions
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.8))
    x = np.arange(len(sets))
    for ax, (k, title) in zip(axs, (("WER", "WER ↓"), ("R1", "Recall@1 ↑"), ("count_err", "count error ↓"), ("acc_prec", "accepted-word precision ↑"))):
        for j, tag in enumerate(VERSIONS):
            ax.bar(x + (j - 1) * 0.27, [rep["sets"][n][tag][k] for n in sets], 0.27, color=COL[tag], label=tag)
        ax.set_xticks(x); ax.set_xticklabels([lab[n] for n in sets], rotation=40, ha="right"); ax.set_title(title)
    axs[0].legend()
    fig.tight_layout(); fig.savefig(out / "sets.png", dpi=130); plt.close(fig)
    # 2. reliability of P(correct) (test sets pooled), three versions
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, tag in zip(axs, VERSIONS):
        conf, acc, n = np.zeros(10), np.zeros(10), np.zeros(10)
        for name in TEST_SETS:
            for k, b_ in enumerate(rep["calibration"][name][tag]["reliability"]):
                if b_["n"]:
                    conf[k] += b_["conf"] * b_["n"]; acc[k] += b_["acc"] * b_["n"]; n[k] += b_["n"]
        m = n > 0
        e = float(np.sum(n[m] / n.sum() * np.abs(conf[m] / n[m] - acc[m] / n[m])))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.bar(np.arange(10)[m] / 10 + 0.05, acc[m] / n[m], 0.09, color=COL[tag], alpha=0.8)
        ax.plot(conf[m] / n[m], acc[m] / n[m], "o-", color="#dc2626", ms=4)
        ax.set_title(f"{tag}: ECE {e:.3f} (10 equal-width bins, n={int(n.sum())})"); ax.set_xlabel("predicted P(correct)"); ax.set_ylabel("observed accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(out / "reliability.png", dpi=130); plt.close(fig)
    # 3. tune: every recognition / segmentation choice (mean tune WER; chosen = indigo)
    seg = tune["segmentation"]
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(seg) + 1))
    ch = tune["chosen"]["spot"]
    ax.barh(range(len(seg)), [-s_["objective"] for s_ in seg], color=[COL["v8.1"] if s_["spot"] == ch else COL["v8"] for s_ in seg])
    ax.set_yticks(range(len(seg))); ax.set_yticklabels([s_["tag"] for s_ in seg]); ax.invert_yaxis()
    ax.set_xlabel("mean WER over the 5 tune sets ↓"); ax.set_xlim(min(-s_["objective"] for s_ in seg) - 0.01, max(-s_["objective"] for s_ in seg) + 0.005)
    ax.set_title("choices on TUNE data (chosen = indigo)")
    fig.tight_layout(); fig.savefig(out / "tune_choices.png", dpi=130); plt.close(fig)
    _p("figures → %s", out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache"); c.add_argument("--sets", default=""); c.add_argument("--force", action="store_true")
    b = sub.add_parser("baseline"); b.add_argument("--sets", default=""); b.add_argument("--out", default="")
    t = sub.add_parser("tune"); t.add_argument("--accept_precision", type=float, default=0.9); t.add_argument("--write", action="store_true")
    t.add_argument("--start", default="v8.1", choices=["v8", "v8.1"]); t.add_argument("--out", default=str(V9)); t.add_argument("--version", default="v9")
    t.add_argument("--grammar", default="off", choices=["off", "tune"])
    t.add_argument("--spot_from", default="", help="reuse the segmentation of this spotting.json (decision stages only)")
    r = sub.add_parser("report"); r.add_argument("--tag", default="v9"); r.add_argument("--versions", default="v8.1,v9")
    f = sub.add_parser("figures"); f.add_argument("--tag", default="v8.1")
    hs = sub.add_parser("hands"); hs.add_argument("--shrink", type=float, default=0.1); hs.add_argument("--dim", type=int, default=64)
    lb = sub.add_parser("lmbank"); lb.add_argument("--out", default=str(ROOT / "cache" / "v9" / "model")); lb.add_argument("--bank", default=str(MODEL / "bank.npz"))
    a = p.parse_args()
    {"cache": cmd_cache, "baseline": cmd_baseline, "tune": cmd_tune, "report": cmd_report, "figures": cmd_figures, "hands": cmd_hands,
     "lmbank": cmd_lmbank}[a.cmd](a)


if __name__ == "__main__":
    main()
