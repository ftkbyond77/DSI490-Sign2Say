"""v11 minimal-pair verifier (prompt-opt-v11 §3B). Runs only when a segment's conformal set holds 2–3 words of ONE confusion group.

Per (segment, candidate word) features:
  emb_proto   cosine of the segment embedding to the candidate's form prototypes (max over forms)
  emb_exem    mean of the two best exemplar cosines of different signers
  sub_k       the four phonology sub-space cosines when the encoder is factorised (R6), else 0
  dtw         soft-DTW-like cost (DTW on cosine distance, length-normalised) between the segment's frame states and the candidate's
              exemplar frame sequences (best over ≤ 6 exemplars of different signers)
  hand        hand-shape descriptor distance to the candidate's mean descriptor (modules/handshape.descriptor, standardised)
  loc / mov   distance of the dominant hand centre to nose / chin at the hold, net wrist direction cosine with the candidate's mean
Model: gradient-boosted trees per group (sklearn), trained leave-signers-out on TRAIN clips of the group's words (real + 2 virtual
signer copies each), checked on TUNE. A group keeps its verifier only if its TUNE R@1 rises and no word outside it changes.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path as Path_

FEATS = ["emb_proto", "emb_exem", "sub0", "sub1", "sub2", "sub3", "dtw", "hand", "loc_nose", "loc_chin", "mov_cos", "len_ratio"]


def dtw_cost(A, B):
    """A [n,d], B [m,d] L2-normalised rows → mean cosine distance along the best monotone alignment"""
    n, m = len(A), len(B)
    if n == 0 or m == 0:
        return 1.0
    C = 1.0 - A @ B.T
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0.0
    for i in range(1, n + 1):
        ci = C[i - 1]
        for j in range(1, m + 1):
            D[i, j] = ci[j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def _sub(n_frames, k=12):
    return np.linspace(0, n_frames - 1, min(k, n_frames)).round().astype(int)


class WordRef:
    """per word: form prototypes, exemplar embeddings / signers, exemplar frame sequences, mean hand descriptor, location / movement"""

    def __init__(self):
        self.P, self.E, self.S, self.F, self.D, self.loc, self.mov, self.L = [], [], [], [], [], [], [], []


def phon_stats(kp, sc, f0, f1):
    from .phonology import raw_features
    r = raw_features(kp, sc, f0, f1)
    if r is None:
        return np.zeros(2), np.zeros(2)
    c = r["c"]; sw = r["sw"]
    loc = np.array([np.linalg.norm(c - r["nose"]) / sw, np.linalg.norm(c - r["chin"]) / sw])
    nv = r["net_v"]; mv = nv / (np.linalg.norm(nv) + 1e-6)
    return loc, mv


def pair_features(seg, ref, n_sub=0):
    """seg: dict(z [D], Hs [k,d] subsampled normalised frame states, desc [DIM], loc [2], mov [2], len)"""
    z = seg["z"]
    proto = max(float(z @ p) for p in ref.P) if ref.P else 0.0
    sims = sorted(((float(z @ e), s) for e, s in zip(ref.E, ref.S)), reverse=True)
    best, seen = [], set()
    for v, s in sims:
        if s not in seen:
            best.append(v); seen.add(s)
        if len(best) == 2:
            break
    exem = float(np.mean(best)) if best else 0.0
    subs = [0.0] * 4
    if n_sub:
        zs = np.split(z * np.sqrt(n_sub), n_sub)
        ps = [np.split(p * np.sqrt(n_sub), n_sub) for p in ref.P]
        subs = [max(float(zs[k] @ pp[k]) for pp in ps) if ps else 0.0 for k in range(n_sub)]
    dtw = min((dtw_cost(seg["Hs"], F) for F in ref.F[:6]), default=1.0)
    hand = float(np.linalg.norm(seg["desc"] - np.mean(ref.D, 0))) if ref.D else 0.0
    loc = seg["loc"] - (np.mean(ref.loc, 0) if ref.loc else 0)
    mvc = float(seg["mov"] @ (np.mean(ref.mov, 0) if ref.mov else np.zeros(2)))
    lr = seg["len"] / (np.mean(ref.L) if ref.L else seg["len"])
    return [proto, exem] + subs + [dtw, hand, abs(loc[0]), abs(loc[1]), mvc, lr]


class Verifier:
    """trained per-group models + word references; __call__(seg, candidates) → chosen word or None (not confident)"""

    def __init__(self, refs, models, n_sub=0, thr=0.65, margin=0.15):
        self.refs, self.models, self.n_sub, self.thr, self.margin = refs, models, n_sub, thr, margin

    def choose(self, segfeat, cands):
        key = tuple(sorted(cands))
        model = next((m for g, m in self.models.items() if set(cands) <= set(g)), None)
        if model is None or any(c not in self.refs for c in cands):
            return None
        X = np.array([pair_features(segfeat, self.refs[c], self.n_sub) for c in cands])
        p = model.predict_proba(X)[:, 1]
        o = np.argsort(-p)
        if p[o[0]] >= self.thr and (len(p) == 1 or p[o[0]] - p[o[1]] >= self.margin):
            return cands[o[0]]
        return None


class RuntimeVerifier:
    """pipeline hook: SignPipeline.verifier(seg, candidates) → chosen word or None. Reads the utterance landmarks the pipeline sets in
    `_cur_kp` before decide(); features with the encoder the verifier was fitted on (torch, CPU)."""

    def __init__(self, pkl, enc_ckpt, pipe, device="cpu"):
        import pickle
        from .lmenc import load_model, prep_for
        d = pickle.load(open(pkl, "rb"))
        self.v = Verifier(d["refs"], d["models"], d.get("n_sub", 0))
        self.model = load_model(enc_ckpt, device); self.prep = prep_for(self.model.cfg); self.dev = device
        self.pipe = pipe
        self.calls = self.changed = 0

    def __call__(self, seg, cands):
        cur = getattr(self.pipe, "_cur_kp", None)
        if cur is None:
            return None
        kp, sc = cur
        f0, f1 = int(seg["f0"]), max(int(seg["f1"]), int(seg["f0"]) + 4)
        if f1 > len(kp):
            return None
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
        from v11_verifier import segfeat
        fe = segfeat(self.model, self.prep, kp, sc, f0, min(f1, len(kp)), self.dev)
        self.calls += 1
        w = self.v.choose(fe, list(cands))
        self.changed += w is not None and w != seg.get("nearest")
        return w


# ------------------------------------------------------------------ production path (ONNX Runtime + NumPy, no torch / sklearn)
def export_trees(clf):
    """sklearn GradientBoostingClassifier (binary) → plain arrays"""
    trees = []
    for est in clf.estimators_[:, 0]:
        t = est.tree_
        trees.append(dict(f=t.feature.tolist(), th=t.threshold.tolist(), l=t.children_left.tolist(), r=t.children_right.tolist(),
                          v=t.value[:, 0, 0].tolist()))
    init = clf.init_
    prior = float(np.log(init.class_prior_[1] / init.class_prior_[0])) if hasattr(init, "class_prior_") else float(clf._raw_predict_init(np.zeros((1, clf.n_features_in_)))[0, 0])
    return dict(trees=trees, lr=float(clf.learning_rate), init=prior)


class NumpyTrees:
    def __init__(self, d):
        self.t, self.lr, self.init = d["trees"], d["lr"], d["init"]

    def predict_proba(self, X):
        X = np.asarray(X, float)
        raw = np.full(len(X), self.init)
        for t in self.t:
            for i, x in enumerate(X):
                n = 0
                while t["l"][n] != -1:
                    n = t["l"][n] if x[t["f"][n]] <= t["th"][n] else t["r"][n]
                raw[i] += self.lr * t["v"][n]
        p = 1 / (1 + np.exp(-raw))
        return np.stack([1 - p, p], 1)


def export_runtime(pkl, out_dir, enc_ckpt):
    """verifier.pkl (+ encoder) → <out_dir>/verifier/{verifier.pkl.json refs + trees, frames.onnx}"""
    import json
    import pickle
    from .lmenc import export_frames_onnx
    out = Path_(out_dir) / "verifier"; out.mkdir(parents=True, exist_ok=True)
    d = pickle.load(open(pkl, "rb"))
    refs = {w: {k: [np.asarray(x).tolist() for x in getattr(r, k)] if k != "S" else list(r.S) for k in ("P", "E", "S", "F", "D", "loc", "mov", "L")}
            for w, r in d["refs"].items()}
    models = [dict(group=list(g), model=export_trees(m)) for g, m in d["models"].items()]
    (out / "verifier.json").write_text(json.dumps(dict(refs=refs, models=models, n_sub=d.get("n_sub", 0)), ensure_ascii=False), encoding="utf-8")
    err = export_frames_onnx(enc_ckpt, out / "frames.onnx")
    return err


class OnnxVerifier:
    """production verifier: embeddings from the shipped lmenc.onnx, frame states from verifier/frames.onnx, trees in NumPy"""

    def __init__(self, model_dir, pipe):
        import json
        import onnxruntime as ort
        from .lmenc import OnnxLM
        vd = Path_(model_dir) / "verifier"
        d = json.loads((vd / "verifier.json").read_text(encoding="utf-8"))
        refs = {}
        for w, r in d["refs"].items():
            ref = WordRef()
            for k, v in r.items():
                setattr(ref, k, [np.asarray(x, np.float32) for x in v] if k != "S" else list(v))
            refs[w] = ref
        self.v = Verifier(refs, {tuple(m["group"]): NumpyTrees(m["model"]) for m in d["models"]}, d.get("n_sub", 0))
        self.lm = OnnxLM(Path_(model_dir) / "onnx" / "lmenc.onnx")
        self.frames = ort.InferenceSession(str(vd / "frames.onnx"), providers=["CPUExecutionProvider"])
        self.pipe = pipe

    def _segfeat(self, kp, sc, f0, f1):
        from .handshape import descriptor
        from .lmenc import feats_fn
        z = self.lm.spans(kp, sc, [(f0, f1)])[0]
        nf = self.frames.get_inputs()[0].shape[-1]
        f = (feats_fn(11) if nf != 122 else feats_fn(10))(kp[f0:f1], sc[f0:f1])
        H = self.frames.run(None, {"x": f[None].astype(np.float32), "m": np.ones((1, len(f)), np.float32)})[0][0]
        H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
        loc, mov = phon_stats(kp, sc, f0, f1)
        return dict(z=z, Hs=H[_sub(len(H))], desc=descriptor(kp, sc, f0, f1), loc=loc, mov=mov, len=f1 - f0)

    def __call__(self, seg, cands):
        cur = getattr(self.pipe, "_cur_kp", None)
        if cur is None:
            return None
        kp, sc = cur
        f0, f1 = int(seg["f0"]), min(max(int(seg["f1"]), int(seg["f0"]) + 4), len(kp))
        return self.v.choose(self._segfeat(kp, sc, f0, f1), list(cands))
