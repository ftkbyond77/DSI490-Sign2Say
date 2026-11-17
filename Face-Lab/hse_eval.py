"""Off-the-shelf AffectNet model (HSEmotion, Savchenko 2022: github.com/av-savchenko/face-emotion-recognition, code Apache-2.0, weights trained on
AffectNet = research / education use) against our own FER2013 models, on the SAME face boxes:
  * RAVDESS actors 3-4 (labelled acted video, never used to pick a checkpoint)  -> expression recall, neutral kept, clip Macro-F1
  * the 18 signer clips (unlabelled; hand-judged neutral frames + 3 smiles)     -> over-reacting on neutral faces, smiles caught

HSEmotion needs a colour face picture (224 px for B0, 260 px for B2, ImageNet normalisation). Variants tried, to see what the web page would have to send:
  colour   - colour crop (what a changed web page would send)
  gray     - the same crop in gray, copied to 3 channels
  gray48   - the 48x48 gray crop the web page sends TODAY, scaled up (no web change at all)
Its 8 classes: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise -> contempt is dropped and the rest renormalised.

  python hse_eval.py            (env with onnxruntime + cv2; caches from video_eval.py / real_check.py)
  -> results/hse_eval.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "Face-Lab"))
from modules.face_fer import CLASSES, FaceFER   # noqa: E402
from video_eval import scores                    # noqa: E402
import real_check                                # noqa: E402

HSE_DIR = ROOT / "data_test" / "hsemotion"
HSE_ORDER = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
TO7 = [HSE_ORDER.index(c) for c in CLASSES]       # our class order, contempt left out
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


class HSE:
    def __init__(self, name):
        self.name = name
        self.sess = ort.InferenceSession(str(HSE_DIR / f"{name}.onnx"), providers=["CPUExecutionProvider"])
        self.size = self.sess.get_inputs()[0].shape[2]

    def logits(self, imgs):
        """imgs uint8 [N,H,W,3] RGB (or [N,H,W] gray) -> logits over OUR 7 classes (contempt column dropped)."""
        out = []
        for i in range(0, len(imgs), 16):
            b = imgs[i:i + 16]
            if b.ndim == 3:
                b = np.repeat(b[..., None], 3, -1)
            x = np.stack([cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_CUBIC) for im in b]).astype(np.float32) / 255.0
            x = ((x - MEAN) / STD).transpose(0, 3, 1, 2)
            out.append(self.sess.run(None, {"input": np.ascontiguousarray(x)})[0])
        return np.concatenate(out)[:, TO7]


def probs(z, prior=0.0):
    z = z.astype(np.float64) + np.array([prior if c == "neutral" else 0.0 for c in CLASSES])
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def gray(col):
    return np.stack([cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) for c in col])


def main():
    out = dict(note=__doc__.split("\n\n")[0], ravdess={}, signers={})
    R = np.load(ROOT / "Face-Lab" / "cache" / "ravdess_faces.npz")
    ok = (R["q"] >= 0.25) & np.isin(R["actor"], [3, 4])
    clip, lab, g48, col = R["clip"][ok], R["label"][ok], R["g"][ok], R["col"][ok]
    clips = sorted(set(clip)); cy = np.array([lab[clip == c][0] for c in clips])
    S = np.load(real_check.CACHE, allow_pickle=True)
    jl = real_check.judged(S["clip"], S["t"], S["q"]); n, h = jl == CLASSES.index("neutral"), jl == CLASSES.index("happy")

    def run(nm, zr, zs, priors):
        out["ravdess"][nm], out["signers"][nm] = {}, {}
        for pr in priors:
            P = probs(zr, pr); cp = np.stack([P[clip == c].mean(0) for c in clips]).argmax(1)
            s = scores(cy, cp); out["ravdess"][nm][f"prior_{pr}"] = s
            ps = probs(zs, pr).argmax(1)
            sg = dict(neutral_frames_answered_neutral=round(float((ps[n] == CLASSES.index("neutral")).mean()), 3),
                      neutral_frames_wrong_expression=round(float((ps[n] != CLASSES.index("neutral")).mean()), 3),
                      smiles_happy=f"{int((ps[h] == CLASSES.index('happy')).sum())}/{int(h.sum())}")
            out["signers"][nm][f"prior_{pr}"] = sg
            print(f"{nm:34s} prior {pr:+.1f} | RAV a3-4 F1 {s['macro_f1']:.3f} acc {s['accuracy']:.3f} exprRec {s['expression_recall']:.3f} neuKept {s['neutral_kept']}"
                  f" | signers neutral {sg['neutral_frames_answered_neutral']:.3f} wrong {sg['neutral_frames_wrong_expression']:.3f} smiles {sg['smiles_happy']}", flush=True)

    priors = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    for b in [ROOT / "models" / "face-model-efficientnetb0-v1", ROOT / "Face-Lab" / "candidates" / "face-model-efficientnetb0-v2-ferplus"]:
        m = FaceFER(b); nm = b.name.replace("face-model-", "")
        zr = np.concatenate([m.logits(g48[i:i + 256]) for i in range(0, len(g48), 256)]) / m.T
        zs = np.concatenate([m.logits(S["g"][i:i + 256]) for i in range(0, len(S["g"]), 256)]) / m.T
        run(nm, zr, zs, priors)
    for name in ("enet_b0_8_best_vgaf", "enet_b2_8"):
        H = HSE(name)
        run(f"{name}/colour", H.logits(col), H.logits(S["col"]), priors)
        run(f"{name}/gray", H.logits(gray(col)), H.logits(gray(S["col"])), priors)
        run(f"{name}/gray48", H.logits(g48), H.logits(S["g"]), priors)
    (ROOT / "Face-Lab" / "results" / "hse_eval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
