"""Round 2: the head retrained WITHOUT the smiling 'neutral' signing clips (rav_head.sign_neutral(max_happy=0.2)). Same CV / metrics as
rav_head.py + rav_prior.py, then the team clips through the app path (check_ravhead.app_path) for a few neutral priors.
  python rav_head2.py → candidates/rav_head_signing_nosmile.npz, results/rav_head2.json"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "Face-Lab"))
from modules.face_fer import FaceFER                     # noqa: E402
from modules.face_model import EMOTIONS                  # noqa: E402
from face_model import fit_lr                            # noqa: E402
from compare24 import metrics                            # noqa: E402
import rav_head as R                                     # noqa: E402
from rav_prior import load_samples                       # noqa: E402
import check_ravhead as C                                # noqa: E402
import real_check                                        # noqa: E402

MAX_HAPPY = 0.2


def main():
    XP, y, actor, whole, inten, meta, to7, e_idx = load_samples()
    Xd = R.sign_neutral(to7, e_idx, max_happy=MAX_HAPPY); print("neutral signing samples", len(Xd), flush=True)
    acts = np.unique(actor); folds = [acts[i::6] for i in range(6)]
    L = np.zeros((len(XP), len(EMOTIONS)))
    for te in folds:
        m = np.isin(actor, te)
        L[m] = fit_lr(np.r_[XP[~m], Xd], np.r_[y[~m], np.full(len(Xd), 0)], 0.1).logits(XP[m])
    rep = dict(max_happy=MAX_HAPPY, n_neutral_signing=len(Xd))
    for ps in (0.0, 0.5, 1.0):
        Lb = L.copy(); Lb[:, 0] += ps; rep[f"ravdess_prior_{ps}"] = r = metrics(Lb, y, whole, inten)
        print("RAVDESS prior", ps, r["clips_macro_f1"], r["per_class_recall"], flush=True)
    head = fit_lr(np.r_[XP, Xd], np.r_[y, np.full(len(Xd), 0)], 0.1)
    out = ROOT / "Face-Lab" / "candidates" / "rav_head_signing_nosmile.npz"; head.save(out)
    m = FaceFER(ROOT / "models" / "face-model-ravhead-v1")
    z = np.load(out); m.head.mu, m.head.sd, m.head.W, m.head.b = (z[k].astype(np.float64) for k in ("mu", "sd", "W", "b"))
    import cv2
    S = np.load(real_check.CACHE, allow_pickle=True); jl = real_check.judged(S["clip"], S["t"], S["q"])
    s96 = list(np.stack([cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) for c in S["col"]]))
    for ps in (0.0, 0.5, 1.0):
        m.head.bias = np.array([ps if c == "neutral" else 0.0 for c in m.classes])
        rep[f"team_prior_{ps}"] = r = C.app_path(m, S, s96, jl); print("team prior", ps, r, flush=True)
    (ROOT / "Face-Lab" / "results" / "rav_head2.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
