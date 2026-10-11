"""pitty: make real TSL51 sentences more fluent by cutting the still holds between signs (no synthetic joins).

TSL51 "no_space" signers do not lower their hands between signs, but they hold still ~1 s and sign slower (2.4 s / word vs
1.7 s in data_test) — enough for the tagger to see every boundary. cut_holds() removes runs of frames where both hands are
nearly still (max hand speed < thr) inside the signing span, keeping `keep` frames of each, so the real signs and real
transitions stay and only the waiting is removed. Used by sentence_eval.py (env HOLD_CUT="thr,keep,min_run").
    python pitty/scripts/hold_cut.py      # pick the setting: v7 tagger runs/word on TSL51 test no_space (target ≈ data_test fluent)"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def cut_holds(kp, sc, hw, thr=0.1, keep=2, min_run=4):
    """kp / sc as stored (pose_mp), hw = (H, W) → kp, sc with long still runs shortened to `keep` frames."""
    from modules.encoder import to_iso
    from modules.segment import kin_features
    F = kin_features(to_iso(kp, hw), sc)
    up = np.flatnonzero(F[:, 11] > -1.0)                 # signing span: a hand above the lower chest
    if len(up) < 2:
        return kp, sc
    a, b = up[0], up[-1] + 1
    still = F[:, 12] < thr
    keepmask = np.ones(len(kp), bool)
    t = a
    while t < b:
        if not still[t]:
            t += 1; continue
        u = t
        while u < b and still[u]:
            u += 1
        if u - t >= min_run:
            keepmask[t + keep:u] = False
        t = u
    return kp[keepmask], sc[keepmask]


if __name__ == "__main__":
    import json
    import logging
    logging.disable(logging.WARNING)
    import onnxruntime as ort
    import pandas as pd
    from modules.encoder import to_iso
    from modules.pipeline import SignPipeline
    from modules.schema import SH_L, SH_R, canonical_hands
    pipe = SignPipeline(backend="onnx", device="cpu", vocab="conversation", use_llm=False)
    pipe.be.tagger = ort.InferenceSession(str(ROOT / "cache/backup_models_v7/tagger.onnx"), providers=["CPUExecutionProvider"])
    A = json.loads((ROOT / "artifacts/runs/seq-v7/tsl51_alignment.json").read_text(encoding="utf-8"))
    items = [(cid, a) for cid, a in A.items() if cid.startswith("t51s:") and a["role"] != "tagger_train" and a["variation"] == "no_space"]
    rows = []
    for name, cfg in {"ไม่ตัด": None, "thr0.10 keep2": (0.10, 2, 4), "thr0.15 keep2": (0.15, 2, 4), "thr0.20 keep2": (0.20, 2, 4),
                      "thr0.15 keep0": (0.15, 0, 3), "thr0.20 keep0": (0.20, 0, 3), "thr0.25 keep0": (0.25, 0, 3)}.items():
        R = W = 0; secs = []
        for cid, a in items:
            z = np.load(ROOT / "cloud_s3/data_prep/pose_mp/tsl51_user_sentence" / f"{cid.split(':', 1)[1]}.npz")
            kp, sc = z["kp"].astype(np.float32), z["sc"].astype(np.float32); hw = tuple(int(v) for v in z["hw"])
            if cfg:
                kp, sc = cut_holds(kp, sc, hw, *cfg)
            present = sc[:, [0, SH_L, SH_R]].mean(1) > 0.5
            sc2 = sc.copy(); sc2[~present] = 0; kp2 = kp.copy(); kp2[~present] = 0
            kp2, sc2, _ = canonical_hands(kp2, sc2, hw)
            prob, _ = pipe._tag(to_iso(kp2, hw), sc2, present)
            act = (prob[:, 1] + prob[:, 2]) > 0.5
            R += int((act[1:] & ~act[:-1]).sum() + act[0]); W += len(a["glosses"]); secs.append(len(kp) / 25 / len(a["glosses"]))
        rows.append({"แบบ": name, "ท่อนต่อคำ (tagger v7)": R / W, "วินาทีต่อคำ": float(np.mean(secs))})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows).set_index("แบบ")
    print(out.round(3).to_string()); print("เป้า: data_test ทำท่าต่อเนื่อง = 0.25 ท่อนต่อคำ · 1.69 วินาทีต่อคำ")
    out.to_csv(ROOT / "cache/analysis/hold_cut_check.csv", encoding="utf-8-sig")
