"""v12 4F: open BLIND-v12 ONCE (+ legacy BLIND-v11, 3rd opening) and score v10 / v11 / v12 in the same run. No change after this.

  THAISLM_OPEN_BLIND=1 THAISLM_OPEN_BLIND12=1 python scripts/v12_blind.py

1. log both openings (manifests) · 2. cache nsig_blind12 (isolated in-vocabulary clips of the two BLIND-v12 people) · 3. paired v10 vs v11 / v12
on nsig_blind12 (E2), t51_blind x1/x1.5/x2 + flu_blind (E1 pooled), tail_blind (E5) · 4. caption-bag recall / precision of accepted words on
the BLIND-v12 single-person natural sentences (Royal Institute captions; order ignored) · matched-precision coverage comes from
`diagnose v12 --sets legacy` (run separately with the same flags).
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from modules.sealed import blind12_open, blind_open, log_opening
from modules.utils import ART, PREP


def caption_bag(tags):
    import logging; logging.disable(logging.WARNING)
    from modules.mediapipe_pose import load_raw
    from modules.pipeline import SignPipeline
    N = pd.read_parquet(PREP / "manifest" / "clips_v12_new.parquet")
    q = N[N.role.eq("blind_v12") & N.kind.eq("sentence") & ~N.dialog & N.extracted & N.caption_words.notna() & N.signer.isin(["S_rid_00", "S_rid_02"])]
    out = {}
    per = {}
    for tag in tags:
        p = SignPipeline(model_dir=ROOT / "models" / f"model-{tag}", backend="onnx", device="cpu", vocab="conversation", use_llm=False, use_face=False)
        rows = []
        for r in q.itertuples():
            res = p.run_landmarks(load_raw(PREP / r.pose_raw_file))
            acc = [s["nearest"] for s in res["segments"] if s["status"] == "accepted"]
            top = [s["nearest"] for s in res["segments"] if s["status"] != "transition"]
            cw = set(str(r.caption_words).split("|"))
            rows.append(dict(id=r.clip_id, n_cap=len(cw), hit=len(set(acc) & cw), n_acc=len(acc), ok_acc=sum(w in cw for w in acc),
                             hit_top=len(set(top) & cw), n_top=len(top)))
        D = pd.DataFrame(rows); per[tag] = D
        out[tag] = dict(n_sentences=len(D), caption_words=int(D.n_cap.sum()), recall=float(D.hit.sum() / D.n_cap.sum()),
                        precision=float(D.ok_acc.sum() / max(D.n_acc.sum(), 1)), accepted_words=int(D.n_acc.sum()),
                        top1_recall=float(D.hit_top.sum() / D.n_cap.sum()))
        print("caption-bag", tag, out[tag], flush=True)
    rng = np.random.RandomState(0)
    for tag in tags[1:]:
        a, b = per[tags[0]], per[tag]
        d = []
        for _ in range(2000):
            i = rng.randint(0, len(a), len(a))
            d.append(b.hit.values[i].sum() / b.n_cap.values[i].sum() - a.hit.values[i].sum() / a.n_cap.values[i].sum())
        out[tag]["recall_diff_vs_" + tags[0]] = float(out[tag]["recall"] - out[tags[0]]["recall"])
        out[tag]["recall_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return out


def main():
    assert blind_open() and blind12_open(), "set THAISLM_OPEN_BLIND=1 and THAISLM_OPEN_BLIND12=1 (TPH 4, once)"
    log_opening("v12 TPH 4F: v10 / v11 / v12 scored once (E1, E2, E5, caption bag); no change after this", which="v12")
    log_opening("v12 TPH 4F: legacy-blind 3rd opening (v10 / v11 / v12 continuity, never used to select)", which="v11")
    py = sys.executable
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    subprocess.run([py, "main.py", "decision", "cache", "--sets", "nsig_blind12"], cwd=ROOT, env=env, check=True)
    sets = "nsig_blind12,t51_blind_x1.0,t51_blind_x1.5,t51_blind_x2.0,flu_blind,iso_blind,tail_blind"
    subprocess.run([py, "scripts/v12_paired.py", "--ref", "v10", "--tags", "v11,v12", "--sets", sets, "--out", "blind"], cwd=ROOT, env=env, check=True)
    cb = caption_bag(["v10", "v11", "v12"])
    (ART / "reports" / "v12_blind_caption_bag.json").write_text(json.dumps(cb, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
