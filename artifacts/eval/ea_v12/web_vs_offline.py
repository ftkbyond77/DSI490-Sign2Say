"""Same source video: offline MediaPipe raw (25 fps, corpus) vs browser-rate re-extraction (13 fps, cache/v12_web) — hand presence + geometry."""
import sys, glob, numpy as np, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"D:/SLM-labs/SLM-Labs-SignDINO"); S = Path(sys.argv[1])
off_dirs = {"thsl": "th_sl", "ttrs": "ttrs", "t51s": "tsl51_user_sentence", "t51u": "tsl51_user_sign", "agent": "agent"}
rows = []
for f in sorted(glob.glob(str(ROOT / "cache/v12_web/*.npz"))):
    name = Path(f).stem; pre = name.split("_")[0]
    cands = [ROOT / "cloud_s3/data_prep/pose_mp_raw" / off_dirs[pre] / (x + ".npz") for x in (name, name[len(pre) + 1:])]
    o = next((c for c in cands if c.exists()), None)
    if o is None:
        continue
    w = np.load(f, allow_pickle=True); d = np.load(o, allow_pickle=True)
    def st(r):
        lh = np.abs(r["lh"]).sum((1, 2)) > 0; rh = np.abs(r["rh"]).sum((1, 2)) > 0; t = np.asarray(r["t"], float)
        fps = (len(t) - 1) / max(t[-1] - t[0], 1e-6)
        # runs of "no hand" inside the signing part (between first and last hand frame)
        any_ = lh | rh; idx = np.where(any_)[0]
        inner = any_[idx[0]:idx[-1] + 1] if len(idx) else np.array([True])
        # per-frame wrist speed in shoulder widths / s (dominant = right hand of the body pose)
        p = r["pose"]; sw = np.median(np.hypot(p[:, 11, 0] - p[:, 12, 0], p[:, 11, 1] - p[:, 12, 1])) or 1
        return dict(fps=fps, any=float(any_.mean()), both=float((lh & rh).mean()), inner_missing=float(1 - inner.mean()), dur=float(t[-1] - t[0]), sw=float(sw),
                    flips=float(np.abs(np.diff(any_.astype(int))).sum() / max(t[-1] - t[0], 1e-6)))
    a, b = st(d), st(w)
    rows.append(dict(name=name, src=pre, **{"off_" + k: v for k, v in a.items()}, **{"web_" + k: v for k, v in b.items()}))
print("pairs", len(rows))
import collections
for src in ["all"] + sorted({r["src"] for r in rows}):
    R = [r for r in rows if src == "all" or r["src"] == src]
    m = lambda k: float(np.mean([r[k] for r in R]))
    print(f"{src:6s} n={len(R):4d} fps off/web {m('off_fps'):.1f}/{m('web_fps'):.1f} | any-hand {m('off_any'):.3f}/{m('web_any'):.3f} | both {m('off_both'):.3f}/{m('web_both'):.3f} | "
          f"missing inside sign {m('off_inner_missing'):.3f}/{m('web_inner_missing'):.3f} | hand on/off flips per s {m('off_flips'):.2f}/{m('web_flips'):.2f}")
json.dump(rows, open(S / "web_vs_offline.json", "w"), ensure_ascii=False)
