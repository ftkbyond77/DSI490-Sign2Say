"""Non-interference test against the RUNNING system (docker compose up): the same sign landmarks are sent to /api/v1/translate
  A  without any face data (face branch idle)            = baseline
  B  with the real face crops of the video
  C  with garbage crops (noise / black / wrong size / bad base64)
  D  with the crops' timestamps shifted by -3 s and +3 s
  E  with the face fully covered (occ = 1, tiny face)
and compares everything the sign path produces (segments incl. scores and embeddings, sentence, language, v9 face cues, duration) —
byte for byte after JSON normalisation. Only `timings_ms`, `face_fer` and the voice choice (speak_*; the speech layer MAY use the
expression) are allowed to differ. Also reports the sign-side latency difference.

  python non_interference.py --url http://localhost:8080/api --clips <mp4> [<mp4> ...]
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.face_fer import crops_from_video            # noqa: E402
from modules.mediapipe_pose import extract                # noqa: E402

IGNORE = {"timings_ms", "face_fer", "speak_emotion", "speak_intensity", "audio_wav_b64"}


def frames_from_raw(raw):
    fr = []
    for i in range(len(raw["t"])):
        pose = raw["pose"][i]; face = raw["face"][i]; bs = raw["bs"][i]
        fr.append(dict(t=float(raw["t"][i] * 1000), pose=pose.tolist() if np.abs(pose).sum() else [], lh=raw["lh"][i].tolist() if np.abs(raw["lh"][i]).sum() else [],
                       rh=raw["rh"][i].tolist() if np.abs(raw["rh"][i]).sum() else [], face=face.tolist() if np.abs(face).sum() else [],
                       bs=bs.tolist() if np.abs(bs).sum() else []))
    return fr


def b64(g):
    return base64.b64encode(bytes(g)).decode()


def crops_payload(crops, shift_ms=0.0, occ=None, side=None):
    return [dict(t=c["t"] * 1000 + shift_ms, g=b64(c["g"]), side=c["side_px"] if side is None else side, occ=c["occluded"] if occ is None else occ) for c in crops]


def garbage(n=40):
    rng = np.random.RandomState(0)
    g = [dict(t=i * 200.0, g=b64(rng.randint(0, 256, 2304, np.uint8)), side=80, occ=0) for i in range(n // 2)]
    g += [dict(t=i * 200.0 + 100, g=b64(np.zeros(2304, np.uint8)), side=80, occ=0) for i in range(n // 4)]
    g += [dict(t=5.0, g=b64(b"short"), side=80, occ=0), dict(t=6.0, g="%%%not-base64%%%", side=80, occ=0)]
    return g


def post(url, body):
    t = time.perf_counter()
    r = requests.post(f"{url}/v1/translate", json=body, timeout=300)
    r.raise_for_status()
    return r.json(), (time.perf_counter() - t) * 1000


def norm(o):
    return json.dumps({k: v for k, v in o.items() if k not in IGNORE}, sort_keys=True, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080/api")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "Face-Lab" / "results" / "non_interference.json"))
    ap.add_argument("--repeat", type=int, default=2, help="repeats per condition for the latency comparison")
    a = ap.parse_args()
    res = dict(url=a.url, clips=[], conditions=["A baseline", "B real crops", "C garbage crops", "D shifted +-3 s", "E covered face"])
    for clip in a.clips:
        raw = extract(clip)
        if raw is None:
            continue
        hw = raw["hw"]
        base = dict(frames=frames_from_raw(raw), width=int(hw[1]), height=int(hw[0]), vocab="conversation", llm=False, speak=True, speak_min_p=0.0)
        crops = crops_from_video(clip, raw)
        conds = {"A": [], "B": crops_payload(crops), "C": garbage(), "D-": crops_payload(crops, -3000), "D+": crops_payload(crops, 3000),
                 "E": crops_payload(crops, occ=1.0, side=18)}
        outs, lat = {}, {}
        for k, c in conds.items():
            ts = []
            for _ in range(a.repeat):
                o, ms = post(a.url, dict(base, face_crops=c)); ts.append(ms)
            outs[k], lat[k] = o, float(np.median(ts))
        ref = norm(outs["A"])
        row = dict(clip=Path(clip).name, n_frames=len(base["frames"]), n_crops=len(crops), sentence=outs["A"].get("sentence"),
                   words=[s["word"] for s in outs["A"]["segments"] if s["status"] != "transition"],
                   identical={k: norm(o) == ref for k, o in outs.items() if k != "A"}, latency_ms=lat,
                   fer={k: (outs[k].get("face_fer") or {}).get("summary", {}).get("expression") for k in outs},
                   fer_valid_frac={k: round((outs[k].get("face_fer") or {}).get("summary", {}).get("valid_frac", 0), 2) for k in outs},
                   speak_emotion={k: outs[k].get("speak_emotion") for k in outs},
                   sign_ms={k: outs[k]["timings_ms"].get("total_ms") for k in outs}, face_ms={k: (outs[k].get("face_fer") or {}).get("latency_ms") for k in outs})
        res["clips"].append(row)
        print(row["clip"], "identical:", row["identical"], "FER:", row["fer"], "voice:", row["speak_emotion"], flush=True)
    res["all_identical"] = all(all(r["identical"].values()) for r in res["clips"])
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("ALL IDENTICAL:", res["all_identical"], "->", a.out)
    sys.exit(0 if res["all_identical"] else 1)


if __name__ == "__main__":
    main()
