"""Translate any sign-language video with the deployable model in models/.

  python main.py infer --video path/to/video.mp4 [--out dir] [--vocab conversation|full] [--no_llm] [--tts] [--backend onnx|torch]

Writes result.json (segments with status / p_correct / nearest / top-5, sentence or withheld reading, face cues, timings),
subtitle.srt (word · word? · [?] ≈ nearest), timeline.png (signer present / P(sign) / segments) and, with --tts, speech.wav.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--out")
    ap.add_argument("--vocab", default="conversation", choices=["conversation", "full"])
    ap.add_argument("--no_llm", action="store_true"); ap.add_argument("--no_face", action="store_true"); ap.add_argument("--tts", action="store_true")
    ap.add_argument("--backend", default="onnx", choices=["onnx", "torch"]); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    from modules.pipeline import SignPipeline
    video = Path(a.video)
    out = Path(a.out) if a.out else ROOT / "result_reporting" / "inference_cli" / video.stem
    pipe = SignPipeline(backend=a.backend, device=a.device, vocab=a.vocab, use_llm=not a.no_llm, use_face=not a.no_face)
    r = pipe.run(video, out, tts=a.tts)
    f = r.get("fusion") or {}
    print(json.dumps(dict(segments=[(s["t0"], s["t1"], s["status"], s["nearest"]) for s in r["segments"]], evidence=f.get("evidence_gloss"),
                          sentence=r.get("sentence") or None, tentative=f.get("tentative_sentence_th"), meaning=f.get("meaning_th"),
                          face={k: (r.get("face") or {}).get(k) for k in ("question_yesno", "negation", "emotion")}, timings_ms=r["timings_ms"],
                          out=str(out)), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
