# affect_v2 — which emotion the voice should carry (input side)

`affect_v2.py` replaces `modules.face_fer.speech_affect` (v1: one rule, non-neutral and p >= 0.5). Pure Python. `python test_affect_v2.py` runs 15 scenario checks.
Status: v0. Thresholds come from Face-Lab validation precision, not from labelled signing clips (none exist yet). Not run inside the web app yet.

## Try it in the web app (one edit, on a branch, in `webapp/api/app.py::_voice_affect`)
```python
def _voice_affect(out):
    import sys; sys.path.insert(0, "TTS-Lab/affect_v2")            # inside docker: mount TTS-Lab/affect_v2 and use that path
    from affect_v2 import speech_affect_v2
    fer = out.get("face_fer") or {}
    emo, inten, why = speech_affect_v2(fer.get("summary") if fer.get("ok") else None, (out.get("face") or {}).get("affect"),
                                       grammar=out.get("face"), segs=out.get("segments"))
    out["speak_why"] = why                                          # shows in the saved request, for tuning
    return emo, inten
```
## Tuning once labelled clips exist
Record N utterances per intended emotion through the web app (requests are saved to `cache/live_requests/`), note the intended emotion per file,
replay v1 and v2 on the saved `face_fer.summary` / `face.affect` / `segments`, then grid-search `CFG["thr"]`, `margin`, `agree_bonus`, `word_bonus` on a tune split.
