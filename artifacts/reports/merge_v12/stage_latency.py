"""Per-stage latency of the merged stack in docker (gateway :8080), LLM off vs on.
Replays recorded utterances (landmarks only) -> /api/v1/translate, then the product's next calls: /api/v1/face (face picture branch)
and /tts/v1/speak (voice). Writes rows to argv[1] (jsonl)."""
import sys, os, json, gzip, glob, time, base64, random
import numpy as np, httpx
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"D:\SLM-labs\SLM-Labs-SignDINO\cache"
G = "http://127.0.0.1:8080"
C = httpx.Client(timeout=180)
OUT = sys.argv[1]
N_LIVE = int(sys.argv[2]) if len(sys.argv) > 2 else 30


def utterances():
    rows = sorted(glob.glob(ROOT + r"\replay_src\*.json.gz"))
    live = sorted(glob.glob(ROOT + r"\live_requests\*.json.gz"))
    rows += live[::max(1, len(live) // N_LIVE)][:N_LIVE]
    for f in rows:
        d = json.load(gzip.open(f, "rt", encoding="utf-8"))
        if len(d["frames"]) < 10 or d["frames"][-1]["t"] - d["frames"][0]["t"] > 15000:
            continue
        yield os.path.basename(f), d


def translate(d, llm):
    t0 = d["frames"][0]["t"]
    body = dict(frames=[{**f, "t": f["t"] - t0} for f in d["frames"]], width=d["width"], height=d["height"], vocab=d.get("vocab", "conversation"),
                llm=llm, speak=True, speak_min_p=0.8, user_bank=[], alphabet=d.get("alphabet", "en"))
    t = time.perf_counter(); r = C.post(G + "/api/v1/translate", json=body); ms = (time.perf_counter() - t) * 1000
    r.raise_for_status()
    return r.json(), ms


rng = np.random.default_rng(0)


def face_call(d):
    """one live /v1/face call as the web sends it (5 crops ≈ 1 s of camera at 180 ms); crops are synthetic (no pictures are saved), points real"""
    fr = [f for f in d["frames"] if f.get("face") and len(f["face"]) == 68][:5]
    if not fr:
        return None
    crops = []
    for f in fr:
        g = rng.integers(60, 200, 2304, dtype=np.uint8); g96 = rng.integers(60, 200, 9216, dtype=np.uint8)
        pts = [v for p in f["face"] for v in (p[0] * d["width"], p[1] * d["height"])]
        crops.append(dict(t=f["t"], g=base64.b64encode(g.tobytes()).decode(), g96=base64.b64encode(g96.tobytes()).decode(), side=120, occ=0.0, pts=pts))
    t = time.perf_counter(); r = C.post(G + "/api/v1/face", json=dict(crops=crops, state=None)); ms = (time.perf_counter() - t) * 1000
    return ms if r.status_code == 200 else None


SPOKEN = set()


def speak(text, emo, inten):
    if not text or text in SPOKEN:                 # the tts service caches by text: only first-time sentences are timed
        return None
    SPOKEN.add(text)
    t = time.perf_counter()
    r = C.post(G + "/tts/v1/speak", json=dict(text=text, voice="male", emotion=emo, intensity=inten))
    ms = (time.perf_counter() - t) * 1000
    if r.status_code != 200:
        return dict(err=r.status_code, body=r.text[:200])
    n = len(r.content); sec = (n - 44) / 2 / 24000 if r.headers.get("x-engine", "").startswith("thonburian") else None
    return dict(ms=ms, engine=r.headers.get("x-engine"), emotion=r.headers.get("x-emotion"), audio_s=sec)


if __name__ == "__main__":
    U = list(utterances())
    print("utterances", len(U), flush=True)
    for name, d in U[:2]:                           # warm-up (not recorded)
        translate(d, False); translate(d, True); face_call(d)
    speak("ทดสอบเสียง", "neutral", 0.5)
    with open(OUT, "w", encoding="utf-8") as fo:
        for i, (name, d) in enumerate(U):
            dur = (d["frames"][-1]["t"] - d["frames"][0]["t"]) / 1000
            for llm in (False, True):
                r, ms = translate(d, llm)
                row = dict(name=name, llm=llm, dur_s=dur, n_frames=len(d["frames"]), rt_ms=ms, timings=r.get("timings_ms"),
                           sentence=r.get("sentence"), provider=(r.get("language") or {}).get("provider"),
                           speak_text=r.get("speak_text"), speak_emotion=r.get("speak_emotion"),
                           n_seg=len(r.get("segments", [])), n_acc=sum(s["status"] == "accepted" for s in r.get("segments", [])))
                row["face_ms"] = face_call(d) if llm is False else None
                row["tts"] = speak(r.get("speak_text"), r.get("speak_emotion") or "neutral", r.get("speak_intensity") or 0.5)
                fo.write(json.dumps(row, ensure_ascii=False) + "\n"); fo.flush()
            print(i + 1, "/", len(U), name, round(ms), r.get("sentence"), flush=True)
