"""Overall-system (end-to-end) evaluation (prompt-opt-v12 4F-2): camera landmarks → sign → face → sentence → speech, three modules joined.
Measurement only (frozen models, nothing tuned, never changes the ship decision). One re-runnable command; every module version is an
argument so the owner can re-run it after the team merges new Face / TTS versions:

  python main.py e2e-eval --sign v10,v11,v12 [--set e2e_v12] [--asr elevenlabs|local|none] [--max-asr 150] [--face on|off] [--llm on|off]
                          [--tts http://127.0.0.1:8001] [--build]

Set (artifacts/eval/e2e_v12.json, built once with --build, ids frozen): TSL51 sentence templates of the BLIND people (legacy BLIND-v11
templates, already opened), Royal Institute captioned sentences of the BLIND-v12 people (single-person clips), negative controls (TSL51
not-signing clips). Input = raw MediaPipe landmarks of the video (pose + hands + face + blendshapes) → SignPipeline.run_landmarks.
Final text = what the product would SPEAK (webapp/api/app.py _spoken_text at speak_min_p 0.8). Text metrics on pythainlp newmm tokens:
BLEU-4 (sacrebleu, tokenize none) · chrF++ (characters) · ROUGE-1 / ROUGE-L F1 · METEOR (exact match only; no Thai stemmer / synonyms) ·
token WER · character CER · exact match · speech-produced / empty rates. Speech: the product's TTS (/v1/speak, the voice / emotion the chain
chose) → ASR (ElevenLabs scribe_v2, Thai; cached by sha1(text+voice+emotion+intensity); hard cap --max-asr calls and 15 audio minutes;
one retry; on an auth / quota error it stops calling and falls back to the local ASR, rows marked asr=local) → ASR-CER / WER vs the TTS
input and vs the ground truth; floor = TTS(ground truth) → ASR on 20 items. Latency / RTF: sign + face + language compute per utterance.
Outputs: artifacts/reports/v12_e2e.json + e2e_rows.jsonl. The API key is read from .env and never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import unicodedata
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
SIDE = ROOT / "cache" / "side_pkgs"
if SIDE.exists():
    sys.path.append(str(SIDE))
if (ROOT / "cache" / "side_api").exists():                # httpx (TTS / ASR calls) lives with the local API side packages
    sys.path.append(str(ROOT / "cache" / "side_api"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from modules.utils import ART, CACHE, PREP, read_json, write_json

SET = ART / "eval" / "e2e_v12.json"
ACACHE = CACHE / "e2e_audio"
REP = ART / "reports"
PUNCT = re.compile(r"[\s\.,!?\"'“”‘’…:;\-\(\)\[\]]+")
DIGITS = {"0": "ศูนย์", "1": "หนึ่ง", "2": "สอง", "3": "สาม", "4": "สี่", "5": "ห้า", "6": "หก", "7": "เจ็ด", "8": "แปด", "9": "เก้า"}


# ----------------------------------------------------------------------------- set
def build_set():
    from modules.sealed import blind_open, blind12_open
    assert blind_open() and blind12_open(), "the e2e set uses BLIND people: run after the BLIND opening (THAISLM_OPEN_BLIND=1 THAISLM_OPEN_BLIND12=1)"
    import decision as D
    sent = read_json(PREP / "params" / "tsl51_thai_sentences.json")
    items = []
    rows = D.load_set("t51_blind_x1.0")
    seen = set(); seen_list = []
    for r in rows:
        cid = r["id"].split("@")[0]
        key = " ".join(sorted(r["ref"]))
        ref = sent.get(key) or sent.get(" ".join(r["ref"]))
        raw = PREP / "pose_mp_raw" / "tsl51_user_sentence" / f"{cid.split(':', 1)[1]}.npz"
        if ref and raw.exists() and seen_list.count(key) < 3:          # up to 3 recordings of a template (different takes)
            seen.add(key) if False else None
            seen = seen | {key} if isinstance(seen, set) else seen
            seen_list.append(key)
            items.append(dict(id=cid, part="tsl51_blind", raw=str(raw.relative_to(PREP)), gt=ref, gloss=r["ref"],
                              question=bool(re.search(r"(ไหม|อะไร|ที่ไหน|ใคร|เมื่อไร|กี่|ทำไม|อย่างไร|หรือเปล่า)", ref))))
    items = items[:30]
    N = pd.read_parquet(PREP / "manifest" / "clips_v12_new.parquet")
    q = N[N.role.eq("blind_v12") & N.kind.eq("sentence") & ~N.dialog & N.signer.isin(["S_rid_00", "S_rid_02"]) & N.extracted & N.caption.notna()]
    q = q[q.caption_words.fillna("").str.count(r"\|") >= 1]
    def clean(c):
        return re.sub(r"\s+", " ", re.sub(r"(ถาม|ตอบ)\s*:", " ", str(c))).strip()
    qs = q[q.caption.str.contains("ไหม|อะไร|ที่ไหน|ใคร|กี่|ทำไม|\\?")]
    nq = min(15, len(qs))
    rest = q.drop(qs.index, errors="ignore")
    pick = pd.concat([qs.sample(nq, random_state=12), rest.sample(min(30 - nq, len(rest)), random_state=12)])
    for r in pick.itertuples():
        items.append(dict(id=r.clip_id, part="rid_blind", raw=r.pose_raw_file, gt=clean(r.caption), gloss=str(r.caption_words).split("|"),
                          question=bool(re.search(r"ไหม|อะไร|ที่ไหน|ใคร|กี่|ทำไม|\?", str(r.caption)))))
    idle = D.load_set("idle_test")[:6]
    for r in idle:
        cid = str(r["id"]).split("@")[0]
        raw = PREP / "pose_mp_raw" / "tsl51_user_sign" / f"{cid.split(':', 1)[-1]}.npz"
        if raw.exists():
            items.append(dict(id=cid, part="negative", raw=str(raw.relative_to(PREP)), gt="", gloss=[], question=False))
    write_json(SET, dict(created=time.strftime("%Y-%m-%d %H:%M"), n=len(items), parts={p: sum(i["part"] == p for i in items) for p in ("tsl51_blind", "rid_blind", "negative")},
                         note="built once after the BLIND opening; ids frozen", items=items))
    print("e2e set", len(items), {p: sum(i["part"] == p for i in items) for p in ("tsl51_blind", "rid_blind", "negative")})


# ----------------------------------------------------------------------------- metrics
def toks(s):
    from pythainlp.tokenize import word_tokenize
    return [t for t in word_tokenize(norm(s), engine="newmm", keep_whitespace=False) if t.strip()]


def norm(s):
    s = unicodedata.normalize("NFC", str(s or ""))
    s = "".join(DIGITS.get(ch, ch) for ch in s)
    return PUNCT.sub(" ", s).strip()


def edit(a, b):
    d = np.arange(len(b) + 1)
    for i, x in enumerate(a, 1):
        p = d.copy(); d[0] = i
        for j, y in enumerate(b, 1):
            d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (x != y))
    return int(d[-1])


def wer(ref, hyp):
    r, h = toks(ref), toks(hyp)
    return edit(r, h) / max(len(r), 1) if r else float(len(h) > 0)


def cer(ref, hyp):
    r, h = norm(ref).replace(" ", ""), norm(hyp).replace(" ", "")
    return edit(list(r), list(h)) / max(len(r), 1) if r else float(len(h) > 0)


def lcs(a, b):
    m = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            m[i + 1][j + 1] = m[i][j] + 1 if a[i] == b[j] else max(m[i][j + 1], m[i + 1][j])
    return m[-1][-1]


def rouge(ref, hyp):
    r, h = toks(ref), toks(hyp)
    if not r or not h:
        return 0.0, 0.0
    from collections import Counter
    ov = sum((Counter(r) & Counter(h)).values())
    f1 = 2 * ov / (len(r) + len(h))
    l = lcs(r, h); p_, r_ = l / len(h), l / len(r)
    return f1, (2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0)


def meteor(ref, hyp):
    """METEOR with exact matching only (Banerjee & Lavie 2005: Fmean = 10PR/(R+9P), penalty 0.5·(chunks/matches)^3)"""
    r, h = toks(ref), toks(hyp)
    if not r or not h:
        return 0.0
    used, align = set(), []
    for i, t in enumerate(h):
        for j, u in enumerate(r):
            if j not in used and t == u:
                used.add(j); align.append((i, j)); break
    m = len(align)
    if not m:
        return 0.0
    P, R = m / len(h), m / len(r)
    fmean = 10 * P * R / (R + 9 * P)
    chunks = 1 + sum(1 for (a, b), (c, d) in zip(align, align[1:]) if not (c == a + 1 and d == b + 1))
    return fmean * (1 - 0.5 * (chunks / m) ** 3)


def text_metrics(rows):
    import sacrebleu
    pos = [r for r in rows if r["gt"]]
    hyp = [" ".join(toks(r["text"])) for r in pos]; ref = [" ".join(toks(r["gt"])) for r in pos]
    out = dict(n=len(pos), BLEU4=round(sacrebleu.corpus_bleu(hyp, [ref], tokenize="none").score, 2),
               sentence_BLEU=round(float(np.mean([sacrebleu.sentence_bleu(a, [b], tokenize="none", smooth_method="exp").score for a, b in zip(hyp, ref)])), 2),
               chrF=round(sacrebleu.corpus_chrf([norm(r["text"]).replace(" ", "") for r in pos], [[norm(r["gt"]).replace(" ", "") for r in pos]], word_order=2).score, 2))
    rg = [rouge(r["gt"], r["text"]) for r in pos]
    out.update(ROUGE1=float(np.mean([x[0] for x in rg])), ROUGEL=float(np.mean([x[1] for x in rg])), METEOR=float(np.mean([meteor(r["gt"], r["text"]) for r in pos])),
               WER=float(np.mean([wer(r["gt"], r["text"]) for r in pos])), CER=float(np.mean([cer(r["gt"], r["text"]) for r in pos])),
               exact=float(np.mean([norm(r["gt"]).replace(" ", "") == norm(r["text"]).replace(" ", "") for r in pos])),
               speech_produced=float(np.mean([bool(norm(r["text"]).strip()) for r in pos])),
               empty=float(np.mean([not norm(r["text"]).strip() for r in pos])))
    qs = [r for r in pos if r["question"]]
    out["question_accuracy"] = float(np.mean([bool(re.search(r"ไหม|อะไร|ที่ไหน|ใคร|กี่|ทำไม|\?|หรือ", r["text"])) for r in qs])) if qs else None
    out["question_n"] = len(qs)
    neg = [r for r in rows if not r["gt"]]
    out["negative_spoken_rate"] = float(np.mean([bool(norm(r["text"]).strip()) for r in neg])) if neg else None
    return out


# ----------------------------------------------------------------------------- speech
class ASR:
    def __init__(self, mode, max_calls, max_minutes=15.0):
        self.mode, self.max_calls, self.max_s = mode, max_calls, max_minutes * 60
        self.calls, self.seconds, self.dead = 0, 0.0, False
        self.key = None
        if mode == "elevenlabs":
            for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
                if line.startswith("ELEVEN_LABS_API="):
                    self.key = line.split("=", 1)[1].strip().strip('"')
        self.local = None
        self.cache_f = ACACHE / "asr_cache.json"
        self.cache = json.loads(self.cache_f.read_text(encoding="utf-8")) if self.cache_f.exists() else {}

    def _local(self, wav):
        if self.local is None:
            from modules.speech import ThaiASR
            self.local = ThaiASR()
        f = ACACHE / "_asr_tmp.wav"; ACACHE.mkdir(parents=True, exist_ok=True); f.write_bytes(wav)
        return self.local(str(f))

    def __call__(self, key, wav, dur):
        if key in self.cache:
            return self.cache[key]
        if self.mode == "elevenlabs" and not self.dead and self.calls < self.max_calls and self.seconds + dur <= self.max_s:
            import httpx
            for attempt in range(2):
                try:
                    r = httpx.post("https://api.elevenlabs.io/v1/speech-to-text", headers={"xi-api-key": self.key},
                                   files={"file": ("a.wav", wav, "audio/wav")}, data={"model_id": "scribe_v2", "language_code": "tha", "tag_audio_events": "false"},
                                   timeout=60)
                    self.calls += 1
                    if r.status_code in (401, 402, 403, 429):
                        print(f"ASR: ElevenLabs refused ({r.status_code}) -> local ASR from here on", flush=True); self.dead = True; break
                    r.raise_for_status()
                    self.seconds += dur
                    out = dict(text=r.json().get("text", ""), asr="elevenlabs")
                    self.cache[key] = out; self._save()
                    print(f"  ASR calls {self.calls}/{self.max_calls}, audio {self.seconds / 60:.2f} / {self.max_s / 60:.0f} min", flush=True)
                    return out
                except Exception as e:  # noqa: BLE001
                    print("ASR error:", type(e).__name__, str(e)[:120], flush=True)
                    if attempt:
                        self.dead = True
        if self.mode == "none":
            return dict(text=None, asr="none")
        out = dict(text=self._local(wav), asr="local")
        self.cache[key] = out; self._save()
        return out

    def _save(self):
        ACACHE.mkdir(parents=True, exist_ok=True)
        self.cache_f.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")


def tts(url, text, voice, emotion, intensity):
    import httpx
    key = hashlib.sha1(f"{text}|{voice}|{emotion}|{intensity}".encode("utf-8")).hexdigest()
    f = ACACHE / f"{key}.wav"
    if not f.exists():
        t = time.perf_counter()
        r = httpx.post(f"{url}/v1/speak", json=dict(text=text, voice=voice, emotion=emotion, intensity=float(intensity or 0.5)), timeout=180)
        r.raise_for_status()
        ACACHE.mkdir(parents=True, exist_ok=True); f.write_bytes(r.content)
        (ACACHE / f"{key}.ms").write_text(str((time.perf_counter() - t) * 1000))
    wav = f.read_bytes()
    with wave.open(io.BytesIO(wav)) as w:
        dur = w.getnframes() / w.getframerate()
    ms = float((ACACHE / f"{key}.ms").read_text()) if (ACACHE / f"{key}.ms").exists() else None
    return key, wav, dur, ms


# ----------------------------------------------------------------------------- run
def spoken_text(r, min_p=0.8):
    signs = [s for s in r.get("segments", []) if s["status"] != "transition"]
    shows = lambda s: s["status"] == "accepted" or (s.get("p_correct") or 0) >= min_p      # noqa: E731
    if any(s["status"] != "accepted" and shows(s) for s in signs):
        return re.sub(r"(…\s*){2,}", "… ", " ".join(s["nearest"] if shows(s) else "…" for s in signs)).strip()
    return r.get("sentence") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sign", default="v10,v11,v12"); ap.add_argument("--set", default="e2e_v12"); ap.add_argument("--asr", default="elevenlabs")
    ap.add_argument("--max-asr", dest="max_asr", type=int, default=150); ap.add_argument("--face", default="on"); ap.add_argument("--llm", default="off")
    ap.add_argument("--tts", default="http://127.0.0.1:8001"); ap.add_argument("--build", action="store_true"); ap.add_argument("--voice", default="male")
    ap.add_argument("--ablations", action="store_true"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default="v12_e2e")
    a = ap.parse_args()
    if a.build:
        build_set(); return
    import logging; logging.disable(logging.WARNING)
    from modules.mediapipe_pose import load_raw
    from modules.pipeline import SignPipeline
    S = read_json(SET)
    if a.limit:                                          # dry run (prompt: 3 clips first)
        S = dict(S, items=[i for i in S["items"] if i["gt"]][:a.limit], n=a.limit)
    asr = ASR(a.asr, a.max_asr)
    rows, summ = [], dict(created=time.strftime("%Y-%m-%d %H:%M"), set=a.set, n_items=S["n"], parts=S["parts"], asr=a.asr, face=a.face, llm=a.llm,
                          modules=dict(face=str((ROOT / "models" / "face-model").resolve().name), tts=os.environ.get("SPEECH_MODEL_DIR", "models/Speech-model-v4-TTS"),
                                       voice=a.voice), systems={})
    for tag in a.sign.split(","):
        pipe = SignPipeline(model_dir=ROOT / "models" / f"model-{tag}", backend="onnx", device="cpu", vocab="conversation", use_llm=a.llm == "on",
                            use_face=a.face == "on")
        R = []
        for it in S["items"]:
            raw = load_raw(PREP / it["raw"])
            t = time.perf_counter(); r = pipe.run_landmarks(raw); ms = (time.perf_counter() - t) * 1000
            dur = float(raw["t"][-1] - raw["t"][0]) if len(raw["t"]) > 1 else 0.0
            text = spoken_text(r)
            face = r.get("face") or {}
            emo = (face.get("affect") or {}).get("emotion") or face.get("emotion")
            top = " ".join(s["nearest"] for s in r.get("segments", []) if s["status"] != "transition")
            R.append(dict(system=tag, id=it["id"], part=it["part"], gt=it["gt"], question=it["question"], text=text, top1_words=top,
                          accepted_words=" ".join(s["nearest"] for s in r.get("segments", []) if s["status"] == "accepted"), emotion=emo,
                          compute_ms=ms, input_s=dur, timings=r.get("timings_ms")))
        m = text_metrics(R)
        m["RTF_recognition"] = float(np.sum([x["compute_ms"] for x in R]) / 1000 / max(np.sum([x["input_s"] for x in R]), 1e-6))
        m["compute_ms_p50"], m["compute_ms_p90"] = float(np.percentile([x["compute_ms"] for x in R], 50)), float(np.percentile([x["compute_ms"] for x in R], 90))
        if a.ablations and tag == a.sign.split(",")[-1]:
            ab = {}
            ab["oracle_gloss_rules"] = text_metrics([dict(x, text=" ".join(it["gloss"])) for x, it in zip(R, S["items"])])
            ab["top1_no_acceptance"] = text_metrics([dict(x, text=x["top1_words"].replace(" ", "")) for x in R])
            ab["accepted_only_words"] = text_metrics([dict(x, text=x["accepted_words"].replace(" ", "")) for x in R])
            m["ablations"] = ab
        # speech
        if a.asr != "none":
            for x in R:
                if not norm(x["text"]).strip():
                    continue
                emo = x["emotion"] if x["emotion"] not in (None, "neutral") else None
                try:
                    key, wav, d, tms = tts(a.tts, x["text"].replace("…", " "), a.voice, emo, 0.5)
                except Exception as e:  # noqa: BLE001
                    x["tts_error"] = str(e)[:100]; continue
                tr = asr(key, wav, d)
                x.update(audio_s=d, tts_ms=tms, asr_text=tr["text"], asr=tr["asr"])
                if tr["text"] is not None:
                    x.update(asr_cer_vs_tts=cer(x["text"].replace("…", " "), tr["text"]), asr_wer_vs_tts=wer(x["text"].replace("…", " "), tr["text"]),
                             asr_cer_vs_gt=cer(x["gt"], tr["text"]) if x["gt"] else None, asr_wer_vs_gt=wer(x["gt"], tr["text"]) if x["gt"] else None)
            sp = [x for x in R if x.get("asr_text") is not None]
            for src in ("elevenlabs", "local"):
                ss = [x for x in sp if x["asr"] == src]
                if ss:
                    m[f"speech_{src}"] = dict(n=len(ss), ASR_CER_vs_TTS=float(np.mean([x["asr_cer_vs_tts"] for x in ss])),
                                              ASR_WER_vs_TTS=float(np.mean([x["asr_wer_vs_tts"] for x in ss])),
                                              ASR_CER_vs_GT=float(np.mean([x["asr_cer_vs_gt"] for x in ss if x["asr_cer_vs_gt"] is not None])),
                                              ASR_WER_vs_GT=float(np.mean([x["asr_wer_vs_gt"] for x in ss if x["asr_wer_vs_gt"] is not None])),
                                              RTF_speech=float(np.sum([x["tts_ms"] or 0 for x in ss]) / 1000 / max(np.sum([x["audio_s"] for x in ss]), 1e-6)))
        summ["systems"][tag] = m
        rows += R
        print(tag, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if not isinstance(v, dict)}, ensure_ascii=False), flush=True)
    if a.asr != "none" and not a.limit:                  # floor: TTS(ground truth) -> ASR on 20 items
        fl = []
        for it in [i for i in S["items"] if i["gt"]][:20]:
            try:
                key, wav, d, _ = tts(a.tts, it["gt"], a.voice, None, 0.5)
            except Exception:  # noqa: BLE001
                continue
            tr = asr(key, wav, d)
            if tr["text"] is not None:
                fl.append(dict(cer=cer(it["gt"], tr["text"]), wer=wer(it["gt"], tr["text"]), asr=tr["asr"]))
        summ["floor_tts_gt_asr"] = dict(n=len(fl), CER=float(np.mean([f["cer"] for f in fl])) if fl else None, WER=float(np.mean([f["wer"] for f in fl])) if fl else None,
                                        asr_sources=sorted({f["asr"] for f in fl}))
        summ["asr_calls"], summ["asr_audio_minutes"] = asr.calls, round(asr.seconds / 60, 2)
    write_json(REP / (f"{a.out}_dryrun.json" if a.limit else f"{a.out}.json"), summ)
    with open(REP / f"{a.out.replace('v12_e2e', 'e2e_rows')}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    print("->", REP / "v12_e2e.json", "| ASR calls", asr.calls, "audio min", round(asr.seconds / 60, 2))


if __name__ == "__main__":
    main()
