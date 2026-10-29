"""Speech v1 (models/speech-model-v1/, modules/speech.py) — build, data, train, evaluate. Run with the speech env
(D:/SLM-labs/envs/speech: the hugging env + transformers 4.57 / qwen-asr / f5-tts, kept apart so the sign pipeline is untouched).

    python main.py speech refs          # emotion reference prompts: ElevenLabs v3 (audio tags) → models/speech-model-v1/refs/*.wav + refs.json
    python main.py speech asr-data      # ASR fine-tuning data (labels known by construction): short daily-conversation lines spoken by
                                        #   ElevenLabs (teacher voices) + F5-TTS-THAI cloned from many references → cloud_s3/speech-data/asr/
    python main.py speech asr-eval [--lora]   # Qwen3-ASR CER on FLEURS-th (real speech) + reverse part's asr_th (short phrases) + held-out synth
    python main.py speech asr-train     # LoRA on Qwen3-ASR-0.6B (local RTX 2050, bf16) → models/speech-model-v1/asr_lora
    python main.py speech demo          # real clips end to end: forward v9 → face affect → speech plan → voice (result_reporting/speech_samples_v1)
    python main.py speech tts-eval      # voice: intelligibility (ASR CER of the spoken sentence), emotion carried (prosody vs neutral),
                                        #   real-time factor; vs the old MMS-TTS voice → artifacts/reports/speech_v1.json + sample wavs
ElevenLabs is used only to make labelled audio (the user's .env ELEVEN_LABS_API); its output stays in cloud_s3/speech-data/.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_CACHE", "D:/SLM-labs/hf_cache")

from modules.speech import EMOTIONS, cer  # noqa: E402
from modules.utils import ART, write_json  # noqa: E402

MD = ROOT / "models" / "speech-model-v1"
DATA = ROOT / "cloud_s3" / "speech-data"
EVAL = ROOT / "reverse_text2sign" / "data" / "eval"
VOICES = {"female": "EXAVITQu4vr4xnSDxMaL", "male": "nPczCjzI2devNBz1zQrb"}       # ElevenLabs default voices (free tier)
EL_TAG = {"neutral": "", "happy": "[happy] ", "sad": "[sad] ", "angry": "[angry] ", "fear": "[scared] ", "disgust": "[disgusted] ",
          "surprise": "[surprised] "}
REF_LINES = {
    "neutral": "วันนี้ฉันไปทำงานตามปกติ แล้วก็กลับบ้านตอนเย็น",
    "happy": "ดีใจจังเลย วันนี้ได้เจอเพื่อนเก่าที่ไม่ได้เจอมานาน",
    "sad": "ฉันเสียใจมาก ที่ไม่ได้ไปลาเขาเป็นครั้งสุดท้าย",
    "angry": "ทำไมถึงทำแบบนี้ ฉันบอกแล้วว่าอย่าทำอีก",
    "fear": "เดี๋ยวก่อน ได้ยินเสียงอะไรไหม ฉันกลัวจังเลย",
    "disgust": "อี๋ กลิ่นอะไรเนี่ย เหม็นมากจนทนไม่ไหว",
    "surprise": "จริงเหรอ ไม่น่าเชื่อเลยว่าเขาจะมาด้วย",
}


def _key():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("ELEVEN_LABS_API="):
            return line.split("=", 1)[1].strip().strip('"')
    sys.exit("ELEVEN_LABS_API missing in .env")


def eleven(text, voice_id, out, model="eleven_v3"):
    """ElevenLabs TTS → wav (mono, 24 kHz). Returns seconds; skips when out exists."""
    import requests
    import soundfile as sf
    import librosa
    if out.exists():
        return sf.info(str(out)).duration
    r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128",
                      headers={"xi-api-key": _key(), "Content-Type": "application/json"}, json=dict(text=text, model_id=model), timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"ElevenLabs {r.status_code}: {r.text[:200]}")
    y, sr = librosa.load(io.BytesIO(r.content), sr=24000, mono=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y, 24000)
    return len(y) / 24000


def cmd_refs(a):
    raw = DATA / "elevenlabs" / "refs"
    refs, chars = {}, 0
    for voice, vid in VOICES.items():
        for emo in EMOTIONS:
            w = raw / f"{voice}_{emo}.wav"
            if not w.exists():
                chars += len(EL_TAG[emo] + REF_LINES[emo])
            sec = eleven(EL_TAG[emo] + REF_LINES[emo], vid, w)
            dst = MD / "refs" / w.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(w.read_bytes())
            refs.setdefault(voice, {})[emo] = dict(wav=f"refs/{w.name}", text=REF_LINES[emo], sec=round(sec, 2), source="ElevenLabs eleven_v3 " + vid)
            print(voice, emo, f"{sec:.1f}s", flush=True)
    write_json(MD / "refs.json", refs)
    print("ElevenLabs characters used this run:", chars)


# ----------------------------------------------------------------------------- ASR fine-tuning data
EL_VOICES = {"JBFqnCBsd6RMkjVDRZzb": "m", "EXAVITQu4vr4xnSDxMaL": "f", "XrExE9yKIg1WjnnlVkGX": "f", "FGY2WhTYpPnrIDTdsKH5": "f",
             "cgSgspJ2msm6clMCkdW9": "f", "nPczCjzI2devNBz1zQrb": "m", "onwK4e9ZLuTAKqWW03F9": "m", "pFZP5JQG7iQjIQuC4Bku": "f",
             "TX3LPaxmHKxFdv7VOQHJ": "m", "cjVigY5qzO86Huf0OWal": "m"}
HELD_OUT_VOICE = "Xb7hH8MSUJpSbSDYk0k2"          # never in training (test voice)


def phrases(seed=0):
    """Short everyday utterances (what a hearing partner says to a deaf signer): greetings, answers, questions, 2–5-word statements
    built from the daily-conversation lexicon. Any text equal to a test sentence (reverse part's dev / test-A, FLEURS) is dropped."""
    rng = np.random.RandomState(seed)
    subj = ["ฉัน", "ผม", "คุณ", "เขา", "เรา", "พ่อ", "แม่", "พี่", "น้อง", "เพื่อน", "ครู", "ยาย"]
    pred = ["กินข้าว", "ดื่มน้ำ", "ไปโรงเรียน", "ไปทำงาน", "ไปตลาด", "กลับบ้าน", "เรียนภาษามือ", "ชอบแมว", "รักเพื่อน", "คิดถึงแม่",
            "ไม่สบาย", "เหนื่อย", "หิวข้าว", "ง่วงนอน", "ดีใจมาก", "เสียใจ", "ไปเที่ยว", "อ่านหนังสือ", "ดูหนัง", "ทำการบ้าน", "นอนหลับ",
            "ไปหาหมอ", "ซื้อขนมปัง", "ชอบกินไข่", "อยู่บ้าน", "มาสาย", "ร้องไห้", "โกรธ", "สบายดี", "ทำงานหนัก", "ขับรถ", "ว่ายน้ำ"]
    when = ["", "", "วันนี้", "พรุ่งนี้", "เมื่อวาน", "ตอนนี้", "เดี๋ยวนี้"]
    ask = ["ไหม", "หรือยัง", "หรือเปล่า", "ใช่ไหม"]
    wh = ["ไปไหน", "ทำอะไร", "ชื่ออะไร", "อยู่ที่ไหน", "มาเมื่อไร", "กินอะไร", "ชอบอะไร", "อายุเท่าไร", "เป็นอะไร"]
    fixed = ["สวัสดีค่ะ", "สวัสดีครับทุกคน", "ขอบคุณค่ะ", "ขอบคุณมากนะ", "ขอโทษนะคะ", "ไม่เป็นไรครับ", "ใช่ครับ", "ใช่ค่ะ", "ไม่ใช่ครับ",
             "ได้ครับ", "ได้ค่ะ", "ไม่ได้ครับ", "ไม่รู้", "ไม่รู้ค่ะ", "เข้าใจแล้ว", "เข้าใจไหม", "รอแป๊บนึงนะ", "เดี๋ยวก่อน", "โอเค",
             "ตกลง", "แน่นอน", "จริงเหรอ", "ไม่ต้อง", "ไปกันเถอะ", "มาที่นี่", "นั่งก่อน", "ช้าๆหน่อย", "อีกครั้งนะ", "ยินดีด้วย",
             "เก่งมาก", "สู้ๆ", "ระวังนะ", "หิวไหม", "ง่วงไหม", "เจ็บไหม", "เป็นอะไรหรือเปล่า", "ไปด้วยกันไหม", "กลับบ้านกัน",
             "พบกันใหม่", "ฝันดีนะ", "อรุณสวัสดิ์", "ราตรีสวัสดิ์", "ยินดีที่ได้รู้จัก", "ไม่ต้องห่วง", "ใจเย็นๆ", "ขอบใจนะ"]
    out = set(fixed)
    while len(out) < 420:
        k = rng.randint(3)
        s, w = subj[rng.randint(len(subj))], when[rng.randint(len(when))]
        if k == 0:
            t = f"{w}{s}{pred[rng.randint(len(pred))]}"
        elif k == 1:
            t = f"{s}{pred[rng.randint(len(pred))]}{ask[rng.randint(len(ask))]}"
        else:
            t = f"{w}{s}{wh[rng.randint(len(wh))]}"
        out.add(t)
    test_texts = {r.replace(" ", "") for r in _test_texts()}
    out = sorted(t for t in out if t.replace(" ", "") not in test_texts)
    rng.shuffle(out)
    return out


def _test_texts():
    import csv
    t = [r["ref"] for r in csv.DictReader(open(EVAL / "asr_th.csv", encoding="utf-8-sig"))]
    t += [m["text"] for m in json.loads((EVAL / "asr_test" / "meta.json").read_text(encoding="utf-8"))["short"]]
    t += [r["ref"] for r in _fleurs()]
    return t


def cmd_asr_data(a):
    """train = 85 % of the phrases (ElevenLabs: one teacher voice each, random emotion tag; F5: 3 clones each from random references),
    test = the other 15 % (phrases never trained on) in the held-out ElevenLabs voice. Transcripts are exact by construction."""
    import random
    import soundfile as sf
    P = phrases()
    n_te = int(len(P) * 0.15)
    te, tr = P[:n_te], P[n_te:]
    out = DATA / "asr"; (out / "wav").mkdir(parents=True, exist_ok=True)
    rnd = random.Random(0)
    rows_tr, rows_te, chars = [], [], 0
    voices = list(EL_VOICES)
    for i, t in enumerate(tr[:a.n_eleven]):
        v, emo = voices[i % len(voices)], rnd.choice(EMOTIONS)
        f = out / "wav" / f"el_tr_{i:04d}.wav"
        if not f.exists():
            chars += len(EL_TAG[emo] + t)
        try:
            eleven(EL_TAG[emo] + t, v, f)
        except RuntimeError as e:
            print("ElevenLabs stopped:", e, flush=True); break
        rows_tr.append(dict(audio=str(f), ref=t, source=f"elevenlabs:{v}:{emo}"))
    for i, t in enumerate(te):
        f = out / "wav" / f"el_te_{i:04d}.wav"
        if not f.exists():
            chars += len(t)
        try:
            eleven(t, HELD_OUT_VOICE, f)
        except RuntimeError as e:
            print("ElevenLabs stopped:", e, flush=True); break
        rows_te.append(dict(audio=str(f), ref=t, source=f"elevenlabs:{HELD_OUT_VOICE}"))
    print(f"ElevenLabs: {len(rows_tr)} train + {len(rows_te)} test utterances, {chars} new characters", flush=True)
    # local F5-TTS-THAI clones: every train phrase × 3 references (teacher emotional refs + the train utterances of the teacher voices)
    from modules.speech import ExpressiveTTS
    tts = ExpressiveTTS(MD)
    refs = [(str(MD / r["wav"]), r["text"]) for v in tts.refs.values() for r in v.values()]
    refs += [(r["audio"], r["ref"]) for r in rows_tr if len(r["ref"]) >= 8][:40]
    from f5_tts_th.utils_infer import infer_process, preprocess_ref_audio_text
    import torch
    t0 = time.time()
    for i, t in enumerate(tr):
        for k in range(a.f5_per_phrase):
            f = out / "wav" / f"f5_tr_{i:04d}_{k}.wav"
            if not f.exists():
                ra, rt = refs[rnd.randrange(len(refs))]
                ra, rt = preprocess_ref_audio_text(ra, rt)
                torch.manual_seed(i * 10 + k)
                wav, sr, _ = infer_process(ra, rt, t, tts.model, tts.vocoder, mel_spec_type="vocos", nfe_step=16,
                                           speed=rnd.uniform(0.85, 1.15), cfg_strength=2.0, use_ipa=False)
                sf.write(str(f), wav, sr)
            rows_tr.append(dict(audio=str(f), ref=t, source="f5-tts-thai"))
        if i % 50 == 0:
            print(f"F5 {i}/{len(tr)} ({time.time() - t0:.0f} s)", flush=True)
    for name, rows in (("train", rows_tr), ("test", rows_te)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"written {out}: train {len(rows_tr)} · test {len(rows_te)}", flush=True)


# ----------------------------------------------------------------------------- ASR evaluation / fine-tuning
def _fleurs():
    import csv
    """Google FLEURS Thai dev (real read speech; the reverse part's set): the first 100 utterances with audio."""
    d = EVAL / "fleurs"
    tsv = d / "data" / "th_th" / "dev.tsv"
    rows = []
    if tsv.exists():
        for r in csv.reader(open(tsv, encoding="utf-8"), delimiter="\t"):
            if len(r) > 2 and (d / "dev" / r[1]).exists():
                rows.append(dict(audio=str(d / "dev" / r[1]), ref=r[2]))
    return rows[:100]


def eval_sets():
    import csv
    sets = {}
    rows = list(csv.DictReader(open(EVAL / "asr_th.csv", encoding="utf-8-sig")))
    wavs = sorted((EVAL / "asr_th").glob("*.wav"))
    sets["asr_th (short daily phrases)"] = [dict(audio=str(w), ref=r["ref"]) for w, r in zip(wavs, rows)]
    meta = json.loads((EVAL / "asr_test" / "meta.json").read_text(encoding="utf-8"))
    sets["test-A (held-out sentences, male voice)"] = [dict(audio=str(EVAL / "asr_test" / f"{m['id']}.wav"), ref=m["text"]) for m in meta["short"]]
    fl = _fleurs()
    if fl:
        sets["FLEURS-th (real read speech)"] = fl
    ho = DATA / "asr" / "test.jsonl"
    if ho.exists():
        sets["held-out synthetic conversation"] = [json.loads(l) for l in open(ho, encoding="utf-8")]
    return sets


def cmd_asr_eval(a):
    from modules.speech import ThaiASR
    asr = ThaiASR(MD, use_lora=a.lora)
    out = {}
    for name, rows in eval_sets().items():
        t = time.time()
        hyp = []
        for i in range(0, len(rows), 8):
            hyp += asr([r["audio"] for r in rows[i:i + 8]])
        c = [cer(r["ref"], h) for r, h in zip(rows, hyp)]
        out[name] = dict(n=len(rows), CER=float(np.mean(c)), exact=int(sum(x == 0 for x in c)), empty=int(sum(not h.strip() for h in hyp)),
                         sec_per_utt=(time.time() - t) / len(rows), examples=[dict(ref=r["ref"], hyp=h) for r, h in list(zip(rows, hyp))[:12]])
        print(f"{name}: CER {out[name]['CER']:.4f} exact {out[name]['exact']}/{len(rows)} empty {out[name]['empty']}", flush=True)
    write_json(ART / "reports" / f"speech_asr_{'lora' if a.lora else 'base'}.json", out)


def cmd_asr_train(a):
    """LoRA (r 16) on the Qwen3 decoder of Qwen3-ASR-0.6B (attention + MLP of every layer; the audio encoder frozen), bf16 on the
    local GPU. Target = "language Thai<asr_text>" + transcript + <|im_end|> after the model's own chat prompt; loss on the transcript
    only. Model selection on the held-out synthetic phrases (unseen texts, unseen voice) — the real-speech sets are only reported."""
    import random
    import librosa
    import torch
    from peft import LoraConfig, get_peft_model
    from qwen_asr import Qwen3ASRModel
    rows = [json.loads(l) for l in open(DATA / "asr" / "train.jsonl", encoding="utf-8")]
    test = [json.loads(l) for l in open(DATA / "asr" / "test.jsonl", encoding="utf-8")]
    m = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-0.6B", dtype=torch.bfloat16, device_map="cuda", max_inference_batch_size=8, max_new_tokens=128)
    proc, tok = m.processor, m.processor.tokenizer
    tok.padding_side = "right"
    prompt = m._build_text_prompt(context="", force_language="Thai")
    m.model = get_peft_model(m.model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=
                             r".*thinker\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"))
    m.model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in m.model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    wav = {}

    def load(path):
        if path not in wav:
            wav[path] = librosa.load(path, sr=16000)[0].astype(np.float32)
        return wav[path]

    def batch_loss(b):
        texts = [prompt + r["ref"] + "<|im_end|>" for r in b]
        inp = proc(text=texts, audio=[load(r["audio"]) for r in b], return_tensors="pt", padding=True).to("cuda")
        inp = {k: (v.to(torch.bfloat16) if v.dtype.is_floating_point else v) for k, v in inp.items()}
        labels = torch.full_like(inp["input_ids"], -100)
        for i, r in enumerate(b):
            n = len(tok(r["ref"] + "<|im_end|>", add_special_tokens=False)["input_ids"])
            L = int(inp["attention_mask"][i].sum())
            labels[i, L - n:L] = inp["input_ids"][i, L - n:L]
        return m.model.base_model.model.thinker(**inp, labels=labels).loss

    def test_cer():
        m.model.eval()
        with torch.no_grad():
            hyp = []
            for i in range(0, len(test), 8):
                hyp += [r.text for r in m.transcribe([r["audio"] for r in test[i:i + 8]], language="Thai")]
        m.model.train()
        return float(np.mean([cer(r["ref"], h) for r, h in zip(test, hyp)]))
    best = (test_cer(), -1)
    print(f"epoch -1 (base): held-out synthetic CER {best[0]:.4f}", flush=True)
    rnd = random.Random(0)
    m.model.train()
    out = MD / "asr_lora"
    for ep in range(a.epochs):
        rnd.shuffle(rows)
        t, tot = time.time(), 0.0
        for i in range(0, len(rows), a.bs):
            loss = batch_loss(rows[i:i + a.bs])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            tot += float(loss)
        c = test_cer()
        print(f"epoch {ep}: loss {tot / max(1, len(rows) // a.bs):.4f} · held-out synthetic CER {c:.4f} ({time.time() - t:.0f} s)", flush=True)
        if c < best[0] - 1e-4:
            best = (c, ep)
            m.model.save_pretrained(str(out))
    print(f"best epoch {best[1]} CER {best[0]:.4f}" + (f" → {out}" if best[1] >= 0 else " (base model kept: no adapter written)"), flush=True)


TTS_TEST = ["สวัสดี ฉันชื่อแจ็ค ฉันชอบแมว", "พรุ่งนี้ฉันจะไปเที่ยวประเทศจีน", "วันนี้ฉันอยากเรียนภาษามือ", "ฉันรักเพื่อน",
            "ฉันคิดถึงแม่", "พ่อดื่มน้ำ", "วันหยุดพ่อชอบไปตลาด", "ฉันไม่สบาย ปวดหัวมาก", "ขอบคุณมากที่ช่วยฉัน", "ทำไมคุณมาสาย",
            "ฉันเสียใจที่เพื่อนร้องไห้", "เราจะเป็นเพื่อนกันตลอดไป"]


def prosody(y, sr):
    """F0 median / spread (semitones re 100 Hz), loudness (dB RMS of voiced frames), voiced fraction, duration."""
    import librosa
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
    f0, vf, _ = librosa.pyin(y16, fmin=70, fmax=500, sr=16000, frame_length=1024)
    f = f0[vf & np.isfinite(f0)]
    st = 12 * np.log2(f / 100.0) if len(f) else np.array([0.0])
    rms = librosa.feature.rms(y=y16, frame_length=1024, hop_length=256)[0]
    return dict(f0_st=float(np.median(st)), f0_range_st=float(np.percentile(st, 90) - np.percentile(st, 10)),
                loud_db=float(20 * np.log10(np.percentile(rms, 90) + 1e-6)), dur_s=len(y16) / 16000)


def cmd_tts_eval(a):
    """Voice v1 vs the old MMS-TTS voice: 12 sentences × 7 emotions × 2 voices (v1) — CER by the BASE Qwen3-ASR (an independent ear,
    not the fine-tuned one), prosody per emotion relative to the same voice's neutral reading, compared with the teacher references'
    own shift (does the clone carry the emotion?), real-time factor on this GPU / CPU."""
    import soundfile as sf
    import torch
    from modules.speech import ExpressiveTTS, SpeechPlan, ThaiASR
    sdir = ROOT / "result_reporting" / "speech_samples_v1"; sdir.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / "cache" / "v9" / "tts_eval"; tmp.mkdir(parents=True, exist_ok=True)
    tts = ExpressiveTTS(MD, nfe_step=a.nfe)
    gen = []
    for voice in ("female", "male"):
        for emo in EMOTIONS:
            for i, t in enumerate(TTS_TEST[: a.n]):
                t0 = time.time()
                y, sr = tts.speak(SpeechPlan(text=t, emotion=emo, intensity=0.8, voice=voice))
                dt = time.time() - t0
                f = tmp / f"{voice}_{emo}_{i:02d}.wav"; sf.write(str(f), y, sr)
                gen.append(dict(voice=voice, emotion=emo, i=i, text=t, wav=str(f), rtf=dt / max(len(y) / sr, 1e-3), **prosody(y, sr)))
                if i == 0:
                    sf.write(str(sdir / f"{voice}_{emo}.wav"), y, sr)
        print(voice, "done", flush=True)
    # CPU real-time factor (Docker runs without a GPU): 3 sentences
    del tts; torch.cuda.empty_cache()
    cpu = ExpressiveTTS(MD, device="cpu", nfe_step=a.nfe)
    rt_cpu = []
    for t in TTS_TEST[:3]:
        t0 = time.time(); y, sr = cpu.speak(SpeechPlan(text=t)); rt_cpu.append((time.time() - t0) / (len(y) / sr))
    del cpu
    # old voice: MMS-TTS Thai (the current web TTS service), neutral only
    from transformers import AutoTokenizer, VitsModel
    mt, mm = AutoTokenizer.from_pretrained("facebook/mms-tts-tha"), VitsModel.from_pretrained("facebook/mms-tts-tha")
    old = []
    for i, t in enumerate(TTS_TEST[: a.n]):
        with torch.no_grad():
            y = mm(**mt(t, return_tensors="pt")).waveform[0].numpy()
        f = tmp / f"mms_{i:02d}.wav"; sf.write(str(f), y, mm.config.sampling_rate)
        old.append(dict(text=t, wav=str(f), **prosody(y, mm.config.sampling_rate)))
    sf.write(str(sdir / "old_mms_neutral.wav"), *sf.read(old[0]["wav"]))
    asr = ThaiASR(None, use_lora=False)
    for g in gen + old:
        g["hyp"] = asr(g["wav"]); g["cer"] = cer(g["text"], g["hyp"])
    import pandas as pd
    d = pd.DataFrame(gen)
    refs = json.loads((MD / "refs.json").read_text(encoding="utf-8"))
    rp = {(v, e): prosody(*sf.read(str(MD / r["wav"]))) for v, es in refs.items() for e, r in es.items()}
    shift = []
    for (v, e), g in d.groupby(["voice", "emotion"]):
        n = d[(d.voice == v) & (d.emotion == "neutral")]
        row = dict(voice=v, emotion=e, CER=g.cer.mean(), rtf=g.rtf.mean())
        for k in ("f0_st", "f0_range_st", "loud_db"):
            row[f"d_{k}"] = float(g[k].mean() - n[k].mean())
            row[f"ref_d_{k}"] = float(rp[(v, e)][k] - rp[(v, "neutral")][k])
        row["d_rate"] = float(n.dur_s.mean() / g.dur_s.mean() - 1)
        shift.append(row)
    S = pd.DataFrame(shift)
    nz = S[S.emotion != "neutral"]
    corr = {k: float(np.corrcoef(nz[f"d_{k}"], nz[f"ref_d_{k}"])[0, 1]) for k in ("f0_st", "f0_range_st", "loud_db")}
    out = dict(v1=dict(CER=float(d.cer.mean()), CER_by_emotion=d.groupby("emotion").cer.mean().round(4).to_dict(), rtf_gpu=float(d.rtf.mean()),
                       rtf_cpu=float(np.mean(rt_cpu)), n=len(d)),
               old_mms=dict(CER=float(np.mean([o["cer"] for o in old])), n=len(old)),
               emotion_shift=S.round(3).to_dict("records"), shift_follows_reference_corr=corr, nfe=a.nfe,
               examples=[dict(text=g["text"], emotion=g["emotion"], voice=g["voice"], hyp=g["hyp"]) for g in gen[:: max(1, len(gen) // 12)]])
    print(json.dumps({k: v for k, v in out.items() if k not in ("emotion_shift", "examples")}, ensure_ascii=False, indent=1))
    print(S.round(2).to_string())
    write_json(ART / "reports" / "speech_tts_v1.json", out)


def cmd_demo(a):
    """End to end on real clips: landmarks → forward v9 (words + calibrated status) → face model + fusion (affect) → speech plan →
    expressive voice → result_reporting/speech_samples_v1/demo_<clip>.wav + demo.json (what was said, how, and why)."""
    import logging
    import soundfile as sf
    from modules.mediapipe_pose import load_raw
    from modules.pipeline import SignPipeline
    from modules.speech import ExpressiveTTS, plan_speech
    logging.disable(logging.WARNING)
    pipe = SignPipeline(model_dir=ROOT / "models" / "model-v9", use_llm=False, use_face=True)
    tts = ExpressiveTTS(MD)
    sdir = ROOT / "result_reporting" / "speech_samples_v1"; sdir.mkdir(parents=True, exist_ok=True)
    clips = {"iphone_3307": ROOT / "cache/iphone_test_mp/sentence_IMG_3307.npz", "iphone_3302": ROOT / "cache/iphone_test_mp/sentence_IMG_3302.npz",
             "data_test_รักเพื่อน": ROOT / "cache/data_test_mp/ฉันรักเพื่อน.npz",
             "team_pitty_คิดถึงแม่": ROOT / "cache/modeling_compare/mp/Pitty__ฉันคิดถึงแม่.npz",
             "team_boom_แอบชอบ": ROOT / "cache/modeling_compare/mp/Boom__ฉันแอบชอบคุณ.npz"}
    out = []
    for name, f in clips.items():
        r = pipe.run_landmarks(load_raw(f))
        aff = r["face"].get("affect", {})
        plan = plan_speech(r, aff, voice="female", speak_uncertain=True)
        y, sr = tts.speak(plan)
        if len(y):
            sf.write(str(sdir / f"demo_{name}.wav"), y, sr)
        out.append(dict(clip=name, words=[(s["nearest"], s["status"], round(s.get("p_correct", 0), 2)) for s in r["segments"] if s["status"] != "transition"],
                        said=plan.text, emotion=plan.emotion, intensity=round(plan.intensity, 2), speed=round(plan.speed, 3), why=plan.reason,
                        affect_explanation=aff.get("explanation_th"), face_only=(aff.get("face_only") or {}).get("explanation_th")))
        print(json.dumps(out[-1], ensure_ascii=False), flush=True)
    write_json(sdir / "demo.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["refs", "asr-data", "asr-eval", "asr-train", "tts-eval", "demo"])
    ap.add_argument("--nfe", type=int, default=32); ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--n_eleven", type=int, default=160)
    ap.add_argument("--f5_per_phrase", type=int, default=3)
    a = ap.parse_args()
    {"refs": cmd_refs, "asr-data": cmd_asr_data, "asr-eval": cmd_asr_eval, "asr-train": cmd_asr_train, "tts-eval": cmd_tts_eval, "demo": cmd_demo}[a.step](a)


if __name__ == "__main__":
    main()
