"""Fixed evaluation for TTS v2 (same sets / seeds for every model):  refs · gen · score · affect · latency · pack · compare

  python evaluate.py gen     --model_dir D --out O [--sets main,onset,short,emo] [--voices a,b] [--seeds 0,1,2] [--nfe 32]
  python evaluate.py score   --dirs v3=O1 v4=O2           # CER (Whisper turbo + Pathumma), first-word hit, onset timing, speaker similarity, F0, gap span, RTF
  python evaluate.py affect  --dirs v3=O1 v4=O2           # perceived age (audeering) + emotion (evaluator SER, NOT the labeller) on the generated audio
  python evaluate.py latency --model_dir D --out O.json   # cold start, warm p50/p95 and RTF for ~1/3/6 s of speech, time split
  python evaluate.py pack    --dir O                      # blinded listening pack (HTML + answer key) for the user to rate
  python evaluate.py compare --dirs v3=O1 v4=O2 --out data/results.json   # tables + paired bootstrap

Metrics are automatic; nothing here is a MOS. Limits are written next to the numbers in the report.
"""
import argparse, json, re, sys, time
from pathlib import Path

import numpy as np, pandas as pd, soundfile as sf

HERE = Path(__file__).parent
DATA = HERE / "data"
CLEAN = lambda t: re.sub(r"[\s\.,!?\-\"'…ๆ]", "", t or "")

SENTENCES = [
    "สวัสดีครับ วันนี้อากาศดีมาก ไปเดินเล่นที่สวนสาธารณะกันไหม",
    "ฉันอยากไปโรงพยาบาลเพราะรู้สึกไม่ค่อยสบาย",
    "คุณแม่ทำอาหารอร่อยมากเลยค่ะ ฉันชอบกินข้าวผัดที่สุด",
    "พรุ่งนี้ตอนเช้าเจ็ดโมงครึ่ง เราจะนั่งรถไฟไปเชียงใหม่",
    "ขอบคุณมากนะคะ ที่ช่วยสอนภาษามือให้ฉัน",
    "ทำไมวันนี้ถึงมาสายล่ะ มีอะไรเกิดขึ้นหรือเปล่า",
    "เด็กๆ ดีใจมากเมื่อได้รับของขวัญวันเกิดจากคุณพ่อ",
]
WORDS = {"ครอบครัว": "พ่อ แม่ พี่ น้อง ลูก", "อาหาร": "ข้าว น้ำ ขนม ผลไม้ ก๋วยเตี๋ยว", "สถานที่": "บ้าน โรงเรียน โรงพยาบาล ตลาด ประเทศ",
         "กริยา": "กิน นอน เดิน ซื้อ เรียน", "ความรู้สึก": "ดีใจ เสียใจ เหนื่อย หิว รัก"}
GAPS = ["ฉัน … แม่", "ฉันอยากไปประเทศ …", "พรุ่งนี้ … จะไปตลาดกับ … ที่บ้าน"]
SHORT = ["ฉันชอบแมว", "ผมชอบแมว", "ฉันรักแม่", "ผมหิวแล้ว", "คุณชอบไหม", "วันนี้ฝนตก", "อยากกินข้าว", "ขอน้ำหน่อย"]
EMO_SENTS = ["ฉันไม่คิดว่าจะเป็นแบบนี้", "วันนี้เป็นวันที่ฉันรอคอยมานาน", "เขาบอกว่าพรุ่งนี้จะมาที่บ้านเรา", "ทำไมเรื่องถึงเป็นอย่างนี้ได้",
             "ฉันเห็นของสิ่งนั้นอยู่ตรงหน้า", "ตอนนี้ฉันอยู่ที่หน้าประตูบ้าน", "คุณพูดอย่างนั้นกับฉันได้อย่างไร", "ผลออกมาแล้ว ดูสิว่าเป็นอย่างไร"]
EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
LAT = {"1s": "ฉันชอบแมว", "3s": "วันนี้อากาศดีมาก ไปเดินเล่นที่สวนสาธารณะกันไหม", "6s": "พรุ่งนี้ตอนเช้าเจ็ดโมงครึ่ง เราจะนั่งรถไฟไปเชียงใหม่ แล้วค่อยไปหาคุณยายที่บ้านสวน"}
# onset set: 60 short sentences, first sound class tagged (first-word reliability)
ONSET = {
    "vowel": ["อากาศดีมากเลย", "อยากกินข้าวผัด", "อร่อยมากเลยครับ", "อีกสักครู่นะคะ", "เอาอันนี้ค่ะ", "ออกไปข้างนอกกัน", "อ่านหนังสือเล่มนี้"],
    "pronoun": ["ฉันชอบแมว", "ฉันหิวข้าวแล้ว", "ผมชอบหมา", "ผมไปตลาดครับ", "คุณชื่ออะไร", "คุณสบายดีไหม", "วันนี้ร้อนมาก", "วันนี้ไปโรงเรียน", "เธอสวยจัง", "เรากินข้าวกัน"],
    "stop": ["ปิดประตูด้วยครับ", "ไปไหนมาคะ", "ตื่นเช้ามากเลย", "ถึงบ้านแล้วครับ", "กินข้าวหรือยัง", "ทำงานเสร็จแล้ว", "บ้านอยู่ไกลมาก", "เดินทางปลอดภัยนะ"],
    "nasal": ["แมวนอนอยู่บนเตียง", "หมาวิ่งเล่นในสวน", "น้ำเย็นมากเลย", "หนูตัวเล็กมาก", "งานเยอะมากครับ", "นกบินสูงมาก", "มาแล้วค่ะ"],
    "fricative": ["สวัสดีครับ", "ฝนตกหนักมาก", "ซื้อของที่ตลาด", "ห้องนี้เย็นดี", "ฟังเพลงเพราะมาก", "ช่วยด้วยครับ", "ขอบคุณมากค่ะ", "ศาลาอยู่ตรงนั้น"],
    "glide": ["ยินดีที่ได้รู้จัก", "วันเกิดของฉันวันนี้", "ว่างไหมวันนี้", "หยุดตรงนี้ครับ", "ใช่แล้วครับ", "ใกล้จะถึงแล้ว", "รอสักครู่นะคะ", "ลืมกุญแจไว้ที่บ้าน", "เรียนภาษามือสนุกมาก", "ร้อนจังเลย"],
    "affricate": ["จะไปไหนครับ", "ใจเย็นๆ นะคะ", "ชอบกินส้มตำมาก", "เช้านี้อากาศดี", "ช้าๆ หน่อยครับ"],
    "particle": ["ไม่เป็นไรครับ", "ไม่ชอบเลย", "ได้เลยค่ะ", "อะไรนะครับ", "ใครอยู่ที่นั่น"],
}
ONSET_CASES = [(f"o{i + 1:02d}", t, c) for i, (c, t) in enumerate((c, t) for c, ts in ONSET.items() for t in ts)]


def cases(sets, test_n=10):
    c = []
    if "main" in sets:
        c += [(f"s{i + 1:02d}", t, "sent", None) for i, t in enumerate(SENTENCES)]
        c += [(f"g{i + 1:02d}", t, "gap", None) for i, t in enumerate(GAPS)]
        c += [(f"w{gi + 1}{wi + 1}", w, "word", None) for gi, ws in enumerate(WORDS.values()) for wi, w in enumerate(ws.split())]
        m = pd.read_csv(DATA / "manifest_v1.csv"); m = m[m.split == "test"].sample(test_n, random_state=0)       # v1's held-out speakers, same 10 transcripts
        c += [(f"t{i + 1:02d}", r.text.strip(), "test", None) for i, r in enumerate(m.itertuples())]
    if "onset" in sets:
        c += [(cid, t, "onset:" + cl, None) for cid, t, cl in ONSET_CASES[::2]]      # every 2nd sentence: all classes, 30 sentences
    if "short" in sets:
        c += [(f"h{i + 1:02d}", t, "short", None) for i, t in enumerate(SHORT)]
    if "emo" in sets:
        c += [(f"e{i + 1}{EMOTIONS.index(e)}{int(it * 10)}", t, "emo", (e, it)) for i, t in enumerate(EMO_SENTS) for e in EMOTIONS for it in ((0.5,) if e == "neutral" else (0.5, 1.0))]
    return c


# ----------------------------------------------------------------------------- gen
def gen(a):
    sys.path.insert(0, str(HERE))
    from tts import Speech
    import torch
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); s = Speech(a.model_dir); load_s = time.time() - t0
    voices = a.voices.split(",") if a.voices else list(s.voices)
    seeds = [int(x) for x in a.seeds.split(",")]
    rows = []
    has_emo = bool(s.tags)
    for cid, text, kind, emo in cases(a.sets.split(","), a.test_n):
        if emo and not has_emo:
            continue                                                  # emotion set only for models that have the tags
        vs = [v for v in voices if kind != "emo" or v in a.emo_voices.split(",")] if kind == "emo" else voices
        for vid in vs:
            if kind in ("word", "test") and vid != vs[0] and not (vid.endswith("_1") or vid in ("male", "female")):
                continue                                              # words / test sentences: one voice per presentation
            for sd in (seeds if kind.startswith(("onset", "short")) else seeds[:1]):
                t = time.time(); s.timing = {}
                kw = dict(emotion=emo[0], intensity=emo[1]) if emo else {}
                wav, sr = s.say(text, voice=vid, seed=sd, nfe=a.nfe, **kw); dt = time.time() - t
                f = f"{cid}_{vid}_s{sd}.wav"; s.save(out / f, wav, sr)
                rows.append(dict(case=cid, kind=kind, voice=vid, seed=sd, text=text, emotion=emo[0] if emo else "", intensity=emo[1] if emo else np.nan, file=f,
                                 audio_s=len(wav) / sr, gen_s=dt, rtf=dt / (len(wav) / sr), **{"t_" + k: v for k, v in s.timing.items()}))
        print(cid, len(rows), flush=True)
    pd.DataFrame(rows).to_csv(out / "gen.csv", index=False, encoding="utf-8")
    (out / "gen_meta.json").write_text(json.dumps(dict(load_s=load_s, vram_gb=torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                                                       gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", nfe=a.nfe, model=str(a.model_dir)), indent=1))
    print("generated", len(rows), "clips, load", round(load_s, 1), "s")


# ----------------------------------------------------------------------------- score
def edit(a, b):
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        c = [i]
        for j, y in enumerate(b, 1):
            c.append(min(p[j] + 1, c[-1] + 1, p[j - 1] + (x != y)))
        p = c
    return p[-1]


def spans(y, sr):
    """Longest low-energy span (s) strictly inside the utterance."""
    h = int(0.02 * sr); r = np.sqrt(np.convolve(y ** 2, np.ones(h) / h, "valid"))[::h]
    thr = 0.15 * np.percentile(r, 90); q = r < thr
    idx = np.where(~q)[0]
    if len(idx) == 0:
        return 0.0
    best = cur = 0
    for x in q[idx[0]:idx[-1] + 1]:
        cur = cur + 1 if x else 0; best = max(best, cur)
    return best * 0.02


def onset_stats(y, sr):
    h = int(0.01 * sr); n = len(y) // h
    r = np.sqrt((y[: n * h].reshape(n, h) ** 2).mean(1) + 1e-12); db = 20 * np.log10(r / np.percentile(r, 95))
    idx = np.where(db > -25)[0]
    return dict(t_first_ms=int(idx[0]) * 10 if len(idx) else -1, db0=float(db[:2].mean()), clipped_onset=int(len(idx) > 0 and idx[0] == 0 and db[:3].mean() > -12))


def first_word(text):
    from pythainlp.tokenize import word_tokenize
    toks = [t for t in word_tokenize(text, engine="newmm") if CLEAN(t)]
    w = CLEAN(toks[0]) if toks else ""
    if len(w) < 2 and len(toks) > 1:
        w += CLEAN(toks[1])
    return w


def asr_pipe(name, dev):
    import torch
    from transformers import pipeline
    mid = {"turbo": "openai/whisper-large-v3-turbo", "pathumma": "nectec/Pathumma-whisper-th-large-v3"}[name]
    return pipeline("automatic-speech-recognition", model=mid, device=0 if dev == "cuda" else -1, torch_dtype=torch.float16 if dev == "cuda" else torch.float32)


def score(a):
    import librosa, torch
    from transformers import AutoFeatureExtractor, WavLMForXVector
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv"); sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
    pad = lambda y: np.pad(y, (0, max(0, 16000 - len(y))))
    emb = lambda y: torch.nn.functional.normalize(sv(**fe(pad(y), sampling_rate=16000, return_tensors="pt")).embeddings, dim=-1)[0]
    res = []
    for name, d in (x.split("=") for x in a.dirs):
        d = Path(d); g = pd.read_csv(d / "gen.csv"); f = d / "score.csv"
        mdir = Path(dict(x.split("=") for x in a.model_dirs)[name])                            # model dir that holds voices.json + refs/
        vo = json.loads((mdir / "voices.json").read_text(encoding="utf-8"))
        with torch.no_grad():
            ref_emb = {k: emb(librosa.load(mdir / v["ref"], sr=16000)[0]) for k, v in vo.items()}
        done = pd.read_csv(f) if f.exists() else pd.DataFrame()
        rows = done.to_dict("records"); have = {r["file"] for r in rows}
        todo = g[~g.file.isin(have)]
        for r in todo.itertuples():                                                          # pass 1 (cheap): speaker similarity, F0, energy, onset, gap span
            y, sr = sf.read(d / r.file, dtype="float32"); y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
            row = dict(model=name, file=r.file, case=r.case, kind=r.kind, voice=r.voice, seed=r.seed, text=r.text, emotion=r.emotion, intensity=r.intensity,
                       rtf=r.rtf, gen_s=r.gen_s, audio_s=r.audio_s)
            with torch.no_grad():
                row["spk_sim"] = float(ref_emb[r.voice] @ emb(y16))
            f0 = librosa.yin(y, fmin=60, fmax=450, sr=sr, frame_length=2048, hop_length=480)
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=480)[0][: len(f0)]
            v = f0[rms > 0.5 * np.median(rms)]
            ref = CLEAN(r.text.replace("…", ""))
            row.update(f0_mean=float(np.median(v)) if len(v) else np.nan, f0_std=float(np.std(v)) if len(v) else np.nan, energy_db=float(20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-9)),
                       rate_cps=len(ref) / max(len(y) / sr, 1e-3), gap_span_s=spans(y, sr), **onset_stats(y, sr))
            rows.append(row)
        for n, bs in (("turbo", 4), ("pathumma", 1)):                                         # pass 2: ONE ASR model on the GPU at a time (both do not fit in 4 GB), batched
            todo_r = [r for r in rows if pd.isna(r.get(f"cer_{n}", np.nan)) and (n == "turbo" or (str(r["kind"]).startswith(("onset", "short")) and r["seed"] == 0))]   # Pathumma only where the first word matters, seed 0 (4 GB GPU)
            if not todo_r:
                continue
            p = asr_pipe(n, dev)
            for k in range(0, len(todo_r), 16):
                chunk = todo_r[k:k + 16]
                wavs = [librosa.resample(sf.read(d / r["file"], dtype="float32")[0], orig_sr=sf.info(d / r["file"]).samplerate, target_sr=16000) for r in chunk]
                outs = p([{"raw": w, "sampling_rate": 16000} for w in wavs], batch_size=bs, generate_kwargs={"language": "th", "task": "transcribe"})
                for r, o in zip(chunk, outs):
                    hyp = o["text"]; h = CLEAN(hyp); ref = CLEAN(r["text"].replace("…", "")); fw = first_word(r["text"].replace("…", " "))
                    r[f"hyp_{n}"] = hyp; r[f"cer_{n}"] = edit(ref, h) / max(len(ref), 1); r[f"fw_{n}"] = int(h.startswith(fw)) if fw else np.nan
                if (k // 16) % 5 == 4:
                    pd.DataFrame(rows).to_csv(f, index=False, encoding="utf-8")
            del p; import gc; gc.collect(); torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(f, index=False, encoding="utf-8")
        res.append(pd.DataFrame(rows))
    r = pd.concat(res)
    r["cer_turbo_c"] = r.cer_turbo.clip(upper=1.0)
    print(r.groupby(["model", "kind"])[[c for c in ("cer_turbo_c", "cer_pathumma", "fw_turbo", "fw_pathumma", "spk_sim", "t_first_ms") if c in r]].median().round(3).to_string())


# ----------------------------------------------------------------------------- affect (age + emotion on generated audio)
EVAL_SER = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"        # evaluator: a different model family from the labeller (emotion2vec)
SER_MAP = {"angry": "angry", "disgust": "disgust", "fearful": "fear", "happy": "happy", "neutral": "neutral", "sad": "sad", "surprised": "surprise"}


def affect(a):
    import librosa, torch
    sys.path.insert(0, str(HERE))
    from prep import Age
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    age = Age(dev)
    fe = AutoFeatureExtractor.from_pretrained(EVAL_SER); ser = AutoModelForAudioClassification.from_pretrained(EVAL_SER).to(dev).eval()
    for name, d in (x.split("=") for x in a.dirs):
        d = Path(d); g = pd.read_csv(d / "gen.csv"); rows = []
        for r in g.itertuples():
            if r.kind != "emo" and not r.case.startswith(("s0", "h0")):
                continue
            y = librosa.load(d / r.file, sr=16000)[0]
            row = dict(file=r.file, **age(y))
            x = fe(y, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                p = torch.softmax(ser(**{k: v.to(dev) for k, v in x.items()}).logits, -1)[0].cpu().numpy()
            for i, l in ser.config.id2label.items():
                row["ser_" + SER_MAP.get(l.lower(), l.lower())] = float(p[int(i)])
            rows.append(row)
        pd.DataFrame(rows).to_csv(d / "affect.csv", index=False)
        print(name, len(rows), "clips")


# ----------------------------------------------------------------------------- latency
def latency(a):
    sys.path.insert(0, str(HERE))
    import torch
    from tts import Speech
    t0 = time.time(); s = Speech(a.model_dir, warm=False); load_s = time.time() - t0
    v = a.voice or next(iter(s.voices))
    t0 = time.time(); s.say("ทดสอบ", voice=v, nfe=a.nfe); cold_first_s = time.time() - t0               # first call incl. reference preparation / kernel warm-up
    res = dict(load_s=load_s, cold_first_s=cold_first_s, nfe=a.nfe, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", sets={})
    for k, text in LAT.items():
        ts, rtf, tm = [], [], {}
        for i in range(a.n):
            s.timing = {}; t = time.perf_counter(); wav, sr = s.say(text, voice=v, seed=i, nfe=a.nfe)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t; ts.append(dt); rtf.append(dt / (len(wav) / sr))
            for kk, vv in s.timing.items():
                tm.setdefault(kk, []).append(vv)
        res["sets"][k] = dict(audio_s=len(wav) / sr, p50=float(np.percentile(ts, 50)), p95=float(np.percentile(ts, 95)), rtf=float(np.median(rtf)),
                              split={kk: float(np.median(vv)) for kk, vv in tm.items()})
        print(k, res["sets"][k], flush=True)
    res["vram_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    Path(a.out).write_text(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------- listening pack
def pack(a):
    d = Path(a.dir); g = pd.read_csv(d / "gen.csv"); g = g[g.kind == "emo"].sample(frac=1, random_state=0).reset_index(drop=True)
    g["id"] = [f"c{i:03d}" for i in range(len(g))]
    g[["id", "file", "voice", "emotion", "intensity", "text"]].to_csv(d / "listening_key.csv", index=False, encoding="utf-8")       # answer key: do NOT open before rating
    opts = "".join(f"<option>{e}</option>" for e in ["", *EMOTIONS])
    rows = "".join(f"<tr><td>{r.id}</td><td><audio controls preload=none src='{r.file}'></audio></td><td><select>{opts}</select></td>"
                   f"<td><select><option></option>{''.join(f'<option>{i}</option>' for i in range(1, 6))}</select></td></tr>" for r in g.itertuples())
    (d / "listening_pack.html").write_text("<meta charset=utf-8><title>listening pack</title><h3>Listen, pick the emotion you hear and rate naturalness (1 bad - 5 excellent)</h3>"
                                           "<table border=1 cellpadding=4><tr><th>id<th>audio<th>emotion heard<th>naturalness</tr>" + rows + "</table>"
                                           "<p>Write your answers into listening_answers.csv (id,emotion,naturalness).", encoding="utf-8")
    print("pack:", len(g), "clips ->", d / "listening_pack.html")


# ----------------------------------------------------------------------------- compare (tables + paired bootstrap + acceptance numbers)
def boot(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(x, float) - np.asarray(y, float); d = d[~np.isnan(d)]
    if len(d) == 0:
        return None
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return dict(mean=round(float(d.mean()), 4), lo=round(float(np.percentile(m, 2.5)), 4), hi=round(float(np.percentile(m, 97.5)), 4), n=int(len(d)))


PRES = lambda v: "masculine" if v.startswith(("masc", "male")) else "feminine"
DEFAULT = {"v3": ("masculine_1", "feminine_1"), "v4": ("male", "female")}
GROUP = lambda k: "onset" if k.startswith("onset") else k


def load(d):
    r = pd.read_csv(Path(d) / "score.csv"); r.loc[r.seed != 0, ["cer_pathumma", "fw_pathumma"]] = np.nan; r["pres"] = r.voice.map(PRES); r["grp"] = r.kind.map(GROUP); r["cer"] = r.cer_turbo.clip(upper=1.0)
    return r


def compare(a):
    D_ = {n: d for n, d in (x.split("=") for x in a.dirs)}
    S = {n: load(d) for n, d in D_.items()}
    out = {}
    v3, v4 = S["v3"], S["v4"]
    dv = lambda r, m: r[r.voice.isin(DEFAULT[m])]                                     # the two default voices of each model

    def kinds(r):
        return {g: dict(n=int(len(x)), cer_median=round(float(x.cer.median()), 4), cer_mean=round(float(x.cer.mean()), 4), cer_path_mean=round(float(x.cer_pathumma.clip(upper=1).mean()), 4) if x.cer_pathumma.notna().any() else None,
                        fw_turbo=round(float(x.fw_turbo.mean()), 4), fw_pathumma=round(float(x.fw_pathumma.mean()), 4) if x.fw_pathumma.notna().any() else None, sim=round(float(x.spk_sim.mean()), 4),
                        t_first_ms=round(float(x.t_first_ms.mean()), 1), clipped_onset=round(float(x.clipped_onset.mean()), 4), gap_span_s=round(float(x.gap_span_s.mean()), 3), rtf=round(float(x.rtf.median()), 3))
                for g, x in r.groupby("grp")}
    out["default_voices"] = {m: kinds(dv(S[m], m)) for m in ("v3", "v4")}
    out["all_voices"] = {m: kinds(S[m]) for m in ("v3", "v4")}
    fwset = lambda r: r[r.grp.isin(["onset", "short"])]
    out["first_word_by_presentation"] = {m: {p: dict(turbo=round(float(x.fw_turbo.mean()), 4), pathumma=round(float(x.fw_pathumma.mean()), 4), n=int(len(x)))
                                             for p, x in fwset(S[m]).groupby("pres")} for m in S if "n" not in m[:1] or m in ("v3", "v4", "v3y32", "v3y16")}
    out["first_word_by_voice"] = {m: {v: dict(turbo=round(float(x.fw_turbo.mean()), 3), pathumma=round(float(x.fw_pathumma.mean()), 3), n=int(len(x))) for v, x in fwset(S[m]).groupby("voice")} for m in ("v3", "v4")}
    rep = lambda r, v: r[(r.text == "ฉันชอบแมว") & (r.voice == v)]
    out["reported_sentence"] = {m: {v: dict(turbo=round(float(rep(S[m], v).fw_turbo.mean()), 3), pathumma=round(float(rep(S[m], v).fw_pathumma.mean()), 3), n=int(len(rep(S[m], v)))) for v in DEFAULT[m]} for m in ("v3", "v4")}
    # paired bootstrap on the same sentence / seed / presentation, default voices
    a3 = dv(v3, "v3").set_index(["case", "seed", "pres"]); a4 = dv(v4, "v4").set_index(["case", "seed", "pres"]); j = a3.join(a4, lsuffix="_v3", rsuffix="_v4", how="inner")
    out["paired_v4_minus_v3"] = {c: boot(j[c + "_v4"], j[c + "_v3"]) for c in ("cer", "fw_turbo", "fw_pathumma", "spk_sim", "t_first_ms", "f0_mean", "energy_db")}
    for g in ("sent", "test", "word", "gap", "short", "onset"):
        jj = j[j.grp_v3 == g]
        out["paired_v4_minus_v3"][f"cer[{g}]"] = boot(jj.cer_v4, jj.cer_v3)
    # reference effect vs model effect (onset+short, seed 0, default voices): v3 old refs / v3 new refs / v4
    ref = {}
    for nm, r, vs in (("v3 old refs, 32 steps", v3, DEFAULT["v3"]), ("v3 new refs, 32 steps", S.get("v3y32"), ("male", "female")), ("v3 new refs, 16 steps", S.get("v3y16"), ("male", "female")), ("v4, 16 steps", v4, DEFAULT["v4"])):
        if r is None:
            continue
        x = fwset(r[r.voice.isin(vs)]); x = x[x.seed == 0]
        ref[nm] = dict(n=int(len(x)), fw_turbo=round(float(x.fw_turbo.mean()), 3), fw_pathumma=round(float(x.fw_pathumma.mean()), 3), cer_mean=round(float(x.cer.mean()), 3), sim=round(float(x.spk_sim.mean()), 3))
    out["reference_vs_model"] = ref
    # v3 AS SHIPPED (old references, 50 ms tail, 32 steps) from the local ablation runs (data/ablation): the cloud "v3" rows above already use the new 0.30 s tail
    A_ = Path(__file__).parent / "data" / "ablation"
    if (A_ / "v3_old_refs_nfe32.csv").exists():
        o = pd.read_csv(A_ / "v3_old_refs_nfe32.csv"); sh = pd.read_csv(A_ / "tail005.csv"); both = pd.concat([o.assign(grp="onset"), sh.assign(grp="short")])
        both["pres"] = both.voice.map(PRES)
        ship = {}
        for nm, x in (("all seeds", both), ("seed 0", both[both.seed == 0])):
            ship[nm] = {p_: dict(turbo=round(float(y.fw_ok_turbo.mean()), 3), pathumma=round(float(y.fw_ok_pathumma.mean()), 3), n=int(len(y))) for p_, y in x.groupby("pres")}
        r_ = sh[sh.text == "ฉันชอบแมว"]
        ship["reported_sentence"] = {v: dict(turbo=round(float(y.fw_ok_turbo.mean()), 3), pathumma=round(float(y.fw_ok_pathumma.mean()), 3), n=int(len(y))) for v, y in r_.groupby("voice")}
        out["v3_as_shipped"] = ship
    # step sweep for v4 (seed 0, onset+short, male/female)
    sw = {}
    base = fwset(v4[v4.voice.isin(DEFAULT["v4"])]); base = base[base.seed == 0]
    for nm, r in (("16 (default)", base), ("32", S.get("n32")), ("12", S.get("n12"))):
        if r is None:
            continue
        x = fwset(r[r.voice.isin(DEFAULT["v4"])]) if nm != "16 (default)" else r
        sw[nm] = dict(n=int(len(x)), fw_turbo=round(float(x.fw_turbo.mean()), 3), fw_pathumma=round(float(x.fw_pathumma.mean()), 3), cer_mean=round(float(x.cer.mean()), 3), sim=round(float(x.spk_sim.mean()), 3),
                      rtf_median=round(float(x.rtf.median()), 3), gen_s_median=round(float(x.gen_s.median()), 3))
    out["step_sweep_v4"] = sw
    # emotion + age on the generated audio
    af = Path(D_["v4"]) / "affect.csv"
    if af.exists():
        e = v4[v4.kind == "emo"].merge(pd.read_csv(af), on="file"); neu = e[e.emotion == "neutral"]
        em = {}
        for it in (0.5, 1.0):
            x = e[((e.intensity == it) | (e.emotion == "neutral"))]; M = np.zeros((7, 7))
            for r_, emo in enumerate(EMOTIONS):
                p = x[x.emotion == emo][[f"ser_{c}" for c in EMOTIONS]].values
                if len(p):
                    M[r_] = np.bincount(p.argmax(1), minlength=7) / len(p)
            em[f"confusion_intensity_{it}"] = dict(matrix=np.round(M, 3).tolist(), rows=EMOTIONS, accuracy=round(float(np.trace(M) / 7), 3), recall={EMOTIONS[i]: round(float(M[i, i]), 3) for i in range(7)})
        base_f = neu.groupby("voice")[["f0_mean", "f0_std", "energy_db", "rate_cps"]].mean()
        sh = {}
        for emo in EMOTIONS[1:]:
            sh[emo] = {}
            for it in (0.5, 1.0):
                x = e[(e.emotion == emo) & (e.intensity == it)].groupby("voice")[["f0_mean", "f0_std", "energy_db", "rate_cps"]].mean()
                sh[emo][str(it)] = dict(f0_semitones=round(float(np.mean([12 * np.log2(x.loc[v, "f0_mean"] / base_f.loc[v, "f0_mean"]) for v in x.index])), 3),
                                        f0_std_ratio=round(float(np.mean([x.loc[v, "f0_std"] / base_f.loc[v, "f0_std"] for v in x.index])), 3),
                                        energy_db=round(float(np.mean([x.loc[v, "energy_db"] - base_f.loc[v, "energy_db"] for v in x.index])), 3),
                                        rate_ratio=round(float(np.mean([x.loc[v, "rate_cps"] / base_f.loc[v, "rate_cps"] for v in x.index])), 3))
        em["prosody_shift_vs_neutral"] = sh
        em["cer_by_emotion"] = {emo: round(float(e[e.emotion == emo].cer.mean()), 4) for emo in EMOTIONS}
        em["intensity_monotonic"] = {emo: bool(abs(sh[emo]["1.0"]["f0_semitones"]) >= abs(sh[emo]["0.5"]["f0_semitones"]) and abs(sh[emo]["1.0"]["energy_db"]) >= abs(sh[emo]["0.5"]["energy_db"] - 0.05)) for emo in EMOTIONS[1:]}
        out["emotion"] = em
        ag = v4.merge(pd.read_csv(af), on="file"); ag = ag[ag.voice.isin(DEFAULT["v4"]) & ag.kind.isin(["sent", "short", "emo"]) & (ag.emotion.fillna("") .isin(["", "neutral"]))]
        out["age_generated"] = {v: dict(mean=round(float(x.age.mean()), 1), median=round(float(x.age.median()), 1), p10=round(float(x.age.quantile(.1)), 1), p90=round(float(x.age.quantile(.9)), 1), share_18_23=round(float(x.age.between(18, 23).mean()), 3), n=int(len(x)))
                                for v, x in ag.groupby("voice")}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False)); print(json.dumps({k: out[k] for k in ("first_word_by_presentation", "reported_sentence", "reference_vs_model", "step_sweep_v4")}, indent=1)[:3500])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["gen", "score", "affect", "latency", "pack", "compare"])
    ap.add_argument("--model_dir"); ap.add_argument("--out"); ap.add_argument("--dirs", nargs="+"); ap.add_argument("--model_dirs", nargs="+"); ap.add_argument("--dir"); ap.add_argument("--voices", default="")
    ap.add_argument("--emo_voices", default="male,female"); ap.add_argument("--sets", default="main,onset,short,emo"); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--nfe", type=int, default=None); ap.add_argument("--test_n", type=int, default=10); ap.add_argument("--n", type=int, default=7); ap.add_argument("--voice")
    a = ap.parse_args()
    {"gen": gen, "score": score, "affect": affect, "latency": latency, "pack": pack, "compare": compare}[a.cmd](a)
