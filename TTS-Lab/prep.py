"""Data prep for TTS v2: S+, S and A+ clips of Chalermdej/yodas2_sidon_th_tts, noise-free, with per-clip onset/F0/age/emotion annotations.

  python prep.py select   # metadata only (CPU): grade-wise filters -> data/select.parquet
  python prep.py fetch    # download shard by shard, decode selected rows, quality checks, onset + F0 -> data/clips/*.flac (raw shard deleted)
  python prep.py label    # GPU passes: perceived age (audeering) per speaker sample, emotion (emotion2vec+) per clip -> data/labels.parquet
  python prep.py build    # manifest.csv (anonymised speakers, splits, voice presentation, age bucket, emotion pseudo-label, exclusion reason)

Rules: text agrees (>=0.9) with BOTH whisper_large and whisper_typhoon, 2-12 s, DNSMOS per grade (S+ >= 3.2, S >= 3.3, A+ >= 3.3), <= 40 clips/speaker,
no clipping / near-silence. Speakers (YouTube video ids) are anonymised in the manifest. Splits are speaker-disjoint and keep v1's test/val speakers.
"""
import difflib, hashlib, io, re, sys, time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd, pyarrow.parquet as pq, soundfile as sf

REPO = "Chalermdej/yodas2_sidon_th_tts"
HERE = Path(__file__).parent
OUT = HERE / "data"
DNSMOS_MIN = {"S+": 3.2, "S": 3.3, "A+": 3.3}
MAX_PER_SPK, SEED, SALT = 40, 0, "tts-v2"
norm = lambda x: re.sub(r"[\s\.,!?\-\"'…]", "", x or "")
sim = lambda a, b: difflib.SequenceMatcher(None, norm(a), norm(b), autojunk=False).ratio()
anon = lambda s: "spk" + hashlib.sha1(f"{SALT}{s}".encode()).hexdigest()[:8]


def select():
    from huggingface_hub import HfFileSystem
    d = pd.read_parquet(OUT / "meta.parquet")
    d = d[d.duration.between(2, 12)].copy()
    n0 = len(d)
    ag = OUT / "agree.parquet"
    if ag.exists():
        d = d.merge(pd.read_parquet(ag), on="utt_id")
    else:
        d["agree"] = [min(sim(a, b), sim(a, c)) for a, b, c in zip(d.text, d.whisper_large, d.whisper_typhoon)]
    d["reason"] = ""
    d.loc[d.agree < 0.9, "reason"] = "asr_disagree"
    d.loc[(d.reason == "") & (d.dnsmos_overall < d.grade_avg.map(DNSMOS_MIN)), "reason"] = "dnsmos"
    d.loc[(d.reason == "") & (d.text.str.strip().str.len() < 2), "reason"] = "empty_text"
    d.loc[(d.reason == "") & d.text.duplicated(keep="first"), "reason"] = "dup_text"
    ok = d[d.reason == ""].sample(frac=1, random_state=SEED)
    ok = ok.groupby("speaker_id").head(MAX_PER_SPK)
    d.loc[~d.utt_id.isin(ok.utt_id) & (d.reason == ""), "reason"] = "speaker_cap"
    print(f"{n0} clips 2-12 s -> {len(ok)} selected, {ok.duration.sum() / 3600:.1f} h, {ok.speaker_id.nunique()} speakers")
    print(ok.groupby("grade_avg").agg(n=("utt_id", "size"), h=("duration", lambda x: x.sum() / 3600)).round(1).to_string())
    print(d.reason.value_counts().to_dict())
    ok.to_parquet(OUT / "select.parquet"); d[["utt_id", "reason"]].to_parquet(OUT / "reasons.parquet")


# ----------------------------------------------------------------------------- fetch
def _proc(args):
    """decode one clip, quality checks, onset/offset silence + median F0 of voiced frames; write flac."""
    import librosa
    utt, raw, dest = args
    y, sr = sf.read(io.BytesIO(raw), dtype="float32")
    y = y.mean(1) if y.ndim > 1 else y
    if sr != 24000:
        y = librosa.resample(y, orig_sr=sr, target_sr=24000)
    pk = float(np.abs(y).max())
    if pk >= 0.999 or (np.abs(y) > 0.995).mean() > 1e-4:
        return utt, dict(reason="clipping")
    if pk < 0.05:
        return utt, dict(reason="too_quiet")
    h = 240
    n = len(y) // h
    r = np.sqrt((y[: n * h].reshape(n, h) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(r / np.percentile(r, 95))
    act = np.where(db > -30)[0]
    lead, trail = int(act[0]) * 10, int(n - 1 - act[-1]) * 10
    f0 = librosa.yin(y, fmin=60, fmax=450, sr=24000, frame_length=2048, hop_length=480)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=480)[0][: len(f0)]
    v = f0[rms > 0.5 * np.median(rms)]
    sf.write(dest, y, 24000, subtype="PCM_16")
    return utt, dict(reason="", lead_ms=lead, trail_ms=trail, f0=float(np.median(v)) if len(v) else np.nan, f0_std=float(np.std(v)) if len(v) else np.nan,
                     peak=pk, rms_db=float(20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-9)))


def fetch():
    from huggingface_hub import hf_hub_download
    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    sel = pd.read_parquet(OUT / "select.parquet")
    shards = sorted(sel.shard.unique())
    done_f = OUT / "fetched.parquet"
    done = pd.read_parquet(done_f) if done_f.exists() else pd.DataFrame(columns=["utt_id"])
    todo = [s for s in shards if not sel[sel.shard == s].utt_id.isin(done.utt_id).all()]
    print(len(todo), "shards to fetch", flush=True)
    get = lambda sh: hf_hub_download(REPO, f"data/{sh}", repo_type="dataset", local_dir=str(OUT / "raw"))
    rows = done.to_dict("records")
    with ThreadPoolExecutor(1) as dl, ProcessPoolExecutor(4) as pool:
        fut = dl.submit(get, todo[0]) if todo else None
        for i, sh in enumerate(todo):
            p = fut.result()
            fut = dl.submit(get, todo[i + 1]) if i + 1 < len(todo) else None                   # prefetch the next shard
            sub = sel[sel.shard == sh]
            t = pq.ParquetFile(p).read(columns=["audio"]).column("audio")
            jobs = [(r.utt_id, t[int(r.row)].as_py()["bytes"], str(OUT / "clips" / f"{r.utt_id}.flac")) for r in sub.itertuples()]
            for utt, info in pool.map(_proc, jobs, chunksize=16):
                rows.append(dict(utt_id=utt, **info))
            del t
            Path(p).unlink()
            pd.DataFrame(rows).to_parquet(done_f)
            print(f"{sh}: {len(rows)} clips so far ({i + 1}/{len(todo)})", flush=True)


# ----------------------------------------------------------------------------- label (GPU)
class Age:
    """audeering/wav2vec2-large-robust-24-ft-age-gender (CC-BY-NC-SA-4.0): age in years (0-1 -> 0-100) + P(female, male, child). 'Perceived', not true age."""
    def __init__(self, dev):
        import torch, torch.nn as nn
        from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

        class Head(nn.Module):
            def __init__(s, c, n):
                super().__init__(); s.dense = nn.Linear(c.hidden_size, c.hidden_size); s.dropout = nn.Dropout(c.final_dropout); s.out_proj = nn.Linear(c.hidden_size, n)
            def forward(s, x):
                return s.out_proj(s.dropout(torch.tanh(s.dense(s.dropout(x)))))

        class M(Wav2Vec2PreTrainedModel):
            def __init__(s, c):
                super().__init__(c); s.wav2vec2 = Wav2Vec2Model(c); s.age = Head(c, 1); s.gender = Head(c, 3); s.post_init()
            def forward(s, x):
                h = s.wav2vec2(x)[0].mean(1)
                return s.age(h), torch.softmax(s.gender(h), 1)

        self.torch, self.dev = torch, dev
        self.m = M.from_pretrained("audeering/wav2vec2-large-robust-24-ft-age-gender").to(dev).half().eval()

    def __call__(self, y):
        t = self.torch
        x = (y - y.mean()) / (y.std() + 1e-7)
        with t.no_grad():
            a, g = self.m(t.from_numpy(x)[None].to(self.dev).half())
        return dict(age=float(a[0, 0]) * 100, p_female=float(g[0, 0]), p_male=float(g[0, 1]), p_child=float(g[0, 2]))        # head order: female, male, child (verified against F0)


class Emotion:
    """emotion2vec_plus_large (FunASR): utterance-level posteriors. Labeller only -- evaluation uses another model (evaluate.py)."""
    MAP = {"angry": "angry", "disgusted": "disgust", "fearful": "fear", "happy": "happy", "neutral": "neutral", "sad": "sad", "surprised": "surprise", "other": "other", "<unk>": "unk"}

    def __init__(self, dev):
        from funasr import AutoModel
        self.m = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf", disable_update=True, device=dev)

    def __call__(self, y):
        r = self.m.generate(y, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
        return {"emo_" + self.MAP[l.split("/")[-1]]: float(s) for l, s in zip(r["labels"], r["scores"])}

def label(limit=None):
    import librosa, torch
    sel = pd.read_parquet(OUT / "select.parquet").drop(columns="reason").merge(pd.read_parquet(OUT / "fetched.parquet"), on="utt_id")
    sel = sel[sel.reason == ""]
    lf = OUT / "labels.parquet"
    lab = pd.read_parquet(lf) if lf.exists() else pd.DataFrame(columns=["utt_id"])
    todo = sel[~sel.utt_id.isin(lab.utt_id)]
    if limit:
        todo = todo.head(max(limit - len(lab), 0))                                                  # exactly `limit` labelled clips in total
    print(len(todo), "clips to label", flush=True)
    age, emo = Age("cuda"), Emotion("cuda")
    spk_n = lab.merge(sel[["utt_id", "speaker_id"]], on="utt_id").groupby("speaker_id").age.count() if "age" in lab else pd.Series(dtype=int)
    rows = lab.to_dict("records")
    for k, r in enumerate(todo.itertuples()):
        y = librosa.load(OUT / "clips" / f"{r.utt_id}.flac", sr=16000)[0]
        row = dict(utt_id=r.utt_id, **emo(y))
        if spk_n.get(r.speaker_id, 0) < 6:                                                         # age: <= 6 clips per speaker (speaker-level aggregate)
            row.update(age(y)); spk_n[r.speaker_id] = spk_n.get(r.speaker_id, 0) + 1
        rows.append(row)
        if k % 2000 == 1999:
            pd.DataFrame(rows).to_parquet(lf); print(k + 1, flush=True)
    pd.DataFrame(rows).to_parquet(lf)


# ----------------------------------------------------------------------------- build
E7 = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]


def emo_label(r):
    """natural pseudo-label: emotion2vec's top class AND the clip's speaker-normalised prosody agree (thresholds read off the data, see the report)."""
    if r.n < 8:
        return None
    t, c = r.top, r.conf
    if t == "happy" and c >= .8 and (r.f0_z >= .3 or r.f0_std_z >= .3): return "happy"
    if t == "angry" and c >= .7 and (r.rms_db_z >= .2 or r.f0_z >= .3): return "angry"
    if t == "surprise" and c >= .6 and r.f0_z >= .3: return "surprise"
    if t == "sad" and c >= .7 and r.f0_z <= -.2 and r.rms_db_z <= 0: return "sad"
    if t == "disgust" and c >= .8 and r.rms_db_z <= -.3: return "disgust"
    if t == "fear" and c >= .5: return "fear"
    if t == "neutral" and c >= .8 and abs(r.f0_z) < .5 and abs(r.rms_db_z) < .5: return "neutral"
    return None


def build():
    sel = pd.read_parquet(OUT / "select.parquet").drop(columns="reason").merge(pd.read_parquet(OUT / "fetched.parquet"), on="utt_id", how="left")
    lab = pd.read_parquet(OUT / "labels.parquet") if (OUT / "labels.parquet").exists() else pd.DataFrame(columns=["utt_id"])
    m = sel.merge(lab, on="utt_id", how="left")
    m["reason"] = m.reason.fillna("not_fetched")
    m["labelled"] = m.emo_neutral.notna() if "emo_neutral" in m else False
    # --- speaker-level: median F0, perceived age (audeering), gender head -> voice presentation
    g = m.groupby("speaker_id")
    spk = pd.DataFrame(dict(f0=g.f0.median(), age=g.age.median(), n_age=g.age.count(), pf=g.p_female.mean() if "p_female" in m else np.nan, pm=g.p_male.mean() if "p_male" in m else np.nan))
    def vp(r):
        by_f0 = "feminine" if r.f0 >= 180 else ("masculine" if r.f0 <= 150 else None)
        by_head = None if np.isnan(r.pf) else ("feminine" if r.pf > r.pm else "masculine")
        if by_f0 and by_head:
            return by_f0 if by_f0 == by_head else "unk"
        if by_f0:
            return by_f0
        if by_head and max(r.pf, r.pm) > 0.9:
            return by_head
        return "unk"
    spk["voice"] = spk.apply(vp, axis=1)
    m["voice"] = m.speaker_id.map(spk.voice)
    m["voice_how"] = "speaker-median F0 (>=180 feminine, <=150 masculine) must agree with audeering gender head; F0 150-180 -> head if >0.9; else unk"
    m["spk_age"] = m.speaker_id.map(spk.age.where(spk.n_age >= 3))
    # --- emotion: top class + speaker-normalised prosody
    if m.labelled.any():
        P = m[["emo_" + e for e in E7]].values
        m["top"] = [E7[i] if not np.isnan(r).any() else None for i, r in zip(np.nan_to_num(P).argmax(1), P)]
        m["conf"] = np.nanmax(P, axis=1)
        g = m.groupby("speaker_id")
        for c in ("f0", "f0_std", "rms_db"):
            m[c + "_z"] = (m[c] - g[c].transform("mean")) / (g[c].transform("std") + 1e-6)
        m["n"] = g.utt_id.transform("size")
        m["emo"] = [emo_label(r) if r.labelled else None for r in m.itertuples()]
    old = pd.read_csv(OUT / "manifest_v1.csv") if (OUT / "manifest_v1.csv").exists() else pd.DataFrame(columns=["speaker_id", "split"])
    keep = old.drop_duplicates("speaker_id").set_index("speaker_id").split.to_dict()
    h = lambda s: int(hashlib.md5(f"{SEED}{s}".encode()).hexdigest(), 16) % 100
    m["split"] = m.speaker_id.map(lambda s: keep.get(str(s)) or ("test" if h(s) < 2 else ("val" if h(s) < 4 else "train")))
    rs = OUT / "ref_speakers.json"                                                               # default-voice reference speakers are held out of training
    if rs.exists():
        import json
        m.loc[m.speaker_id.map(anon).isin(json.loads(rs.read_text())), "split"] = "test"
    m["spk"] = m.speaker_id.map(anon)
    cols = ["utt_id", "spk", "grade_avg", "dnsmos_overall", "text", "duration", "voice", "voice_how", "f0", "f0_std", "lead_ms", "trail_ms", "rms_db", "split", "reason", "spk_age", "labelled"]
    cols += [c for c in m.columns if c.startswith("emo_") or c in ("age", "p_female", "p_male", "p_child", "emo", "top", "conf")]
    out = m[cols].rename(columns={"utt_id": "sample_id", "spk": "speaker_id", "dnsmos_overall": "dnsmos"})
    out["path"] = "clips/" + out.sample_id + ".flac"
    out.to_csv(OUT / "manifest.csv", index=False)
    up = out[(out.reason == "") & out.labelled & out.split.isin(["train", "val"])]                 # what the training job gets (labelled clips, held-out speakers excluded)
    up[["sample_id"]].to_csv(OUT / "upload_set.csv", index=False)
    ok = out[(out.reason == "") & out.labelled] if out.labelled.any() else out[out.reason == ""]
    print(ok.groupby(["grade_avg", "voice"]).duration.agg(["count", lambda x: x.sum() / 3600]).round(1).to_string())
    print(ok.groupby("split").duration.agg(["count", lambda x: x.sum() / 3600]).round(1).to_string())
    if "emo" in out:
        print(out.emo.value_counts().to_dict())
    print(pd.crosstab(spk.voice, spk.age.between(18, 23)).to_string())


if __name__ == "__main__":
    {"select": select, "fetch": fetch, "label": lambda: label(int(sys.argv[2]) if len(sys.argv) > 2 else None), "build": build}[sys.argv[1]]()
