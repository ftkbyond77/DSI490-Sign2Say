"""v4 data transformation: Standard Schema v4, cleaning decisions, weak word labels, vocabulary bank sources.

Label sources (no manual annotation):
  gloss      TTRS curated gloss per clip (strong)
  title      single-sign YouTube clips whose title/filename names the sign (e.g. ".../หมวดกิริยาอาการ/ช่วย", 'คำว่า "ดีใจ"')
  vtt_word   YouTube th-orig auto-captions: word-level timestamps of what the teacher SAYS while teaching the sign
  asr_word   Thai Whisper (biodatlab/whisper-th-medium-combined) for lessons without Thai captions
Speech words are only *candidates*: a sign instance is accepted after pose-based verification (modules/mining.py).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .data import normalize_text
from .utils import ART, ROOT, get_logger, write_json

log = get_logger("lexicon")
MAN4 = ART / "manifests_v4"
TS = re.compile(r"<(\d+):(\d{2}):(\d{2})\.(\d{3})>")
CUE = re.compile(r"(\d+):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})\.(\d{3})")
FINGERSPELL_KW = ("สะกดนิ้ว", "ก-ฮ", "ก ฮ", "a-z", "a - z", "พยัญชนะ", "สระ", "วรรณยุกต์", "ตัวอักษร", "fingerspell",
                  "consonant", "alphabet", "ตัวสะกด", "มาตรา")
SONG_KW = ("เพลง", "song")


def _sec(g):
    return int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000


# ----------------------------------------------------------------------------- VTT with word timestamps
def parse_vtt_words(path: Path) -> list[dict]:
    """YouTube auto-caption VTT → [{w, t0, t1}] using inline <hh:mm:ss.mmm><c>word</c> timestamps.
    Rolling duplicate lines (the previous line repeated without tags) are skipped; manual caption files (no inline
    tags) fall back to cue-level timing split evenly by characters."""
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    words, manual = [], []
    for block in re.split(r"\n\s*\n", txt):
        m = CUE.search(block)
        if not m:
            continue
        g = m.groups(); c0, c1 = _sec(g[:4]), _sec(g[4:])
        lines = [l for l in block[m.end():].splitlines() if l.strip()]
        tagged = [l for l in lines if "<c>" in l]
        for line in tagged:
            parts = re.split(r"(<\d+:\d{2}:\d{2}\.\d{3}>)", line)
            t = c0
            for p in parts:
                tm = TS.fullmatch(p)
                if tm:
                    t = _sec(tm.groups()); continue
                w = normalize_text(re.sub(r"</?c>", "", p))
                if w and not w.startswith("["):
                    words.append(dict(w=w, t0=t))
        if not tagged and lines and c1 - c0 > 0.05:
            manual.append((c0, c1, normalize_text(re.sub(r"<[^>]+>", "", " ".join(lines)))))
    if words:
        for i, w in enumerate(words):
            w["t1"] = words[i + 1]["t0"] if i + 1 < len(words) and words[i + 1]["t0"] - w["t0"] < 2.0 else w["t0"] + 0.6
        return words
    out, last = [], None
    for c0, c1, s in manual:
        if s == last or not s or s.startswith("["):
            continue
        last = s
        out.append(dict(w=s, t0=c0, t1=c1))
    return out


def best_vtt(media_path: str) -> tuple[Path | None, str | None]:
    base = (ROOT / media_path).with_suffix("")
    for tag in ("th-orig", "th", "th-th"):
        p = Path(f"{base}.{tag}.vtt")
        if p.exists():
            return p, tag
    return None, None


# ----------------------------------------------------------------------------- ASR (lessons without Thai captions)
def run_asr(items, model="biodatlab/whisper-th-medium-combined", max_s=320.0, out_path=None):
    """items: [(clip_id, media_path)] → {clip_id: [{w,t0,t1}]} via Thai Whisper with chunk timestamps."""
    import subprocess
    import torch
    from transformers import pipeline
    out_path = out_path or MAN4 / "asr_whisper.json"
    done = json.loads(Path(out_path).read_text(encoding="utf-8")) if Path(out_path).exists() else {}
    pipe = pipeline("automatic-speech-recognition", model=model, torch_dtype=torch.float16, device=0, chunk_length_s=30)
    for cid, mp in items:
        if cid in done:
            continue
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(ROOT / mp), "-t", str(max_s), "-ac", "1", "-ar", "16000",
                              "-f", "s16le", "-"], capture_output=True).stdout
        audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        if len(audio) < 16000 or np.abs(audio).mean() < 1e-3:
            done[cid] = []; continue
        try:
            r = pipe({"raw": audio, "sampling_rate": 16000}, return_timestamps=True, batch_size=4,
                     generate_kwargs={"language": "th", "task": "transcribe"})
            words = []
            for ch in r.get("chunks", []):
                t0, t1 = ch["timestamp"]
                t1 = t1 if t1 is not None else t0 + 2.0
                txt = normalize_text(ch["text"])
                if txt:
                    words.append(dict(w=txt, t0=float(t0), t1=float(t1)))
            done[cid] = words
        except Exception as e:  # keep going; record failure
            log.warning("asr %s failed: %s", cid, e)
            done[cid] = []
        write_json(out_path, done)
        log.info("asr %s: %d chunks", cid, len(done[cid]))
    return done


# ----------------------------------------------------------------------------- tokenisation against the lexicon
class LexTokenizer:
    def __init__(self, lemmas: list[str]):
        from pythainlp.corpus.common import thai_words
        from pythainlp.tokenize import Trie
        self.lemmas = {l for l in lemmas if l and " " not in l and len(l) >= 1}
        self.trie = Trie(set(thai_words()) | self.lemmas)

    def tokens(self, text: str):
        from pythainlp.tokenize import word_tokenize
        return [t for t in word_tokenize(text, custom_dict=self.trie, engine="newmm", keep_whitespace=False) if t.strip()]

    def hits(self, timed_words: list[dict]):
        """Re-tokenise the timed stream (ASR splits words into syllables) and return lexicon hits with times."""
        chars, times = [], []
        for w in timed_words:
            dur = max(w["t1"] - w["t0"], 0.05)
            for k, ch in enumerate(w["w"].replace(" ", "")):
                chars.append(ch); times.append(w["t0"] + dur * k / max(len(w["w"]), 1))
        text = "".join(chars)
        out, pos = [], 0
        for tok in self.tokens(text):
            i = text.find(tok, pos)
            if i < 0:
                continue
            pos = i + len(tok)
            if tok in self.lemmas:
                out.append(dict(word=tok, t0=round(times[i], 2), t1=round(times[min(pos, len(times)) - 1] + 0.3, 2)))
        return out


# ----------------------------------------------------------------------------- titles → single-sign labels
def title_word(title: str, duration_s: float, max_dur=15.0):
    """Short clips whose title names exactly one sign. Returns the word or None."""
    if duration_s > max_dur:
        return None
    t = normalize_text(title)
    m = re.search(r"คำว่า\s*[\"“”'‘’«]?\s*([^\"“”'‘’»|()\[\]]+?)\s*[\"“”'‘’»]?\s*(?:\||$|\()", t)
    if m:
        return m.group(1).strip()
    m = re.search(r"[\"“]([^\"”]{1,20})[\"”]", t)
    if m:
        return m.group(1).strip()
    m = re.search(r"\(ศัพท์ภาษามือ\)\s*(.+)$", t)
    if m:
        return m.group(1).strip()
    if "/" in t:
        return t.split("/")[-1].strip()
    if "-" in t:
        last = t.split("-")[-1].strip()
        return re.sub(r"\s*\d+$", "", last) or None
    m = re.search(r"\[([^\]]{1,20})\]", t)
    if m:
        return m.group(1).strip()
    return None


# ----------------------------------------------------------------------------- schema v4 + cleaning
def build_schema_v4(asr: dict | None = None):
    """Writes manifests_v4/{clips_v4, speech_words, lexicon_v4}.parquet + cleaning_report.json."""
    MAN4.mkdir(parents=True, exist_ok=True)
    c = pd.read_parquet(ART / "manifests" / "clips_signers.parquet")
    ttrs = c[c.source == "ttrs"]
    lemmas = sorted(set(ttrs.lemma))
    tok = LexTokenizer(lemmas)
    rows, speech = [], []
    for r in c.itertuples():
        rec = dict(clip_id=r.clip_id, source=r.source, subset=r.subset, media_path=r.media_path, duration_s=r.duration_s,
                   width=r.width, height=r.height, signer_id=r.signer_id if r.source == "ttrs" else f"yt:{r.channel}",
                   title=r.title, label=None, label_source=None, usable_for="", exclude_reason=None)
        if r.source == "ttrs":
            rec.update(label=r.lemma, label_source="gloss", sign_variant_id=r.sign_variant_id)
            if r.duration_s > 15:
                rec["exclude_reason"] = "ttrs_long_clip(>15s: explanation / several signs)"
            elif isinstance(r.lemma, str) and r.lemma.strip() in ("", "ทดสอบ"):
                rec["exclude_reason"] = "test_entry"
            else:
                rec["usable_for"] = "vocab"
        else:
            tl = (r.title or "").lower()
            if r.subset in ("bigsign", "youtube_continuous"):
                rec["exclude_reason"] = "broadcast/song interpretation: caption = announcer speech with lag, 2-person layouts"
            elif any(k in tl for k in SONG_KW):
                rec["exclude_reason"] = "song interpretation (rhythm-driven signing)"
            elif any(k in tl for k in FINGERSPELL_KW):
                rec["exclude_reason"] = "fingerspelling/alphabet drill (letters, not lexical signs)"
            else:
                w = title_word(r.title or "", r.duration_s)
                if w:
                    rec.update(label=w, label_source="title", usable_for="vocab")
                vtt, tag = best_vtt(r.media_path)
                timed = parse_vtt_words(vtt) if vtt else []
                src = f"vtt_{tag}" if timed else None
                if not timed and asr and r.clip_id in asr:
                    timed, src = asr[r.clip_id], "asr_whisper"
                if timed:
                    hs = tok.hits(timed)
                    for h in hs:
                        speech.append(dict(clip_id=r.clip_id, source=src, **h))
                    if hs:
                        rec["usable_for"] = ",".join(x for x in (rec["usable_for"], "weak_word") if x)
                if not rec["usable_for"]:
                    rec["exclude_reason"] = "no usable label (no Thai speech captions/ASR hits, title is a topic)"
        rows.append(rec)
    clips = pd.DataFrame(rows)
    sp = pd.DataFrame(speech)
    clips.to_parquet(MAN4 / "clips_v4.parquet", index=False)
    sp.to_parquet(MAN4 / "speech_words.parquet", index=False)
    rep = dict(
        by_source_usable=clips.groupby(["source", "subset"]).usable_for.apply(lambda s: (s != "").sum()).to_dict(),
        excluded=clips.exclude_reason.value_counts().to_dict(),
        title_labels=clips[clips.label_source == "title"][["clip_id", "title", "label"]].to_dict("records"),
        speech_hits=int(len(sp)), speech_hit_words=int(sp.word.nunique()) if len(sp) else 0,
        speech_hits_by_source=sp.source.value_counts().to_dict() if len(sp) else {},
    )
    write_json(MAN4 / "cleaning_report.json", {k: (str(v) if isinstance(v, dict) and any(isinstance(x, tuple) for x in v) else v) for k, v in rep.items()})
    log.info("schema v4: %d clips, usable %d, speech hits %d (%d words)", len(clips), (clips.usable_for != "").sum(),
             len(sp), rep["speech_hit_words"])
    return clips, sp
