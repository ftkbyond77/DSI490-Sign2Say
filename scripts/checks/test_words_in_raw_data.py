"""Are the data_test words present anywhere in raw_data? (independent of the v4 manifests)
   python scripts/checks/test_words_in_raw_data.py  → artifacts/reports/test_words_in_raw_data.json + printed summary"""
import glob, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from pythainlp.tokenize import word_tokenize

TARGETS = {"ฉัน": ["ฉัน", "ผม", "ดิฉัน", "ข้าพเจ้า"], "ปลอบใจ": ["ปลอบใจ", "ปลอบ", "ปลอบโยน"], "เพื่อน": ["เพื่อน"],
           "ร้องไห้": ["ร้องไห้"], "ไป": ["ไป"], "กิน": ["กิน", "ทาน", "รับประทาน"], "ข้าว": ["ข้าว"],
           "ด้วยกัน": ["ด้วยกัน", "ร่วมกัน"], "ไหม": ["ไหม", "มั้ย", "หรือเปล่า", "หรือไม่"]}
out = open(sys.argv[1] if len(sys.argv) > 1 else ROOT / "artifacts/reports/test_words_in_raw_data.txt", "w", encoding="utf-8")
P = lambda *a: print(*a, file=out)

# ---- TTRS raw json
ttrs = []
for p in glob.glob(str(ROOT / "raw_data/word_level/ttrs_dictionary/*.json")):
    d = json.load(open(p, encoding="utf-8"))
    ttrs.append(dict(file=Path(p).stem, gloss=d.get("gloss_th", "").strip(), syn=", ".join(d["synonyms_th"]) if isinstance(d.get("synonyms_th"), list) else str(d.get("synonyms_th") or ""), cat=d.get("category_th", ""), mp4=(Path(p).with_suffix(".mp4")).exists()))
ttrs = pd.DataFrame(ttrs)
b = np.load(ROOT / "artifacts/v4/bank.npz"); bank_words = set(b["words"].tolist())
cs = pd.read_parquet(ROOT / "artifacts/manifests/clips_signers.parquet")
emb = pd.read_parquet(ROOT / "artifacts/v4/emb_ttrs.parquet")
cv4 = pd.read_parquet(ROOT / "artifacts/manifests_v4/clips_v4.parquet").set_index("clip_id")
P(f"TTRS raw json: {len(ttrs)} clips, {ttrs.gloss.nunique()} distinct glosses | bank entries: {len(bank_words)}")

# ---- YouTube titles + captions
yt = []
for sub in ("word_level/youtube_word", "sentence_level/youtube_sentence"):
    for p in glob.glob(str(ROOT / f"raw_data/{sub}/*.json")):
        d = json.load(open(p, encoding="utf-8")); stem = Path(p).stem
        vtts = sorted(glob.glob(str(Path(p).with_suffix("")) + ".th*.vtt"))
        text = ""
        for v in vtts[:1]:
            text = " ".join(l for l in open(v, encoding="utf-8").read().splitlines() if l and "-->" not in l and not l.startswith(("WEBVTT", "Kind:", "Language:")))
            text = re.sub(r"<[^>]+>", "", text)
        yt.append(dict(stem=stem, sub=sub, title=str(d.get("title") or ""), caption=text, has_vtt=bool(vtts)))
yt = pd.DataFrame(yt)
P(f"YouTube raw: {len(yt)} videos ({yt.has_vtt.sum()} with Thai captions)")
sw = pd.read_parquet(ROOT / "artifacts/manifests_v4/speech_words.parquet")

report = {}
for target, forms in TARGETS.items():
    P(f"\n=== {target}  (forms checked: {', '.join(forms)})")
    r = {}
    for f in forms:
        exact = ttrs[ttrs.gloss == f]
        syn = ttrs[ttrs.syn.apply(lambda s: f in [x.strip() for x in re.split(r"[,;/]", s)])]
        contains = ttrs[ttrs.gloss.str.contains(f, regex=False) & (ttrs.gloss != f)]
        P(f"  [{f}] TTRS gloss exact: {len(exact)} {exact.file.tolist()[:5]}  | listed as synonym of: {syn.gloss.tolist()[:8]}")
        P(f"        TTRS glosses containing '{f}' ({len(contains)}): {contains.gloss.tolist()[:15]}")
        P(f"        in v4 bank as entry: {f in bank_words}")
        if len(exact) and f not in bank_words:
            for fl in exact.file:
                cid = cs[cs.media_path.str.contains(fl, regex=False)].clip_id.tolist()
                why = [(c, cv4.loc[c, "usable_for"] if c in cv4.index else None, cv4.loc[c, "exclude_reason"] if c in cv4.index else None, c in set(emb.clip_id)) for c in cid]
                P(f"        !! raw exists but not in bank: {fl} → manifest {why}")
        tt = yt[yt.title.str.contains(f, regex=False)]
        cap = yt[yt.caption.str.contains(f, regex=False)]
        tok_hits = sum(word_tokenize(c, engine="newmm").count(f) for c in yt.caption)
        P(f"        YouTube titles containing: {len(tt)} {tt.title.tolist()[:4]}")
        P(f"        YouTube Thai captions containing: {len(cap)} videos, {tok_hits} tokenised occurrences | v4 speech_words hits: {int((sw.word == f).sum())}")
        r[f] = dict(ttrs_exact=len(exact), ttrs_synonym_of=syn.gloss.tolist(), ttrs_contains=contains.gloss.tolist(), in_bank=f in bank_words,
                    youtube_titles=len(tt), caption_videos=len(cap), caption_token_occurrences=tok_hits, speech_word_hits=int((sw.word == f).sum()))
    report[target] = r
json.dump(report, open(ROOT / "artifacts/reports/test_words_in_raw_data.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
