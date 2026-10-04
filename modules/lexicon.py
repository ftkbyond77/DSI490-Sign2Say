"""L0b: unified gloss → Thai lemma → concept table for TTRS + TSL51 + TSL-ONE-S (and the words of data_test).

Three problems, three mechanisms:
  1. spelling / annotation noise inside a source  (TTRS "ตลาด (ภาษามือวิทยาลัยราชสุดา)", "ยาสระผม (2)", TSL51 "ฉัน_var_2",
     "ขอบคุณ_เปิดมือสองข้าง")  → deterministic `clean_lemma` / variant split; the removed part is kept as `sign_variant`.
  2. English labels (TSL-ONE-S: "Gor Gai", "Seven", "Safari World", "Uncle (Mother's younger brother or sister)")
     → closed-set tables for letters / vowels / numbers (no model involved), an LLM only for the rest. The LLM sees the
       English label, the category and the 20 nearest TTRS lemmas, and must answer with a Thai lemma plus the TTRS lemma
       that means *the same thing* (or none). It never sees poses and never decides what a sign looks like.
  3. synonyms across sources (กิน / ทาน / รับประทาน, ประเทศเวียดนาม / เวียดนาม, บาสเกตบอล / บาสเก็ตบอล)
     → `concept` groups: text-embedding nearest neighbours propose pairs, an LLM keeps only pairs that are interchangeable
       in translation (not broader/narrower, not merely related), union-find joins them.
The bank scores concepts (a sign of any member counts), outputs show the canonical lemma, evaluation matches concepts.
Every automatic decision carries `source` (rule | llm | embedding+llm) and `confidence`, and is exported to vocab/v5 for review.
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np
import pandas as pd

from .utils import ART, ROOT, get_logger, read_json, write_json

log = get_logger("lexicon")
NULL = "<null>"

THAI_CONSONANT_NAMES = {
    "gor gai": "ก", "kor kai": "ข", "kor kwaii": "ค", "kor ra-kung": "ฆ", "tor tao": "ต", "thor thoong": "ถ", "thor tharn": "ฐ",
    "thor phoo thao": "ฒ", "thor mon-tho": "ฑ", "tor pa-tuk": "ฏ", "sor sua": "ส", "sor saalaa": "ศ", "sor rue si": "ษ", "sor so": "ซ",
    "phor phaan": "พ", "por plaa": "ป", "phor pheung": "ผ", "phor samphao": "ภ", "hor heep": "ห", "hor nok hoog": "ฮ", "bor bai mai": "บ",
    "ror rua": "ร", "wor whaen": "ว", "dor dek": "ด", "dor cha-daa": "ฎ", "for fun": "ฟ", "fhor fhaa": "ฝ", "lor ling": "ล", "lor chulaa": "ฬ",
    "jor jaan": "จ", "yor yuk": "ย", "yor ying": "ญ", "mor maa": "ม", "nor nhoo": "น", "nor nayn": "ณ", "ngor ngoo": "ง",
    "tor thaharn": "ท", "thor thohng": "ธ", "chor ching": "ฉ", "chor chaang": "ช", "chor cher": "ฌ", "or aang": "อ"}
# fingerspelled vowels; ai/aii order follows the Thai vowel table (ใ ไม้ม้วน precedes ไ ไม้มลาย) — flagged for review
THAI_VOWEL_NAMES = {"i": ("สระอิ", 0.9), "ee": ("สระอี", 0.9), "u": ("สระอุ", 0.9), "uu": ("สระอู", 0.9), "o": ("สระโอ", 0.9),
                    "ai": ("สระไอ", 0.5), "aii": ("สระใอ", 0.5)}
NUMBERS = {"zero": "ศูนย์", "one": "หนึ่ง", "two": "สอง", "three": "สาม", "four": "สี่", "five": "ห้า", "six": "หก", "seven": "เจ็ด",
           "eight": "แปด", "nine": "เก้า", "ten": "สิบ"}
TTRS_FIXES = {"ปรเศรษฐเสถียรในพระราชูปถัมภ์)ะโยคปฎิเสธ (ภาษามือโรงเรียน": ("ประโยคปฏิเสธ", "ภาษามือโรงเรียนเศรษฐเสถียรในพระราชูปถัมภ์")}
PAREN = re.compile(r"\s*\(([^)]*)\)\s*\d*\s*$")

# reference glosses of data_test (from file names + the on-screen word cards of test1.mp4; used ONLY for scoring)
DATA_TEST_REFS = {
    "test1": ["สวัสดี", "สบายดี", "ชอบ", "โกรธ", "แมว", "แฟน", "ปรบมือ"],
    "ฉันปลอบเพื่อนร้องไห้": ["ฉัน", "ปลอบใจ", "เพื่อน", "ร้องไห้"],
    "ฉันรักเพื่อน": ["ฉัน", "รัก", "เพื่อน"],
    "พ่อดื่มน้ำ": ["พ่อ", "ดื่ม", "น้ำ"],
    "ไปทานข้าวด้วยกันมั้ย": ["ไป", "กิน", "ข้าว", "ด้วยกัน", "ไหม"],
}
TEST_SURFACE = {"ทาน": "กิน", "รับประทาน": "กิน", "มั้ย": "ไหม", "ปลอบ": "ปลอบใจ", "เป็นแฟน": "แฟน"}

# manual review of the automatic table (vocab/v5/gloss_mapping.csv) — corrections a reader of Thai kinship / place names makes
MANUAL_LEMMA = {  # TSL-ONE-S English gloss → Thai lemma
    "Country": ("ประเทศ", "generic 'country' (LLM had ประเทศไทย)"),
    "Children": ("ลูก", "family category → offspring, not เด็ก"),
    "Football": ("ฟุตบอล", "LLM matched ลูกบอล (ball)"),
    "Victory Monument": ("อนุสาวรีย์ชัยสมรภูมิ", "different place from วงเวียนใหญ่"),
    "Grandfather": ("ปู่", "paternal/maternal pairing not documented by the dataset; see sign-evidence note (v5 encoder)"),
    "Grandmother": ("ย่า", "pairing checked against TTRS signs (see sign-evidence note)"),
    "Grandpa": ("ตา", "pairing checked against TTRS signs (see sign-evidence note)"),
    "Grandma": ("ยาย", "pairing checked against TTRS signs (see sign-evidence note)"),
    "Great-grandmother": ("ย่าทวด-ยายทวด", "TTRS lemma for great-grandmother"),
    "Great-grandfather": ("ปู่ทวด-ตาทวด", "TTRS lemma for great-grandfather"),
}
MANUAL_NOT_SAME = {frozenset(x) for x in [("ยาย", "ย่า"), ("ลูก", "เด็ก"), ("ญาติพี่น้อง", "ลูกพี่ลูกน้อง"), ("วงเวียนใหญ่", "อนุสาวรีย์ชัยสมรภูมิ"),
                                         ("กางเกง", "กางเกงชั้นใน"), ("กางเกง", "กางเกงใน"), ("กางเกง", "กางเกงในผู้ชาย"), ("มะม่วง", "มะม่วงสุก"),
                                         ("เนื้อวัว", "เนื้อ"), ("ปู่", "ตา"), ("หมู", "เนื้อหมู")]}


def apply_manual_review(lex: pd.DataFrame) -> pd.DataFrame:
    lex = lex.copy()
    for eng, (lem, why) in MANUAL_LEMMA.items():
        m = (lex.dataset == "tslone") & (lex.gloss == eng)
        if m.any():
            lex.loc[m, "lemma"] = lem
            lex.loc[m, "lemma_source"] = "manual_review"
            lex.loc[m, "note"] = why
            lex.loc[m, "confidence"] = 0.6 if "unverified" in why else 1.0
    return lex


# ----------------------------------------------------------------------------- deterministic cleaning
def clean_lemma(word: str):
    """→ (lemma, variant_note). Strips school/source annotations and duplicate counters, keeps content parentheses inside."""
    w = str(word).strip()
    if w in TTRS_FIXES:
        return TTRS_FIXES[w]
    note = None
    m = PAREN.search(w)
    if m and (m.group(1).startswith("ภาษามือ") or m.group(1).isdigit()):
        note = m.group(1); w = w[:m.start()].strip()
    w = re.sub(r"(?<=[฀-๿])\d+$", "", w).strip()
    w = re.sub(r"\s+", " ", w).replace("่่", "่")
    return w, note


def tsl51_lemma(sign_id: str):
    s = str(sign_id)
    if s == "null_act":
        return NULL, None
    if "_var_" in s:
        a, b = s.split("_var_", 1)
        return a, f"var_{b}"
    a, _, b = s.partition("_")
    return a, (b or None)


_VARIANT = re.compile(r"\s*\(?\s*(ท่ามือที่|ท่าที่|แบบที่)\s*(\d+)[^)]*\)?\s*")
_SENSE = re.compile(r"\s*-?\s*\(([^()]*)\)\s*$")
_DIGITS = {0: "ศูนย์", 1: "หนึ่ง", 2: "สอง", 3: "สาม", 4: "สี่", 5: "ห้า", 6: "หก", 7: "เจ็ด", 8: "แปด", 9: "เก้า"}


def thai_number(n: int) -> str:
    """Integer → Thai number word (สิบ, ยี่สิบ, สิบเอ็ด, ร้อย, …) up to 999,999."""
    if n < 10:
        return _DIGITS[n]
    out = ""
    for val, name in ((100000, "แสน"), (10000, "หมื่น"), (1000, "พัน"), (100, "ร้อย")):
        q, n = divmod(n, val)
        if q:
            out += ("" if (q == 1 and val == 100 and not out) else _DIGITS[q]) + name if q > 1 or val != 100 or out else "ร้อย"
    q, r = divmod(n, 10)
    if q:
        out += ("ยี่" if q == 2 else "" if q == 1 else _DIGITS[q]) + "สิบ"
    if r:
        out += "เอ็ด" if (r == 1 and (q or out)) else _DIGITS[r]
    return out


def thsl_lemma(word: str):
    """th-sl.com entry → (lemma, sign_variant, sense_note, aliases).
    'ขอโทษ(ท่าที่ 2)' → ขอโทษ, 'ท่าที่2';  'หยิก (แขน)' → หยิก, sense 'แขน';  'แต่งงาน/สมรส' → แต่งงาน, aliases [สมรส];  '20' → ยี่สิบ."""
    w = re.sub(r"\s+", " ", str(word)).strip()
    var = None
    m = _VARIANT.search(w)
    if m:
        var = f"ท่าที่{m.group(2)}"
        w = (w[:m.start()] + " " + w[m.end():]).strip()
    sense = None
    if w.count("(") > w.count(")"):              # unclosed note: 'แย่ง (ใช้กับคน'
        w = w + ")"
    m = _SENSE.search(w)
    if m and m.start() > 0:
        sense = m.group(1).strip(); w = w[:m.start()].strip()
    w = w.strip(" -")
    parts = [x.strip() for x in re.split(r"[/,]", w) if x.strip()]
    lemma, aliases = (parts[0], parts[1:]) if parts else (w, [])
    if lemma.isdigit():
        lemma = thai_number(int(lemma))
    return lemma, var, sense, aliases


# ----------------------------------------------------------------------------- LLM helpers
def _client():
    from openai import OpenAI
    from .env import load_env
    load_env()
    return OpenAI(timeout=180)


def embed_texts(texts, model="text-embedding-3-small", bs=500):
    cli = _client()
    out = []
    for i in range(0, len(texts), bs):
        r = cli.embeddings.create(model=model, input=texts[i:i + bs])
        out += [d.embedding for d in r.data]
    X = np.array(out, np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def chat_json(system, user, model="gpt-4.1", retries=3):
    cli = _client()
    for k in range(retries):
        try:
            r = cli.chat.completions.create(model=model, temperature=0, response_format={"type": "json_object"},
                                            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return json.loads(r.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            log.warning("LLM retry %d: %s", k, e); time.sleep(3 * (k + 1))
    raise RuntimeError("LLM failed")


TSLONE_SYS = """คุณเป็นผู้เชี่ยวชาญภาษามือไทยและพจนานุกรมไทย งานคือแปลงชื่อ gloss ภาษาอังกฤษของชุดข้อมูลภาษามือไทย TSL-ONE-S เป็นคำไทยมาตรฐาน
สำหรับแต่ละรายการ ตอบ:
- thai: คำไทยที่คนไทยใช้เรียกสิ่งนั้นจริง (ชื่อสถานที่ใช้ชื่อไทยทางการที่สั้นที่นิยม เช่น "Safari World" → "ซาฟารีเวิลด์", "Grandpa" ใช้ความหมายไทยที่ถูกต้องตามบริบทครอบครัว)
- aliases: คำไทยอื่นที่หมายถึงสิ่งเดียวกันทุกประการ (ถ้ามี) เช่นชื่อเต็ม/ชื่อย่อ/การสะกดอื่น
- ttrs_same: คำจากรายการ candidates ที่หมายถึง "สิ่งเดียวกันทุกประการ" (ใช้แทนกันได้ในการแปล) ถ้าไม่มีให้ null — ห้ามเลือกคำที่กว้างกว่า/แคบกว่า/แค่เกี่ยวข้อง
- confidence: 0-1
- note: ข้อสังเกตสั้นๆ (เช่น ความกำกวม ปู่/ตา)
ข้อมูลช่วย: Grandfather/Grandmother ในชุดนี้แยกกับ Grandpa/Grandma — ให้ใช้ ปู่/ย่า (ฝั่งพ่อ) กับ ตา/ยาย (ฝั่งแม่) ตามลำดับที่สมเหตุสมผลและระบุใน note;
Uncle/Aunt ที่มีคำอธิบายในวงเล็บให้แปลตามคำอธิบาย (ลุง ป้า น้า อา)
ตอบ JSON: {"items": [{"id": ..., "thai": ..., "aliases": [...], "ttrs_same": ... , "confidence": ..., "note": ...}]}"""

SYN_SYS = """คุณเป็นนักภาษาศาสตร์ไทยที่ช่วยสร้างคลังศัพท์สำหรับระบบแปลภาษามือไทย
จะได้รับคู่คำไทย (a, b) ให้ตัดสินว่าใช้แทนกันได้ในการแปลประโยคทั่วไปหรือไม่:
- "same": ความหมายเดียวกันทุกประการ ต่างกันแค่การสะกด/ระดับภาษา/คำนำหน้าที่ไม่เปลี่ยนความหมาย (กิน=รับประทาน=ทาน, ประเทศเวียดนาม=เวียดนาม, บาสเกตบอล=บาสเก็ตบอล)
- "different": อย่างอื่นทั้งหมด รวมถึง กว้างกว่า/แคบกว่า (ผลไม้ vs มะม่วง, พ่อ vs พ่อเลี้ยง), เกี่ยวข้องแต่ไม่ใช่สิ่งเดียวกัน (ครู vs นักเรียน), ตัวเลขต่างกัน
ระวังมาก: ถ้าไม่แน่ใจให้ตอบ different
ตอบ JSON: {"pairs": [{"i": เลขคู่, "label": "same"|"different"}]}"""


# ----------------------------------------------------------------------------- build
def build_lexicon(use_llm=True):
    """→ DataFrame one row per (dataset, gloss): dataset, gloss, lemma, sign_variant, lemma_source, confidence, note, english, category."""
    rows = []
    e = pd.read_parquet(ART / "v4" / "emb_ttrs.parquet")
    for w, v in e.groupby("word").sign_variant_id.first().items():
        lem, note = clean_lemma(w)
        rows.append(dict(dataset="ttrs", gloss=w, lemma=lem, sign_variant=note, lemma_source="gloss" if note is None else "rule", confidence=1.0))
    ex = pd.read_csv(ROOT / "raw_data/TSL51/metadata/expert_metadata.csv")
    us = pd.read_csv(ROOT / "raw_data/TSL51/metadata/user_sign_metadata.csv")
    for sid in sorted(set(ex.sign_id) | set(us.sign_id)):
        lem, var = tsl51_lemma(sid)
        rows.append(dict(dataset="tsl51", gloss=sid, lemma=lem, sign_variant=var, lemma_source="rule", confidence=1.0))
    ds = read_json(ROOT / "raw_data/TSL-ONE-S/tsl_one_s_dataset.json")
    cats = {}
    for g in ds:
        cats[g["gloss"]] = g["instances"][0]["video_id"][3:5]
    ttrs_lemmas = sorted({r["lemma"] for r in rows if r["dataset"] == "ttrs"})
    pending = []
    for eng, cat in cats.items():
        k = eng.strip().lower()
        if k in THAI_CONSONANT_NAMES:
            rows.append(dict(dataset="tslone", gloss=eng, english=eng, category=cat, lemma=THAI_CONSONANT_NAMES[k], lemma_source="rule:consonant", confidence=1.0))
        elif k in THAI_VOWEL_NAMES:
            th, cf = THAI_VOWEL_NAMES[k]
            rows.append(dict(dataset="tslone", gloss=eng, english=eng, category=cat, lemma=th, lemma_source="rule:vowel", confidence=cf,
                             note="ai/aii (ไ/ใ) order unverified" if cf < 1 else None))
        elif k in NUMBERS:
            rows.append(dict(dataset="tslone", gloss=eng, english=eng, category=cat, lemma=NUMBERS[k], lemma_source="rule:number", confidence=1.0))
        else:
            pending.append((eng, cat))
    if use_llm and pending:
        E_t = embed_texts(ttrs_lemmas)
        E_q = embed_texts([f"{eng} (Thai sign language gloss; category {cat})" for eng, cat in pending])
        S = E_q @ E_t.T
        items = []
        for i, (eng, cat) in enumerate(pending):
            cand = [ttrs_lemmas[j] for j in np.argsort(-S[i])[:20]]
            items.append(dict(id=i, english=eng, category=cat, candidates=cand))
        out = {}
        for b in range(0, len(items), 20):
            r = chat_json(TSLONE_SYS, json.dumps(dict(items=items[b:b + 20]), ensure_ascii=False))
            for it in r.get("items", []):
                out[int(it["id"])] = it
        for i, (eng, cat) in enumerate(pending):
            it = out.get(i, {})
            same = it.get("ttrs_same")
            same = same if same in ttrs_lemmas else None
            lem = same or (it.get("thai") or eng)
            rows.append(dict(dataset="tslone", gloss=eng, english=eng, category=cat, lemma=lem, lemma_source="llm+ttrs" if same else "llm",
                             confidence=float(it.get("confidence") or 0.5), note=it.get("note"), llm_thai=it.get("thai"),
                             aliases="|".join(it.get("aliases") or []), candidates="|".join(items[i]["candidates"][:8])))
    df = pd.DataFrame(rows)
    return df


def build_concepts(lex: pd.DataFrame, extra_words=(), use_llm=True, k=6, min_sim=0.72, prior=None, new_lemmas=None, same_pairs=()):
    """Union-find over lemmas with verified 'same' pairs → (lemma → concept, decisions, groups).
    prior      earlier decisions (DataFrame a, b, label, source) — kept, not re-asked
    new_lemmas only pairs touching these lemmas are proposed to the LLM (None = all lemmas)
    same_pairs pairs that are the same sign by construction (e.g. a dictionary entry 'แต่งงาน/สมรส')"""
    lemmas = sorted(set(lex.lemma) - {NULL} | set(extra_words))
    parent = {w: w for w in lemmas}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    decisions = [] if prior is None else prior.to_dict("records")
    seen = {frozenset((d["a"], d["b"])) for d in decisions}
    alias_pairs = set()
    # TSL-ONE-S aliases proposed by the mapping LLM are only candidates: they go through the same pair verification
    if "aliases" in lex:
        for r in lex[lex.dataset == "tslone"].itertuples():
            if isinstance(r.aliases, str):
                for al in r.aliases.split("|"):
                    if al in parent and al != r.lemma:
                        alias_pairs.add(tuple(sorted((r.lemma, al))))
    for s_, c in TEST_SURFACE.items():
        if s_ in parent and c in parent and frozenset((c, s_)) not in seen:
            decisions.append(dict(a=c, b=s_, label="same", source="rule-test-surface"))
    for a_, b_, src in same_pairs:
        if a_ in parent and b_ in parent and a_ != b_:
            decisions.append(dict(a=a_, b=b_, label="same", source=src)); seen.add(frozenset((a_, b_)))
    if use_llm:
        E = embed_texts(lemmas)
        pos = {w: i for i, w in enumerate(lemmas)}
        query = [pos[w] for w in (new_lemmas if new_lemmas is not None else lemmas) if w in pos]
        pairs = set()
        for i in range(0, len(query), 2000):
            q = query[i:i + 2000]
            S = E[q] @ E.T
            for r_, row in enumerate(S):
                a = q[r_]
                for j in np.argsort(-row)[1:k + 1]:
                    if row[j] >= min_sim and a != j:
                        pairs.add(tuple(sorted((lemmas[a], lemmas[j]))))
        pairs = sorted(p for p in (pairs | alias_pairs) if frozenset(p) not in seen)
        log.info("synonym candidate pairs to verify: %d (incl. mapping aliases)", len(pairs))
        for b in range(0, len(pairs), 120):
            chunk = pairs[b:b + 120]
            r = chat_json(SYN_SYS, json.dumps(dict(pairs=[dict(i=i, a=x, b=y) for i, (x, y) in enumerate(chunk)]), ensure_ascii=False))
            lab = {int(p["i"]): p.get("label") for p in r.get("pairs", [])}
            for i, (x, y) in enumerate(chunk):
                decisions.append(dict(a=x, b=y, label=lab.get(i, "different"), source="embedding+llm"))
    for d in decisions:
        if d["label"] == "same" and frozenset((d["a"], d["b"])) in MANUAL_NOT_SAME:
            d["label"] = "different"; d["source"] = str(d["source"]) + "+manual_review"
    for d in decisions:
        if d["label"] == "same" and d["a"] in parent and d["b"] in parent:
            union(d["a"], d["b"])
    groups = {}
    for w in lemmas:
        groups.setdefault(find(w), []).append(w)
    # canonical = member used by TSL51, else the most frequent / shortest form, else alphabetical
    t51 = set(lex[lex.dataset == "tsl51"].lemma)
    counts = lex.groupby("lemma").size().to_dict()
    concept = {}
    for members in groups.values():
        canon = sorted(members, key=lambda w: (w not in t51, -counts.get(w, 0), len(w), w))[0]
        for w in members:
            concept[w] = canon
    concept[NULL] = NULL
    return concept, pd.DataFrame(decisions), {c: sorted(m) for c, m in ((concept[m[0]], m) for m in groups.values()) if len(m) > 1}


# ----------------------------------------------------------------------------- role sources for the grammar layer
def ttrs_pos(lex):
    """[(concept, part-of-speech)] for TTRS glosses (harvest catalogue if present, else the .json sidecars from S3)."""
    from .utils import RAW
    concept_of = dict(zip(lex[lex.dataset == "ttrs"].gloss, lex[lex.dataset == "ttrs"].concept))
    mw = ROOT / "metadata" / "metadata_word.csv"
    if mw.exists():
        m = pd.read_csv(mw)
        m = m[m.uid.str.startswith("ttrs")]
        return [(concept_of.get(w, w), p) for w, p in zip(m.label, m.parts_of_speech_th)]
    out = []
    for j in (RAW / "word_level" / "ttrs_dictionary").glob("ttrs_*.json"):
        d = json.loads(j.read_text(encoding="utf-8"))
        out.append((concept_of.get(d.get("gloss_th"), d.get("gloss_th")), d.get("parts_of_speech_th")))
    return out


def thsl_tags(lex):
    """[(concept, th-sl.com category tag)]."""
    from .utils import RAW
    m = pd.read_csv(RAW / "word_level" / "th_sl_dictionary" / "metadata" / "th_sl_metadata.csv")
    concept_of = dict(zip(lex[lex.dataset == "th_sl"].gloss, lex[lex.dataset == "th_sl"].concept))
    return [(concept_of.get(w, w), t) for w, t in zip(m.word, m.tags)]
