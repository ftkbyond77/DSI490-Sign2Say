"""L5a: Thai Sign Language sentence patterns learned from TSL51 (76 templates, 252 recordings).

TSL51 sentences are recorded in TSL order, e.g. "ขนมปัง ฉัน กิน" (bread I eat), "ภาษามือ ฉัน เรียน ชอบ", "พรุ่งนี้ คุณ ข้าว กิน ไป ที่ไหน".
From them we learn, at the level of grammatical roles (not words, so the pattern transfers to the whole vocabulary):
  * role bigram model  P(role_i | role_{i-1})   with roles SUBJ OBJ PLACE TIME VERB STATE ADJ GREET NEG QUEST NUM …
  * slot templates     "SUBJ OBJ VERB", "TIME SUBJ OBJ VERB VERB QUEST", "GREET SUBJ STATE" … with counts and examples
  * few-shot pairs     TSL gloss order → natural Thai sentence (for the LLM layer)
Use: (1) the LLM prompt carries the learned patterns and examples, (2) an optional re-ranker lets an *uncertain* segment move
to its 2nd/3rd visual candidate when that makes the role sequence much more probable. The re-ranker is only enabled if it lowers
WER on the tune sentences (scripts/eval.py); accepted segments are never changed and no word is ever added.
"""
from __future__ import annotations

import math
from collections import Counter

# TSL51 README categories → roles
TSL51_ROLE = {
    "แม่": "SUBJ", "คุณ": "SUBJ", "แฟน": "SUBJ", "พี่": "SUBJ", "น้อง": "SUBJ", "ฉัน": "SUBJ", "ยาย": "SUBJ", "แมว": "SUBJ",
    "ภาษามือ": "OBJ", "ขนมปัง": "OBJ", "ข้าว": "OBJ", "ไข่": "OBJ",
    "ถาม": "VERB", "เรียก": "VERB", "ชอบ": "VERB", "เที่ยว": "VERB", "ไป": "VERB", "กิน": "VERB", "เรียน": "VERB", "ชื่อ": "VERB",
    "เหงา": "STATE", "แต่งงาน": "STATE", "ร้อน": "STATE", "โสด": "STATE", "โกรธ": "STATE", "กลัว": "STATE", "เหนื่อย": "STATE", "สบายดี": "STATE",
    "ทำงาน": "STATE", "เกิด": "STATE", "ง่วง": "STATE", "อยู่บ้าน": "STATE",
    "ดี": "ADJ", "ด้วยกัน": "ADJ", "หูหนวก": "ADJ",
    "สวัสดี": "GREET", "ขอโทษ": "GREET", "ขอบคุณ": "GREET",
    "วันนี้": "TIME", "เช้า": "TIME", "วันหยุด": "TIME", "พรุ่งนี้": "TIME",
    "อย่า": "NEG", "ทำไม": "QUEST", "อะไร": "QUEST", "ที่ไหน": "QUEST",
    "กรุงเทพ": "PLACE", "โรงเรียน": "PLACE", "บ้าน": "PLACE", "ตลาด": "PLACE",
}
ONE_CAT_ROLE = {"01": "NUM", "04": "PLACE", "06": "LETTER", "07": "LETTER", "08": "SUBJ", "13": "SUBJ", "25": "OBJ", "28": "OBJ"}
POS_ROLE = [("สรรพนาม", "SUBJ"), ("กริยา", "VERB"), ("วิเศษณ์", "ADJ"), ("ลักษณนาม", "OBJ"), ("นาม", "OBJ"), ("อุทาน", "GREET"), ("บุพบท", "ADJ")]
THSL_TAG_ROLE = [("กริยา", "VERB"), ("ความรู้สึก", "STATE"), ("อารมณ์", "STATE"), ("การทักทาย", "GREET"), ("คำถาม", "QUEST"),
                 ("จังหวัด", "PLACE"), ("อำเภอ", "PLACE"), ("อําเภอ", "PLACE"), ("ตำบล", "PLACE"), ("ประเทศ", "PLACE"), ("สถานที่", "PLACE"),
                 ("เวลา", "TIME"), ("วัน", "TIME"), ("เดือน", "TIME"), ("ฤดู", "TIME"), ("ตัวเลข", "NUM"), ("จำนวน", "NUM"),
                 ("ครอบครัว", "SUBJ"), ("บุคคล", "SUBJ"), ("อาชีพ", "SUBJ"), ("สรรพนาม", "SUBJ"), ("นาม", "OBJ"), ("สัตว์", "OBJ"),
                 ("ผลไม้", "OBJ"), ("อาหาร", "OBJ"), ("ขนม", "OBJ"), ("ของใช้", "OBJ"), ("ยานพาหนะ", "OBJ")]
QUESTION_WORDS = {"ไหม", "อะไร", "ที่ไหน", "ทำไม", "ใคร", "เมื่อไหร่", "อย่างไร", "เท่าไร"}
NEG_WORDS = {"ไม่", "อย่า", "ไม่มี", "ไม่ใช่"}


def build_role_table(lex, ttrs_meta=None, thsl_tags=None):
    """concept → role, from TSL51 categories, TSL-ONE-S categories, TTRS part of speech, th_sl category tags (in that priority)."""
    role = {}
    for r in lex.itertuples():
        c = r.concept
        if c in role or not isinstance(c, str):
            continue
        if r.lemma in TSL51_ROLE:
            role[c] = TSL51_ROLE[r.lemma]
        elif c in QUESTION_WORDS:
            role[c] = "QUEST"
        elif c in NEG_WORDS:
            role[c] = "NEG"
        elif r.dataset == "tslone" and isinstance(r.category, str):
            role[c] = ONE_CAT_ROLE.get(r.category, "OBJ")
    if ttrs_meta is not None:
        for w, pos in ttrs_meta:
            if not isinstance(pos, str):
                continue
            for key, rl in POS_ROLE:
                if key in pos:
                    role.setdefault(w, rl); break
    for w, tag in (thsl_tags or []):
        if isinstance(tag, str):
            for key, rl in THSL_TAG_ROLE:
                if key in tag:
                    role.setdefault(w, rl); break
    return role


def learn_patterns(sentences, role):
    """sentences: list of concept lists (TSL order) → dict(bigram, templates, n)."""
    big, uni, tmpl = Counter(), Counter(), Counter()
    examples = {}
    for s in sentences:
        rs = ["<s>"] + [role.get(w, "OBJ") for w in s] + ["</s>"]
        for a, b in zip(rs[:-1], rs[1:]):
            big[(a, b)] += 1; uni[a] += 1
        t = " ".join(rs[1:-1]); tmpl[t] += 1
        examples.setdefault(t, " ".join(s))
    roles = sorted({r for pair in big for r in pair} | set(role.values()) | {"</s>"})
    return dict(bigram={f"{a}>{b}": c for (a, b), c in big.items()}, unigram=dict(uni), roles=roles,
                templates=[dict(template=t, count=c, example=examples[t]) for t, c in tmpl.most_common()], n=len(sentences))


def role_logprob(seq_roles, pat, k=0.5):
    V = len(pat["roles"])
    lp = 0.0
    rs = ["<s>"] + seq_roles + ["</s>"]
    for a, b in zip(rs[:-1], rs[1:]):
        lp += math.log((pat["bigram"].get(f"{a}>{b}", 0) + k) / (pat["unigram"].get(a, 0) + k * V))
    return lp


def rerank(segs, role, pat, weight=0.5, topn=3, min_gap=0.03):
    """Greedy left-to-right: an uncertain/unknown segment may switch to a candidate within `min_gap` visual score of its top-1
    when the role sequence log-probability gain × weight exceeds the visual loss. Returns new nearest list."""
    chosen = [s["nearest"] for s in segs]
    for i, s in enumerate(segs):
        if s.get("status") == "accepted":
            continue
        base_roles = [role.get(w, "OBJ") for w in chosen]
        best_w, best_v = chosen[i], role_logprob(base_roles, pat) * weight
        top = s["candidates"][0]["score"]
        for c in s["candidates"][1:topn]:
            if top - c["score"] > min_gap:
                continue
            rs = base_roles.copy(); rs[i] = role.get(c["word"], "OBJ")
            v = role_logprob(rs, pat) * weight - (top - c["score"]) * 20
            if v > best_v + 1e-6:
                best_w, best_v = c["word"], v
        chosen[i] = best_w
    return chosen


DAILY_ROLE = dict(pronoun_people="SUBJ", family="SUBJ", greeting_social="GREET", question="QUEST", negation_modal="VERB", time="TIME",
                  verb="VERB", feeling_state="STATE", adjective="ADJ", food_drink="OBJ", place_thing="PLACE", number="NUM",
                  animal_nature="OBJ", misc="OBJ")
DAILY_FIX = {"ใคร": "QUEST", "สบายดี": "STATE", "ชื่อ": "VERB", "อายุ": "OBJ", "รู้จัก": "VERB", "ไม่": "NEG", "แล้ว": "ADV", "ยัง": "ADV",
             "ด้วยกัน": "ADV", "คนเดียว": "ADV", "อีก": "ADV"}


def language_roles(role, daily):
    """Roles for the sentence composer: the dictionary-derived table (also used by the grammar re-ranker, unchanged) with the
    daily-conversation words set from their category in vocab/daily_conversation.json (the dictionary table had e.g. น้ำ = TIME)."""
    out = dict(role)
    for cat, words in daily.items():
        if cat in DAILY_ROLE and isinstance(words, list):
            for w in words:
                out[w] = DAILY_FIX.get(w, DAILY_ROLE[cat])
    return out
