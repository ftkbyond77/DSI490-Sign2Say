"""Text side of the dictionary: every way a signable concept can be written → that concept.

Built offline (`python main.py lexicon`) into models/lexicon.json from the data lake's own labels — never from an LLM's
imagination: manifest lexicon (source gloss → lemma → concept), the reviewed synonym groups (concepts.json), TTRS `synonyms_th`
and `parts_of_speech_th`, English glosses (TSL-ONE-S). Only concepts that exist in the motion library are signable.

Each surface entry keeps HOW it maps (lemma · gloss · alias · group · ttrs_synonym · english) so the IR can show the user why a
word became a sign, and the translation confidence can differ between an exact lemma and a synonym.
"""
from __future__ import annotations

import glob
import json
import re
from functools import lru_cache
from pathlib import Path

from .. import paths

LEX_PATH = paths.MODELS / "lexicon.json"
PAREN = re.compile(r"\s*[\(（][^\)）]*[\)）]\s*")
VARIANT = re.compile(r"\(?\s*ท่า(?:มือ)?ที่\s*\d+\s*\)?")
MATCH_RANK = {"lemma": 0, "gloss": 1, "alias": 2, "group": 3, "ttrs_synonym": 4, "english": 5}

POS_FROM_CATEGORY = [                      # TTRS / th-sl category or part-of-speech text → coarse role
    (re.compile(r"กริยา|การกระทำ|กิริยา"), "verb"),
    (re.compile(r"วิเศษณ์|ความรู้สึก|ลักษณะ|บุคลิก|สี$|รสชาติ|อารมณ์"), "adj"),
    (re.compile(r"ตัวเลข|จำนวน"), "num"),
    (re.compile(r"ปฏิทิน|เวลา|วันสำคัญ|ฤดู|เดือน"), "time"),
    (re.compile(r"สถานที่|จังหวัด|อำเภอ|ประเทศ|ทวีป|ที่ว่าการ|in the city"), "place"),
    (re.compile(r"นาม|สัตว์|ผัก|ผลไม้|อาหาร|อาชีพ|ร่างกาย|ของใช้|เสื้อผ้า|ยานพาหนะ|เครื่อง|ห้อง|ครอบครัว|บุคคล|พรรค|กีฬา|พืช"), "noun"),
]


PHRASE_DROP = re.compile(r"ครับ|ค่ะ|คะ|นะ|จ้ะ|จ๊ะ|ฮะ|[\s\?\!\.,\"'“”]")
PHRASE_VAR = {"ไหร่": "ไร", "แท๊กซี่": "แท็กซี่", "มั้ย": "ไหม", "ยังไง": "อย่างไร"}


def phrase_key(text):
    """normalised form for matching a whole clause to a recorded phrase (politeness particles, spaces, punctuation, spelling)."""
    t = PHRASE_DROP.sub("", text)
    for a, b in PHRASE_VAR.items():
        t = t.replace(a, b)
    return t


def clean_surface(s):
    if not isinstance(s, str):
        return []
    s = VARIANT.sub("", s)
    s = PAREN.sub(" ", s).strip()
    out = []
    for part in re.split(r"[/,;、]|\s+หรือ\s+", s):
        part = re.sub(r"\s+", " ", part).strip(" .-")
        if part and len(part) <= 40:
            out.append(part)
    return out


def coarse_pos(*texts):
    for t in texts:
        if not isinstance(t, str) or not t:
            continue
        for rx, pos in POS_FROM_CATEGORY:
            if rx.search(t):
                return pos
    return None


def build(library_index=None):
    lib = json.loads((paths.MODELS / "motion" / "index.json").read_text(encoding="utf-8"))["clips"] if library_index is None \
        else library_index
    import pandas as pd
    signable = set(lib)
    lx = pd.read_parquet(paths.PREP / "manifest" / "lexicon.parquet")
    groups = json.loads((paths.PREP / "manifest" / "concepts.json").read_text(encoding="utf-8"))["groups"]
    ttrs = {}
    for f in glob.glob(str(paths.RAW / "word_level" / "ttrs_dictionary" / "*.json")):
        j = json.loads(Path(f).read_text(encoding="utf-8"))
        ttrs.setdefault(j.get("gloss_th", ""), j)
    entries: dict[str, list] = {}

    def add(surface, concept, how, sense=None):
        surface = surface.strip()
        if not surface or concept not in signable:
            return
        key = surface.lower()
        lst = entries.setdefault(key, [])
        if any(e["concept"] == concept for e in lst):
            for e in lst:
                if e["concept"] == concept and MATCH_RANK[how] < MATCH_RANK[e["how"]]:
                    e["how"] = how
            return
        lst.append(dict(concept=concept, how=how, sense=sense))

    concepts = {}
    for c in signable:
        add(c, c, "lemma")
        concepts[c] = dict(pos=None, category=None, senses=[], english=[])
    for r in lx.itertuples():
        c = r.concept
        if c not in signable:
            continue
        info = concepts[c]
        add(r.lemma, c, "lemma", r.sense)
        for s in clean_surface(r.gloss):
            add(s, c, "gloss", r.sense)
        for s in clean_surface(r.aliases):
            add(s, c, "alias", r.sense)
        if isinstance(r.sense, str) and r.sense and r.sense not in info["senses"]:
            info["senses"].append(r.sense)
        if isinstance(r.english, str) and r.english:
            for s in clean_surface(r.english):
                add(s, c, "english")
                if s not in info["english"]:
                    info["english"].append(s)
        tj = ttrs.get(r.gloss)
        if tj:
            for s in clean_surface(tj.get("synonyms_th")):
                add(s, c, "ttrs_synonym")
            info["pos"] = info["pos"] or coarse_pos(tj.get("parts_of_speech_th"), tj.get("category_th"))
        info["category"] = info["category"] or (r.category if isinstance(r.category, str) else None)
        info["pos"] = info["pos"] or coarse_pos(r.category)
    for c, members in groups.items():
        for m in members:
            for s in clean_surface(m):
                add(s, c, "group")
    for lst in entries.values():
        lst.sort(key=lambda e: MATCH_RANK[e["how"]])
    # fingerspelling: th-sl.com letter entries ("ค ควาย", vowels "อะ", tone marks "อ่") and TTRS English letters
    letters = {}
    for c in signable:
        m = re.match(r"^([ก-ฮ])\.?\s+[ก-๙]+$", c)
        if m:
            letters[m.group(1)] = c
        elif re.match(r"^(อ[ะ-ฺ็-๎]|[เแโใไ]อ|อ[ั-ื]|ฤ|ๆ|ฯ)$", c):
            letters[c] = c
        elif re.match(r"^[A-Z](\s*\(.*\))?$", c):          # TTRS: "A (พจนานุกรมสารสนเทศภาษามือไทย)"
            letters[c[0]] = c
    # a YouTube-sourced compound must not hide its decomposition into dictionary signs ("กินข้าว" → ข้าว + กิน, as TSL51's
    # deaf expert signs it): its written form only maps to it when no full split into dictionary-sourced signs exists
    dict_src = {k for k, v in lib.items() if v.get("dataset") in ("ttrs", "th_sl", "tsl51_user_sign")}

    def splits(w):
        n = len(w)
        ok = [False] * (n + 1); ok[0] = True
        for i in range(n):
            if ok[i]:
                for j in range(i + 1, n + 1):
                    if w[i:j] in entries and any(e["concept"] in dict_src for e in entries[w[i:j]]) and j - i < n:
                        ok[j] = True
        return ok[n]
    for w in [w for w, lst in entries.items() if lst and all(e["concept"] not in dict_src for e in lst)]:
        if not w.startswith("phrase:") and splits(w):
            del entries[w]
    phrases = {phrase_key(c[len("phrase:"):]): c for c in signable if c.startswith("phrase:")}
    for c in list(entries):                                    # phrase recordings are reached through `phrases`, not as words
        entries[c] = [e for e in entries[c] if not e["concept"].startswith("phrase:")]
        if not entries[c]:
            del entries[c]
    for c in [c for c in concepts if c.startswith("phrase:")]:
        del concepts[c]
    out = dict(entries=entries, concepts=concepts, letters=letters, phrases=phrases)
    LEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEX_PATH.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"lexicon: {len(entries)} surfaces → {len(signable)} concepts; letters {len(letters)}; "
          f"pos known {sum(1 for v in concepts.values() if v['pos'])}")
    return out


@lru_cache(maxsize=1)
def load():
    return json.loads(LEX_PATH.read_text(encoding="utf-8"))
