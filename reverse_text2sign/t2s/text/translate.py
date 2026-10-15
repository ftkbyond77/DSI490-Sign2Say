"""Thai text → IR (TSL structure + sign units). English input is first translated to Thai (t2s.text.english).

Per clause: normalise → lexicon-aware tokenise → classify → TSL reorder (grammar.py) → map each word to a signable concept:
  exact surface (lemma / gloss / alias / reviewed synonym group / TTRS synonym)  →  prefix-stripped form (การ-, ความ-)
  → compound split into signable parts (all parts must be signable)
  → fingerspelling (names, short unknown words; only if every letter has a recorded sign)  →  text card (shown, never signed).
Nothing in this chain can produce a sign that is not a recorded dictionary sign.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from ..ir import IR, SignUnit
from . import grammar as G
from . import curated, senses
from .lexicon import load as load_lexicon, phrase_key

TRANSLATION_CONF = {"sense": senses.EVIDENCE_CONF, "lemma": 1.0, "gloss": 0.97, "alias": 0.95, "pronoun": 0.95, "group": 0.9, "ttrs_synonym": 0.85, "curated": 0.85,
                    "english": 0.85, "aspect": 0.85, "compound": 0.8, "fingerspell": 0.6}
CLAUSE_SPLIT = re.compile(r"[.!?;。！？\n]+|,(?=\s)|，")
THAI_CHAR = re.compile(r"[฀-๿]")


def normalize(text):
    from pythainlp.util import normalize as th_norm
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"[​‌‍﻿]", "", t)
    t = th_norm(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _num_words(tok):
    from pythainlp.util import num_to_thaiword
    try:
        w = num_to_thaiword(int(tok))
    except Exception:  # noqa: BLE001
        return [tok]
    from pythainlp.tokenize import word_tokenize
    return [x for x in word_tokenize(w, engine="newmm") if x.strip()]


def _known(w, mapper):
    return ((w in mapper.entries and w not in mapper.letter_keys) or w in G.CLOSED_WORDS or w in mapper.variants
            or w in mapper.fixed)


def _repair(toks, mapper):
    """newmm ties are broken badly sometimes ("บอกว่า" → บอ|กว่า). Re-split each adjacent pair at the point that makes the most
    known words (signable or closed-class); keep the original unless strictly better."""
    out = list(toks)
    i = 0
    while i < len(out) - 1:
        a, b = out[i], out[i + 1]
        sc0 = _known(a, mapper) + _known(b, mapper)
        if sc0 < 2:
            s_ = a + b
            best, bs = None, sc0
            for k in range(1, len(s_)):
                x, y = s_[:k], s_[k:]
                sc = _known(x, mapper) + _known(y, mapper)
                if sc > bs:
                    best, bs = (x, y), sc
            if best and bs == 2:                       # only a split into two known words replaces the original
                out[i], out[i + 1] = best
        i += 1
    return out


def tokenize(clause, mapper=None):
    """pythainlp newmm with its own dictionary (well-tuned segmentation), then adjacent tokens are merged when together they
    spell a signable dictionary entry (longest first, ≤ 4 tokens): โรงเรียน+เทคนิค → โรงเรียนเทคนิค, ไม่+ชอบ → ไม่ชอบ."""
    from pythainlp.tokenize import word_tokenize
    raw = []
    for w in word_tokenize(clause, engine="newmm", keep_whitespace=False):
        w = w.strip()
        if not w or re.fullmatch(r"[\"'()\[\]{}:–—\-…]+", w):
            continue
        if w.isdigit():
            raw += _num_words(w)
        elif w.endswith("ๆ") and len(w) > 1:
            raw += [w[:-1].strip()]                       # reduplication: same sign (repetition is in the sign itself)
        elif w == "ๆ":
            continue
        else:
            raw.append(w)
    if mapper is None:
        return raw
    raw = _repair(raw, mapper)
    out, i = [], 0
    while i < len(raw):
        for j in range(min(len(raw), i + 4), i + 1, -1):
            cand = "".join(raw[i:j])
            if (cand in mapper.fixed or cand in mapper.variants or cand in G.CLOSED_WORDS
                    or (cand in mapper.entries and cand not in G.PRON_SIGN and cand not in mapper.letter_keys)):
                out.append(cand); i = j
                break
        else:
            out.append(raw[i]); i += 1
    # possessive pronoun words → possessive + pronoun ("ของคุณ" → ของ คุณ: the grammar puts the owner first)
    res = []
    for w in out:
        if w.startswith("ของ") and w[3:] in G.PRON_SIGN:
            res += ["ของ", w[3:]]
        else:
            res.append(w)
    return res


def clauses_of(text):
    """split at punctuation; spaces separate clauses only when the text is not a space-separated word list (gloss-style input)."""
    parts = [p.strip() for p in CLAUSE_SPLIT.split(text) if p and p.strip()]
    out = []
    for p in parts:
        chunks = p.split(" ")
        if len(chunks) > 1 and sum(len(tokenize(c)) > 1 for c in chunks) >= 1:
            out += [c for c in chunks if c]
        else:
            out.append(p)
    return out


class Fixed(str):
    """a word that comes from a curated fixed expression (keeps its order)."""


class Mapper:
    """word → (concept, how) with the fallback chain of the module docstring."""

    def __init__(self):
        self.lex = load_lexicon()
        self.entries = self.lex["entries"]
        self.concepts = self.lex["concepts"]
        self.letters = self.lex["letters"]
        self.letter_keys = set(self.letters.values())
        self.phrases = self.lex.get("phrases", {})
        self.variants, self.fixed = curated.load(self.entries)

    def pos(self, w):
        e = self.entries.get(w.lower())
        if e:
            return self.concepts.get(e[0]["concept"], {}).get("pos")
        return None

    def exact(self, w):
        e = self.entries.get(w.lower())
        return (e[0]["concept"], e[0]["how"]) if e else None

    def split(self, w, max_parts=3):
        """w as a concatenation of signable surfaces (fewest parts, each ≥ 2 chars)."""
        n = len(w)
        best = [None] * (n + 1); best[0] = []
        for i in range(n):
            if best[i] is None:
                continue
            for j in range(i + 2, n + 1):
                piece = w[i:j]
                ok = piece in self.entries or piece in G.FREE_PARTS
                if ok and (best[j] is None or len(best[i]) + 1 < len(best[j])):
                    best[j] = best[i] + [piece]
        parts = best[n]
        if not parts or not any(p in self.entries for p in parts):
            return None
        return parts if 1 < len(parts) <= max_parts + sum(p in G.FREE_PARTS for p in parts) else None

    def map(self, w, kind=None):
        """→ list of (surface, concept, how, score); empty = unknown."""
        if kind == "pron" and w in G.PRON_SIGN and G.PRON_SIGN[w] in self.entries:
            c = self.exact(G.PRON_SIGN[w])[0]
            return [(w, c, "lemma" if c == w else "pronoun", 1.0 if c == w else 0.95)]
        if w == "แล้ว" and self.exact("เสร็จแล้ว"):
            return [(w, self.exact("เสร็จแล้ว")[0], "aspect", 0.85)]
        hit = self.exact(w)
        if hit:
            return [(w, hit[0], hit[1], TRANSLATION_CONF[hit[1]])]
        if w in self.variants:
            hit = self.exact(self.variants[w])
            return [(w, hit[0], "curated", TRANSLATION_CONF["curated"])]
        for pre in ("การ", "ความ", "นัก"):
            if w.startswith(pre) and len(w) > len(pre) + 1:
                hit = self.exact(w[len(pre):]) if pre != "นัก" else None
                if hit:
                    return [(w, hit[0], "lemma" if hit[1] == "lemma" else hit[1], TRANSLATION_CONF[hit[1]] * 0.95)]
        parts = self.split(w)
        if parts:
            return [(p, self.exact(p)[0], "compound", TRANSLATION_CONF["compound"]) for p in parts]
        return []

    def spell(self, w):
        """Thai / Latin letters of w → letter concepts, or None if any letter (other than a tone / silent mark) has no recorded sign."""
        out = []
        self.skipped = []
        for ch in w:
            if ch.isspace():
                continue
            key = ch.upper() if ch.isascii() else ch
            if key in self.letters:
                out.append(self.letters[key]); continue
            alt = ch + "อ" if ch in "เแโใไ" else ("อ" + ch if "ะ" <= ch <= "๎" else None)   # leading vs following vowels
            if alt and alt in self.letters:
                out.append(self.letters[alt]); continue
            if ch in "่้๊๋์":         # a tone / silent mark without a recorded clip: skipped (noted)
                self.skipped.append(ch)
                continue
            return None
        return out or None


ROLE = {"time": "time", "wh": "wh", "neg": "neg", "pron": "subject", "verb": "verb", "modal": "verb", "motion": "verb",
        "noun": "noun", "adj": "adjective", "num": "number", "greeting": "greeting", "aspect": "aspect", "word": None}


def translate_thai(text, mapper=None, topicalize=True, spell_max=10, source=None, phrasebook=True, grammar="tsl51"):
    """grammar: tsl51 (TSL51-derived ordering rules) | thai (Normal mode: same signs in Thai word order, no recorded-phrase
    shortcut — a real signer's phrase is in TSL order, not Thai order)."""
    if grammar not in G.MODES:
        raise ValueError(f"grammar must be one of {G.MODES}")
    mapper = mapper or Mapper()
    if grammar == "thai":
        phrasebook = False
    thai = normalize(text)
    units, clauses, warnings, all_tokens = [], [], [], []
    whole = phrase_key(thai)
    if phrasebook and whole in mapper.phrases:               # the whole input is a recorded conversational phrase
        parts = [thai]
    else:
        parts = clauses_of(thai)
    for cl in parts:
        key = phrase_key(cl)
        if phrasebook and key in mapper.phrases:
            # a real signer's recording of this exact phrase (natural TSL order, co-articulation and face) — played as one unit
            toks = tokenize(cl, mapper)
            stype = G.reorder(G.classify(toks, mapper.pos)).sentence_type
            units.append(SignUnit(gloss=mapper.phrases[key][len("phrase:"):], kind="phrase", concept=mapper.phrases[key],
                                  source_words=toks, match="phrase", role="phrase", confidence={"translation": 0.95},
                                  note="ประโยคที่บันทึกจากผู้ใช้ภาษามือจริง (recorded phrase, natural TSL)",
                                  provenance=dict(origin="recorded_phrase")))
            clauses.append(dict(text=cl, type=stype, negated=False, recorded=True, units=[len(units) - 1],
                                moves=["whole clause = a recorded phrase of a real signer (natural TSL order and face)"]))
            all_tokens += toks
            continue
        toks = tokenize(cl, mapper)
        expanded_fixed = []
        for w in toks:                                     # fixed expressions → their sign sequence (kept in this order)
            if w in mapper.fixed:
                expanded_fixed += [Fixed(x) for x in mapper.fixed[w]]
            else:
                expanded_fixed.append(w)
        toks = expanded_fixed
        all_tokens += toks
        # 1) compound split first (so "กินข้าว" takes part in the grammar as กิน + ข้าว), 2) classify, 3) map with the word's kind
        words, comp, names = [], {}, set()
        closed = (G.DROP, G.YNQ, G.PREP_DROP, G.COPULA, G.COMITATIVE, G.POSSESSIVE, G.WH, G.NEG)
        k = next((i for i, w in enumerate(toks) if w in G.NAME_TRIGGER), None)
        if k is not None:
            # the name = the tokens right after ชื่อ, up to the first particle / question / degree word or a known adjective /
            # verb ("หมอชื่อวิชัยใจดีมาก": the name is วิชัย, ใจดี มาก describe him) — the rest keeps its place
            j, name = k + 1, []
            while j < len(toks):
                w = toks[j]
                if w in G.DROP or w in G.ADV or any(w in S for S in closed) or (name and mapper.pos(w) in ("adj", "verb")):
                    break
                name.append(w); j += 1
            if len(name) > 1 or (name and mapper.exact(name[0]) is None):
                toks = toks[:k + 1] + ["".join(name)] + toks[j:]
                name_at = k + 1
            else:
                name_at = None
        else:
            name_at = None
        for ti, w in enumerate(toks):
            if name_at is not None and ti == name_at:
                names.add(len(words)); words.append(w)            # a name: fingerspelled as written
                continue
            if (mapper.exact(w) is None and w not in G.PRON_SIGN and w not in mapper.variants and w not in mapper.fixed
                    and not any(w in S for S in closed)):
                parts = mapper.split(w)
                if parts:
                    for p_ in parts:
                        comp[len(words)] = w
                        words.append(p_)
                    continue
            words.append(w)
        ctoks = G.classify([str(w) for w in words], mapper.pos)
        for i, w in enumerate(words):
            if isinstance(w, Fixed) and ctoks[i].kind != "ynq":
                ctoks[i].kind = "fixed"; ctoks[i].notes.append("fixed expression")
        # homographs: which meaning does this clause use (senses.py) — before the grammar, a noun reading changes the role
        sense_of = {}
        plain = [t.text for t in ctoks]
        for i, t in enumerate(ctoks):
            r = senses.resolve(t.text, plain, i, as_pronoun=t.kind in ("pron", "drop"))
            if r:
                sense_of[i] = r
                if t.kind in ("pron", "word", "drop"):
                    t.kind = "noun"
                t.notes.append(f"sense: {r['label']}" + (f" (context: {' '.join(r['evidence'])})" if r["evidence"] else " (no context cue)"))
        info = {}
        for i in names:
            ctoks[i].kind = "name"
        for i, t in enumerate(ctoks):
            if t.kind == "name" and mapper.exact(t.text) is None:
                info[i] = None
                continue
            if t.kind in ("drop", "ynq", "prep", "copula", "comitative", "possessive", "cls"):
                info[i] = None
                continue
            if i in comp and t.kind in ("drop", "prep"):
                info[i] = None
                continue
            if i in sense_of:
                r = sense_of[i]
                conf = senses.AMBIGUOUS_CONF if r["ambiguous"] else TRANSLATION_CONF["sense"]
                info[i] = (t.text, r["concept"], "sense", conf) if r["concept"] and mapper.exact(r["concept"]) else None
                continue
            if i in comp:
                c = mapper.exact(t.text)
                info[i] = (t.text, c[0], "compound", TRANSLATION_CONF["compound"]) if c else None
                continue
            m = mapper.map(t.text, kind=t.kind if t.kind == "pron" else None)
            info[i] = m[0] if len(m) == 1 else None
            if info[i] and t.kind == "fixed" and info[i][2] in ("lemma", "gloss"):
                info[i] = (info[i][0], info[i][1], "curated", TRANSLATION_CONF["curated"])
        for i, t in enumerate(ctoks):                     # incorporated object of a compound verb (กินข้าว → ข้าว กิน, no fronting)
            if i in comp and t.kind == "noun" and i > 0 and (i - 1) in comp and comp[i - 1] == comp[i] \
                    and ctoks[i - 1].kind in ("verb", "modal"):
                t.kind = "iobj"
                t.notes.append("incorporated object")
        clause = G.reorder(ctoks, topicalize=topicalize, mode=grammar)
        idx = []
        for t in clause.signs:
            m = info.get(t.src)
            if any(n.startswith("post-verbal") for n in t.notes):
                mm = mapper.map(t.text)
                m = mm[0] if mm else m
            role = ROLE.get(t.kind, t.kind)
            if t.kind == "noun" and "incorporated object" not in t.notes:
                role = "noun"
            if "incorporated object" in t.notes:
                role = "object"
            sres = sense_of.get(t.src)
            if m:
                u = SignUnit(gloss=m[1], kind="sign", concept=m[1], source_words=[t.text], match=m[2], role=role,
                             confidence={"translation": round(float(m[3]), 3)})
                if t.src in comp or "fixed expression" in t.notes:
                    whole = comp.get(t.src) or "fixed expression"
                    u.provenance = dict(origin="composed", assembled_from=whole)
                if sres:
                    u.sense = sres
                    if sres["ambiguous"]:
                        u.note = (f"คำกำกวม: ใช้ความหมาย '{sres['label']}' (ท่าที่มีในคลัง) — อาจหมายถึง "
                                  f"{' / '.join(sres['alternatives'])}; ตรวจสอบบริบท")
                        warnings.append(f"'{t.text}': ambiguous word — the recorded meaning '{sres['label']}' was used without a context cue")
            else:
                letters = mapper.spell(t.text) if len(t.text) <= spell_max else None
                if letters and t.kind in ("word", "noun", "pron", "name", "adj", "verb", "iobj"):
                    note = "ไม่มีท่าในคลัง — สะกดนิ้ว (no recorded sign — fingerspelled)"
                    if sres and not sres["concept"]:
                        note = (f"ความหมาย '{sres['label']}' ไม่มีท่าในคลัง (ท่า '{t.text}' ในคลังหมายถึงความหมายอื่น) — สะกดนิ้ว "
                                f"(the meaning in this sentence has no recorded sign; the recorded '{t.text}' means something else)")
                    if mapper.skipped:
                        note += f"; ไม่มีท่าของเครื่องหมาย {' '.join('อ' + c for c in mapper.skipped)} — ข้าม"
                    u = SignUnit(gloss=t.text, kind="fingerspell", source_words=[t.text], match="fingerspell", role=role,
                                 letters=letters, confidence={"translation": TRANSLATION_CONF["fingerspell"]}, note=note,
                                 sense=sres, provenance=dict(origin="fingerspelling"))
                    warnings.append(f"'{t.text}': no recorded sign → fingerspelled")
                else:
                    u = SignUnit(gloss=t.text, kind="text", source_words=[t.text], match="none", role=role,
                                 confidence={"translation": 0.0}, note="ไม่มีท่าในคลัง — แสดงเป็นข้อความ (no recorded sign — shown as text)",
                                 sense=sres, provenance=dict(origin="text"))
                    warnings.append(f"'{t.text}': no recorded sign → shown as text")
            idx.append(len(units)); units.append(u)
        clauses.append(dict(text=cl, type=clause.sentence_type, negated=clause.negated, moves=clause.moves, units=idx))
    signed = [u for u in units if u.kind in ("sign", "phrase")]
    content = [u for u in units if u.role not in ("greeting",)]
    cov = (sum(u.kind in ("sign", "phrase") for u in content) / len(content)) if content else (1.0 if units else 0.0)
    tconf = (min(u.confidence["translation"] for u in units) if units else 0.0)
    ir = IR(input=dict(text=text, lang="th", source=source or "text"), thai=thai, tokens=all_tokens, clauses=clauses, units=units,
            warnings=warnings, confidence=dict(translation=round(tconf, 3), coverage=round(cov, 3),
                                               signs=len(signed), units=len(units)),
            grammar=dict(mode=grammar, rules=G.RULES_VERSION if grammar == "tsl51" else "thai-order",
                         phrasebook=bool(phrasebook)))
    return ir
