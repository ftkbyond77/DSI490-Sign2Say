"""Thai sentence → TSL sign order (the "TSL structure" stage of the IR). Deterministic, explainable rules — every move is recorded.

Sources for the rules (result_v1.md §4):
  * TSL51 (Namonpas/thai-sign-language-tsl51): 62 Thai sentences with deaf-expert gloss order — verb-final clauses, serial verbs and
    attitude/modal verbs after the main verb ("ชอบ เที่ยว กรุงเทพ" → "กรุงเทพ เที่ยว ชอบ"), time first, WH last, possessor before the
    possessed ("แฟนฉัน" → "ฉัน แฟน"), copula / future / politeness words not signed, comitative partner first ("แต่งงานกับคุณ").
  * Thai PBS interview with TSL interpreters (2023): Deaf signers put the object/topic before subject and verb ("ข้าว … กิน") so the
    picture comes first; word-for-word Thai order is wrong.
  * Sign-language typology: negation sign after the predicate; yes/no questions are marked non-manually on the clause.
Thai word order itself is never "fixed" by guessing — a word the rules do not know keeps its Thai position.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---- closed classes (Thai) -------------------------------------------------------------------------------------------------------
DROP = {  # politeness / discourse particles, future & progressive auxiliaries, complementisers: not signed
    "ครับ", "ค่ะ", "คะ", "ค่า", "คับ", "ฮะ", "จ้ะ", "จ๊ะ", "จ้า", "นะ", "น่ะ", "นะคะ", "นะครับ", "เถอะ", "หรอก", "ล่ะ", "สิ", "ซิ",
    "เนอะ", "จ๊ะ", "จะ", "กำลัง", "หน่อย", "ซะ", "ไว้", "ได้โปรด", "โปรด", "กรุณา", "บ้าง", "ถึง", "จ้ะ", "คง", "ว่า", "ซึ่ง", "อัน", "ก็", "และ", "แล้วก็", "การ", "ความ", "ๆ", "เลย", "เอง", "น้า",
}
COPULA = {"เป็น", "คือ"}
YNQ = {"ไหม", "มั้ย", "หรือเปล่า", "รึเปล่า", "หรือไม่", "ใช่ไหม", "ใช่มั้ย", "เหรอ", "หรอ", "มั๊ย", "ไม๊"}
WH = {"อะไร", "ที่ไหน", "ไหน", "ใคร", "เมื่อไร", "เมื่อไหร่", "ทำไม", "อย่างไร", "ยังไง", "เท่าไร", "เท่าไหร่", "กี่", "แบบไหน",
      "อันไหน", "คนไหน", "ตอนไหน"}
NEG = {"ไม่", "ไม่ได้", "ไม่ใช่", "ไม่เคย", "ไม่มี", "ไม่ต้อง", "มิ"}
TIME = {"วันนี้", "พรุ่งนี้", "เมื่อวาน", "เมื่อวานนี้", "มะรืน", "มะรืนนี้", "เมื่อวานซืน", "ตอนนี้", "เดี๋ยวนี้", "ตอนเช้า", "เช้า",
        "เช้านี้", "ตอนเที่ยง", "เที่ยง", "ตอนบ่าย", "บ่าย", "ตอนเย็น", "เย็น", "เย็นนี้", "กลางคืน", "คืนนี้", "เมื่อคืน", "ตอนกลางคืน",
        "ปีนี้", "ปีหน้า", "ปีที่แล้ว", "ปีก่อน", "เดือนนี้", "เดือนหน้า", "เดือนที่แล้ว", "สัปดาห์หน้า", "อาทิตย์หน้า", "สัปดาห์นี้",
        "อาทิตย์นี้", "สัปดาห์ที่แล้ว", "อาทิตย์ที่แล้ว", "เมื่อก่อน", "ต่อไป", "อนาคต", "อดีต", "วันจันทร์", "วันอังคาร", "วันพุธ",
        "วันพฤหัสบดี", "วันพฤหัส", "วันศุกร์", "วันเสาร์", "วันอาทิตย์", "วันหยุด", "ทุกวัน", "เสมอ", "บ่อย", "ตอนเด็ก"}
TIME_RX = re.compile(r"^(วัน(นี้|จันทร์|อังคาร|พุธ|พฤหัส(บดี)?|ศุกร์|เสาร์|อาทิตย์|หยุด|เกิด)|(เดือน|ปี|สัปดาห์|อาทิตย์|วัน)(นี้|หน้า|ที่แล้ว|ก่อน)"
                     r"|ตอน(เช้า|สาย|เที่ยง|บ่าย|เย็น|ค่ำ|ดึก|กลางคืน|นี้)|เมื่อ(วาน|คืน|เช้า|กี้|ก่อน)|(มกรา|กุมภา|มีนา|เมษา|พฤษภา|มิถุนา|กรกฎา|สิงหา|กันยา|"
                     r"ตุลา|พฤศจิกา|ธันวา)(คม|ยน|พันธ์))$")
# Thai pronouns encode gender, age and politeness; TSL points (to self / addressee / other). Thai form → the pointing sign recorded
# in the dictionaries (ผม = "hair", มัน = "potato", หนู = "mouse" there — the pronoun readings must not become those signs).
PRON_SIGN = {"ผม": "ฉัน", "ดิฉัน": "ฉัน", "ฉัน": "ฉัน", "กู": "ฉัน", "ข้า": "ฉัน", "ข้าพเจ้า": "ฉัน", "หนู": "ฉัน", "เค้า": "ฉัน",
             "คุณ": "คุณ", "มึง": "คุณ", "แก": "คุณ", "ท่าน": "คุณ", "ตัวเอง": "คุณ", "เอ็ง": "คุณ", "เธอ": "เธอ",
             "เขา": "เขา", "มัน": "เขา", "หล่อน": "เขา", "พวกเขา": "เขา", "เรา": "เรา", "พวกเรา": "เรา", "พวกคุณ": "คุณ"}
PRON = {"ฉัน", "ผม", "ดิฉัน", "เรา", "พวกเรา", "คุณ", "เธอ", "เขา", "พวกเขา", "มัน", "ท่าน", "หนู", "กู", "มึง", "แก", "ข้า", "เค้า",
        "ตัวเอง", "พวกคุณ", "เอ็ง"}
# attitude / modal verbs that take a VP: signed AFTER the main verb (TSL51: X ชอบ Y → Y X... ชอบ last)
MODAL = {"ชอบ", "อยาก", "ต้องการ", "ต้อง", "ควร", "ได้", "เคย", "สามารถ", "ไม่ชอบ", "อยากจะ", "ช่วย", "ลอง", "เริ่ม", "หยุด", "เลิก",
         "พยายาม", "ชอบที่จะ", "รัก", "เกลียด", "กลัว"}
MOTION = {"ไป", "มา", "กลับ"}               # serial-verb heads ("ไปกินข้าว" → "ข้าว กิน ไป")
# verbs whose complement is a state / name / measure, not an object: the complement keeps its place after the verb
# (TSL51: "คุณ เกิด หูหนวก", "ฉัน ชื่อ แมว")
LINKING = {"เกิด", "รู้สึก", "กลายเป็น", "ชื่อ", "อายุ", "เหมือน", "ดูเหมือน", "หน้าตา", "นามสกุล", "ชื่อเล่น"}
NAME_TRIGGER = {"ชื่อ", "นามสกุล", "ชื่อเล่น"}         # what follows is a name → fingerspelled, never split into dictionary words
PREP_DROP = {"ที่", "แก่", "โดย", "ยัง", "สู่", "ไปยัง"}
COMITATIVE = {"กับ"}
POSSESSIVE = {"ของ"}
CLASSIFIER = {"คน", "ตัว", "อัน", "ใบ", "เล่ม", "คัน", "หลัง", "ลูก", "ชิ้น", "แก้ว", "จาน", "ขวด", "ครั้ง", "ด้าม", "เม็ด", "แผ่น",
              "ห้อง", "ชุด", "คู่", "กล่อง", "ถุง", "ดอก", "ต้น", "เครื่อง", "บาท"}
ADV = {"มาก", "มากๆ", "นิดหน่อย", "เกินไป", "จริงๆ", "ที่สุด", "ค่อนข้าง", "ด้วย", "เลย", "จัง", "สุดๆ", "ทั้งหมด", "อีก"}
ASPECT_END = {"แล้ว"}                       # perfective: kept as a final sign if signable, dropped after ไป/มา (TSL51 "ไปแล้ว" → ไป)
GREETING = {"สวัสดี", "ขอบคุณ", "ขอโทษ", "ลาก่อน", "ยินดีที่ได้รู้จัก", "ยินดีต้อนรับ", "โชคดี", "บ๊ายบาย", "ขอบใจ", "ไม่เป็นไร"}
NUM_WORDS = {"ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า", "สิบ", "ยี่สิบ", "ร้อย", "พัน", "หมื่น", "แสน",
             "ล้าน", "เอ็ด"}


@dataclass
class Tok:
    text: str
    kind: str = "word"            # word · time · wh · ynq · neg · pron · verb · modal · motion · noun · adj · num · cls · drop ·
    pos: str | None = None        # greeting · copula · comitative · possessive · aspect · prep
    src: int = 0                  # index in the tokenised input (traceability)
    lexpos: str | None = None     # part of speech from the sign dictionary (None = unknown there)
    notes: list = field(default_factory=list)


FREE_PARTS = {"ที่", "ได้", "การ", "ความ", "และ", "ของ", "ก็", "จะ"}   # may glue signable parts of an unknown compound
HAIR_CONTEXT = {"ยาว", "สั้น", "หงอก", "ร่วง", "ดำ", "หยิก", "ตรง", "สวย", "ทอง", "บาง"}


def _pythainlp_pos(tokens):
    try:
        from pythainlp.tag import pos_tag
        return [p for _, p in pos_tag(tokens, engine="perceptron", corpus="orchid_ud")]
    except Exception:  # noqa: BLE001
        return [None] * len(tokens)


def classify(tokens, lexicon_pos):
    """tokens (str list) → Tok list with kinds; lexicon_pos(word) → coarse POS of the sign concept (or None).
    Closed classes first (they are reliable), then the dictionary's own part of speech, then pythainlp's tagger as a fallback."""
    out = []
    ud = None
    for i, w in enumerate(tokens):
        t = Tok(w, src=i)
        p = lexicon_pos(w)
        t.lexpos = p
        if p == "time" and not (w in TIME or TIME_RX.match(w)):
            p = "noun"                                   # e.g. รัก filed under "วันสำคัญ" (Valentine's) is not a time word
        if p is None and w not in GREETING:
            if ud is None:
                ud = _pythainlp_pos(tokens)
            p = {"VERB": "verb", "ADJ": "adj", "NOUN": "noun", "PROPN": "noun"}.get(ud[i])
        t.pos = p
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if ud is None:
            ud = _pythainlp_pos(tokens)
        nxt_verb = nxt is not None and (lexicon_pos(nxt) == "verb" or nxt in MODAL or nxt in MOTION or
                                        (lexicon_pos(nxt) is None and i + 1 < len(ud) and ud[i + 1] == "VERB"))
        prv = out[-1] if out else None
        if w == "ได้" and nxt is not None and nxt in WH:
            t.kind = "drop"; t.notes.append("ได้ before a WH word (ได้ยังไง) not signed")
        elif w == "ให้" and prv is not None and prv.kind in ("verb", "modal", "motion"):
            t.kind = "drop"; t.notes.append("benefactive ให้ after a verb not signed")
        elif w == "มัน" and i == 0 and nxt is not None and (lexicon_pos(nxt) in ("adj", "verb") or
                                                              (i + 1 < len(ud) and ud[i + 1] in ("VERB", "ADJ"))):
            t.kind = "drop"; t.notes.append("expletive มัน ('it is …') not signed")
        elif w == "ใน" and prv is not None and prv.text in ("อยู่", "ไป", "มา", "อาศัย"):
            t.kind = "drop"; t.notes.append("ใน after a location verb not signed")
        elif w == "แล้ว" and (i == 0 or any(x in WH or x in YNQ for x in tokens)):
            t.kind = "drop"; t.notes.append("แล้ว as 'then' / in a question not signed")
        elif w in ADV:
            t.kind = "adv"
        elif w == "ได้" and nxt_verb:
            t.kind = "drop"; t.notes.append("ได้ before a verb = past/opportunity, not 'can'")
        elif w == "ผม" and nxt in HAIR_CONTEXT:
            t.kind = "noun"; t.notes.append("ผม = hair here")
        elif w == "หนู" and not (i == 0 and nxt is not None and (lexicon_pos(nxt) in ("verb", "adj") or nxt in MODAL or nxt in MOTION)):
            t.kind = "noun"                              # "mouse" unless it opens the clause as the subject of a verb
        elif w in GREETING:
            t.kind = "greeting"
        elif w in YNQ:
            t.kind = "ynq"
        elif w in WH:
            t.kind = "wh"
        elif w in NEG:
            t.kind = "neg"
        elif w in TIME or TIME_RX.match(w):
            t.kind = "time"
        elif w in PRON:
            t.kind = "pron"
        elif w in COPULA:
            t.kind = "copula"
        elif w in COMITATIVE:
            t.kind = "comitative"
        elif w in POSSESSIVE:
            t.kind = "possessive"
        elif w in ASPECT_END:
            t.kind = "aspect"
        elif w in DROP:
            t.kind = "drop"
        elif w in PREP_DROP:
            t.kind = "prep"
        elif w in MOTION or w in ("ไปแล้ว", "มาแล้ว"):
            t.kind = "motion"
        elif w in MODAL:
            t.kind = "modal"
        elif w.isdigit() or w in NUM_WORDS:
            t.kind = "num"
        elif w in CLASSIFIER and out and out[-1].kind == "num":
            t.kind = "cls"
        elif p == "verb":
            t.kind = "verb"
        elif p == "adj":
            t.kind = "adj"
        elif p in ("noun", "place", "time"):
            t.kind = "noun"
        out.append(t)
    # a content word right after a modal / motion verb heads the VP complement → verb, unless the dictionary says noun / place
    for i, t in enumerate(out):
        if i > 0 and out[i - 1].kind in ("modal", "motion") and t.kind in ("word", "adj", "noun") \
                and t.lexpos not in ("noun", "place", "time"):
            t.kind = "verb"; t.notes.append("after modal/motion → verb")
    # a stative word followed by an object behaves as a transitive verb ("เข้าใจ ภาษามือ", "ชอบ แมว")
    for i, t in enumerate(out):
        if t.kind == "adj" and not any(x.kind in ("verb", "modal", "motion") for x in out) and i + 1 < len(out) \
                and out[i + 1].kind in ("noun", "word") and i > 0 and out[i - 1].kind in ("pron", "noun", "word"):
            t.kind = "verb"; t.notes.append("stative with object → verb")
    return out


@dataclass
class Clause:
    signs: list            # Tok in TSL order
    sentence_type: str     # statement · wh_question · yn_question · negative · greeting
    negated: bool
    moves: list            # human-readable record of every rule applied


def _np_spans(toks):
    """group consecutive nominal tokens (noun / pron / word / num / adj after a noun) into NPs."""
    spans, cur = [], []
    for t in toks:
        if t.kind in ("noun", "pron", "word", "num", "cls") or (t.kind == "adj" and cur):
            cur.append(t)
        else:
            if cur:
                spans.append(cur)
            cur = []
            spans.append([t])
    if cur:
        spans.append(cur)
    return spans


def reorder(toks, topicalize=True):
    """one clause of classified tokens → Clause in TSL order."""
    moves = []
    keep = []
    for t in toks:
        if t.kind in ("drop", "cls", "prep"):
            moves.append(f"drop '{t.text}' ({t.kind})")
            continue
        keep.append(t)
    toks = keep
    greet = [t for t in toks if t.kind == "greeting"]
    time = [t for t in toks if t.kind == "time"]
    wh = [t for t in toks if t.kind == "wh"]
    ynq = [t for t in toks if t.kind == "ynq"]
    neg = [t for t in toks if t.kind == "neg"]
    rest = [t for t in toks if t.kind not in ("greeting", "time", "wh", "ynq", "neg", "adv")]
    adv = [t for t in toks if t.kind == "adv"]
    # a fixed formula at the start / end of the clause stays there, in its own order
    fixed_head, fixed_tail = [], []
    while rest and rest[0].kind == "fixed":
        fixed_head.append(rest.pop(0))
    while rest and rest[-1].kind == "fixed":
        fixed_tail.insert(0, rest.pop())
    for t in rest:
        if t.kind == "fixed":
            t.kind = "word"
    if time:
        moves.append("time first: " + " ".join(t.text for t in time))
    if wh:
        moves.append("WH last: " + " ".join(t.text for t in wh))
    if neg:
        moves.append("negation after the predicate: " + " ".join(t.text for t in neg))
    if ynq:
        moves.append("yes/no particle → non-manual question marking: " + " ".join(t.text for t in ynq))
    # comitative "กับ X" → X first (topic)
    topic = []
    i = 0
    while i < len(rest):
        if rest[i].kind == "comitative" and i + 1 < len(rest):
            j = i + 1
            grp = []
            while j < len(rest) and rest[j].kind in ("noun", "pron", "word"):
                grp.append(rest[j]); j += 1
            if grp:
                topic += grp
                moves.append("comitative partner first: " + " ".join(t.text for t in grp))
                rest = rest[:i] + rest[j:]
                continue
        i += 1
    rest = [t for t in rest if t.kind != "comitative"]
    # possessive "N ของ P" / "N P(pron)" → "P N"
    out = []
    i = 0
    while i < len(rest):
        t = rest[i]
        if t.kind == "possessive" and out and i + 1 < len(rest):
            owner = rest[i + 1]
            head = out.pop()
            out += [owner, head]
            moves.append(f"possessor first: {owner.text} {head.text}")
            i += 2; continue
        if t.kind == "pron" and out and out[-1].kind == "noun" and (len(out) >= 2 or i > 1 or any(x.kind in ("verb", "modal", "motion", "copula") for x in rest[:i])):
            head = out.pop()
            out += [t, head]
            moves.append(f"possessor first: {t.text} {head.text}")
            i += 1; continue
        out.append(t); i += 1
    rest = out
    # subject = nominal material before the first verb-like token
    vi = next((k for k, t in enumerate(rest) if t.kind in ("verb", "modal", "motion", "copula")), None)
    if vi is None:                                   # verbless clause (NP + adjective, greetings, single words): Thai order kept
        subj, pred = [], rest
    else:
        subj, pred = rest[:vi], rest[vi:]
    copula = [t for t in pred if t.kind == "copula"]
    if copula:
        moves.append("copula not signed: " + " ".join(t.text for t in copula))
    pred = [t for t in pred if t.kind != "copula"]
    verbs = [t for t in pred if t.kind in ("verb", "modal", "motion", "aspect_motion")]
    others = [t for t in pred if t.kind not in ("verb", "modal", "motion", "aspect", "aspect_motion")]
    aspect = [t for t in pred if t.kind == "aspect"]
    # predicative adjective / state / name after the verb keeps its place ("เกิด หูหนวก", "ชื่อ แมว");
    # an object incorporated in a compound verb (กินข้าว) sits right before its verb and is never topicalised
    iobj = [t for t in others if t.kind == "iobj"]
    others = [t for t in others if t.kind != "iobj"]
    comps = [t for t in others if t.kind == "adj"]
    objs = [t for t in others if t.kind != "adj"]
    if verbs and verbs[0].text in LINKING:
        comps, objs = others, []
        moves.append(f"'{verbs[0].text}' takes a complement, not an object: order kept")
    for k in range(1, len(pred)):                    # "กินไปแล้ว": ไปแล้ว right after a verb is aspect, not a second verb
        if pred[k].text in ("ไปแล้ว", "มาแล้ว") and pred[k - 1].kind in ("verb", "modal"):
            pred[k].kind = "aspect_motion"
    first_obj = next((k for k, t in enumerate(pred) if t.kind not in ("verb", "modal", "motion", "aspect")), len(pred))
    pre = [t for t in pred[:first_obj] if t.kind in ("verb", "modal", "motion")]
    post = [t for t in pred[first_obj:] if t.kind in ("verb", "modal", "motion", "aspect_motion")]
    for t in post:                                   # "กินข้าวไปแล้ว": ไปแล้ว after the VP is aspect → ไป (TSL51)
        if t.text == "ไปแล้ว":
            t.text = "ไป"; t.notes.append("post-verbal ไปแล้ว → ไป")
    verbs_tsl = list(reversed(pre)) + post
    if len(verbs) > 1:
        moves.append("verb chain head-final: " + " ".join(t.text for t in verbs) + " → " + " ".join(t.text for t in verbs_tsl))
    if aspect and any(v.kind == "motion" for v in verbs):
        moves.append("'แล้ว' after ไป/มา not signed (perfective carried by ไป)")
        aspect = []
    if objs and verbs:
        moves.append("object before verb: " + " ".join(t.text for t in objs))
    front = []
    if topicalize and objs and verbs and not wh and subj:
        front, objs = objs, []
        moves.append("object topicalised before subject")
    if iobj:
        moves.append("incorporated object before its verb: " + " ".join(t.text for t in iobj))
    signs = greet + time + fixed_head + topic + front + subj + objs + iobj + verbs_tsl + comps + adv + aspect + neg + fixed_tail + wh
    if not verbs and not subj:
        rest = [t for t in rest if t.kind not in ("copula", "aspect")]
        signs = greet + time + fixed_head + topic + rest + adv + neg + fixed_tail + wh
    dedup = []                                       # "เวลา กี่โมง" → เวลา + (เวลา เท่าไร): one sign is enough
    for t in signs:
        if dedup and dedup[-1].text == t.text:
            moves.append(f"repeated '{t.text}' signed once")
            continue
        dedup.append(t)
    signs = dedup
    if adv:
        moves.append("degree word after the predicate: " + " ".join(t.text for t in adv))
    wh_inside = any(any(q in t.text for q in ("ไหน", "อะไร", "ใคร", "เท่าไร", "เมื่อไร", "ทำไม", "อย่างไร")) for t in signs)
    stype = ("wh_question" if (wh or wh_inside) else "yn_question" if ynq else "negative" if neg else
             ("greeting" if greet and len(signs) == len(greet) else "statement"))
    return Clause(signs=signs, sentence_type=stype, negated=bool(neg), moves=moves)


CLOSED_WORDS = (DROP | COPULA | YNQ | WH | NEG | TIME | PRON | MODAL | MOTION | PREP_DROP | COMITATIVE | POSSESSIVE | CLASSIFIER |
                ASPECT_END | GREETING | NUM_WORDS | ADV | {"ว่า", "ที่", "ให้", "แต่", "หรือ", "ถ้า", "เพราะ", "แล้ว"})
