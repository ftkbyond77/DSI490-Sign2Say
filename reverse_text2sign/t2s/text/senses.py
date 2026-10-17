"""Homographs: one written Thai word, several meanings, and the dictionary recorded the sign of only ONE of them (v2).

v1 mapped a word to its dictionary entry by spelling alone, so "ตาของฉันเจ็บ" (my EYE hurts) signed GRANDFATHER — the only
recorded "ตา" (TTRS: พ่อของแม่ / Grandpa). Here every such word lists its meanings, which one the recording shows (evidence:
the dictionary's own definition / category, checked entry by entry) and the context words that select each meaning. The
resolver looks at the clause's other words; a meaning without a recorded sign is never replaced by the other meaning's sign —
it falls back to fingerspelling with a note. With no context cue the recorded meaning is used but marked ambiguous (lower
translation confidence + a warning), so the viewer sees the uncertainty instead of a confident wrong sign.
Pronoun readings (ผม, หนู, มัน, เขา as I / it / he) are decided by the grammar's pronoun rules first; this table covers the rest.
"""
from __future__ import annotations

# word → [(sense id, Thai label, concept with a recording of THIS meaning or None, cue words)]; the first sense is the one the
# library clip shows. cues match whole tokens anywhere in the clause; "<w" / ">w" = immediately before / after the word.
SENSES = {
    "ตา": [("grandfather", "ตา (พ่อของแม่)", "ตา", {"ยาย", "ปู่", "ย่า", "หลาน", "คุณตา", "คุณยาย", "แก่", "<คุณ"}),
           ("eye", "ตา (ดวงตา)", None, {"เจ็บ", "ปวด", "คัน", "แดง", "บอด", "มอง", "เห็น", "หลับ", "กระพริบ", "แว่น", "แว่นตา",
                                       "ข้าง", "เคือง", "อักเสบ", "ลืม", "หยอด", "ตาย", "<ลืม", "<หลับ", "<ปิด", "<เปิด"})],
    "มัน": [("yam", "มัน (หัวมัน อาหาร)", "มัน", {"<กิน", ">ทอด", ">ฝรั่ง", ">เทศ", ">บด", ">เผา", ">ต้ม", ">สำปะหลัง"})],
    "กา": [("crow", "กา (นก)", "กา", {"นก", "บิน", "ดำ", "ร้อง", "ต้นไม้"}),
           ("kettle", "กา (กาน้ำ)", None, {">น้ำ", "ต้ม", "ชา", ">ต้มน้ำ"})],
    "เขา": [("horn", "เขา (เขาสัตว์)", None, {"<วัว", "<ควาย", "<กวาง", "<แพะ", "<แรด"})],
}
AMBIGUOUS_CONF = 0.6                  # translation confidence of a recorded meaning chosen without context evidence
EVIDENCE_CONF = 0.9                   # … chosen because a context word selects it


def _hits(cues, toks, i):
    ev = []
    for c in cues:
        if c.startswith("<"):
            if i > 0 and toks[i - 1] == c[1:]:
                ev.append(c[1:])
        elif c.startswith(">"):
            if i + 1 < len(toks) and toks[i + 1] == c[1:]:
                ev.append(c[1:])
        elif c in toks[:i] + toks[i + 1:]:
            ev.append(c)
    return ev


def resolve(word, toks, i, as_pronoun=False):
    """meaning of toks[i] (= word) in this clause → dict(word, sense, label, concept, evidence, ambiguous, alternatives) or None
    when the word is not a listed homograph, or the grammar read it as a pronoun and no noun meaning has context evidence."""
    if word not in SENSES:
        return None
    opts = SENSES[word]
    scored = [(len(_hits(c, toks, i)), k) for k, (_, _, _, c) in enumerate(opts)]
    best_n, best = max(scored, key=lambda x: (x[0], -x[1]))
    if as_pronoun and best_n == 0:                            # the pronoun reading stands unless a noun meaning has evidence
        return None
    if best_n == 0:
        if len(opts) == 1:                                    # only the pronoun reading competes (handled by the grammar)
            sid, label, concept, cues = opts[0]
            return dict(word=word, sense=sid, label=label, concept=concept, evidence=[], ambiguous=False, alternatives=[])
        best = 0
    sid, label, concept, cues = opts[best]
    return dict(word=word, sense=sid, label=label, concept=concept, evidence=_hits(cues, toks, i),
                ambiguous=best_n == 0 and len(opts) > 1, alternatives=[o[1] for k, o in enumerate(opts) if k != best])
