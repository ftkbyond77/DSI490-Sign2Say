"""Curated conversational Thai → existing recorded signs. Written by hand from the conversational coverage test
(tests/conversation.tsv, result_v1.md §4.4) — every target must be a sign that exists in the motion library, otherwise the entry is
ignored at load time (nothing here can create a sign). Each use is marked `match: curated` in the IR (translation confidence 0.85)
so the Deaf reviewers can check exactly these choices.

VARIANTS   one written form → another written form of the same meaning that the dictionaries recorded
           (spelling / register / colloquial variants; politeness prefixes of kinship terms).
FIXED      multi-word conversational formulas → their TSL sign sequence (kept in this order, not re-ordered by the grammar);
           a "ไหม" in the sequence marks a yes/no question (non-manual marking), it is not signed.
"""
from __future__ import annotations

VARIANTS = {
    # spelling / colloquial
    "เท่าไหร่": "เท่าไร", "เมื่อไหร่": "เมื่อไร", "ยังไง": "อย่างไร", "ไง": "อย่างไร", "อะไรบ้าง": "อะไร",
    "นิดนึง": "นิดหน่อย", "หน่อยนึง": "นิดหน่อย", "ป่าว": "เปล่า", "บายบาย": "บ๊ายบาย", "ไหว้": "สวัสดี",
    # same meaning, recorded under another word
    "ไหน": "ที่ไหน", "เจอกัน": "เจอ", "พบกัน": "พบ", "เยอะ": "มาก", "เยอะแยะ": "มาก", "หลาย": "มาก", "มากมาย": "มาก",
    "ภาษาสัญลักษณ์": "ภาษามือ", "เรียนรู้": "เรียน", "อาศัยอยู่": "อยู่", "อาศัย": "อยู่", "พักอาศัย": "อยู่",
    "รับประทาน": "กิน", "ทาน": "กิน", "โทรหา": "โทร", "เจ็บป่วย": "ป่วย", "อันนี้": "นี้", "สิ่งนี้": "นี้", "ตรงนี้": "ที่นี่",
    "สักครู่": "แป๊บเดียว", "แป๊บ": "แป๊บเดียว", "แป๊บนึง": "แป๊บเดียว", "เดี๋ยว": "แป๊บเดียว", "อีกครั้ง": "ซ้ำ", "อีกที": "ซ้ำ",
    "ตอนเช้า": "เช้า", "ตอนเย็น": "เย็น", "ตอนกลางคืน": "กลางคืน", "ตอนเที่ยง": "เที่ยง", "ตอนบ่าย": "บ่าย",
    "คุณแม่": "แม่", "คุณพ่อ": "พ่อ", "ห้องสุขา": "ห้องน้ำ", "สุขา": "ห้องน้ำ", "รถ": "รถยนต์", "ความสุข": "มีความสุข",
    "ยินดีด้วย": "ยินดี", "มาจาก": "มา", "ตื่นนอน": "ตื่น", "นอนหลับ": "นอน", "หลับ": "นอน", "ง่วง": "ง่วงนอน", "เดินทาง": "เดินทาง",
}

FIXED = {
    "ยินดีที่ได้รู้จัก": ["ยินดี", "รู้จัก"], "ไม่เป็นไร": ["ไม่เป็นไร"], "ขอบคุณมาก": ["ขอบคุณ", "มาก"],
    "สวัสดีตอนเช้า": ["สวัสดี", "เช้า"], "ไม่ค่อย": ["ไม่"], "ใช่แล้ว": ["ใช่"], "ช่วยด้วย": ["ช่วย"],
    "ไปไหนมา": ["ไป", "ที่ไหน"], "เป็นยังไงบ้าง": ["สบายดี", "ไหม"], "เป็นไงบ้าง": ["สบายดี", "ไหม"], "ไงบ้าง": ["สบายดี", "ไหม"],
    "สบายดีไหม": ["สบายดี", "ไหม"], "สบายดีมั้ย": ["สบายดี", "ไหม"], "หรือยัง": ["ยัง", "ไหม"], "รึยัง": ["ยัง", "ไหม"],
    "แล้วเจอกันใหม่": ["พบกันใหม่"], "แล้วพบกันใหม่": ["พบกันใหม่"], "เจอกันใหม่": ["พบกันใหม่"], "พบกันใหม่": ["พบกันใหม่"],
    "กี่โมง": ["เวลา", "เท่าไร"], "ราคาเท่าไร": ["เงิน", "เท่าไร"], "ราคาเท่าไหร่": ["เงิน", "เท่าไร"], "ทำไมถึง": ["ทำไม"],
}
FIXED_FALLBACK = {"พบกันใหม่": ["เจอ", "ใหม่"]}       # when the first choice has no recorded sign


def load(entries):
    """keep only entries whose targets are signable (+ the question marker)."""
    v = {k: t for k, t in VARIANTS.items() if t in entries and k not in entries}
    f = {}
    for k, seq in FIXED.items():
        if all(x in entries or x == "ไหม" for x in seq):
            f[k] = seq
        elif k in FIXED_FALLBACK or seq[0] in FIXED_FALLBACK:
            alt = FIXED_FALLBACK.get(k) or FIXED_FALLBACK.get(seq[0])
            if alt and all(x in entries for x in alt):
                f[k] = alt
    return v, f
