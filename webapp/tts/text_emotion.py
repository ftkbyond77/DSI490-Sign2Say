"""Thai text → emotion by lexicon (no model, offline). Used when /v1/speak gets emotion="auto".

Returns one of the 7 FER2013 classes, or None when no cue is found (the caller falls back to the face emotion / neutral).
Substring match (Thai has no spaces); a cue directly after "ไม่" ("ไม่เสียใจ") is skipped. Extend the lists freely.
"""
from __future__ import annotations

# cue → weight (2 = clear, 1 = weak)
LEXICON: dict[str, dict[str, int]] = {
    "happy": {"ดีใจ": 2, "สุขใจ": 2, "มีความสุข": 2, "ยินดี": 2, "สนุก": 2, "เย้": 2, "สุดยอด": 2, "เยี่ยม": 2, "ดีมาก": 2, "ดีจัง": 2,
              "ชอบมาก": 2, "รักมาก": 2, "รักนะ": 1, "ปลื้ม": 2, "โล่งใจ": 2, "สอบผ่าน": 2, "สำเร็จ": 1, "ฮ่าๆ": 2, "555": 2, "หัวเราะ": 1,
              "ขอบคุณมาก": 2, "ขอบคุณ": 1, "เก่งมาก": 2, "วิเศษ": 2, "ชนะ": 1, "สุขสันต์": 2, "ดีใจมาก": 2},
    "sad":   {"เสียใจ": 2, "เศร้า": 2, "ร้องไห้": 2, "น้ำตา": 2, "คิดถึง": 1, "เหงา": 2, "ท้อ": 1, "ผิดหวัง": 2, "จากไป": 2, "สูญเสีย": 2,
              "เสียชีวิต": 2, "ล่วงลับ": 2, "ทุกข์": 2, "หดหู่": 2, "ใจหาย": 2, "สงสาร": 1, "น่าเศร้า": 2, "อกหัก": 2, "แย่จัง": 1, "ขอโทษ": 1,
              "เสียดาย": 1, "ไม่สบายใจ": 1},
    "angry": {"โกรธ": 2, "โมโห": 2, "หงุดหงิด": 2, "ฉุน": 1, "ทำไมทำแบบนี้": 2, "รำคาญ": 2, "เหลือทน": 2, "บ้าเหรอ": 2, "ไอ้": 1, "เลว": 2,
              "หยุดนะ": 2, "ไม่ไหวแล้ว": 1, "ทนไม่ไหว": 2, "อย่ามายุ่ง": 2, "ไม่พอใจ": 2, "ห้ามนะ": 1, "เกลียดที่สุด": 2, "ปั่นหัว": 2, "ทำไมไม่": 1},
    "fear":  {"กลัว": 2, "หวาดกลัว": 2, "น่ากลัว": 2, "ช่วยด้วย": 2, "อันตราย": 1, "ใจสั่น": 2, "ขนลุก": 2, "หนีเร็ว": 2, "ระวัง": 1, "ผี": 1, "หวาดระแวง": 2},
    "disgust": {"ขยะแขยง": 2, "น่าเกลียด": 1, "สกปรก": 2, "เหม็น": 2, "อี๋": 2, "แหวะ": 2, "คลื่นไส้": 2, "ยะแยง": 2, "น่ารังเกียจ": 2, "เกลียด": 1},
    "surprise": {"ว้าว": 2, "โอ้โห": 2, "อะไรนะ": 2, "จริงเหรอ": 2, "จริงหรือ": 2, "จริงดิ": 2, "ไม่อยากเชื่อ": 2, "ไม่น่าเชื่อ": 2, "ตกใจ": 2,
                 "ตายแล้ว": 2, "เฮ้ย": 2, "อ้าว": 1, "แปลกใจ": 2, "ไม่คาดคิด": 2, "เหลือเชื่อ": 2},
}
_ORDER = ["angry", "sad", "fear", "disgust", "surprise", "happy"]          # tie-break
_NEG = ("ไม่", "ไม่ได้", "ไม่ค่อย", "มิ")


def _negated(text: str, i: int) -> bool:
    pre = text[max(0, i - 4):i].replace(" ", "")
    return any(pre.endswith(n) for n in _NEG)


def detect(text: str) -> tuple[str | None, dict[str, int]]:
    """→ (emotion or None, score per emotion)."""
    t = (text or "").lower()
    score = {e: 0 for e in LEXICON}
    for emo, cues in LEXICON.items():
        for cue, w in cues.items():
            start = 0
            while True:
                i = t.find(cue, start)
                if i < 0:
                    break
                if not _negated(t, i):
                    score[emo] += w
                start = i + len(cue)
    best = max(score.values())
    if best <= 0:
        return None, score
    return next(e for e in _ORDER if score[e] == best), score


if __name__ == "__main__":
    for s in ["วันนี้ฉันดีใจมากที่สอบผ่าน", "ฉันเสียใจมากที่เขาจากไป", "ทำไมทำแบบนี้ ฉันโมโหมากแล้วนะ", "ฉันไม่เสียใจเลย", "วันนี้อากาศดี", "อะไรนะ จริงเหรอ", "เหม็นมาก อี๋"]:
        print(s, "→", detect(s))
