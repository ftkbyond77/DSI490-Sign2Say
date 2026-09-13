"""v4 fusion + translation layer: Sign model (segments, candidates, decisions) + Face model (emotion, NMM) → LLM.

The LLM may only use words that the Sign model proposed. Segments rejected by the recogniser stay unknown and are
reported with their nearest known word — the LLM must not fill them with guesses."""
from __future__ import annotations

import json
import os
import time

SYSTEM = """คุณคือล่ามภาษามือไทย → ภาษาไทย ที่ซื่อสัตย์ต่อหลักฐาน
ข้อมูลที่ได้รับ (JSON):
- segments: ช่วงเวลาที่ระบบ Sign model ตรวจพบว่ามี "ท่ามือ 1 คำ" ตามลำดับเวลา แต่ละช่วงมี
    status = "accepted" (มั่นใจ) | "uncertain" (มีตัวเลือกแต่ไม่มั่นใจ) | "unknown" (มีท่าแต่ไม่รู้ว่าคำอะไร)
    candidates = คำที่ใกล้เคียงที่สุด 5 อันดับพร้อมคะแนน (ยิ่งสูงยิ่งใกล้), nearest = คำที่ใกล้ที่สุด
- face: Face model — อารมณ์จากสีหน้า และสัญญาณทางไวยากรณ์บนใบหน้า (คิ้วยกช่วงท้าย = อาจเป็นคำถาม, ส่ายหัว = ปฏิเสธ)
กฎบังคับ (ห้ามเดา — ถ้าไม่มีหลักฐานพอ ให้บอกว่าไม่รู้ ดีกว่าแต่งประโยคที่ฟังดูดีแต่ผิด):
1. ใช้ได้เฉพาะคำใน candidates ของแต่ละ segment เท่านั้น ห้ามเพิ่มคำศัพท์หลักที่ไม่มีในข้อมูล
2. segment ที่ status = accepted ให้ใช้ candidate อันดับ 1 เว้นแต่ขัดกับคำ accepted/uncertain อื่นอย่างชัดเจน (ถ้าขัด ให้ใช้ "[?]")
3. segment ที่ status = uncertain เลือก candidate อันดับ 1–3 ได้ **เฉพาะเมื่อ** คำนั้นประกอบกับคำอื่นที่เลือกแล้วเป็นความหมายที่คนพูดกันจริงในชีวิตประจำวัน
   ถ้าคำไหนทำให้ความหมายแปลก/สุ่ม (เช่น อวัยวะ ตัวเลข หน่วยวัด ที่ไม่เข้ากับคำอื่น) ให้ใช้ "[?]"
4. segment ที่ status = unknown ห้ามเดาคำ ให้ใช้ "[?]"
5. ภาษามือไทยเรียงแบบ topic-comment เรียบเรียงเป็นภาษาไทยพูดที่เป็นธรรมชาติได้ เติมคำเชื่อม/คำลงท้ายเล็กน้อยได้
   เติม "ไหม" เมื่อ face ชี้ว่าเป็นคำถาม หรือ "ไม่" เมื่อชี้ว่าปฏิเสธ
6. ถ้าคำที่เลือกได้ (ไม่ใช่ [?]) น้อยกว่าครึ่งของ segments หรือไม่เป็นความหมายที่สมเหตุสมผล ให้ sentence_th = "" และ confidence ≤ 0.3
   แล้วอธิบายใน meaning_th ว่าจับได้เพียงบางคำ (ระบุคำ) โดยไม่แต่งเรื่องเพิ่ม
7. confidence ต้องสะท้อนหลักฐาน: มีแต่ uncertain → ไม่เกิน 0.5
8. ตอบเป็น JSON เท่านั้น:
{"chosen": ["คำที่เลือกต่อ segment หรือ [?]", ...],
 "sentence_th": "ประโยคภาษาไทยที่เรียบเรียงแล้ว หรือ \"\" ถ้าหลักฐานไม่พอ",
 "meaning_th": "อธิบายสั้นๆ ว่าผู้ใช้ภาษามือกำลังสื่ออะไร เท่าที่หลักฐานรองรับ",
 "emotion_th": "อธิบายอารมณ์/น้ำเสียงจากสีหน้า 1 ประโยค",
 "confidence": 0.0-1.0,
 "notes": "ระบุส่วนที่ไม่แน่ใจ และคำใกล้เคียงที่สุดของช่วงที่เป็น [?]"}"""


def evidence_gloss(segments):
    """What the Sign model alone supports: word / word? / [?]≈nearest."""
    out = []
    for s in segments:
        w = s["candidates"][0]["word"]
        out.append(w if s["status"] == "accepted" else (f"{w}?" if s["status"] == "uncertain" else f"[?]≈{w}"))
    return " | ".join(out)


def rules_compose(segments, face):
    chosen = []
    for s in segments:
        chosen.append(s["candidates"][0]["word"] if s["status"] == "accepted" else "[?]")
    words = [w for w in chosen if w != "[?]"]
    sent = "".join(words)
    if face.get("nmm", {}).get("question_yesno"):
        sent += "ไหม"
    return dict(chosen=chosen, sentence_th=sent or "[ไม่สามารถถอดความได้อย่างมั่นใจ]", meaning_th="", emotion_th=face.get("emotion_th", ""),
                confidence=float(sum(s["status"] == "accepted" for s in segments) / max(len(segments), 1)), notes="rules fallback",
                provider="rules")


def compose(segments, face, model=None, timeout=30):
    from .env import load_env
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        return rules_compose(segments, face)
    from openai import OpenAI
    payload = dict(segments=[dict(t0=s["t0"], t1=s["t1"], status=s["status"], nearest=s["candidates"][0]["word"],
                                  candidates=[dict(word=c["word"], score=round(c["score"], 3)) for c in s["candidates"][:5]])
                             for s in segments],
                   face=dict(emotion=face.get("emotion"), emotion_probs=face.get("probs"), nmm={k: v for k, v in face.get("nmm", {}).items() if k != "per_span"}))
    t = time.time()
    try:
        r = OpenAI(timeout=timeout).chat.completions.create(
            model=model or os.environ.get("OPENAI_MODEL", "gpt-4.1"), temperature=0, max_tokens=500,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        out = json.loads(r.choices[0].message.content)
        out.update(provider="openai", latency_s=round(time.time() - t, 2), request=payload)
        # guard: every chosen word must come from that segment's candidates (or [?])
        viol = []
        for s, w in zip(segments, out.get("chosen", [])):
            if w != "[?]" and w not in [c["word"] for c in s["candidates"][:5]]:
                viol.append(w)
        out["guard_violations"] = viol
        out["evidence_gloss"] = evidence_gloss(segments)
        # evidence guard (deterministic, the LLM is not trusted to follow rule 6): a fluent sentence is only shown when at least
        # one sign is accepted and at least half of the segments were resolved; otherwise it is kept as a tentative reading
        chosen = out.get("chosen", [])
        n_acc = sum(1 for s_, w in zip(segments, chosen) if w != "[?]" and s_["status"] == "accepted")
        n_used = sum(w != "[?]" for w in chosen)
        if out.get("sentence_th") and (n_acc == 0 or n_used < len(segments) / 2):
            out["tentative_sentence_th"] = out["sentence_th"]
            out["sentence_th"] = ""
            out["confidence"] = min(float(out.get("confidence") or 0), 0.3)
            out["evidence_guard"] = "sentence withheld: no accepted sign" if n_acc == 0 else "sentence withheld: fewer than half of the signs resolved"
            out["tentative_meaning_th"] = out.get("meaning_th")
            near = ", ".join(f"ช่วง {i + 1} ≈ {s_['candidates'][0]['word']}" for i, s_ in enumerate(segments))
            out["meaning_th"] = f"ตรวจพบท่ามือ {len(segments)} ช่วง แต่หลักฐานไม่พอจะแปลเป็นประโยคอย่างมั่นใจ — คำที่ใกล้ที่สุดที่ระบบเคยเรียน: {near}"
        return out
    except Exception as e:
        out = rules_compose(segments, face)
        out["notes"] = f"LLM failed ({type(e).__name__}) → rules"
        out["evidence_gloss"] = evidence_gloss(segments)
        return out
