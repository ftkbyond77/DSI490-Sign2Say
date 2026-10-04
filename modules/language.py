"""Language layer: Sign-model evidence + face cues + learned TSL word order → a natural Thai sentence, never beyond the evidence.

Inputs per segment: status (accepted | uncertain | unknown | transition), p_correct, top-5 candidates. The sentence may only use
words the Sign model proposed; an LLM (OpenAI, optional) re-orders TSL gloss order into Thai and adds particles, a deterministic
rules composer does the same without network. A deterministic evidence guard decides what may be shown:

  * accepted → its top-1 word is used
  * uncertain → may be used (top-1..3) only if it fits the sentence; in a run of adjacent uncertain segments only the most
    probable one may be used (v6 failure "พ่อ ตา ดื่ม เด็ก": hand transitions became uncertain words that the LLM strung together)
  * unknown → "[?]" ; transition → ignored
  * the sentence is shown only when ≥ 1 sign is accepted and ≥ half of the sign segments are resolved; otherwise the tentative
    reading is kept (tentative_sentence_th) and the nearest learned words are listed
  * face: yes/no-question cue → "ไหม" ; head shake → "ไม่" (only when no negation sign was recognised)
  * the LLM sentence must contain every word it chose and may not add a person (คุณ / ฉัน …) the signs did not show; otherwise the
    deterministic rules composer writes the sentence (v7.1: "วันนี้ กิน ข้าว ยัง" had become "วันนี้ ยัง ข้าว")
"""
from __future__ import annotations

import json
import os
import time

SYSTEM = """คุณคือล่ามภาษามือไทย → ภาษาไทย ที่ซื่อสัตย์ต่อหลักฐาน
ข้อมูล (JSON):
- segments: ท่ามือที่ตรวจพบตามลำดับเวลา: status (accepted = มั่นใจ | uncertain = ไม่มั่นใจ | unknown = ไม่รู้คำ), p_correct, candidates 5 อันดับ,
  allowed = คำที่อนุญาตให้ใช้ในช่วงนั้น (ว่าง = ต้องใช้ "[?]")
- tsl_grammar: รูปแบบลำดับคำภาษามือไทยที่เรียนจากประโยคจริง และตัวอย่าง "ลำดับภาษามือ → ประโยคไทย"
- face: สัญญาณบนใบหน้า (question_yesno = คิ้วยกท้ายประโยค → เป็นคำถาม, negation = ส่ายหัว → ปฏิเสธ, emotion = อารมณ์)
กฎบังคับ (ห้ามเดา):
1. แต่ละ segment ใช้ได้เฉพาะคำใน allowed ของ segment นั้น ห้ามเพิ่มคำศัพท์หลักที่ไม่มีในข้อมูล
2. accepted → ใช้ candidate อันดับ 1 เสมอ
3. uncertain → เลือกคำใน allowed ได้เฉพาะเมื่อเข้ากับคำอื่นตาม tsl_grammar และเป็นความหมายที่คนพูดกันจริง ไม่เช่นนั้นใช้ "[?]"
4. ภาษามือไทยมักเรียง เวลา → หัวเรื่อง/กรรม → ประธาน → กริยา → คำถาม ผู้ใช้อาจทำท่าเรียงแบบภาษาไทยก็ได้ ให้เรียบเรียงเป็นภาษาไทยพูดที่เป็นธรรมชาติ
   เช่น "วันนี้ กรุงเทพ เธอ อยู่" → "วันนี้เธออยู่กรุงเทพ", "ข้าว กิน ยัง" → "กินข้าวยัง" ("ยัง"/"หรือยัง" ท้ายประโยค = คำถาม)
   คำที่สะกดนิ้ว (ตัวอักษรต่อกัน เช่น JACK) คือชื่อเฉพาะ ใช้ตามนั้นเป๊ะ ๆ
   ต้องใช้ทุกคำที่เลือก (chosen) ในประโยคครบทุกคำ ห้ามตัดทิ้ง ห้ามเติมคำสรรพนาม/ประธานที่ไม่มีในข้อมูล (เช่น คุณ ฉัน เขา)
   เติมได้เฉพาะคำเชื่อม/คำลงท้ายเล็กน้อย (เช่น ที่ กับ ครับ/ค่ะ) เติม "ไหม" เมื่อ face.question_yesno และยังไม่มีคำถามอื่น (อะไร ที่ไหน ยัง …)
   และ "ไม่" เมื่อ face.negation
5. ตอบ JSON เท่านั้น: {"chosen": [คำต่อ segment หรือ "[?]"], "sentence_th": "...", "meaning_th": "...", "confidence": 0-1}"""


def evidence_gloss(segments):
    out = []
    for s in segments:
        if s["status"] == "transition":
            continue
        w = s["candidates"][0]["word"]
        out.append(w if s["status"] == "accepted" else (f"{w}?" if s["status"] == "uncertain" else f"[?]≈{w}"))
    return " | ".join(out)


def allowed_words(segments, tau_strong=None):
    """Deterministic part of the guard: which words each sign segment may contribute."""
    allowed = []
    for s in segments:
        if s["status"] == "accepted":
            allowed.append([s["candidates"][0]["word"]])
        elif s["status"] == "uncertain":
            allowed.append([c["word"] for c in s["candidates"][:3]])
        else:
            allowed.append([])
    # in a run of adjacent uncertain segments only the most probable one keeps its words (unless it is itself strong)
    i = 0
    while i < len(segments):
        if segments[i]["status"] != "uncertain":
            i += 1; continue
        j = i
        while j + 1 < len(segments) and segments[j + 1]["status"] == "uncertain":
            j += 1
        if j > i:
            run = list(range(i, j + 1))
            best = max(run, key=lambda k: segments[k].get("p_correct", 0))
            for k in run:
                if k != best and not (tau_strong and segments[k].get("p_correct", 0) >= tau_strong):
                    allowed[k] = []
        i = j + 1
    return allowed


FINAL = ("หรือยัง", "หรือเปล่า", "ไหม", "มั้ย", "ยัง")          # sentence-final question particles
PRONOUNS = ("คุณ", "ฉัน", "ผม", "ดิฉัน", "เธอ", "เขา", "เรา", "หนู", "พวกเขา", "พวกเรา", "มัน")


def _order(words, roles):
    """Thai word order for signs given in TSL order (time · object · subject · verb · question) or in Thai order:
    greeting → time → subject → (negation +) verbs → objects / places / the rest → question words → final particle."""
    r = [roles.get(w) for w in words]
    vi = next((i for i, x in enumerate(r) if x in ("VERB", "STATE")), len(words))
    if words and words[-1] in FINAL:
        r[-1] = "FINAL"
    subj = [i for i, x in enumerate(r) if x == "SUBJ" and i < vi] or [i for i, x in enumerate(r) if x == "SUBJ"][:1]
    si = subj[-1] if subj else None                     # TSL puts the object first: the last subject-like word before the verb is the subject
    take = lambda keep: [words[i] for i in range(len(words)) if keep(i)]   # noqa: E731
    greet = take(lambda i: r[i] == "GREET")
    time_ = take(lambda i: r[i] == "TIME")
    subj_ = [words[si]] if si is not None else []
    pred = []
    for i in range(len(words)):
        if r[i] in ("VERB", "STATE") or (r[i] == "NEG" and i + 1 < len(words) and r[i + 1] in ("VERB", "STATE")):
            pred.append(words[i])
    rest = take(lambda i: i != si and r[i] not in ("GREET", "TIME", "QUEST", "FINAL", "VERB", "STATE") and words[i] not in pred)
    quest = take(lambda i: r[i] == "QUEST")
    final = take(lambda i: r[i] == "FINAL")
    return greet + time_ + subj_ + pred + rest + quest + final


def rules_compose(segments, face, roles=None, allowed=None, words=None):
    """Offline composer (no LLM): only ACCEPTED signs, put into Thai word order (`_order`); face cues add ไม่ (head shake) /
    ไหม (question cue) when no sign says it."""
    roles = roles or {}
    chosen = [s["candidates"][0]["word"] if s["status"] == "accepted" else "[?]" for s in segments]
    if words is not None:              # recomposition after the LLM: the words that survived the guard
        chosen = list(words)
    seq = _order([w for w in chosen if w != "[?]"], roles)
    if seq and face.get("negation") and not any(roles.get(w) == "NEG" for w in seq):
        k = next((i for i, w in enumerate(seq) if roles.get(w) in ("VERB", "STATE")), len(seq))
        seq.insert(k, "ไม่")
    sent = ""
    for w in seq:                                   # Thai words join without spaces; a spelled Latin name keeps spaces around it
        latin = w.isascii() and w.isalpha()
        if sent and (latin or (sent[-1].isascii() and sent[-1].isalpha())):
            sent += " "
        sent += w
    if sent and face.get("question_yesno") and not any(roles.get(w) == "QUEST" or w in FINAL for w in seq):
        sent += "ไหม"
    n_acc = sum(s["status"] == "accepted" for s in segments)
    return dict(chosen=chosen, sentence_th=sent, meaning_th="", confidence=round(n_acc / max(len(segments), 1), 2), provider="rules")


def merge_letters(signs):
    """Consecutive fingerspelled letters → one spelled word ("J","A","C","K" → "JACK"); its status is the weakest letter's."""
    out, run = [], []

    def flush():
        if run:
            word = "".join(x["nearest"] for x in run)
            st = "accepted" if all(x["status"] == "accepted" for x in run) else "uncertain"
            p = float(min(x.get("p_correct", 0) for x in run))
            out.append(dict(f0=run[0]["f0"], f1=run[-1]["f1"], status=st, p_correct=p, nearest=word, kind="spelled",
                            candidates=[dict(word=word, score=p)], letters=[x["nearest"] for x in run]))
            run.clear()
    for s in signs:
        if s.get("kind") == "letter":
            run.append(s)
        else:
            flush(); out.append(s)
    flush()
    return out


def sentence_problems(sentence, chosen):
    """Words the sentence dropped, and persons it added, compared with the chosen words."""
    used = [w for w in chosen if w and w != "[?]"]
    missing = [w for w in used if w not in sentence]
    rest = sentence
    for w in sorted(used, key=len, reverse=True):
        rest = rest.replace(w, " ")
    added = [p for p in PRONOUNS if p in rest]
    return missing, added


def compose_sentence(segments, face=None, grammar=None, roles=None, use_llm=True, model=None, timeout=None, face_cues=("negation",)):
    """→ dict(chosen, sentence_th (or "" when withheld), tentative_sentence_th, meaning_th, evidence_gloss, evidence_guard, provider).
    face_cues: which face grammar cues may change the sentence (models/spotting.json "face_cues"; only cues that were validated on
    held-out data — the yes/no-question brow cue is off until it is)."""
    from .env import load_env
    face = {k: v for k, v in (face or {}).items() if k not in ("question_yesno", "question_wh", "negation", "affirmation") or k in face_cues}
    grammar = grammar or {}
    signs = merge_letters([s for s in segments if s["status"] != "transition"])
    if not signs:
        return dict(chosen=[], sentence_th="", meaning_th="ไม่พบช่วงที่เป็นภาษามือ", confidence=0.0, provider="none", evidence_gloss="")
    allowed = allowed_words(signs)
    load_env()
    out = None
    # every sign confident → the deterministic composer already gives the sentence (Thai order); the LLM (1–3 s) is only needed to
    # choose among the candidates of uncertain signs
    # no guessing: only confident signs (status accepted, P(correct) ≥ accept_min) enter the sentence; the others are shown as [?]
    sure = [s for s in signs if s["status"] == "accepted"]
    if not sure:
        return dict(chosen=["[?]"] * len(signs), sentence_th="", meaning_th="ยังไม่มั่นใจพอจะแปล — ลองทำท่าช้าลง ให้มืออยู่ในกรอบ หรือกดยืนยันคำที่ถูก",
                    confidence=0.0, provider="none", evidence_gloss=evidence_gloss(signs), evidence_guard="withheld: no confident sign")
    signs_all, signs = signs, sure
    allowed = allowed_words(signs)
    # the LLM only polishes a sentence of ≥ 3 confident words (word order + particles); shorter ones: the rules composer, no wait
    if use_llm and os.environ.get("OPENAI_API_KEY") and len(signs) >= 3:
        payload = dict(
            segments=[dict(status=s["status"], p_correct=round(s.get("p_correct", 0), 2), allowed=a,
                           candidates=[dict(word=c["word"], score=round(c["score"], 3)) for c in s["candidates"][:5]]) for s, a in zip(signs, allowed)],
            tsl_grammar=dict(templates=grammar.get("templates", [])[:12], examples=grammar.get("examples", [])[:12]),
            face={k: face.get(k) for k in ("question_yesno", "question_wh", "negation", "affirmation", "emotion")})
        t = time.time()
        try:
            from openai import OpenAI
            r = OpenAI(timeout=float(timeout or os.environ.get("OPENAI_TIMEOUT_S", 8))).chat.completions.create(
                model=model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0, max_tokens=400,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
            out = json.loads(r.choices[0].message.content)
            out.update(provider="openai", latency_s=round(time.time() - t, 2))
        except Exception as e:  # noqa: BLE001
            out = None
            err = f"LLM failed ({type(e).__name__}) → rules"
    if out is None:
        out = rules_compose(signs, face, roles, allowed)
        if use_llm and os.environ.get("OPENAI_API_KEY"):
            out["notes"] = locals().get("err", "")
    chosen = list(out.get("chosen", []))[:len(signs)] + ["[?]"] * max(0, len(signs) - len(out.get("chosen", [])))
    viol = []
    for i, (w, a) in enumerate(zip(chosen, allowed)):
        if w != "[?]" and w not in a:
            viol.append(w); chosen[i] = "[?]"
    for i, (s, a) in enumerate(zip(signs, allowed)):          # an accepted sign is always used
        if s["status"] == "accepted" and chosen[i] == "[?]":
            chosen[i] = a[0]
    if out.get("provider") == "openai":
        missing, added = sentence_problems(out.get("sentence_th") or "", chosen)
        used = [w for w in chosen if w != "[?]"]
        thai = _order(used, roles or {})
        kept_sign_order = (out.get("sentence_th") or "").replace(" ", "") == "".join(used) and thai != used
        if viol or missing or added or kept_sign_order:   # disallowed word / dropped word / added person / TSL order left as is → rules
            rc = rules_compose(signs, face, roles)                 # fallback: the confident words themselves, Thai order
            out["sentence_th"], out["provider"] = rc["sentence_th"], "openai+rules"
            out["guard_reason"] = dict(not_allowed=viol, dropped=missing, added=added, kept_sign_order=kept_sign_order)
    out["chosen"] = chosen
    out["guard_violations"] = viol
    out["evidence_gloss"] = evidence_gloss(signs_all)
    n_acc = sum(1 for s, w in zip(signs, chosen) if w != "[?]" and s["status"] == "accepted")
    n_used = sum(w != "[?]" for w in chosen)
    if out.get("sentence_th") and (n_acc == 0 or n_used < len(signs) / 2):
        out["tentative_sentence_th"] = out["sentence_th"]; out["sentence_th"] = ""
        out["confidence"] = min(float(out.get("confidence") or 0), 0.3)
        out["evidence_guard"] = "withheld: no accepted sign" if n_acc == 0 else "withheld: fewer than half of the signs resolved"
        near = ", ".join(f"ช่วง {i + 1} ≈ {s['candidates'][0]['word']}" for i, s in enumerate(signs))
        out["meaning_th"] = f"ตรวจพบท่ามือ {len(signs)} ช่วง แต่หลักฐานไม่พอจะแปลเป็นประโยคอย่างมั่นใจ — คำที่ใกล้ที่สุด: {near}"
    return out
