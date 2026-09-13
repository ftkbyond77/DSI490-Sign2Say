"""L5 Language layer: gloss timeline (+NMM, affect) → Thai sentence.
OpenAI LLM (rerank top-k + compose) with a deterministic rules fallback (used when offline / no key)."""
from __future__ import annotations

import json
import os

SYSTEM_TH = """คุณคือผู้ช่วยแปลภาษามือไทยเป็นภาษาไทยพูด
ข้อมูลเข้าเป็น JSON: glosses (คำภาษามือเรียงตามเวลา มี conf และตัวเลือกสำรอง alt), nmm (negation/question), affect
กฎ:
1. ภาษามือไทยเรียงแบบ topic-comment ไม่มีคำบอกกาลครบ ให้เรียบเรียงเป็นภาษาไทยพูดที่เป็นธรรมชาติ
2. ห้ามเพิ่มเนื้อหาที่ไม่มีใน gloss
3. ถ้า conf ต่ำ ให้เลือกจาก alt ที่ทำให้ประโยคสมเหตุสมผล และตั้ง uncertain เป็น true
4. เติม "ไหม" เมื่อ question เป็น true และ "ไม่" เมื่อ negation เป็น true
5. ตอบเป็น JSON เท่านั้น: {"sentence_th": "...", "style_hint": "...", "uncertain": false}"""


class RuleComposer:
    name = "rules"

    def __init__(self, low_conf=0.45):
        self.low_conf = low_conf

    def compose(self, glosses: list[dict], nmm: dict | None = None, affect: dict | None = None, context=None) -> dict:
        nmm, affect = nmm or {}, affect or {}
        words = []
        for g in glosses:
            w = g["lemma"]
            if not words or words[-1] != w:
                words.append(w)
        if nmm.get("negation") and "ไม่" not in words:
            words.insert(max(len(words) - 1, 0), "ไม่")
        sent = "".join(words)
        if nmm.get("question") and not sent.endswith(("ไหม", "มั้ย", "หรือเปล่า")):
            sent += "ไหม"
        uncertain = any(g.get("conf", 1.0) < self.low_conf for g in glosses) or not glosses
        return dict(sentence_th=sent, style_hint=affect.get("emotion", "neutral"), uncertain=bool(uncertain),
                    provider=self.name)


class OpenAIComposer:
    name = "openai"

    def __init__(self, model=None, timeout_s=None, max_tokens=80):
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        self.timeout = float(timeout_s or os.environ.get("OPENAI_TIMEOUT_S", 3))
        self.max_tokens = max_tokens
        self.fallback = RuleComposer()

    def available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def compose(self, glosses, nmm=None, affect=None, context=None) -> dict:
        if not self.available() or not glosses:
            return self.fallback.compose(glosses, nmm, affect, context)
        try:
            from openai import OpenAI
            client = OpenAI(timeout=self.timeout)
            payload = dict(glosses=[{k: g[k] for k in ("lemma", "conf", "alt", "t0", "t1") if k in g} for g in glosses],
                           nmm=nmm or {}, affect=affect or {}, context=context or [])
            r = client.chat.completions.create(
                model=self.model, max_tokens=self.max_tokens, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM_TH},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
            out = json.loads(r.choices[0].message.content)
            out["provider"] = self.name
            return out
        except Exception as e:  # network down / timeout → rules (T12)
            out = self.fallback.compose(glosses, nmm, affect, context)
            out["fallback_reason"] = type(e).__name__
            return out


def get_composer(provider: str = "openai"):
    return OpenAIComposer() if provider == "openai" else RuleComposer()
