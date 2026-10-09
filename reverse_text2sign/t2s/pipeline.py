"""End to end: text (Thai / English) → IR → motion timeline. One Engine per process (the API keeps one); everything it loads is
read-only and memory-mapped, so start-up is fast and several workers can share the page cache.

Latency budget (laptop CPU, measured in result_v1.md §7): Thai text → IR a few ms; IR → motion ~20–60 ms for a sentence;
English adds the local MT model (~0.3–0.8 s).
"""
from __future__ import annotations

import re
import time

import numpy as np

from . import paths
from .ir import IR, unit_from_dict
from .motion.codec import encode
from .motion.compose import FPS, compose
from .motion.library import Library
from .motion.nmm import EVIDENCE
from .text.translate import Mapper, translate_thai

THAI = re.compile(r"[฀-๿]")


def detect_lang(text):
    th = len(THAI.findall(text))
    lat = len(re.findall(r"[A-Za-z]", text))
    return "th" if th >= lat else "en"


def motion_confidence(info):
    """how faithfully the avatar plays this dictionary sign: tier (A clean / B soft tracking issues) and the measured
    retargeting error (wrist in shoulder widths, fingertips in palm lengths)."""
    f = info.get("fidelity") or {}
    c = 0.92 if info.get("tier") == "A" else 0.72
    c -= max(0.0, (f.get("wrist") or 0) - 0.08) * 1.5
    c -= max(0.0, (f.get("fingertip") or 0) - 0.30) * 0.6
    return float(np.clip(c, 0.2, 0.98))


class Engine:
    def __init__(self, models=None, english=True):
        t0 = time.time()
        self.lib = Library((models or paths.MODELS) / "motion")
        # no embedding "nearest word" fallback: on the lake's labelled synonym pairs the best encoder (bge-m3) recalls only 9 % of
        # true synonyms at 90 % precision (result_v1.md §4.3) — it would mostly show a wrong sign
        self.mapper = Mapper()
        self.english = None
        if english:
            try:
                from .text.english import Translator
                self.english = Translator.load()
            except Exception:                                     # noqa: BLE001
                self.english = None
        self.load_s = time.time() - t0

    # ------------------------------------------------------------------ language
    def translate(self, text, lang="auto", source="text", topicalize=True, phrasebook=True):
        lang = detect_lang(text) if lang in (None, "", "auto") else lang
        t0 = time.time()
        warnings = []
        thai = text
        mt = None
        if lang == "en":
            if self.english is None:
                raise RuntimeError("English input needs the translation model (models/mt) — not installed")
            thai, mt = self.english.to_thai(text)
        ir = translate_thai(thai, self.mapper, topicalize=topicalize, source=source, phrasebook=phrasebook)
        ir.input = dict(text=text, lang=lang, source=source)
        if mt:
            ir.input["mt"] = mt
            warnings.append("English → Thai by machine translation before TSL ordering")
        ir.warnings = warnings + ir.warnings
        self._enrich(ir)
        ir.confidence["t_language_ms"] = round((time.time() - t0) * 1000, 1)
        return ir

    def _enrich(self, ir):
        for u in ir.units:
            if u.kind in ("sign", "phrase") and u.concept in self.lib:
                info = self.lib.clips[u.concept]
                u.clip = {k: info.get(k) for k in ("clip_id", "dataset", "signer", "tier", "mirrored", "consensus", "n_signers",
                                                   "n_candidates")}
                u.clip["fidelity"] = info.get("fidelity")
                u.manual = info.get("desc")
                u.nmm = {"face": "from the dictionary clip"}
                u.confidence["motion"] = round(motion_confidence(info), 3)
            elif u.kind == "fingerspell":
                cs = [motion_confidence(self.lib.clips[L]) for L in u.letters if L in self.lib]
                u.confidence["motion"] = round(min(cs), 3) if cs else 0.0
                u.manual = {"dominant": "right", "two_handed": False, "location": "chest", "movement_kind": "fingerspelling"}
            else:
                u.confidence["motion"] = 0.0
        for c in ir.clauses:
            if c["type"] in EVIDENCE and not c.get("recorded"):
                ev = EVIDENCE[c["type"]]
                c["nmm"] = ev
                for i in c["units"]:
                    ir.units[i].nmm = dict(ir.units[i].nmm or {}, sentence=c["type"], brows=ev.get("brows"), source=ev["source"])
        mot = [u.confidence["motion"] for u in ir.units if u.kind != "text"]
        ir.confidence["motion"] = round(min(mot), 3) if mot else 0.0
        if "signs" not in ir.confidence:                      # an edited IR / a single sign: the same summary as translate()
            content = [u for u in ir.units if u.role != "greeting"] or ir.units
            ir.confidence.update(
                signs=sum(u.kind in ("sign", "phrase") for u in ir.units), units=len(ir.units),
                coverage=round(sum(u.kind in ("sign", "phrase") for u in content) / max(len(content), 1), 3),
                translation=round(min((u.confidence.get("translation", 1.0) for u in ir.units), default=0.0), 3))

    # ------------------------------------------------------------------ motion
    def render(self, ir, speed=1.0, nmm=True, fit=None):
        """fit: a t2s.motion.fit.Fit for another character's body (None = the library avatar)."""
        t0 = time.time()
        units = ir.units
        clause_ends = [c["units"][-1] for c in ir.clauses if c["units"]]
        overlays = [dict(type=c["type"], units=c["units"]) for c in ir.clauses if not c.get("recorded")] if nmm else None
        q, m, segs = compose(units, self.lib, speed=speed, clause_ends=clause_ends, overlays=overlays)
        for s in segs:
            if s["kind"] != "transition" and 0 <= s["unit"] < len(units):
                s["label"] = units[s["unit"]].gloss
        if fit is not None:
            q = fit(q)
        payload = encode(q, m, fps=FPS, segments=segs)
        ir.confidence["t_motion_ms"] = round((time.time() - t0) * 1000, 1)
        ir.confidence["duration_s"] = round(len(q) / FPS, 2)
        return payload

    def run(self, text, lang="auto", speed=1.0, source="text", nmm=True, phrasebook=True, fit=None):
        ir = self.translate(text, lang=lang, source=source, phrasebook=phrasebook)
        motion = self.render(ir, speed=speed, nmm=nmm, fit=fit)
        return ir, motion

    def render_ir_dict(self, d, speed=1.0, nmm=True, fit=None):
        """re-render an edited IR (expert edits the gloss list in the UI)."""
        ir = IR(input=d.get("input", {}), thai=d.get("thai", ""), tokens=d.get("tokens", []), clauses=d.get("clauses", []),
                units=[unit_from_dict(u) for u in d.get("units", [])], warnings=d.get("warnings", []), confidence={})
        for u in ir.units:
            if u.kind == "sign" and u.concept not in self.lib:
                hit = self.mapper.exact(u.gloss) if u.gloss else None
                if hit:
                    u.concept = hit[0]
                else:
                    u.kind, u.concept = "text", None
                    ir.warnings.append(f"'{u.gloss}': no recorded sign → shown as text")
        if not ir.clauses:
            ir.clauses = [dict(text=ir.thai, type="statement", negated=False, moves=["edited by user"], units=list(range(len(ir.units))))]
        self._enrich(ir)
        return ir, self.render(ir, speed=speed, nmm=nmm, fit=fit)
