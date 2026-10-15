"""End to end: text (Thai / English) → IR → motion timeline. One Engine per process (the API keeps one); everything it loads is
read-only and memory-mapped, so start-up is fast and several workers can share the page cache.

Latency budget (laptop CPU, measured in result_v1.md §7): Thai text → IR a few ms; IR → motion ~20–60 ms for a sentence;
English adds the local MT model (~0.3–0.8 s).
"""
from __future__ import annotations

import hashlib
import json
import re
import time

import numpy as np

from . import paths
from .ir import IR, unit_from_dict
from .motion.codec import encode
from .motion.compose import FPS, compose
from .motion.library import Library
from .motion.nmm import EVIDENCE, NOT_APPLIED
from .text import grammar as G
from .text.lexemes import Lexemes
from .text.translate import Mapper, translate_thai

THAI = re.compile(r"[฀-๿]")


def detect_lang(text):
    th = len(THAI.findall(text))
    lat = len(re.findall(r"[A-Za-z]", text))
    return "th" if th >= lat else "en"


def _is_neg(u):
    return u.kind in ("sign", "fingerspell") and (u.role == "neg" or (u.gloss or "").startswith("ไม่"))


def motion_confidence(info):
    """how faithfully the avatar plays this dictionary sign: tier (A clean / B soft tracking issues) and the measured
    retargeting error (wrist in shoulder widths, fingertips in palm lengths)."""
    f = info.get("fidelity") or {}
    c = 0.92 if info.get("tier") == "A" else 0.72
    c -= max(0.0, (f.get("wrist") or 0) - 0.08) * 1.5
    c -= max(0.0, (f.get("fingertip") or 0) - 0.30) * 0.6
    return float(np.clip(c, 0.2, 0.98))


def _h(p, n=12):
    try:
        return hashlib.sha1(p.read_bytes()).hexdigest()[:n]
    except OSError:
        return None


def versions(models=None):
    """what produced an IR (reproducibility, prompt v2 §3): content hashes of the deployable data + rule sources."""
    m = models or paths.MODELS
    src = paths.PKG / "text"
    rules = hashlib.sha1(b"".join((src / f).read_bytes() for f in ("grammar.py", "curated.py", "senses.py", "translate.py")))
    return dict(lexicon=_h(m / "lexicon.json"), motion_index=_h(m / "motion" / "index.json"), motion_q=_h(m / "motion" / "q.bin"),
                timing=_h(m / "timing.json"),
                avatar_rig=_h(m / "avatar.vrm"), grammar_rules=G.RULES_VERSION, language_code=rules.hexdigest()[:12])


class Engine:
    def __init__(self, models=None, english=True):
        t0 = time.time()
        self.lib = Library((models or paths.MODELS) / "motion")
        self.lex = Lexemes(self.lib.clips)
        self.versions = versions(models)
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
    def translate(self, text, lang="auto", source="text", topicalize=True, phrasebook=True, grammar="tsl51", asr=None):
        lang = detect_lang(text) if lang in (None, "", "auto") else lang
        t0 = time.time()
        warnings = []
        thai = text
        mt = None
        if lang == "en":
            if self.english is None:
                raise RuntimeError("English input needs the translation model (models/mt) — not installed")
            thai, mt = self.english.to_thai(text)
        ir = translate_thai(thai, self.mapper, topicalize=topicalize, source=source, phrasebook=phrasebook, grammar=grammar)
        ir.input = dict(text=text, lang=lang, source=source)
        if asr:                                   # the recogniser's own text and the corrections, kept apart from the input
            ir.input["asr"] = asr
        if mt:
            ir.input["mt"] = mt
            warnings.append("English → Thai by machine translation before TSL ordering")
        ir.warnings = warnings + ir.warnings
        self._enrich(ir)
        ir.confidence["t_language_ms"] = round((time.time() - t0) * 1000, 1)
        return ir

    def _enrich(self, ir):
        ir.versions = self.versions
        for u in ir.units:
            hint = (u.provenance or {}).get("origin")
            if u.kind in ("sign", "phrase") and u.concept in self.lib:
                info = self.lib.clips[u.concept]
                u.lexeme = self.lex.lexeme(u.concept)
                origin = "recorded_phrase" if u.kind == "phrase" else (hint if hint == "composed" else "recorded_clip")
                prov = self.lex.provenance(u.concept, origin)
                if hint == "composed":
                    prov["assembled_from"] = (u.provenance or {}).get("assembled_from")
                u.provenance = prov
                u.clip = {k: info.get(k) for k in ("clip_id", "dataset", "signer", "tier", "mirrored", "consensus", "n_signers",
                                                   "n_candidates")}
                u.clip["fidelity"] = info.get("fidelity")
                u.manual = info.get("desc")
                u.nmm = {"face": "from the dictionary clip"}
                u.confidence["motion"] = round(motion_confidence(info), 3)
            elif u.kind == "fingerspell":
                u.provenance = dict(origin="fingerspelling", letters=[self.lex.provenance(L)["clip_id"] for L in u.letters if L in self.lib],
                                    note="recorded letter signs (fallback: no recorded sign for this word / meaning)")
                cs = [motion_confidence(self.lib.clips[L]) for L in u.letters if L in self.lib]
                u.confidence["motion"] = round(min(cs), 3) if cs else 0.0
                u.manual = {"dominant": "right", "two_handed": False, "location": "chest", "movement_kind": "fingerspelling"}
            else:
                u.provenance = dict(origin="text", note="no recorded sign — the word is shown, nothing is signed")
                u.confidence["motion"] = 0.0
        for c in ir.clauses:                              # sentence-level marks planned per clause (applied with scope in render)
            if c.get("recorded"):
                continue
            plan = [dict(type=c["type"], **EVIDENCE[c["type"]])] if c["type"] in EVIDENCE else []
            if any(_is_neg(ir.units[i]) for i in c["units"] if i < len(ir.units)):
                plan.append(dict(type="negation", **EVIDENCE["negation"]))
            if any("topicalised" in mv for mv in c.get("moves", [])):
                plan.append(dict(type="topic", applied=False, reason=NOT_APPLIED["topic"]))
            c["nmm"] = plan
            for i in c["units"]:
                if i < len(ir.units) and ir.units[i].kind != "text":
                    marks = {p["type"]: p for p in plan}
                    if c["type"] in marks:
                        ir.units[i].nmm = dict(ir.units[i].nmm or {}, sentence=c["type"], brows=marks[c["type"]].get("brows"),
                                               source=marks[c["type"]]["source"])
                    if _is_neg(ir.units[i]):
                        ir.units[i].nmm = dict(ir.units[i].nmm or {}, head="shake (negation)", source="data")
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
        overlays = [dict(type=c["type"], units=c["units"], neg_units=[i for i in c["units"] if i < len(units) and _is_neg(units[i])])
                    for c in ir.clauses if not c.get("recorded")] if nmm else None
        q, m, segs, marks = compose(units, self.lib, speed=speed, clause_ends=clause_ends, overlays=overlays)
        ir.nmm = marks
        for s in segs:
            if s["kind"] != "transition" and 0 <= s["unit"] < len(units):
                s["label"] = units[s["unit"]].gloss
        if fit is not None:
            q = fit(q)
        payload = encode(q, m, fps=FPS, segments=segs)
        ir.confidence["t_motion_ms"] = round((time.time() - t0) * 1000, 1)
        ir.confidence["duration_s"] = round(len(q) / FPS, 2)
        return payload

    def run(self, text, lang="auto", speed=1.0, source="text", nmm=True, phrasebook=True, fit=None, grammar="tsl51", asr=None):
        ir = self.translate(text, lang=lang, source=source, phrasebook=phrasebook, grammar=grammar, asr=asr)
        motion = self.render(ir, speed=speed, nmm=nmm, fit=fit)
        return ir, motion

    def render_ir_dict(self, d, speed=1.0, nmm=True, fit=None):
        """re-render an edited IR (expert edits the gloss list in the UI)."""
        ir = IR(input=d.get("input", {}), thai=d.get("thai", ""), tokens=d.get("tokens", []), clauses=d.get("clauses", []),
                units=[unit_from_dict(u) for u in d.get("units", [])], warnings=d.get("warnings", []), confidence={},
                grammar=d.get("grammar") or dict(mode="edited"))
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
