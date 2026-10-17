"""Lexeme layer (v2): concept · sense · variant · clip kept apart (prompt v2 §B.1).

The motion library is keyed by the dictionaries' own entry names, which mix three things: the concept ("กระเทียม"), a variant
of the same concept from another school or region ("กระเทียม (ภาษามือโรงเรียนเศรษฐเสถียร…)", "ใจ (ท่าที่ 2)") and a sense
("ขอโทษ (อายุเท่ากันหรือน้อยกว่า)" — to a peer). Here every library entry is split into
  concept_id   the written form without the parenthesis            gloss   the full dictionary entry
  variant_id   school / region / "form n" tag (None = the main form) sense   any other parenthesis or the dictionary sense
and entries of one concept are grouped, so the IR can show the variants that exist and the gloss editor can switch between
them. The variant the translator picks is the one the lexicon maps the word to (consensus medoid of the signers, result_v1 §3.3);
the others are offered, never substituted silently.

Provenance per clip: dataset, clip, signer, quality tier, the audit's landmark-quality metrics, the clean-up / reconstruction
applied (palm-contact mirroring, face depth) and the usage terms of its source.
"""
from __future__ import annotations

import re
from collections import defaultdict

PAREN = re.compile(r"\s*[\(（]([^\)）]*)[\)）]\s*")
SCHOOL = [(re.compile(r"เศรษฐเสถียร"), "setsatian"), (re.compile(r"ราชสุดา"), "ratchasuda"), (re.compile(r"ท่า(?:มือ)?ที่\s*(\d+)"), "form")]

LICENCE = {
    "ttrs": "TTRS Thai Sign Language dictionary (NECTEC / Ratchasuda) — public dictionary, research use",
    "th_sl": "th-sl.com (NADT) — rights reserved; research use approved by the project owner, not redistributable",
    "tsl51_user_sign": "TSL51 dataset (Hugging Face Namonpas/thai-sign-language-tsl51) — dataset licence, research use",
    "youtube_agent": "YouTube lesson — publisher's rights, research use only (eye-checked)",
    "youtube_harvest": "YouTube lesson — publisher's rights, research use only (eye-checked)",
    "youtube_phrase": "YouTube recorded phrase — publisher's rights, research use only (eye-checked)",
}


def split_entry(key, info):
    """library key + index info → (concept_id, variant_id, sense)."""
    gloss = info.get("gloss") or key
    paren = PAREN.findall(gloss) + PAREN.findall(key)
    concept = PAREN.sub(" ", key).strip() or key
    variant, sense = None, info.get("sense")
    sv = info.get("sign_variant")
    for p in paren:
        for rx, tag in SCHOOL:
            m = rx.search(p)
            if m:
                variant = f"form {m.group(1)}" if tag == "form" else tag
                break
        else:
            sense = sense or p.strip()
    if variant is None and isinstance(sv, str) and sv:
        m = SCHOOL[2][0].search(sv)
        variant = f"form {m.group(1)}" if m else (sv.split("#", 1)[1] if "#" in sv else None)
    return concept, variant, sense


def source_url(info):
    cid = info.get("clip_id") or ""
    ds = info.get("dataset") or ""
    name = cid.split(":", 1)[1] if ":" in cid else cid
    if ds == "ttrs":
        return f"https://dic.ttrs.or.th/video/view/{name}"
    if ds.startswith("youtube"):                     # agent:yt:<id> · yth:<id>  (YouTube ids may contain _ and -)
        vid = cid.rsplit(":", 1)[-1]
        return f"https://www.youtube.com/watch?v={vid}" if len(vid) == 11 else None
    if ds.startswith("tsl51"):
        return "https://huggingface.co/datasets/Namonpas/thai-sign-language-tsl51"
    return None


class Lexemes:
    def __init__(self, clips):
        self.clips = clips
        self.of = {}
        self.groups = defaultdict(list)
        for k, info in clips.items():
            c, v, s = split_entry(k, info)
            self.of[k] = dict(concept_id=c, variant_id=v, sense=s, gloss=info.get("gloss") or k)
            self.groups[c].append(k)

    def lexeme(self, key):
        d = dict(self.of.get(key) or dict(concept_id=key, variant_id=None, sense=None, gloss=key))
        alts = [k for k in self.groups.get(d["concept_id"], []) if k != key]
        d["n_variants"] = 1 + len(alts)
        d["variants"] = [dict(key=k, variant_id=self.of[k]["variant_id"], sense=self.of[k]["sense"],
                              dataset=self.clips[k].get("dataset")) for k in alts][:8]
        return d

    def provenance(self, key, origin="recorded_clip"):
        info = self.clips[key]
        fx = info.get("fixes") or {}
        recon = []
        if fx.get("contact_mirrored"):
            recon.append(f"palm contact: {fx['contact_mirrored']} frames of the hidden hand mirrored from the tracked one")
        if fx.get("face_runs"):
            recon.append(f"face depth: hand moved {fx.get('face_shift_cm', 0)} cm towards the face (camera depth unreliable)")
        return dict(origin=origin, dataset=info.get("dataset"), clip_id=info.get("clip_id"), signer=info.get("signer"),
                    tier=info.get("tier"), mirrored=bool(info.get("mirrored")), n_signers=info.get("n_signers"),
                    licence=LICENCE.get(info.get("dataset"), "unknown — check before any public use"), source=source_url(info),
                    qc=info.get("qc") or {}, reconstruction=recon, fidelity=info.get("fidelity"))
