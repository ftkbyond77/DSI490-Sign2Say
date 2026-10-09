"""The Intermediate Representation between language and motion (prompt point 3).

  Thai/English text ─▶ [normalise · tokenise] ─▶ TSL structure (gloss order, sentence type, non-manual marks)
                    ─▶ sign units (concept · dictionary clip · hand · location · movement · face)  ─▶ motion timeline ─▶ avatar
Every stage writes into this one JSON-serialisable object, so each step can be inspected and corrected (an expert can edit the
gloss list and re-render without touching the language stage: POST /api/render takes an IR).

Two confidences are kept apart (prompt point 7):
  translation  how sure we are that this sign expresses the input word in this sentence (exact lemma 1.0 · synonym 0.85 ·
               fingerspelling 0.6 · text card 0)
  motion       how faithfully the avatar reproduces the recorded dictionary sign (tier A/B, retargeting error, tracking)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class SignUnit:
    gloss: str                        # what is shown under the avatar
    kind: str                         # sign · fingerspell · text (no sign available: shown as a caption, the avatar does not invent)
    concept: str | None = None        # motion-library key (None for text)
    source_words: list = field(default_factory=list)   # input tokens this unit expresses
    match: str = "exact"              # lemma · gloss · alias · group · ttrs_synonym · english · compound · semantic · fingerspell · none
    role: str | None = None           # subject · object · verb · time · wh · neg · greeting · ...
    letters: list = field(default_factory=list)        # fingerspelling: the letter concepts, in order
    clip: dict | None = None          # {clip_id, dataset, signer, tier, mirrored, consensus, n_signers}
    manual: dict | None = None        # {dominant, two_handed, location, movement, movement_kind}  (measured on the avatar)
    nmm: dict = field(default_factory=dict)            # {face: "from_clip"} + sentence overlays {brows, head}
    confidence: dict = field(default_factory=dict)     # {translation, motion}
    timing: dict | None = None        # {start_s, end_s} filled by the composer
    note: str | None = None


@dataclass
class IR:
    input: dict                       # {text, lang, source: text|speech, asr: {...}}
    thai: str                         # the Thai text that was translated (= input for Thai; MT output for English)
    tokens: list                      # Thai tokens after normalisation
    clauses: list                     # [{type, negated, moves[], units: [index…]}]
    units: list                       # SignUnit list in signing order
    warnings: list = field(default_factory=list)
    confidence: dict = field(default_factory=dict)     # sentence-level {translation, motion, coverage}
    version: str = "t2s-1"

    def to_dict(self):
        d = asdict(self)
        return d


def unit_from_dict(d):
    return SignUnit(**{k: v for k, v in d.items() if k in SignUnit.__dataclass_fields__})
