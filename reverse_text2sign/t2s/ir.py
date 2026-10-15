"""The Intermediate Representation between language and motion (prompt point 3; v2: result_v2.md §3).

  Thai/English text ─▶ [normalise · tokenise] ─▶ TSL structure (gloss order, sentence type, non-manual marks)
                    ─▶ sign units (lexeme · sense · variant · clip · hand · location · movement · face)  ─▶ motion timeline ─▶ avatar
Every stage writes into this one JSON-serialisable object, so each step can be inspected and corrected (an expert can edit the
gloss list and re-render without touching the language stage: POST /api/render takes an IR). The language stage never reads
motion data beyond "is there a recording for this concept", and the motion stage never changes the gloss list — grammar can be
fixed without touching motion and vice versa (prompt v2 §2).

Two confidences are kept apart (prompt point 7):
  translation  how sure we are that this sign expresses the input word in this sentence (exact lemma 1.0 · synonym 0.85 ·
               ambiguous word 0.6 · fingerspelling 0.6 · text card 0) — never derived from the ASR confidence
  motion       how faithfully the avatar reproduces the recorded dictionary sign (tier A/B, retargeting error, tracking)

v2 (`t2s-2`) adds, all optional so a v1 IR still renders:
  unit.lexeme      {concept_id, gloss, sense, variant_id, n_variants, variants[]} — the dictionary entry, separate from the clip
  unit.provenance  {origin, dataset, clip_id, signer, tier, licence, source, qc{…}, fixes{…}} — where the motion comes from;
                   origin ∈ recorded_clip · recorded_phrase · composed (a compound / fixed expression assembled from recorded
                   signs) · fingerspelling (recorded letters) · text (no sign — nothing is shown as a sign)
  unit.sense       how a homograph was resolved (evidence words) when the input word has several meanings
  ir.grammar       {mode: tsl51 | thai, rules} — which ordering produced the gloss list
  ir.versions      lexicon / motion library / grammar rules / timing / avatar rig — content hashes
  ir.nmm           sentence-level non-manual marks actually applied (type, frame range, scope units, evidence)
  ir.input.asr     {raw, corrected, corrections[], model, …} — the recogniser's own text kept next to the corrected one
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

IR_VERSION = "t2s-2"


@dataclass
class SignUnit:
    gloss: str                        # what is shown under the avatar
    kind: str                         # sign · phrase · fingerspell · text (no sign available: shown as a caption, the avatar does not invent)
    concept: str | None = None        # motion-library key (None for text)
    source_words: list = field(default_factory=list)   # input tokens this unit expresses
    match: str = "exact"              # lemma · gloss · alias · group · ttrs_synonym · english · compound · curated · sense · fingerspell · none
    role: str | None = None           # subject · object · verb · time · wh · neg · greeting · ...
    letters: list = field(default_factory=list)        # fingerspelling: the letter concepts, in order
    clip: dict | None = None          # {clip_id, dataset, signer, tier, mirrored, consensus, n_signers}
    manual: dict | None = None        # {dominant, two_handed, location, movement, movement_kind}  (measured on the avatar)
    nmm: dict = field(default_factory=dict)            # {face: "from_clip"} + sentence overlays {brows, head}
    confidence: dict = field(default_factory=dict)     # {translation, motion}
    timing: dict | None = None        # {start_s, end_s} filled by the composer
    note: str | None = None
    lexeme: dict | None = None        # v2: dictionary entry (concept / sense / variant), separate from the clip
    provenance: dict | None = None    # v2: origin of the motion (module docstring)
    sense: dict | None = None         # v2: homograph resolution {word, chosen, evidence, alternatives}


@dataclass
class IR:
    input: dict                       # {text, lang, source: text|speech, asr: {...}}
    thai: str                         # the Thai text that was translated (= input for Thai; MT output for English)
    tokens: list                      # Thai tokens after normalisation
    clauses: list                     # [{type, negated, moves[], units: [index…], nmm}]
    units: list                       # SignUnit list in signing order
    warnings: list = field(default_factory=list)
    confidence: dict = field(default_factory=dict)     # sentence-level {translation, motion, coverage}
    version: str = IR_VERSION
    grammar: dict = field(default_factory=dict)        # {mode, rules}
    versions: dict = field(default_factory=dict)       # content hashes of the data / rules / rig that produced it
    nmm: list = field(default_factory=list)            # sentence-level non-manual marks applied: type, frames, scope, evidence

    def to_dict(self):
        return asdict(self)


def unit_from_dict(d):
    return SignUnit(**{k: v for k, v in d.items() if k in SignUnit.__dataclass_fields__})
