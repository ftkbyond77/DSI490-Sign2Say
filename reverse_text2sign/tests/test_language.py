"""Language stage: Thai → TSL structure. Run: python -m pytest reverse_text2sign/tests -q"""
import json

import pytest

from t2s import paths
from t2s.text.translate import Mapper, translate_thai


@pytest.fixture(scope="module")
def mapper():
    return Mapper()


def glosses(text, mapper, phrasebook=False):
    ir = translate_thai(text, mapper, phrasebook=phrasebook)
    return [u.gloss for u in ir.units], ir


def test_tsl51_dev_set(mapper):
    """TSL51's 62 Thai sentences with deaf-expert gloss order (the rules were developed on them: a regression guard, not a test set)."""
    pairs = json.loads((paths.PREP / "params" / "tsl51_thai_sentences.json").read_text(encoding="utf-8"))
    ok = sum(glosses(th, mapper)[0] == gold.split() for gold, th in pairs.items())
    assert ok >= 60, f"{ok}/62"


@pytest.mark.parametrize("text,expected,stype", [
    ("วันนี้คุณไปไหน?", ["วันนี้", "คุณ", "ไป", "ที่ไหน"], "wh_question"),          # the user's example
    ("คุณชื่ออะไร", ["คุณ", "ชื่อ", "อะไร"], "wh_question"),
    ("ฉันชอบกินไข่", ["ไข่", "ฉัน", "กิน", "ชอบ"], "statement"),
    ("ผมรักคุณ", ["คุณ", "ฉัน", "รัก"], "statement"),                              # ผม (I) → the pointing sign, not "hair"
    ("ฉันไม่ชอบกินผัก", None, "statement"),
    ("พรุ่งนี้คุณจะไปโรงเรียนไหม", ["พรุ่งนี้", "โรงเรียน", "คุณ", "ไป"], "yn_question"),
    ("ฉันเหนื่อยมาก", ["ฉัน", "เหนื่อย", "มาก"], "statement"),
])
def test_conversation(mapper, text, expected, stype):
    g, ir = glosses(text, mapper)
    if expected:
        assert g == expected
    assert ir.clauses[-1]["type"] == stype


def test_hair_is_not_a_pronoun(mapper):
    g, _ = glosses("ผมยาวมาก", mapper)
    assert "ฉัน" not in g


def test_no_invented_signs(mapper):
    """every sign unit is a recorded dictionary concept; unknown words become fingerspelling (recorded letters) or text."""
    lib = json.loads((paths.MODELS / "motion" / "index.json").read_text(encoding="utf-8"))["clips"]
    for text in ["ผมชื่อสมชาย", "ฉันอยากซื้อเสื้อคลุมสีม่วงอ่อนๆ", "ควอนตัมคอมพิวเตอร์ทำงานอย่างไร", "กขคงจ"]:
        _, ir = glosses(text, mapper)
        for u in ir.units:
            if u.kind == "sign":
                assert u.concept in lib
            elif u.kind == "fingerspell":
                assert u.letters and all(L in lib for L in u.letters)
            else:
                assert u.kind == "text" and u.concept is None


def test_confidences_are_separate(mapper):
    _, ir = glosses("ผมชื่อสมชาย", mapper)
    spelled = [u for u in ir.units if u.kind == "fingerspell"]
    assert spelled and spelled[0].confidence["translation"] < 1.0
    assert ir.warnings


def test_phrasebook(mapper):
    """an exact conversational phrase recorded by a real signer is played as one natural unit (and only when it matches)."""
    _, ir = glosses("คุณกินข้าวหรือยังครับ", mapper, phrasebook=True)
    assert [u.kind for u in ir.units] == ["phrase"]
    _, ir = glosses("คุณกินข้าวหรือยังที่บ้าน", mapper, phrasebook=True)
    assert "phrase" not in [u.kind for u in ir.units]


# ---------------------------------------------------------------- v2
def test_grammar_modes_same_signs(mapper):
    """Normal mode = the same signs as TSL51 mode, only in Thai word order (the IR says which mode produced it)."""
    for text in ["ฉันไม่ชอบกินผัก", "พรุ่งนี้คุณจะไปโรงเรียนไหม", "แม่ของฉันเป็นครู", "คุณชื่ออะไร"]:
        a = translate_thai(text, mapper, grammar="tsl51", phrasebook=False)
        b = translate_thai(text, mapper, grammar="thai")
        assert sorted(u.gloss for u in a.units) == sorted(u.gloss for u in b.units)
        assert [c["type"] for c in a.clauses] == [c["type"] for c in b.clauses]
        assert a.grammar["mode"] == "tsl51" and b.grammar["mode"] == "thai"
    assert [u.gloss for u in translate_thai("ฉันไม่ชอบกินผัก", mapper, grammar="thai").units] == ["ฉัน", "ไม่ชอบ", "กิน", "ผัก"]


def test_homograph_never_borrows_the_other_meaning(mapper):
    """the only recorded ตา is GRANDFATHER: 'my eye hurts' must not sign it (fingerspelled + note), kinship context does."""
    eye = translate_thai("ตาของฉันเจ็บ", mapper)
    u = next(x for x in eye.units if "ตา" in x.source_words)
    assert u.kind == "fingerspell" and u.sense and u.sense["sense"] == "eye"
    kin = translate_thai("ตากับยายไปตลาด", mapper)
    u = next(x for x in kin.units if x.source_words == ["ตา"])
    assert u.kind == "sign" and u.sense["sense"] == "grandfather" and not u.sense["ambiguous"]
    amb = translate_thai("ตาฉันอายุแปดสิบปี", mapper)
    u = next(x for x in amb.units if x.source_words == ["ตา"])
    assert u.sense["ambiguous"] and u.confidence["translation"] < 0.7 and amb.warnings


def test_pronoun_vs_expletive(mapper):
    assert "เขา" in [u.gloss for u in translate_thai("มันชอบนอน", mapper).units]           # it (an animal) likes to sleep
    assert "เขา" not in [u.gloss for u in translate_thai("มันร้อนมาก", mapper).units]       # "it is hot": not signed


def test_names_numbers_time(mapper):
    ir = translate_thai("ครูของฉันชื่อสมศรี", mapper)
    assert any(u.kind == "fingerspell" and u.gloss == "สมศรี" for u in ir.units)
    g = {u.gloss for u in translate_thai("ฉันอายุสามสิบปี", mapper).units}
    assert "สามสิบ" in g or {"สาม", "สิบ"} <= g                           # the dictionaries record 30 as one sign
    first = translate_thai("เมื่อวานฉันไปตลาด", mapper, phrasebook=False).units[0]
    assert first.role == "time" and first.source_words == ["เมื่อวาน"]


def test_provenance_and_versions():
    """every unit says where its motion comes from; the IR carries the versions that produced it (Engine)."""
    from t2s.pipeline import Engine
    e = Engine(english=False)
    ir = e.translate("ผมชื่อจอห์น ฉันชอบกินข้าว")
    origins = {u.provenance["origin"] for u in ir.units}
    assert origins <= {"recorded_clip", "recorded_phrase", "composed", "fingerspelling", "text"}
    for u in ir.units:
        if u.provenance["origin"] in ("recorded_clip", "recorded_phrase", "composed"):
            p = u.provenance
            assert p["dataset"] and p["clip_id"] and p["signer"] and p["tier"] and p["licence"] and u.lexeme["concept_id"]
    assert ir.version == "t2s-2" and ir.versions["lexicon"] and ir.versions["motion_index"] and ir.versions["grammar_rules"]


def test_v1_ir_still_renders():
    """backward compatibility: an IR saved by v1 (no lexeme / provenance / grammar fields) re-renders."""
    from t2s.pipeline import Engine
    e = Engine(english=False)
    v1 = {"input": {"text": "คุณชื่ออะไร", "lang": "th", "source": "text"}, "thai": "คุณชื่ออะไร", "tokens": ["คุณ", "ชื่อ", "อะไร"],
          "clauses": [{"text": "คุณชื่ออะไร", "type": "wh_question", "negated": False, "moves": [], "units": [0, 1, 2]}],
          "units": [{"gloss": g, "kind": "sign", "concept": g, "source_words": [g], "match": "lemma", "role": None, "letters": [],
                     "clip": None, "manual": None, "nmm": {}, "confidence": {"translation": 1.0}, "timing": None, "note": None}
                    for g in ("คุณ", "ชื่อ", "อะไร")],
          "warnings": [], "confidence": {}, "version": "t2s-1"}
    ir, mo = e.render_ir_dict(v1)
    assert mo["frames"] > 25 and all(u.provenance for u in ir.units)
