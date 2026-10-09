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
