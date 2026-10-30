"""Speech v2 plan (modules/speech.plan_speech): natural disfluency for signs the model could not read — never a guessed word, one
hesitation per run of unread signs, silence after an unread last sign, transitions are not hesitations. Pure Python (no TTS)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.speech import FILLERS, plan_speech, render  # noqa: E402


def seg(word, status, t0, t1, p=0.95):
    return dict(nearest=word, status=status, t0=t0, t1=t1, f0=int(t0 * 25), f1=int(t1 * 25), p_correct=p)


def result(*segs):
    return dict(segments=list(segs), fusion=dict(provider="rules"), sentence="")


class PlanTests(unittest.TestCase):
    def test_gap_in_the_middle_becomes_one_hesitation(self):
        r = result(seg("ฉัน", "accepted", 0, .6), seg("รถไฟ", "unknown", .6, 1.4, .1), seg("พ่อ", "accepted", 1.4, 2.0))
        p = plan_speech(r)
        self.assertEqual([u["kind"] for u in p.units], ["say", "hesitate", "say"])
        self.assertEqual(p.text, "ฉัน พ่อ")
        self.assertNotIn("รถไฟ", " ".join(u.get("text", "") for u in p.units))      # the [?] segment's nearest word is never said

    def test_neighbouring_unknowns_merge(self):
        r = result(seg("ฉัน", "accepted", 0, .5), seg("ก", "unknown", .5, 1.0, .1), seg("ข", "unknown", 1.0, 1.6, .1), seg("กิน", "accepted", 1.6, 2.1))
        p = plan_speech(r)
        self.assertEqual(sum(u["kind"] == "hesitate" for u in p.units), 1)

    def test_unknown_at_the_end_is_silence(self):
        p = plan_speech(result(seg("ฉัน", "accepted", 0, .5), seg("x", "unknown", .5, 1.2, .1)))
        self.assertEqual(p.units, [])                                                   # nothing to hesitate for → plain sentence
        self.assertEqual(p.text, "ฉัน")

    def test_unknown_at_the_start_opens_with_uh(self):
        p = plan_speech(result(seg("x", "unknown", 0, .8, .1), seg("ฉัน", "accepted", .8, 1.3)))
        self.assertEqual(p.units[0]["kind"], "hesitate")
        self.assertEqual(p.units[0]["filler"], "uh")

    def test_transition_is_not_a_hesitation(self):
        r = result(seg("ฉัน", "accepted", 0, .5), seg("·", "transition", .5, .8), seg("พ่อ", "accepted", .8, 1.2))
        self.assertEqual(plan_speech(r).units, [])

    def test_fillers_are_not_content_words(self):
        r = result(seg("ฉัน", "accepted", 0, .5), seg("x", "unknown", .5, 2.5, .1), seg("y", "unknown", 2.5, 3.5, .1), seg("พ่อ", "accepted", 3.5, 4.0))
        p = plan_speech(r)
        h = [u for u in p.units if u["kind"] == "hesitate"][0]
        self.assertIn(h["filler"], FILLERS)
        self.assertLessEqual(h["pause_s"], 1.4)

    def test_uncertain_not_spoken_is_a_hesitation_and_spoken_when_asked(self):
        r = result(seg("ฉัน", "accepted", 0, .5), seg("รัก", "uncertain", .5, 1.0, .6), seg("เพื่อน", "accepted", 1.0, 1.5))
        self.assertEqual([u["kind"] for u in plan_speech(r).units], ["say", "hesitate", "say"])
        p = plan_speech(r, speak_uncertain=True)
        self.assertEqual(p.units, []); self.assertEqual(p.text, "ฉัน รัก เพื่อน")

    def test_skip_mode_is_v1(self):
        r = result(seg("ฉัน", "accepted", 0, .6), seg("x", "unknown", .6, 1.4, .1), seg("พ่อ", "accepted", 1.4, 2.0))
        p = plan_speech(r, disfluency="skip")
        self.assertEqual(p.units, []); self.assertEqual(p.text, "ฉัน พ่อ")

    def test_pause_mode_has_no_filler(self):
        r = result(seg("ฉัน", "accepted", 0, .6), seg("x", "unknown", .6, 1.4, .1), seg("พ่อ", "accepted", 1.4, 2.0))
        h = [u for u in plan_speech(r, disfluency="pause").units if u["kind"] == "hesitate"][0]
        self.assertIsNone(h["filler"])

    def test_render_order_and_length(self):
        r = result(seg("ฉัน", "accepted", 0, .6), seg("x", "unknown", .6, 1.4, .1), seg("พ่อ", "accepted", 1.4, 2.0))
        p = plan_speech(r)
        said = []

        class F:
            def get(self, voice, name):
                return np.ones(100, np.float32) * 0.1
        y, sr = render(p, lambda t: (said.append(t) or np.ones(200, np.float32) * 0.2, 1000), F(), 1000)
        self.assertEqual(said, ["ฉัน", "พ่อ"])
        h = [u for u in p.units if u["kind"] == "hesitate"][0]
        self.assertEqual(len(y), 200 + 100 + int(1000 * h["pause_s"]) + 200)


if __name__ == "__main__":
    unittest.main()
