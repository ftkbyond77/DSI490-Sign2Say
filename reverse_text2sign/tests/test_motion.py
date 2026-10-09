"""Motion stage: IR → continuous timeline; export."""
import json
import struct

import numpy as np
import pytest

from t2s.motion.codec import decode
from t2s.motion.compose import rest_pose
from t2s.pipeline import Engine


@pytest.fixture(scope="module")
def engine():
    return Engine(english=False)


@pytest.mark.parametrize("text", ["สวัสดีครับ ผมชื่อสมชาย", "คุณชื่ออะไร", "วันนี้คุณไปไหน", "ฉันไม่ชอบกินผัก"])
def test_timeline_is_valid(engine, text):
    ir, mo = engine.run(text)
    q, m = decode(mo)
    assert np.isfinite(q).all() and np.isfinite(m).all()
    assert np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=2e-3)
    assert (m >= 0).all() and (m <= 1).all()
    rest = rest_pose()
    for f in (0, -1):                                           # starts and ends at rest
        assert np.abs(np.abs((q[f] * rest).sum(-1)) - 1).max() < 2e-3
    # no pops: body / arm / wrist bones move < 35° per frame (25 fps); fingers may open fast (a real hand opens in ~0.15 s) but must
    # not flicker (a jump that is undone in the next frame)
    ang = np.degrees(2 * np.arccos(np.clip(np.abs((q[1:] * q[:-1]).sum(-1)), 0, 1)))
    big = [i for i, b in enumerate(mo["bones"]) if not any(f in b for f in ("Thumb", "Index", "Middle", "Ring", "Little"))]
    assert ang[:, big].max() < 35, (ang[:, big].max(), mo["bones"][big[int(ang[:, big].max(0).argmax())]])
    there_and_back = np.degrees(2 * np.arccos(np.clip(np.abs((q[2:] * q[:-2]).sum(-1)), 0, 1)))
    steady_before = np.vstack([np.zeros((1, ang.shape[1])), ang[:-2]]) < 10      # the frame before the jump was still
    steady_after = np.vstack([ang[2:], np.zeros((1, ang.shape[1]))]) < 10         # and the frame after the return is still
    flicker = (ang[:-1] > 25) & (ang[1:] > 25) & (there_and_back < 10) & steady_before & steady_after
    assert flicker.sum() == 0
    # every unit has a timing and segments cover the units in order
    starts = [u.timing["start_s"] for u in ir.units]
    assert starts == sorted(starts)


def test_speed(engine):
    _, slow = engine.run("คุณชื่ออะไร", speed=0.75)
    _, fast = engine.run("คุณชื่ออะไร", speed=1.5)
    assert slow["frames"] > fast["frames"]


def test_vrma_export(engine):
    from t2s.motion.vrma import to_vrma
    _, mo = engine.run("ขอบคุณ")
    b = to_vrma(mo)
    magic, ver, n = struct.unpack("<4sII", b[:12])
    assert magic == b"glTF" and ver == 2 and n == len(b)
    jl = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + jl])
    assert "VRMC_vrm_animation" in j["extensions"]
    assert len(j["animations"][0]["channels"]) >= 45


def _bodies():
    from t2s import avatar
    return [p for p in sorted(avatar.SRC_DIR.glob("*.vrm"))]


@pytest.mark.parametrize("vrm", _bodies(), ids=lambda p: p.name)
def test_fit_to_character(engine, vrm):
    """every character in vrm/ signs at the same places as the library avatar: wrists follow the face / chest landmarks (scaled to
    the body), hands keep their orientation, and fitting the library avatar itself changes nothing."""
    from t2s.motion.fit import Fit
    from t2s.motion.rig import Rig
    ref, body = Rig(), Rig(vrm)
    fit = Fit(body, ref)
    q = np.concatenate([decode(engine.run(t)[1])[0] for t in ("คุณชื่ออะไร", "ฉันหิวข้าว", "ขอบคุณมากครับ")])
    q2 = fit(q)
    assert np.isfinite(q2).all() and np.allclose(np.linalg.norm(q2, axis=-1), 1.0, atol=1e-3)
    if fit.identity:
        assert np.array_equal(q2, q)
        return
    names = ["head", "leftHand", "rightHand"]
    fr, raw, fb = ref.fk(q, names), body.fk(q, names), body.fk(q2, names)
    for s in ("left", "right"):
        d = lambda F: np.linalg.norm(F[f"{s}Hand"][0] - F["head"][0], axis=-1)       # noqa: E731
        before = np.abs(d(raw) / fit.scale - d(fr)).mean()
        after = np.abs(d(fb) / fit.scale - d(fr)).mean()
        assert after < 0.02 and after <= before, (s, before, after)                  # metres, library scale
        ang = np.degrees((fb[f"{s}Hand"][1] * fr[f"{s}Hand"][1].inv()).magnitude())
        assert ang.max() < 0.5
