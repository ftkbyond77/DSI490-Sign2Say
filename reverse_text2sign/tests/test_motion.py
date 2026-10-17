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
    fr, fb = ref.fk(q, names), body.fk(q2, names)
    before, after = fit.placement_error(q, fit.fold(np.array(q, np.float64))), fit.placement_error(q, q2)
    for s in ("left", "right"):
        # the hand lands where the library avatar puts it relative to the nearest body part (scaled to this body's proportions)
        # metres, library scale: at most half the error of copying the rotations unchanged, and < 5 cm on average
        assert after[s][0].mean() < 0.05 and after[s][0].mean() <= 0.5 * before[s][0].mean(), (s, before[s][0].mean(), after[s][0].mean())
        ang = np.degrees((fb[f"{s}Hand"][1] * fr[f"{s}Hand"][1].inv()).magnitude())
        assert ang.max() < 0.5


# ---------------------------------------------------------------- v2
def test_strokes_are_not_changed(engine):
    """transitions / tempo may change between signs, never inside a stroke: arm and finger rotations of every stroke frame equal
    the library clip (retimed, held ends possibly shortened); only head / neck (sentence non-manuals) may differ."""
    from t2s.motion.compose import retime, timing
    from t2s.motion.rig import BONE_INDEX
    keep = [i for b, i in BONE_INDEX.items() if b not in ("head", "neck")]
    ir, mo = engine.run("ฉันชอบกินไข่", nmm=False, phrasebook=False)
    q, _ = decode(mo)
    for s in mo["segments"]:
        if s["kind"] != "sign":
            continue
        u = ir.units[s["unit"]]
        lq, lm, info = engine.lib.get(u.concept)
        a, b = info["stroke"]
        ref, _ = retime(lq[a:b], lm[a:b], timing().get("stroke_speed", 1.0))
        got = q[s["start"]:s["end"]]
        best = 1.0
        for off in range(0, max(1, len(ref) - len(got) + 1)):            # the held start may have been shortened (hold cap)
            n = min(len(got), len(ref) - off)
            d = np.abs(np.abs((got[:n, keep] * ref[off:off + n, keep]).sum(-1)) - 1)
            best = min(best, float(np.median(d)))
        assert best < 2e-3, (u.gloss, best)


def test_bone_lengths_and_finite(engine):
    from t2s.motion.rig import get_rig
    rig = get_rig()
    _, mo = engine.run("วันนี้คุณไปไหน")
    q, _ = decode(mo)
    fk = rig.fk(q, ["leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm", "rightLowerArm", "rightHand"])
    for s in ("left", "right"):
        for a, b in (("UpperArm", "LowerArm"), ("LowerArm", "Hand")):
            ln = np.linalg.norm(fk[f"{s}{b}"][0] - fk[f"{s}{a}"][0], axis=1)
            assert np.ptp(ln) < 1e-4 and np.isfinite(ln).all()


def test_wh_brows_scoped_not_whole_sentence(engine):
    """the WH brow raise starts at the 2nd sign (TSL51 data), never on the first sign of the clause."""
    from t2s.motion.retarget import MORPHS
    ir, mo = engine.run("วันนี้คุณไปไหน", phrasebook=False)
    _, m = decode(mo)
    bi = MORPHS.index("Fcl_BRW_Surprised")
    first = next(s for s in mo["segments"] if s["kind"] == "sign")
    mk = next(x for x in ir.nmm if x["type"] == "wh_question")
    assert mk["frames"][0] > first["start"] and first["unit"] not in mk["units"]
    _, mo2 = engine.run("วันนี้คุณไปไหน", phrasebook=False, nmm=False)
    _, m2 = decode(mo2)
    assert np.allclose(m[first["start"]:first["start"] + 3, bi], m2[first["start"]:first["start"] + 3, bi], atol=0.02)


def test_negation_headshake_only_on_negation(engine):
    ir, mo = engine.run("ฉันไม่ชอบกินผัก", phrasebook=False)
    neg = [x for x in ir.nmm if x["type"] == "negation"]
    assert neg and all(ir.units[i].gloss.startswith("ไม่") for x in neg for i in x["units"])


def test_palm_contact_wai(engine):
    """ขอบคุณ: the palms meet facing each other (v1: crossed) and do not pass through each other."""
    from t2s.motion.rig import get_rig
    rig = get_rig()
    q, _, info = engine.lib.get("ขอบคุณ")
    assert info["fixes"]["contact_mirrored"] > 0
    names = [f"{s}{b}" for s in ("left", "right") for b in ("Hand", "IndexProximal", "LittleProximal")]
    fk = rig.fk(q, names)
    n = {}
    for s in ("left", "right"):
        w, i, l = (fk[f"{s}{b}"][0] for b in ("Hand", "IndexProximal", "LittleProximal"))
        v = np.cross(i - w, l - w)
        n[s] = v / np.linalg.norm(v, axis=1, keepdims=True)
    d = np.linalg.norm(fk["leftHand"][0] - fk["rightHand"][0], axis=1)
    c = d < 0.09
    assert c.sum() >= 5
    ang = np.degrees(np.arccos(np.clip((n["left"][c] * n["right"][c]).sum(1), -1, 1)))
    assert np.median(ang) < 25 and d[c].min() > 0.035


def test_face_contact_depth(engine):
    """face-located signs reach the face (v1: ~20 cm in front of it): ห้องน้ำ pinches at the nose."""
    from t2s.motion.retarget import AV_NOSE, HAND_POINTS
    from t2s.motion.rig import get_rig
    rig = get_rig()
    q, _, info = engine.lib.get("ห้องน้ำ")
    fk = rig.fk(q, ["head"] + [f"right{p}" for p in HAND_POINTS])
    hp, hr = fk["head"]
    L = np.stack([hr.inv().apply(fk[f"right{p}"][0] - hp) for p in HAND_POINTS], 1)
    zf = (AV_NOSE - rig.rest["head"])[2]
    inside = (np.abs(L[..., 0]) < 0.07) & (L[..., 1] > -0.05) & (L[..., 1] < 0.13)
    gap = np.where(inside, zf - L[..., 2], np.inf).min()
    assert -0.02 < gap < 0.08, gap
