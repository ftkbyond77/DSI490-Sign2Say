"""Face channel (v7): non-manual markers + emotion from MediaPipe face blendshapes and face landmarks.

Thai Sign Language carries grammar on the face, not only on the hands:
  * yes/no question  — brows raised (often with the head pushed forward) at the end of the sentence; the manual sign ไหม is optional
  * wh-question      — brows lowered / drawn together while signing อะไร / ที่ไหน / ทำไม …
  * negation         — head shake (with or without the sign ไม่)
  * affirmation      — head nod
  * emotion / degree — smile, frown, puffed cheeks ("มาก"), squint …  → tone of the spoken sentence
The extractor (modules/mediapipe_pose.py, and the same model in the browser) gives 52 blendshape scores per frame — a
person-independent, already-normalised expression code — plus 68 face points for head pose. Every cue is measured relative to the
signer's own baseline in the same utterance (median over the clip), because resting faces differ from person to person.

Output (per utterance):
  dict(question_yesno, question_wh, negation, affirmation, emotion, emotion_th, emotion_scores, intensity,
       cues = per-cue strength, per_segment = brow / mouth / smile per sign segment)
"""
from __future__ import annotations

import numpy as np

from .mediapipe_pose import BLENDSHAPES

B = {n: i for i, n in enumerate(BLENDSHAPES)}
EMOTIONS = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]
EMO_TH = {"neutral": "เป็นกลาง", "happy": "ดีใจ/มีความสุข", "sad": "เศร้า", "angry": "โกรธ", "surprise": "ประหลาดใจ", "fear": "กลัว",
          "disgust": "รังเกียจ"}
# emotion score reference on neutral dictionary faces: (median, report threshold = max(97th percentile, floor), 99th percentile)
EMO_REF = {"happy": (0.073, 0.545, 0.696), "sad": (0.003, 0.05, 0.037), "angry": (0.108, 0.365, 0.485), "surprise": (0.0, 0.15, 0.073),
           "fear": (0.005, 0.05, 0.011), "disgust": (0.0, 0.15, 0.0)}
# question furrow (experimental): end-of-utterance browDown minus the signer's resting face (hands down). Threshold = 97th percentile
# of the same measure on 793 neutral dictionary clips (TTRS + TSL51 researcher). On 54 phrase clips of one unseen signer it fired on
# 0/8 yes/no questions and 1/25 statements (that signer's brows relax in questions), so it is OFF unless the user turns it on.
FURROW_Q = 0.18
# iBUG-68 indices inside the stored face array
EYE_R, EYE_L, NOSE_TIP, CHIN = slice(36, 42), slice(42, 48), 30, 8


def _bs(bs, *names):
    return np.mean([bs[:, B[n]] for n in names], 0)


def _z(x, ok):
    if ok.sum() < 3:
        return np.zeros_like(x)
    base = np.median(x[ok]); mad = np.median(np.abs(x[ok] - base)) * 1.4826 + 0.02
    return (x - base) / mad


def _cycles(sig, amp):
    """Number of back-and-forth swings larger than `amp` (head shake / nod counter)."""
    if len(sig) < 5:
        return 0
    s = sig - np.convolve(sig, np.ones(9) / 9, "same")
    big = np.abs(s) > amp
    sg = np.sign(s[big])
    return int((np.diff(sg) != 0).sum() // 2) if len(sg) > 1 else 0


def head_pose(face, ok):
    """Yaw / pitch / roll proxies from 68 face points (x/W, y/H already isotropic-corrected by the caller)."""
    er, el = face[:, EYE_R].mean(1), face[:, EYE_L].mean(1)
    iod = np.linalg.norm(el - er, axis=1) + 1e-4
    mid = (er + el) / 2
    nose = face[:, NOSE_TIP]
    yaw = (nose[:, 0] - mid[:, 0]) / iod
    pitch = (nose[:, 1] - mid[:, 1]) / iod
    roll = np.arctan2(el[:, 1] - er[:, 1], el[:, 0] - er[:, 0])
    for v in (yaw, pitch, roll):
        if ok.any():
            v[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), v[ok])
    return yaw, pitch, roll, iod


def hands_up(raw):
    """Frames where a detected hand is above the lower chest (same rule as the web client's live mode)."""
    pose, T = raw.get("pose"), len(raw["bs"])
    if pose is None or len(pose) != T:
        return np.zeros(T, bool)
    ls, rs = pose[:, 11, :2], pose[:, 12, :2]
    sw = np.linalg.norm(ls - rs, axis=1) + 1e-3
    mid = (ls[:, 1] + rs[:, 1]) / 2
    up = np.zeros(T, bool)
    for k in ("lh", "rh"):
        h = raw.get(k)
        if h is not None and len(h) == T:
            up |= (np.abs(h).sum((1, 2)) > 0) & (h[:, :, 1].mean(1) < mid + 1.5 * sw)
    return up


def analyse(raw, spans=None, fps=25.0, baseline_bs=None):
    """raw = MediaPipe raw dict (modules.mediapipe_pose): uses bs [T,52], face [T,68,2], hw, t. spans = [(f0,f1)] sign segments
    in 25 fps frames of the resampled sequence (mapped onto raw timestamps here). baseline_bs = the signer's resting-face
    blendshapes (52), measured by the client while the hands are down; otherwise the hands-down frames of the clip are used."""
    bs, face, t = raw["bs"].astype(np.float32), raw["face"].astype(np.float32).copy(), np.asarray(raw["t"], np.float64)
    H, W = (int(raw["hw"][0]), int(raw["hw"][1])) if len(raw["hw"]) == 2 else (1, 1)
    face[..., 0] *= W / max(H, 1)
    ok = (bs.sum(1) > 0) & (np.abs(face).sum((1, 2)) > 0)
    T = len(bs)
    out = dict(face_frames=float(ok.mean()) if T else 0.0)
    if ok.sum() < 5:
        out.update(question_yesno=False, question_wh=False, question_furrow=False, negation=False, affirmation=False, emotion="neutral", emotion_th=EMO_TH["neutral"],
                   emotion_scores={}, intensity=0.0, cues={}, per_segment=[])
        return out
    brow_q = _bs(bs, "browOuterUpLeft", "browOuterUpRight")               # question brows: the OUTER brows rise (inner raise alone = sadness)
    brow_up = _bs(bs, "browInnerUp", "browOuterUpLeft", "browOuterUpRight") # any brow raise (emotion scores, calibrated with this definition)
    brow_down = _bs(bs, "browDownLeft", "browDownRight")
    smile = _bs(bs, "mouthSmileLeft", "mouthSmileRight")
    frown = _bs(bs, "mouthFrownLeft", "mouthFrownRight")
    jaw = bs[:, B["jawOpen"]]
    eye_wide = _bs(bs, "eyeWideLeft", "eyeWideRight")
    squint = _bs(bs, "eyeSquintLeft", "eyeSquintRight", "cheekSquintLeft", "cheekSquintRight")
    sneer = _bs(bs, "noseSneerLeft", "noseSneerRight")
    press = _bs(bs, "mouthPressLeft", "mouthPressRight")
    puff = bs[:, B["cheekPuff"]]
    okf = np.abs(face).sum((1, 2)) > 0               # head pose from every face frame (the web client may send blendshapes only every few frames)
    yaw, pitch, roll, iod = head_pose(face, okf if okf.sum() >= 5 else ok)
    zu, zd = _z(brow_q, ok), _z(brow_down, ok)
    tail = slice(int(T * 0.6), T)
    dur = float(t[-1] - t[0]) if T > 1 else 0.0
    # cues (strength in baseline-MAD units / swing counts)
    cues = dict(brow_raise_tail=float(np.median(zu[tail][ok[tail]])) if ok[tail].any() else 0.0,
                brow_raise_peak=float(np.percentile(brow_q[ok], 90) - np.median(brow_q[ok])),
                brow_furrow=float(np.percentile(zd[ok], 90)),
                head_forward_tail=float(np.median(pitch[tail]) - np.median(pitch)) if T else 0.0,
                head_shake_cycles=_cycles(yaw, 0.06), head_nod_cycles=_cycles(pitch, 0.05),
                mouth_open_frac=float((jaw[ok] > 0.35).mean()), cheek_puff=float(np.percentile(puff[ok], 95)), duration_s=round(dur, 2))
    # furrow relative to the resting face, over the last 40 % of the signing frames
    up = hands_up(raw) & ok
    idx = np.flatnonzero(up) if up.sum() >= 6 else np.flatnonzero(ok)
    sign_tail = idx[int(len(idx) * 0.6):]
    rest = (~hands_up(raw)) & ok
    if baseline_bs is not None and len(baseline_bs) == bs.shape[1]:
        base_down = float(np.mean([baseline_bs[B["browDownLeft"]], baseline_bs[B["browDownRight"]]]))
        cues_base = "client"
    elif rest.sum() >= 5:
        base_down, cues_base = float(np.median(brow_down[rest])), "hands_down_frames"
    else:
        base_down, cues_base = float(np.percentile(brow_down[ok], 20)), "clip_20th_percentile"
    cues["brow_furrow_rel"] = float(np.median(brow_down[sign_tail]) - base_down) if len(sign_tail) else 0.0
    cues["furrow_baseline"] = cues_base
    out["question_furrow"] = bool(cues["brow_furrow_rel"] > FURROW_Q)
    out["question_yesno"] = bool(cues["brow_raise_tail"] > 1.0 and cues["brow_raise_peak"] > 0.08)
    out["question_wh"] = bool(cues["brow_furrow"] > 2.0 and not out["question_yesno"])
    out["negation"] = bool(cues["head_shake_cycles"] >= 2)
    out["affirmation"] = bool(cues["head_nod_cycles"] >= 2 and not out["negation"])
    # emotion from absolute blendshape levels (they are already person-normalised by the face model)
    q = lambda x: float(np.percentile(x[ok], 75))      # noqa: E731
    sc = dict(happy=q(smile) * 1.2 + 0.3 * q(squint), sad=q(frown) * 0.9 + 0.5 * q(brow_up) * q(frown) * 4, angry=q(brow_down) * 0.8 + 0.6 * q(press),
              surprise=0.6 * q(brow_up) + 0.6 * q(jaw) + 0.8 * q(eye_wide) - 0.2, fear=0.5 * q(eye_wide) + 0.4 * q(brow_up) * q(eye_wide) * 3,
              disgust=1.2 * q(sneer))
    sc = {k: max(0.0, v) for k, v in sc.items()}
    # calibrated on 580 dictionary clips (TTRS / TSL51 researcher — neutral faces): an emotion is reported only above the 97th
    # percentile of its neutral-face score (floors where neutral faces never show it); intensity spans median → 99th percentile
    ratio = {k: v / EMO_REF[k][1] for k, v in sc.items()}
    best = max(ratio, key=ratio.get)
    if out["question_furrow"] and best == "angry":       # a question furrow is grammar, not anger → do not speak angrily
        ratio["angry"] = 0.0; best = max(ratio, key=ratio.get)
    emo = best if ratio[best] >= 1.0 else "neutral"
    lo, hi = EMO_REF[best][0], max(EMO_REF[best][2], 1.5 * EMO_REF[best][1])
    inten = float(np.clip((sc[best] - lo) / max(hi - lo, 1e-6), 0, 1)) if emo != "neutral" else 0.0
    out.update(emotion=emo, emotion_th=EMO_TH[emo], emotion_scores={k: round(v, 3) for k, v in sc.items()}, intensity=round(inten, 3),
               cues={k: (round(v, 3) if isinstance(v, float) else v) for k, v in cues.items()})
    if spans:
        tt = t - t[0]
        per = []
        for f0, f1 in spans:
            m = (tt >= f0 / fps) & (tt < max(f1, f0 + 1) / fps) & ok
            per.append(dict(brow=round(float(np.median(zu[m])), 2) if m.any() else 0.0, furrow=round(float(np.median(zd[m])), 2) if m.any() else 0.0,
                            smile=round(float(np.median(smile[m])), 3) if m.any() else 0.0, mouth=round(float(np.median(jaw[m])), 3) if m.any() else 0.0))
        out["per_segment"] = per
    return out
