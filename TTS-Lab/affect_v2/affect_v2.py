"""affect_v2: decide which emotion (if any) the voice should carry. Drop-in for modules.face_fer.speech_affect.

    emo, intensity, why = speech_affect_v2(fer_summary, body_affect, grammar=face, segs=out.get("segments"))

What changes against v1 (one global rule: non-neutral and p >= 0.5):
 1. per-emotion thresholds  - the face model is not equally reliable per class (Face-Lab val precision: happy .91, surprise .77,
                              disgust .77, angry .61, sad .56, fear .52), so weak classes need more evidence (selective prediction)
 2. agreement               - the two face branches agreeing, or a signed emotion word agreeing, lowers the threshold;
                              a margin over the runner-up is required (no coin-flip decisions)
 3. grammar                 - brow raise of a yes/no question is not surprise (as v1); a wh-question furrow is not anger / disgust
 4. signed words            - lexicon rebuilt from the 328 conversation concepts (v1: 26 words, many not in the vocabulary);
                              a clear emotion word can voice a mild emotion on its own when the face does not contradict it
 5. voice policy            - only emotions whose voice has been listened to are voiced; the rest stay neutral (CFG["voiced"])
 6. signing energy          - angry and sad are both negative and the face model confuses them; fast / strong signing -> angry,
                              slow / soft -> sad (valence from the face, arousal from the hands)
 7. persistence             - an emotion is on the face for most of the utterance; a grammatical marker sits on one or two signs.
                              With per-sign face summaries, an expression seen on fewer than half of the signs needs more evidence
                              (and is dropped for the grammar-prone classes surprise / angry / disgust)
 8. memory                  - the emotion voiced in the previous utterance (last 20 s) lowers the threshold for the same emotion
All numbers live in CFG: v0 values are set from Face-Lab validation precision, NOT tuned on labelled signing (none exists yet).
Pure Python, no dependencies.
"""
CLASSES = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
ALIAS = {"fearful": "fear", "disgusted": "disgust", "surprised": "surprise", "joy": "happy"}
VALENCE = {"neutral": 0, "happy": 1, "surprise": 0, "sad": -1, "angry": -1, "fear": -1, "disgust": -1}

CFG = dict(
    w_fer=0.5,                                   # face picture vs blendshape branch (as v1)
    thr=dict(happy=0.50, surprise=0.60, disgust=0.65, angry=0.60, sad=0.60, fear=0.65),
    agree_bonus=0.10,                            # both face branches name the same emotion
    word_bonus=0.15,                             # a signed word names the same emotion
    margin=0.12,                                 # lead over the best other non-neutral emotion
    min_valid_frac=0.25,                         # as v1
    word_p_min=0.5,                              # recognition confidence of a word to count
    word_only_min=1.0, word_only_intensity=0.30, # words alone: one clear emotion word -> mild emotion
    contradict=0.50,                             # face shows the opposite valence this strongly -> words alone do not decide
    energy_hi=0.30, energy_lo=-0.30, pair_min=0.30,   # signing energy in [-1, 1] (face_model: wrist speed); angry/sad tie-break
    persist_min=0.50, persist_penalty=0.15, persist_block=("surprise", "angry", "disgust"),
    memory_s=20.0, memory_bonus=0.08,
    voiced=("happy", "sad", "angry"),            # emotions whose voice was listened to; fear / disgust / surprise pending round 3
    intensity=(0.25, 0.75),
)

# (emotion, weight): 1.0 = the word names a feeling; 0.5 = evaluative / situational. Only concepts of models/model-v10/vocab_conversation.json.
LEX = {
    "ดีใจ": ("happy", 1.0), "มีความสุข": ("happy", 1.0), "สนุก": ("happy", 1.0), "หัวเราะ": ("happy", 1.0), "ยิ้ม": ("happy", 1.0),
    "รัก": ("happy", 1.0), "ภูมิใจ": ("happy", 1.0), "ยินดี": ("happy", 1.0), "ยินดีด้วย": ("happy", 1.0), "ขอบคุณ": ("happy", 0.5),
    "ชอบ": ("happy", 0.5), "ความรัก": ("happy", 0.5), "โชคดี": ("happy", 0.5), "น่ารัก": ("happy", 0.5), "อร่อย": ("happy", 0.5),
    "สบายดี": ("happy", 0.5), "ปรบมือ": ("happy", 0.5), "เก่ง": ("happy", 0.5), "ใจดี": ("happy", 0.5), "กอด": ("happy", 0.5),
    "เสียใจ": ("sad", 1.0), "เศร้า": ("sad", 1.0), "ร้องไห้": ("sad", 1.0), "เหงา": ("sad", 1.0), "คิดถึง": ("sad", 1.0),
    "สงสาร": ("sad", 1.0), "ขอโทษ": ("sad", 0.5), "เหนื่อย": ("sad", 0.5), "เจ็บ": ("sad", 0.5), "ป่วย": ("sad", 0.5),
    "ปวดหัว": ("sad", 0.5), "เบื่อ": ("sad", 0.5),
    "โกรธ": ("angry", 1.0), "เกลียด": ("angry", 1.0), "รำคาญ": ("angry", 1.0), "ทะเลาะ": ("angry", 0.5),
    "กลัว": ("fear", 1.0), "กังวล": ("fear", 1.0), "ห่วง": ("fear", 0.5),
    "ตกใจ": ("surprise", 1.0), "สกปรก": ("disgust", 0.5),
}
NEG = {"ไม่", "ไม่ได้", "ไม่ใช่", "อย่า", "ไม่มี"}


def _norm(p):
    q = {c: 0.0 for c in CLASSES}
    for k, v in (p or {}).items():
        k = ALIAS.get(str(k).lower(), str(k).lower())
        if k in q:
            q[k] += float(v or 0)
    s = sum(q.values())
    return {c: v / s for c, v in q.items()} if s > 0 else None


def word_evidence(segs, grammar=None, cfg=CFG):
    """Signed emotion words -> {emotion: weight}. A word right after a negation word (or under a head-shake) is dropped, not flipped."""
    ev, used, prev_neg = {}, [], False
    neg_all = bool((grammar or {}).get("negation"))
    for s in segs or []:
        if s.get("status") == "transition":
            continue
        w = s.get("word") or s.get("nearest") or ""
        p = s.get("p") if s.get("p") is not None else s.get("p_correct", 0)
        ok = s.get("status") in ("accepted", "uncertain", None) and (p or 0) >= cfg["word_p_min"]
        if ok and w in LEX and not prev_neg and not neg_all:
            e, wt = LEX[w]; ev[e] = ev.get(e, 0.0) + wt; used.append(w)
        prev_neg = ok and w in NEG
    return ev, used


def speech_affect_v2(fer_summary=None, body_affect=None, grammar=None, segs=None, cfg=CFG, fer_segments=None, state=None, now=None):
    """-> (emotion | None, intensity | None, why). None = speak neutrally.
    fer_segments = face_fer `per_segment` summaries (one per sign); state = a dict kept by the caller between utterances (memory)."""
    import time as _time
    now = _time.time() if now is None else now
    emo, inten, why = _decide(fer_summary, body_affect, grammar, segs, cfg, fer_segments, state, now)
    if state is not None and emo:
        state.update(emo=emo, t=now)
    return emo, inten, why


def _decide(fer_summary, body_affect, grammar, segs, cfg, fer_segments, state, now):
    g = grammar or {}
    energy = float(((body_affect or {}).get("sources") or {}).get("energy") or 0.0)
    prev = state.get("emo") if state and now - state.get("t", -1e9) <= cfg["memory_s"] else None
    fer = _norm(fer_summary.get("probs")) if (fer_summary and fer_summary.get("probs") and fer_summary.get("valid_frac", 0) >= cfg["min_valid_frac"]
                                               and fer_summary.get("expression") != "unknown") else None
    body = _norm(body_affect.get("probs")) if body_affect and body_affect.get("probs") else None
    words, used = word_evidence(segs, g, cfg)
    wtop = max(words, key=words.get) if words else None

    def grammar_fix(p):
        p = dict(p)
        if g.get("question_yesno"):                               # raised brows of a yes/no question
            p["neutral"] += 0.5 * p["surprise"]; p["surprise"] *= 0.5
        if g.get("question_furrow") or g.get("question_wh"):      # furrowed brows of a wh-question
            for c in ("angry", "disgust"):
                p["neutral"] += 0.5 * p[c]; p[c] *= 0.5
        return p

    lo, hi = cfg["intensity"]
    if fer or body:
        fer_g, body_g = (grammar_fix(fer) if fer else None), (grammar_fix(body) if body else None)
        w = cfg["w_fer"] if (fer_g and body_g) else (1.0 if fer_g else 0.0)
        p = {c: w * (fer_g or {}).get(c, 0) + (1 - w) * (body_g or {}).get(c, 0) for c in CLASSES}
        emo = max((c for c in CLASSES if c != "neutral"), key=p.get)
        notes = []
        if emo in ("angry", "sad") and min(p["angry"], p["sad"]) >= 0.5 * max(p["angry"], p["sad"]) and p["angry"] + p["sad"] >= cfg["pair_min"]:
            pick = "angry" if energy >= cfg["energy_hi"] else ("sad" if energy <= cfg["energy_lo"] else None)
            if pick:                                              # the hands decide between the two negative emotions
                other = "sad" if pick == "angry" else "angry"
                p = dict(p); p[pick] += p[other]; p[other] = 0.0; emo = pick; notes.append(f"energy {energy:+.2f}")
        thr = cfg["thr"][emo]
        valid = [x for x in (fer_segments or []) if x and x.get("probs") and x.get("expression") != "unknown"]
        if len(valid) >= 3:
            def top(x):
                q = grammar_fix(_norm(x["probs"])); return max(q, key=q.get)
            frac = sum(bool(top(x) == emo or (emo in ("angry", "sad") and notes and top(x) in ("angry", "sad"))) for x in valid) / len(valid)
            if frac < cfg["persist_min"]:
                if emo in cfg["persist_block"]:
                    return None, None, f"face: {emo} on {frac:.0%} of the signs only -> local marker, not an emotion"
                thr += cfg["persist_penalty"]; notes.append(f"on {frac:.0%} of signs")
        if prev == emo:
            thr -= cfg["memory_bonus"]; notes.append("same as previous utterance")
        tops = [max(x, key=x.get) for x in (fer_g, body_g) if x]
        agree = len(tops) == 2 and tops[0] == tops[1] == emo
        if agree:
            thr -= cfg["agree_bonus"]
        if wtop == emo:
            thr -= cfg["word_bonus"]
        second = max(p[c] for c in CLASSES if c not in ("neutral", emo))
        if p[emo] >= thr and p[emo] > p["neutral"] and p[emo] - second >= cfg["margin"]:
            if emo not in cfg["voiced"]:
                return None, None, f"{emo} detected (p={p[emo]:.2f}) but its voice is not validated -> neutral"
            inten = lo + (hi - lo) * min(max((p[emo] - p["neutral"]) * 1.5, 0.0), 1.0)
            if any(n.startswith("energy") for n in notes):
                inten = min(inten, (lo + hi) / 2)                 # decided by the hands, not by a clear face: keep it moderate
            src = "face+words" if wtop == emo else ("face (both branches)" if agree else "face")
            return emo, round(inten, 2), f"{src}: {emo} p={p[emo]:.2f} thr={thr:.2f}" + (f" [{'; '.join(notes)}]" if notes else "")
        face_note = f"face: {emo} p={p[emo]:.2f} < thr {thr:.2f} or margin"
    else:
        p, face_note = None, "no face evidence"

    if wtop and words[wtop] >= cfg["word_only_min"] and wtop in cfg["voiced"]:          # words alone
        against = p and any(p[c] >= cfg["contradict"] for c in CLASSES if c != "neutral" and VALENCE[c] * VALENCE[wtop] < 0)
        if not against:
            return wtop, cfg["word_only_intensity"], f"words: {', '.join(used)} ({face_note})"
    return None, None, face_note
