from affect_v2 import speech_affect_v2 as f
def P(**k):
    d = dict(neutral=0, happy=0, sad=0, angry=0, fear=0, disgust=0, surprise=0); d.update(k); return d
def fer(**k): return dict(probs=P(**k), valid_frac=0.9, expression="x")
def body(**k): return dict(probs=P(**k))
def seg(w, p=0.9): return dict(word=w, p=p, status="accepted")
T = [
 ("happy face, both agree",            f(fer(happy=.7, neutral=.3), body(happy=.6, neutral=.4)), "happy"),
 ("sad .55, both branches agree",      f(fer(sad=.55, neutral=.45), body(sad=.55, neutral=.45)), "sad"),
 ("sad one branch only .55",           f(fer(sad=.55, neutral=.45), None), None),
 ("sad .55 + word เสียใจ",              f(fer(sad=.5, neutral=.5), None, segs=[seg("ฉัน"), seg("เสียใจ")]), "sad"),
 ("angry .7 but wh-question furrow",   f(fer(angry=.7, neutral=.3), body(angry=.7, neutral=.3), grammar=dict(question_wh=True)), None),
 ("angry .8 no question",              f(fer(angry=.8, neutral=.2), body(angry=.7, neutral=.3)), "angry"),
 ("surprise .8 yes/no question",       f(fer(surprise=.8, neutral=.2), None, grammar=dict(question_yesno=True)), None),
 ("surprise .8 (voice not validated)", f(fer(surprise=.8, neutral=.2), None), None),
 ("neutral face + ดีใจ",                f(fer(neutral=.9, happy=.1), body(neutral=.8, happy=.2), segs=[seg("ดีใจ")]), "happy"),
 ("no face + ขอบคุณ (weak word)",       f(None, None, segs=[seg("ขอบคุณ")]), None),
 ("ไม่ + ชอบ + ดีใจ under negation",     f(None, None, segs=[seg("ไม่"), seg("ดีใจ")]), None),
 ("sad face .7 vs word ดีใจ",           f(fer(sad=.75, neutral=.25), body(sad=.7, neutral=.3), segs=[seg("ดีใจ")]), "sad"),
 ("angry .45 / sad .40 coin flip",     f(fer(angry=.45, sad=.40, neutral=.15), None), None),
 ("low-confidence word ignored",       f(None, None, segs=[seg("โกรธ", p=0.3)]), None),
 ("nothing",                           f(None, None), None),
]
def bodyE(e, **k): return dict(probs=P(**k), sources=dict(energy=e))
def segf(n_emo, n_neu, emo): return [dict(probs=P(**{emo: .8, "neutral": .2}), expression=emo)] * n_emo + [dict(probs=P(neutral=.9, **{emo: .1}), expression="neutral")] * n_neu
st = {}
T += [
 ("angry .40 / sad .40 + fast hands",   f(fer(angry=.4, sad=.4, neutral=.2), bodyE(.6, angry=.4, sad=.4, neutral=.2)), "angry"),
 ("angry .40 / sad .40 + slow hands",   f(fer(angry=.4, sad=.4, neutral=.2), bodyE(-.6, angry=.4, sad=.4, neutral=.2)), "sad"),
 ("angry .40 / sad .40 + medium hands", f(fer(angry=.4, sad=.4, neutral=.2), bodyE(0, angry=.4, sad=.4, neutral=.2)), None),
 ("angry .7 on 1 of 4 signs",           f(fer(angry=.7, neutral=.3), body(angry=.7, neutral=.3), fer_segments=segf(1, 3, "angry")), None),
 ("angry .7 on 3 of 4 signs",           f(fer(angry=.7, neutral=.3), body(angry=.7, neutral=.3), fer_segments=segf(3, 1, "angry")), "angry"),
 ("happy .62 on 1 of 4 signs",          f(fer(happy=.62, neutral=.38), None, fer_segments=segf(1, 3, "happy")), None),
 ("happy .62 on 4 of 4 signs",          f(fer(happy=.62, neutral=.38), None, fer_segments=segf(4, 0, "happy")), "happy"),
 ("memory 1: sad clear",                f(fer(sad=.7, neutral=.3), body(sad=.7, neutral=.3), state=st, now=100), "sad"),
 ("memory 2: sad .56 5 s later",        f(fer(sad=.56, neutral=.44), None, state=st, now=105), "sad"),
 ("memory 3: sad .56 60 s later",       f(fer(sad=.56, neutral=.44), None, state=st, now=165), None),
]
bad = 0
for name, (e, i, why), want in T:
    ok = e == want; bad += not ok
    print(("ok  " if ok else "FAIL"), f"{name:36s} -> {str(e):6s} {str(i):5s} | {why}")
print("failed:", bad)
