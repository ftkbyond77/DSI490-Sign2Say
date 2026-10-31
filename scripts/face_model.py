"""Train / evaluate the face emotion model (modules/face_model.py) → models/face-model/.

    python main.py face-model train      # RAVDESS features (cloud_s3/face-data/ravdess, `python main.py face-data ravdess`) →
                                         #   leave-actors-out CV (6 folds × 4 actors) to choose the regularisation + temperature,
                                         #   final fit on all 24 actors → models/face-model/face_emotion.npz, model_card.json
    python main.py face-model signs      # in-domain checks on sign videos (no emotion labels exist for them):
                                         #   (a) neutral dictionary signing → how often a non-neutral emotion is claimed (false alarms)
                                         #   (b) dictionary signs of emotion words (โกรธ, ดีใจ, เสียใจ …) → valence agreement with the word
                                         #   both vs the v7 rule-based face channel (modules/face.py) → artifacts/reports/face_model.json

    python main.py face-model v2 [--write]   # v10 face v2 — variants chosen on VAL signers (tune), reported on TEST signers, v9 on the
                                         #   same clips: more in-domain neutral signing, weakly labelled emotion-word signing of
                                         #   training signers, mouth articulation (mouthing words) kept out of the affect evidence
                                         #   → artifacts/reports/face_model_v2.json (+ models/face-model/ with --write)

Samples: every clip as a whole + 2 s windows (hop 1 s) of it (what a sign segment sees). Folds never share an actor.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.face_model import EMOTIONS, FEATURES, RAVDESS_MAP, VA, FaceEmotion, features  # noqa: E402
from modules.utils import ART, write_json  # noqa: E402

DATA = ROOT / "cloud_s3" / "face-data" / "ravdess"
OUT = ROOT / "models" / "face-model"


def load_ravdess(win_s=2.0, hop_s=1.0):
    X, y, actor, whole, inten = [], [], [], [], []
    for f in sorted(DATA.glob("actor_*.npz")):
        z = np.load(f)
        fps = float(z["fps"])
        for i in range(len(z["name"])):
            a, b = int(z["off"][i]), int(z["off"][i + 1])
            bs, face = z["bs"][a:b].astype(np.float32), z["face"][a:b].astype(np.float32)
            face[..., 0] *= 1280 / 720                   # RAVDESS is 1280×720: face x in y units (as modules/face.py)
            lab = EMOTIONS.index(RAVDESS_MAP[str(z["emotion"][i])])
            spans = [(0, b - a)] + [(s, s + int(win_s * fps)) for s in range(0, max(1, b - a - int(win_s * fps)) + 1, int(hop_s * fps))]
            for k, (s, e) in enumerate(spans):
                X.append(features(bs[s:e], face[s:e])); y.append(lab); actor.append(int(z["actor"][i])); whole.append(k == 0)
                inten.append(int(z["intensity"][i]))
    return np.array(X), np.array(y), np.array(actor), np.array(whole), np.array(inten)


def fit_lr(X, y, l2, w=None, iters=400, mask=None):
    """Class-balanced multinomial LR (+ per-sample weights w; mask = features kept out of the model: zero weight, so they can never
    be part of an explanation either)."""
    import torch
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Z = (X - mu) / sd
    if mask is not None:
        mask = np.r_[mask, np.zeros(X.shape[1] - len(mask), bool)]
        Z[:, mask] = 0.0
    Xt = torch.tensor(Z, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.long)
    cnt = np.bincount(y, minlength=len(EMOTIONS)).astype(np.float32)
    cw = torch.tensor(cnt.sum() / np.maximum(cnt, 1) / len(EMOTIONS))
    sw = torch.tensor(np.ones(len(y), np.float32) if w is None else np.asarray(w, np.float32))
    W = torch.zeros(X.shape[1], len(EMOTIONS), requires_grad=True); b = torch.zeros(len(EMOTIONS), requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        ce = torch.nn.functional.cross_entropy(Xt @ W + b, yt, weight=cw, reduction="none")
        loss = (ce * sw).sum() / (cw[yt] * sw).sum() + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    Wn = W.detach().numpy()
    if mask is not None:
        Wn[mask] = 0.0
    return FaceEmotion(mu.astype(np.float32), sd.astype(np.float32), Wn, b.detach().numpy())


def macro_f1(y, p):
    f = []
    for k in range(len(EMOTIONS)):
        tp = ((p == k) & (y == k)).sum(); fp = ((p == k) & (y != k)).sum(); fn = ((p != k) & (y == k)).sum()
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def sign_neutral(n=300, seed=0):
    """In-domain neutral faces: dictionary sign clips (TTRS / th_sl) of TRAINING-role signers that never sign an emotion word —
    signers mouth words and use grammatical faces, which a model trained on acted speech alone reads as emotion (v9 finding:
    60 % false alarms on neutral signing). Features of the whole clip + 2 s windows, label neutral."""
    from modules.face_model import EMO_LEX
    c = _dict_clips(None, n, seed=seed, held_out=False)
    c = c[~c.concept.isin(list(EMO_LEX))]
    X = []
    for row in c.itertuples():
        z = np.load(ROOT / "cloud_s3/data_prep" / row.mp)
        bs, face = z["bs"].astype(np.float32), z["face"].astype(np.float32).copy()
        H, W = (int(z["hw"][0]), int(z["hw"][1])) if len(z["hw"]) == 2 else (1, 1)
        face[..., 0] *= W / max(H, 1)
        X.append(features(bs, face))
        for s0 in range(0, max(1, len(bs) - 50) + 1, 50):
            X.append(features(bs[s0:s0 + 50], face[s0:s0 + 50]))
    return np.array(X)


def cmd_train(a):
    X, y, actor, whole, inten = load_ravdess()
    actors = np.unique(actor)
    print(f"RAVDESS: {len(actors)} actors, {whole.sum()} clips, {len(X)} samples", flush=True)
    Xd = sign_neutral(a.domain) if a.domain else np.zeros((0, X.shape[1]))
    yd = np.zeros(len(Xd), int)
    print(f"in-domain neutral signing samples: {len(Xd)}", flush=True)
    folds = [actors[i::6] for i in range(6)]
    res = {}
    for l2 in a.l2:
        P = np.zeros((len(X), len(EMOTIONS)))
        for te in folds:
            m = np.isin(actor, te)
            fm = fit_lr(np.r_[X[~m], Xd], np.r_[y[~m], yd], l2)
            P[m] = fm.logits(X[m])
        res[l2] = P
        pw = P[whole].argmax(1); pa = P.argmax(1)
        print(f"l2={l2:g}: clips acc {np.mean(pw == y[whole]):.3f} macro-F1 {macro_f1(y[whole], pw):.3f} | windows acc {np.mean(pa == y):.3f} "
              f"macro-F1 {macro_f1(y, pa):.3f}", flush=True)
    best = max(res, key=lambda k: macro_f1(y[whole], res[k][whole].argmax(1)) + macro_f1(y, res[k].argmax(1)))
    L = res[best]
    # temperature on the out-of-fold logits (NLL)
    Ts = np.linspace(0.3, 8, 78)
    nll = [-np.mean(np.log(np.exp(L / T - (L / T).max(1, keepdims=True))[np.arange(len(y)), y] /
                           np.exp(L / T - (L / T).max(1, keepdims=True)).sum(1) + 1e-9)) for T in Ts]
    T = float(Ts[int(np.argmin(nll))])
    pw = L[whole].argmax(1)
    conf = np.zeros((len(EMOTIONS),) * 2, int)
    for t_, p_ in zip(y[whole], pw):
        conf[t_, p_] += 1
    E = np.exp(L / T - (L / T).max(1, keepdims=True)); E /= E.sum(1, keepdims=True)
    strong = whole & (inten == 2)
    cv = dict(l2=best, temperature=T, clips_acc=float(np.mean(pw == y[whole])), clips_macro_f1=macro_f1(y[whole], pw),
              windows_acc=float(np.mean(L.argmax(1) == y)), windows_macro_f1=macro_f1(y, L.argmax(1)),
              strong_intensity_acc=float(np.mean(L[strong].argmax(1) == y[strong])),
              neutral_vs_emotion_acc=float(np.mean((pw == 0) == (y[whole] == 0))),
              valence_sign_agreement=float(np.mean(np.sign([VA[EMOTIONS[k]][0] for k in pw]) == np.sign([VA[EMOTIONS[k]][0] for k in y[whole]]))),
              confusion=dict(labels=EMOTIONS, matrix=conf.tolist()), chance=float(np.max(np.bincount(y[whole])) / whole.sum()))
    print(json.dumps({k: v for k, v in cv.items() if k != "confusion"}, indent=1), flush=True)
    fm = fit_lr(np.r_[X, Xd], np.r_[y, yd], best)
    OUT.mkdir(parents=True, exist_ok=True)
    fm.save(OUT / "face_emotion.npz", temperature=np.float32(T))
    card = dict(name="face-model", version="v1 (ThaiSLM v9)", task="signer facial emotion (7 classes) + valence / arousal + Thai explanation",
                input="MediaPipe Holistic face blendshapes (52) + 68 face points — same extractor as the sign pipeline (browser and server)",
                model="standardise → multinomial logistic regression (L2 %g), temperature %.2f; %d features" % (best, T, len(FEATURES)),
                classes=EMOTIONS, training_data="RAVDESS video speech, 24 actors × 60 clips (calm folded into neutral), clips + 2 s windows; "
                "Livingstone & Russo 2018, CC BY-NC-SA 4.0 — research / non-commercial use; + %d neutral samples of dictionary signing "
                "(TTRS / th_sl training-role signers, no emotion words)" % len(Xd), evaluation="leave-actors-out (6 folds × 4 actors); sign "
                "videos of held-out signers: python main.py face-model signs",
                cv=cv, code="modules/face_model.py · scripts/face_model.py", features_kept="cloud_s3/face-data/ravdess (features only, no video)")
    write_json(OUT / "model_card.json", card)
    write_json(ART / "reports" / "face_model_cv.json", cv)
    print("written", OUT, flush=True)


def _dict_clips(words=None, n_neutral=300, seed=0, held_out=True):
    """Dictionary sign clips (TTRS / th_sl: one signer, a citation sign, usually a neutral face) with their raw MediaPipe arrays.
    held_out=True: only signers who have no training-role clip (evaluation); False: only training-role clips of the other signers."""
    import pandas as pd
    c = pd.read_parquet(ROOT / "cloud_s3/data_prep/manifest/clips.parquet")
    c = c[c.dataset.isin(["ttrs", "th_sl"]) & c.pose_raw_file.notna()]
    train_signers = set(c[c.role == "train"].signer)
    if held_out is not None:     # None: every signer (emotion-word clips — never used in training whoever signs them)
        c = c[~c.signer.isin(train_signers)] if held_out else c[(c.role == "train")]
    c = c.assign(mp=c.pose_raw_file.str.replace("pose_raw/", "pose_mp_raw/", regex=False))
    c = c[[(ROOT / "cloud_s3/data_prep" / p).exists() for p in c.mp]]
    if words is not None:
        return c[c.concept.isin(words)]
    return c.sample(min(n_neutral, len(c)), random_state=seed)


def cmd_signs(a):
    from modules.face import analyse as rule_analyse
    from modules.face_model import EMO_LEX
    fm = FaceEmotion.load(OUT)
    neg_words = [w for w, e in EMO_LEX.items() if VA[e][0] < 0]
    pos_words = [w for w, e in EMO_LEX.items() if VA[e][0] > 0]
    emo = _dict_clips(neg_words + pos_words, held_out=None)
    neu = _dict_clips(None, a.n)
    neu = neu[~neu.concept.isin(list(EMO_LEX))]

    def read(row):
        z = dict(np.load(ROOT / "cloud_s3/data_prep" / row.mp))
        r = fm.analyse(z)
        q = rule_analyse(z)
        return r, q
    out = {}
    rows = []
    for name, df in (("neutral_dictionary", neu), ("emotion_words", emo)):
        for row in df.itertuples():
            try:
                r, q = read(row)
            except Exception:                 # noqa: BLE001 — a clip without face arrays
                continue
            rows.append(dict(set=name, concept=row.concept, signer=row.signer, model=r["emotion"], model_p=r.get("confidence", 0), model_v=r.get("valence", 0),
                             rule=q["emotion"], expected=EMO_LEX.get(row.concept, "neutral")))
    import pandas as pd
    d = pd.DataFrame(rows)
    n = d[d.set == "neutral_dictionary"]; e = d[d.set == "emotion_words"]
    sign = lambda s: np.sign([VA[x][0] for x in s])        # noqa: E731
    out = dict(neutral_clips=len(n), emotion_word_clips=len(e),
               false_alarm_model=float((n.model != "neutral").mean()), false_alarm_rule=float((n.rule != "neutral").mean()),
               emotion_words_valence_agree_model=float(np.mean(sign(e.model) == sign(e.expected))),
               emotion_words_valence_agree_rule=float(np.mean(sign(e.rule) == sign(e.expected))),
               emotion_words_nonneutral_model=float((e.model != "neutral").mean()), emotion_words_nonneutral_rule=float((e.rule != "neutral").mean()),
               emotion_words_exact_model=float((e.model == e.expected).mean()), emotion_words_exact_rule=float((e.rule == e.expected).mean()),
               mean_model_valence_pos_words=float(e[e.expected.map(lambda x: VA[x][0] > 0)].model_v.mean()),
               mean_model_valence_neg_words=float(e[e.expected.map(lambda x: VA[x][0] < 0)].model_v.mean()),
               per_clip=d.to_dict("records"))
    print(json.dumps({k: v for k, v in out.items() if k != "per_clip"}, indent=1, ensure_ascii=False), flush=True)
    write_json(ART / "reports" / "face_model_signs.json", out)


# ----------------------------------------------------------------------------- v10: face v2
MOUTHING = ["jawOpen", "jawForward", "jawLeft", "jawRight", "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
            "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthLowerDownLeft", "mouthLowerDownRight",
            "mouthPressLeft", "mouthPressRight"]          # lip / jaw articulation: signers mouth the Thai word — grammar, not affect


def _clip_feats(rows, windows=True):
    X, keep = [], []
    for row in rows.itertuples():
        try:
            z = np.load(ROOT / "cloud_s3/data_prep" / row.mp)
        except Exception:                                 # noqa: BLE001 — a missing raw file
            continue
        bs, face = z["bs"].astype(np.float32), z["face"].astype(np.float32).copy()
        H, W = (int(z["hw"][0]), int(z["hw"][1])) if len(z["hw"]) == 2 else (1, 1)
        face[..., 0] *= W / max(H, 1)
        X.append(features(bs, face)); keep.append(row.Index)
        if windows:
            for s0 in range(0, max(1, len(bs) - 50) + 1, 50):
                X.append(features(bs[s0:s0 + 50], face[s0:s0 + 50])); keep.append(row.Index)
    return np.array(X).reshape(-1, len(FEATURES)), keep


def _sign_pool():
    """Dictionary / YouTube sign clips with raw MediaPipe face arrays (th_sl, TTRS, agent YouTube words), role from the manifest."""
    import pandas as pd
    c = pd.read_parquet(ROOT / "cloud_s3/data_prep/manifest/clips.parquet")
    c = c[c.dataset.isin(["ttrs", "th_sl"]) & c.pose_raw_file.notna()].copy()
    c["mp"] = c.pose_raw_file.str.replace("pose_raw/", "pose_mp_raw/", regex=False)
    ag = pd.read_csv(ROOT / "cloud_s3/agent_data/youtube_words/items.csv")
    ag = ag[ag.status == "accepted"].copy()
    ag["mp"] = ["pose_mp_raw/agent/" + Path(str(v)).stem + ".npz" for v in ag.video]
    ag["dataset"] = "agent"
    c = pd.concat([c[["clip_id", "dataset", "signer", "concept", "role", "mp"]], ag[["clip_id", "dataset", "signer", "concept", "role", "mp"]]])
    return c[[(ROOT / "cloud_s3/data_prep" / p).exists() for p in c.mp]].reset_index(drop=True)


def _eval_signs(fm, pool, split, n_neutral=300, seed=0):
    """split = role of the signers (val = tune, test = report). Neutral signing: non-emotion-word clips → share read as an emotion
    (false alarm). Emotion words (EMO_LEX): sign of the predicted valence = the word's (neutral counts as a miss, v9's metric),
    plus the mean valence of positive / negative words (does the face lean the right way at all)."""
    from modules.face_model import EMO_LEX
    m = pool.role == split
    neu = pool[m & ~pool.concept.isin(list(EMO_LEX))]
    neu = neu.sample(min(n_neutral, len(neu)), random_state=seed)
    emo = pool[m & pool.concept.isin(list(EMO_LEX))]
    Xn, _ = _clip_feats(neu, windows=False); Xe, ke = _clip_feats(emo, windows=False)
    pn = fm.proba(Xn).argmax(1) if len(Xn) else np.zeros(0, int)
    Pe = fm.proba(Xe) if len(Xe) else np.zeros((0, len(EMOTIONS)))
    exp = np.array([np.sign(VA[EMO_LEX[c]][0]) for c in emo.loc[ke].concept]) if len(ke) else np.zeros(0)
    pred_v = np.array([np.sign(VA[EMOTIONS[k]][0]) for k in Pe.argmax(1)]) if len(Pe) else np.zeros(0)
    val = Pe @ np.array([VA[e][0] for e in EMOTIONS]) if len(Pe) else np.zeros(0)
    return dict(neutral_n=int(len(Xn)), false_alarm=float(np.mean(pn != 0)) if len(pn) else None, emotion_n=int(len(Xe)),
                valence_agree=float(np.mean(pred_v == exp)) if len(exp) else None,
                mean_valence_pos=float(val[exp > 0].mean()) if (exp > 0).any() else None,
                mean_valence_neg=float(val[exp < 0].mean()) if (exp < 0).any() else None,
                valence_gap=float(val[exp > 0].mean() - val[exp < 0].mean()) if (exp > 0).any() and (exp < 0).any() else None)


def cmd_v2(a):
    """Face v2 variants (all multinomial LR, explainable). Tune = RAVDESS leave-actors-out CV (acc, neutral-vs-emotion) + sign clips of
    TRAINING-role signers in leave-signers-out CV (5 folds by signer: a fold's signers are never in that fold's training data) —
    neutral false alarms and emotion-word valence over ~20 signers instead of the 2–4 val signers. A variant is eligible only if
    no tune metric is worse than the v9 recipe under the same CV; among those, the best sum. Report: TEST-role signers (never used)
    with the shipped v9 model on the same clips, + the TSL51 researcher's emotion words (a test signer)."""
    import time
    from modules.face_model import EMO_LEX
    t0 = time.time()
    X, y, actor, whole, inten = load_ravdess()
    actors = np.unique(actor)
    afolds = [actors[i::6] for i in range(6)]
    pool = _sign_pool()
    tr = pool[pool.role == "train"].reset_index(drop=True)
    neu = tr[~tr.concept.isin(list(EMO_LEX))]
    neu = neu.groupby("signer", group_keys=False).apply(lambda g: g.sample(min(len(g), 250), random_state=0))
    emo = tr[tr.concept.isin(list(EMO_LEX))]
    Xn, kn = _clip_feats(neu)                     # clip + 2 s windows
    Xe, ke = _clip_feats(emo)
    sn = neu.loc[kn].signer.values; se = emo.loc[ke].signer.values
    whole_n = np.r_[True, np.array(kn[1:]) != np.array(kn[:-1])]; whole_e = np.r_[True, np.array(ke[1:]) != np.array(ke[:-1])]
    ye = np.array([EMOTIONS.index(EMO_LEX[c]) for c in emo.loc[ke].concept])
    ve = np.array([np.sign(VA[EMO_LEX[c]][0]) for c in emo.loc[ke].concept])
    signers = np.array(sorted(set(sn) | set(se)))
    rng = np.random.RandomState(0); rng.shuffle(signers)
    sfolds = [signers[i::5] for i in range(5)]
    print(f"in-domain: {len(set(sn))} neutral signers / {len(Xn)} samples, {len(set(se))} emotion-word signers / {len(Xe)} samples"
          f" ({time.time() - t0:.0f}s)", flush=True)
    Xn_v9 = sign_neutral(300)
    mouth = np.array([f.split(":")[0] in MOUTHING for f in FEATURES])
    variants = {
        "v9 recipe": dict(neutral="v9", emo=False, mask=None),
        "neutral signing (all training signers)": dict(neutral="all", emo=False, mask=None),
        "+ emotion-word signing (weak labels)": dict(neutral="all", emo=True, mask=None),
        "+ emotion words, mouthing masked": dict(neutral="all", emo=True, mask=mouth),
        "neutral (all), mouthing masked": dict(neutral="all", emo=False, mask=mouth),
        "domain input (acted / signing) + neutral (all)": dict(neutral="all", emo=False, mask=None, domain=True),
        "domain input + neutral (all) + emotion words": dict(neutral="all", emo=True, mask=None, domain=True),
    }

    def data(v, drop=()):
        mn = ~np.isin(sn, list(drop)); me = ~np.isin(se, list(drop))
        Xs = [X, Xn_v9 if v["neutral"] == "v9" else Xn[mn]]
        ys = [y, np.zeros(len(Xs[1]), int)]
        ws = [np.ones(len(X)), np.ones(len(Xs[1]))]
        if v["emo"]:
            Xs.append(Xe[me]); ys.append(ye[me]); ws.append(np.full(me.sum(), a.w_emo))
        if v.get("domain"):                                    # 0 = acted corpus (RAVDESS), 1 = signing
            Xs = [np.c_[x_, np.full(len(x_), 0.0 if i == 0 else 1.0)] for i, x_ in enumerate(Xs)]
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)

    def ravdess_x(v, Xr):
        return np.c_[Xr, np.zeros(len(Xr))] if v.get("domain") else Xr

    def tune_metrics(v, l2):
        P = np.zeros((len(X), len(EMOTIONS)))                      # RAVDESS: leave actors out (in-domain data from everyone)
        Xd, yd, wd = data(v)
        nR = len(X)
        for te in afolds:
            mm = np.isin(actor, te)
            keep = np.r_[~mm, np.ones(len(Xd) - nR, bool)]
            P[mm] = fit_lr(Xd[keep], yd[keep], l2, w=wd[keep], mask=v["mask"]).logits(ravdess_x(v, X[mm]))
        pw = P[whole].argmax(1)
        fa, vag, vgap = [], [], []
        for te in sfolds:                                          # signs: leave signers out (v9 recipe: its neutral sample excludes nobody
            Xf, yf, wf = data(v, drop=te)                          #   by signer, so it is refit without the fold's signers here too)
            if v["neutral"] == "v9":
                keep = np.ones(len(Xf), bool)
            fm = fit_lr(Xf, yf, l2, w=wf, mask=v["mask"])
            fm.meta["temperature"] = 1.0
            m = np.isin(sn, te) & whole_n
            if m.any():
                fa.append(np.mean(fm.proba(Xn[m]).argmax(1) != 0))
            m = np.isin(se, te) & whole_e
            if m.any():
                Pe = fm.proba(Xe[m])
                vag += list(np.array([np.sign(VA[EMOTIONS[k]][0]) for k in Pe.argmax(1)]) == ve[m])
                vgap += list(zip(Pe @ np.array([VA[e][0] for e in EMOTIONS]), ve[m]))
        vg = np.array(vgap)
        return dict(clips_acc=float(np.mean(pw == y[whole])), neutral_vs_emotion=float(np.mean((pw == 0) == (y[whole] == 0))),
                    false_alarm_cv=float(np.mean(fa)), valence_agree_cv=float(np.mean(vag)) if vag else None,
                    valence_gap_cv=float(vg[vg[:, 1] > 0, 0].mean() - vg[vg[:, 1] < 0, 0].mean()) if len(vg) and (vg[:, 1] > 0).any() and (vg[:, 1] < 0).any() else None,
                    n_emotion=len(vag))

    res = {}
    for name, v in variants.items():
        best = None
        for l2 in a.l2:
            m = tune_metrics(v, l2)
            sc = m["clips_acc"] + m["neutral_vs_emotion"] - m["false_alarm_cv"] + (m["valence_agree_cv"] or 0)
            if best is None or sc > best[0]:
                best = (sc, dict(l2=l2, tune=m))
        res[name] = best[1]
        t_ = best[1]["tune"]
        print(f"{name:42s} l2={best[1]['l2']:g} RAVDESS acc {t_['clips_acc']:.3f} n-vs-e {t_['neutral_vs_emotion']:.3f} | signer-CV FA "
              f"{t_['false_alarm_cv']:.3f} valence {t_['valence_agree_cv']:.3f} gap {t_['valence_gap_cv']:+.3f} (n={t_['n_emotion']})", flush=True)
    base = res["v9 recipe"]["tune"]

    def not_worse(m):
        return (m["clips_acc"] >= base["clips_acc"] - 1e-9 and m["neutral_vs_emotion"] >= base["neutral_vs_emotion"] - 1e-9
                and m["false_alarm_cv"] <= base["false_alarm_cv"] + 1e-9 and (m["valence_agree_cv"] or 0) >= (base["valence_agree_cv"] or 0) - 1e-9)
    ok = {k: r for k, r in res.items() if k != "v9 recipe" and not_worse(r["tune"])}
    score = lambda r: r["tune"]["clips_acc"] + r["tune"]["neutral_vs_emotion"] - r["tune"]["false_alarm_cv"] + (r["tune"]["valence_agree_cv"] or 0)  # noqa: E731
    chosen = max(ok, key=lambda k: score(ok[k])) if ok else None
    print("eligible (no tune metric worse than the v9 recipe):", list(ok), "→ chosen:", chosen, flush=True)
    out = dict(variants=res, chosen=chosen)
    v9 = FaceEmotion.load(ART / "baselines" / "v9" / "face-model")
    if chosen:
        v = variants[chosen]
        Xd, yd, wd = data(v)
        fm = fit_lr(Xd, yd, res[chosen]["l2"], w=wd, mask=v["mask"])
        # temperature on RAVDESS leave-actors-out logits (as v9)
        P = np.zeros((len(X), len(EMOTIONS)))
        for te in afolds:
            mm = np.isin(actor, te); keep = np.r_[~mm, np.ones(len(Xd) - len(X), bool)]
            P[mm] = fit_lr(Xd[keep], yd[keep], res[chosen]["l2"], w=wd[keep], mask=v["mask"]).logits(ravdess_x(v, X[mm]))
        Ts = np.linspace(0.3, 8, 78)
        nll = [-np.mean(np.log(np.exp(P / T - (P / T).max(1, keepdims=True))[np.arange(len(y)), y] / np.exp(P / T - (P / T).max(1, keepdims=True)).sum(1) + 1e-9)) for T in Ts]
        fm.meta["temperature"] = float(Ts[int(np.argmin(nll))])
    else:
        fm = v9
    out["test"] = dict(v10=_eval_signs(fm, pool, "test"), v9=_eval_signs(v9, pool, "test"))
    t51 = _t51_emotion()
    if t51 is not None:
        out["test"]["tsl51_researcher_emotion_words"] = dict(v10=_valence(fm, *t51), v9=_valence(v9, *t51))
    xx = np.stack([features(np.random.rand(75, 52).astype(np.float32), np.random.rand(75, 68, 2).astype(np.float32)) for _ in range(20)])
    t1 = time.perf_counter(); [fm.proba(x) for x in xx]; out["latency_ms_per_stretch"] = (time.perf_counter() - t1) * 1000 / 20
    print(json.dumps({k: v for k, v in out.items() if k != "variants"}, indent=1, ensure_ascii=False, default=float), flush=True)
    write_json(ART / "reports" / "face_model_v2.json", out)
    if a.write and chosen:
        fm.save(OUT / "face_emotion.npz", temperature=np.float32(fm.meta["temperature"]))
        card = read_json_safe(OUT / "model_card.json")
        card.update(version="v2 (ThaiSLM v10)", recipe=chosen,
                    model=f"standardise → multinomial logistic regression (L2 {res[chosen]['l2']:g}), temperature {fm.meta['temperature']:.2f}, "
                          f"{len(FEATURES)} features" + ("; mouth articulation features masked (mouthing = grammar, not affect)" if variants[chosen]["mask"] is not None else ""),
                    v2=dict(tune=res[chosen]["tune"], test=out["test"]))
        write_json(OUT / "model_card.json", card)
        print("written", OUT, flush=True)


def _t51_emotion():
    """The TSL51 researcher's isolated emotion words (a test signer, never in any training set) → (X, expected valence sign)."""
    import pandas as pd
    from modules.face_model import EMO_LEX
    c = pd.read_parquet(ROOT / "cloud_s3/data_prep/manifest/clips.parquet")
    c = c[(c.dataset == "tsl51_user_sign") & c.concept.isin(list(EMO_LEX)) & c.pose_raw_file.notna()].copy()
    if not len(c):
        return None
    c["mp"] = c.pose_raw_file.str.replace("pose_raw/", "pose_mp_raw/", regex=False)
    c = c[[(ROOT / "cloud_s3/data_prep" / p).exists() for p in c.mp]]
    X, k = _clip_feats(c, windows=False)
    return X, np.array([np.sign(VA[EMO_LEX[w]][0]) for w in c.loc[k].concept])


def _valence(fm, X, exp):
    P = fm.proba(X)
    pv = np.array([np.sign(VA[EMOTIONS[i]][0]) for i in P.argmax(1)])
    v = P @ np.array([VA[e][0] for e in EMOTIONS])
    return dict(n=len(exp), valence_agree=float(np.mean(pv == exp)), valence_gap=float(v[exp > 0].mean() - v[exp < 0].mean())
                if (exp > 0).any() and (exp < 0).any() else None)


def read_json_safe(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:                                     # noqa: BLE001
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "signs", "v2"])
    ap.add_argument("--write", action="store_true"); ap.add_argument("--w_emo", type=float, default=3.0)
    ap.add_argument("--l2", type=float, nargs="+", default=[1e-4, 1e-3, 3e-3, 1e-2])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--domain", type=int, default=300, help="in-domain neutral signing clips (0 = RAVDESS only)")
    a = ap.parse_args()
    dict(train=cmd_train, signs=cmd_signs, v2=cmd_v2)[a.step](a)


if __name__ == "__main__":
    main()
