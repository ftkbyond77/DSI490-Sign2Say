"""Train / evaluate the face emotion model (modules/face_model.py) → models/face-model/.

    python main.py face-model train      # RAVDESS features (cloud_s3/face-data/ravdess, `python main.py face-data ravdess`) →
                                         #   leave-actors-out CV (6 folds × 4 actors) to choose the regularisation + temperature,
                                         #   final fit on all 24 actors → models/face-model/face_emotion.npz, model_card.json
    python main.py face-model signs      # in-domain checks on sign videos (no emotion labels exist for them):
                                         #   (a) neutral dictionary signing → how often a non-neutral emotion is claimed (false alarms)
                                         #   (b) dictionary signs of emotion words (โกรธ, ดีใจ, เสียใจ …) → valence agreement with the word
                                         #   both vs the v7 rule-based face channel (modules/face.py) → artifacts/reports/face_model.json

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


def fit_lr(X, y, l2, w=None, iters=400):
    import torch
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.long)
    cnt = np.bincount(y, minlength=len(EMOTIONS)).astype(np.float32)
    cw = torch.tensor(cnt.sum() / np.maximum(cnt, 1) / len(EMOTIONS))
    W = torch.zeros(X.shape[1], len(EMOTIONS), requires_grad=True); b = torch.zeros(len(EMOTIONS), requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt, weight=cw) + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return FaceEmotion(mu.astype(np.float32), sd.astype(np.float32), W.detach().numpy(), b.detach().numpy())


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "signs"])
    ap.add_argument("--l2", type=float, nargs="+", default=[1e-4, 1e-3, 3e-3, 1e-2])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--domain", type=int, default=300, help="in-domain neutral signing clips (0 = RAVDESS only)")
    a = ap.parse_args()
    dict(train=cmd_train, signs=cmd_signs)[a.step](a)


if __name__ == "__main__":
    main()
