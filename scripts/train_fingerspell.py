"""Train the fingerspelling recogniser (modules/fingerspell.py) and the letter bank — CPU, minutes.

  python scripts/train_fingerspell.py            → models/fingerspell.npz, models/letters_bank.npz, artifacts/reports/fingerspell.json

Data (roles keep people apart: a held-out person is never in training):
  * English A–Z   cloud_s3/agent_data/en_alpha/items.csv (letter holds of alphabet lessons; role by channel)
  * Thai letters  TSL-ONE-S (25 signers × 42 consonants + 7 vowels; official roles) + agent_data/th_alpha/items.csv
  * not a letter  word signs of training signers (TTRS, th_sl, YouTube lessons), windows inside the sign span
Model: windows of 8 frames → features → MLP (2 hidden layers) → softmax; exported as NumPy weights (no sklearn at run time).
Letter bank: encoder embeddings of the training letter examples, matched like the word bank (a second, independent opinion).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from modules.fingerspell import EN_LETTERS, NONE, STRIDE, TH_LETTERS, WIN, window_features
from modules.parts import to_iso
from modules.schema import canonical_hands
from modules.utils import ART, DATA, MODEL, get_logger, write_json

log = get_logger("fingerspell")
PREP = DATA / "data_prep"
AGENT = DATA / "agent_data"


def std_file(path):
    z = np.load(path)
    hw = tuple(int(v) for v in z["hw"])
    kp, sc, _ = canonical_hands(z["kp"].astype(np.float32), z["sc"].astype(np.float32), hw)
    return to_iso(kp, hw).astype(np.float32), sc


_raw_cache: dict = {}


def std_raw(path):
    if path not in _raw_cache:
        from harvest_letters import standard_from_raw
        from modules.mediapipe_pose import load_raw
        _raw_cache.clear()
        _raw_cache[path] = standard_from_raw(load_raw(path))
    return _raw_cache[path]


def windows_in(kpi, sc, f0, f1, step=STRIDE, pad=2):
    a, b = max(0, f0 - pad), min(len(kpi), f1 + pad)
    if b - a < WIN:
        c = (a + b) // 2
        a, b = max(0, c - WIN // 2), min(len(kpi), c + WIN // 2 + 1)
    starts = np.arange(a, max(a + 1, b - WIN + 1), step)
    X, st, keep = window_features(kpi, sc, starts=starts)
    return X[keep]


def collect(max_none_clips=2500):
    """→ X [N,D], y (class names), group (person id), role."""
    X, y, g, role = [], [], [], []

    def add(Xw, label, grp, r):
        X.append(Xw); y.extend([label] * len(Xw)); g.extend([grp] * len(Xw)); role.extend([r] * len(Xw))

    # English letters: alphabet-lesson holds (th_alpha lessons are stored but not used until their labels are checked by eye)
    for s in ("en_alpha",):
        f = AGENT / s / "items.csv"
        if not f.exists():
            continue
        it = pd.read_csv(f)
        for r in it.itertuples():
            raw = AGENT / s / "pose_mp_raw" / f"{r.video}.npz"
            if not raw.exists():
                continue
            kpi, sc = std_raw(raw)
            Xw = windows_in(kpi, sc, int(r.f0), int(r.f1))
            if len(Xw):
                add(Xw, r.letter, f"{s}:{r.channel_id}", r.role)
        log.info("%s: %d windows so far", s, sum(len(x) for x in X))
    # Thai letters: TSL-ONE-S
    m = pd.read_parquet(PREP / "manifest" / "clips.parquet")
    L = m[(m.dataset == "tslone") & m.concept.isin(TH_LETTERS)]
    for r in L.itertuples():
        kpi, sc = std_file(PREP / r.pose_file)
        T = len(kpi)
        Xw = windows_in(kpi, sc, int(T * 0.2), int(T * 0.8), pad=0)
        if len(Xw):
            add(Xw, r.concept, f"tslone:{r.signer}", "test" if r.role == "test" else "train")
    log.info("+ TSL-ONE-S letters: %d windows", sum(len(x) for x in X))
    # not a letter: word signs (train and test people, by their roles)
    W = m[m.dataset.isin(["ttrs", "th_sl", "agent"]) & (m.kind == "isolated") & m.span_f0.notna()]
    W = W.sample(min(max_none_clips, len(W)), random_state=0)
    n0 = sum(len(x) for x in X)
    for r in W.itertuples():
        f = PREP / ("pose_mp/" + str(r.pose_file).split("/", 1)[1])
        if not f.exists():
            continue
        kpi, sc = std_file(f)
        Xw = windows_in(kpi, sc, int(r.span_f0), int(r.span_f1), step=4, pad=0)
        if len(Xw):
            add(Xw[:: max(1, len(Xw) // 6)], NONE, f"{r.dataset}:{r.signer}", "test" if r.role == "test" else "train")
    log.info("+ word signs (not a letter): %d windows", sum(len(x) for x in X) - n0)
    # hard negatives: continuous signing that is not spelling — the TSL51 *tune* sentences (test sentences are never used)
    import pickle
    D = pickle.load(open(ART / "runs" / "seq-v7" / "seq_tables.pkl", "rb"))
    n1 = sum(len(x) for x in X)
    for it in D["items"]:
        if it["kind"] == "real" and it["split"] == "tune" and it["speed"] == 1.0:
            kpi, sc = it["kp"].astype(np.float32), it["sc"].astype(np.float32)
            Xw, _, keep = window_features(kpi, sc, starts=np.arange(0, max(1, len(kpi) - WIN), 4))
            if keep.any():
                add(Xw[keep], NONE, "tsl51_sentences", "train")
    log.info("+ TSL51 tune sentences (not a letter): %d windows", sum(len(x) for x in X) - n1)
    return np.concatenate(X), np.array(y), np.array(g), np.array(role)


def train_mlp(X, y, classes, hidden=(256, 128), epochs=60, seed=0):
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)
    ci = {c: i for i, c in enumerate(classes)}
    yt = torch.tensor([ci[c] for c in y], device=dev)
    w = np.bincount(yt.cpu().numpy(), minlength=len(classes)).astype(np.float32)
    cw = torch.tensor((w.sum() / (len(w) * np.maximum(w, 1))) ** 0.5, device=dev)       # soften class imbalance
    dims = [X.shape[1], *hidden, len(classes)]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [torch.nn.ReLU(), torch.nn.Dropout(0.2)]
    net = torch.nn.Sequential(*layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            xb = Xt[b] + 0.05 * torch.randn_like(Xt[b])                     # feature noise ≈ landmark jitter
            loss = torch.nn.functional.cross_entropy(net(xb), yt[b], weight=cw, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval()
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    return dict(mu=mu.astype(np.float32), sd=sd.astype(np.float32), n_layers=len(lin),
                **{f"W{i}": m.weight.detach().cpu().numpy().T.astype(np.float32) for i, m in enumerate(lin)},
                **{f"b{i}": m.bias.detach().cpu().numpy().astype(np.float32) for i, m in enumerate(lin)})


N_MODELS = 3      # an average of 3 independently trained nets: held-out Thai letters 75 → 80 %, English 77 → 78 % (MLP alone)


def flat(models):
    out = dict(n_models=len(models))
    for i, m in enumerate(models):
        out.update({f"m{i}_{k}": v for k, v in m.items()})
    return out


def proba(model, X):
    if "models" in model:
        return np.mean([proba(m, X) for m in model["models"]], 0)
    h = (X - model["mu"]) / model["sd"]
    for i in range(model["n_layers"]):
        h = h @ model[f"W{i}"] + model[f"b{i}"]
        if i < model["n_layers"] - 1:
            h = np.maximum(h, 0)
    h -= h.max(1, keepdims=True)
    e = np.exp(h)
    return e / e.sum(1, keepdims=True)


def evaluate(model, classes, X, y, g):
    """Window accuracy within each alphabet, letter-level accuracy (mean probability over a person's windows of one example is
    approximated per group+letter), and how often a word sign is called a letter."""
    P = proba(model, X)
    rep = {}
    for name, letters in (("en", EN_LETTERS), ("th", TH_LETTERS)):
        m = np.isin(y, letters)
        if not m.any():
            continue
        idx = [classes.index(c) for c in letters if c in classes]
        sub = P[m][:, idx]
        pred = np.array([classes[idx[j]] for j in sub.argmax(1)])
        top3 = np.mean([t in [classes[idx[j]] for j in np.argsort(-p)[:3]] for p, t in zip(sub, y[m])])
        # example level: average the windows of the same person + letter
        df = pd.DataFrame(dict(g=g[m], y=y[m]))
        ex_ok = []
        for (gg, yy), ii in df.groupby(["g", "y"]).groups.items():
            pm = sub[np.asarray(ii)].mean(0)
            ex_ok.append(classes[idx[int(pm.argmax())]] == yy)
        rep[name] = dict(windows=int(m.sum()), window_top1=round(float((pred == y[m]).mean()), 3), window_top3=round(float(top3), 3),
                         example_top1=round(float(np.mean(ex_ok)), 3), examples=len(ex_ok), people=int(len(set(g[m]))))
    nm = y == NONE
    if nm.any() and NONE in classes:
        best_letter = np.delete(P[nm], classes.index(NONE), axis=1).max(1)
        rep["word_signs_called_letter"] = {f"p>={t}": round(float(((best_letter >= t) & (best_letter > P[nm][:, classes.index(NONE)])).mean()), 3)
                                           for t in (0.5, 0.6, 0.7, 0.8)}
    return rep


def letter_bank(X_items):
    """Encoder embeddings of the training letter examples → models/letters_bank.npz (Z, letter)."""
    from modules.parts import prep
    from modules.runtime import make_backend
    be = make_backend(MODEL, "onnx", "cpu")
    Z, lab = [], []
    batch = []
    for letter, kpi, sc in X_items:
        batch.append((letter, prep(kpi, sc)))
        if len(batch) == 64:
            Z.append(be.embed([b[1] for b in batch])); lab += [b[0] for b in batch]; batch = []
    if batch:
        Z.append(be.embed([b[1] for b in batch])); lab += [b[0] for b in batch]
    Z = np.concatenate(Z)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    return Z.astype(np.float16), np.array(lab)


def bank_items():
    """Training letter examples (letter, kpi, sc) for the letter bank."""
    out = []
    for s in ("en_alpha",):
        f = AGENT / s / "items.csv"
        if not f.exists():
            continue
        it = pd.read_csv(f)
        it = it[it.role == "train"]
        for r in it.itertuples():
            raw = AGENT / s / "pose_mp_raw" / f"{r.video}.npz"
            if raw.exists():
                kpi, sc = std_raw(raw)
                a, b = max(0, int(r.f0) - 3), min(len(kpi), int(r.f1) + 3)
                out.append((r.letter, kpi[a:b], sc[a:b]))
    m = pd.read_parquet(PREP / "manifest" / "clips.parquet")
    L = m[(m.dataset == "tslone") & m.concept.isin(TH_LETTERS) & (m.role != "test")]
    for r in L.itertuples():
        kpi, sc = std_file(PREP / r.pose_file)
        out.append((r.concept, kpi, sc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--no_bank", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    X, y, g, role = collect()
    classes = [c for c in EN_LETTERS + TH_LETTERS if c in set(y)] + [NONE]
    log.info("windows %d · classes %d · people %d (test %d)", len(X), len(classes), len(set(g)), len(set(g[role == "test"])))
    tr = role != "test"
    models = [train_mlp(X[tr], y[tr], classes, epochs=a.epochs, seed=s) for s in range(N_MODELS)]
    model = dict(models=models)
    rep = dict(held_out_people=evaluate(model, classes, X[~tr], y[~tr], g[~tr]), train=evaluate(model, classes, X[tr], y[tr], g[tr]))
    log.info("held-out people: %s", json.dumps(rep["held_out_people"], ensure_ascii=False))
    (ART / "runs" / "fingerspell").mkdir(parents=True, exist_ok=True)       # the train-roles-only model: honest comparisons
    np.savez(ART / "runs" / "fingerspell" / "heldout_model.npz", classes=np.array(classes), thr=np.float32(0.6), **flat(models))
    # final model on everyone (the held-out numbers above are the honest estimate)
    final = [train_mlp(X, y, classes, epochs=a.epochs, seed=s) for s in range(N_MODELS)]
    np.savez(MODEL / "fingerspell.npz", classes=np.array(classes), thr=np.float32(0.6), **flat(final))
    if not a.no_bank:
        Z, lab = letter_bank(bank_items())
        np.savez(MODEL / "letters_bank.npz", Z=Z, letter=lab)
        rep["letter_bank"] = dict(examples=int(len(lab)), letters=int(len(set(lab))))
    rep.update(classes=len(classes), windows=int(len(X)), minutes=round((time.time() - t0) / 60, 1))
    write_json(ART / "reports" / "fingerspell.json", rep)
    log.info("saved models/fingerspell.npz (+ letters_bank.npz) in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
