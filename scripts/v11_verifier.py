"""v11 minimal-pair verifier fit (prompt-opt-v11 §3B; modules/verifier.py).

  python main.py v11-verifier fit --enc <run> [--virtual 2]   → artifacts/runs/<run>/verifier.pkl + artifacts/reports/v11_verifier_<run>.json

TRAIN clips of every confusion-group word (real + `virtual` augmented signer copies, modules/v11data.augment_v11) → pair features
against every other group member's reference (built from TRAIN clips of OTHER signers, so a clip never meets its own signer's
exemplars) → gradient-boosted trees per group, leave-signers-out CV. TUNE check: val-role isolated clips of the group words, R@1 inside
the group with the encoder alone (argmax of emb_proto) vs with the verifier. Kept per group only if TUNE R@1 rises.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch

from modules.utils import ART, PREP, get_logger, read_json, write_json

log = get_logger("v11_verifier")


def segfeat(model, prep, kp, sc, f0, f1, dev):
    from modules.handshape import descriptor
    from modules.lmenc import embed, frames_long
    from modules.verifier import _sub, phon_stats
    z = embed(model, [prep(kp[f0:f1], sc[f0:f1])], device=dev)[0]
    H = frames_long(model, kp[f0:f1], sc[f0:f1], device=dev)
    H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    loc, mov = phon_stats(kp, sc, f0, f1)
    return dict(z=z, Hs=H[_sub(len(H))], desc=descriptor(kp, sc, f0, f1), loc=loc, mov=mov, len=f1 - f0)


def cmd_fit(a):
    from sklearn.ensemble import GradientBoostingClassifier
    from train_encoder import PoseStoreMM, ensure_mm
    from modules.lmenc import load_model, prep_for
    from modules.parts import to_iso
    from modules.schema import canonical_hands
    from modules.sealed import sealed_signers
    from modules.v11data import augment_v11
    from modules.verifier import FEATS, Verifier, WordRef, pair_features
    from v11 import GROUPS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ART / "runs" / a.enc / "encoder_best.pt", dev)
    prep = prep_for(model.cfg)
    n_sub = int((model.cfg.get("members") or [model.cfg])[0].get("phon", 0) if "members" in model.cfg else model.cfg.get("phon", 0))
    gf = PREP / "manifest" / "confusion_groups_tune.json"
    groups = [g for g in GROUPS + (read_json(gf)["groups"] if gf.exists() else [])]
    words = sorted({w for g in groups for w in g})
    ensure_mm(); S = PoseStoreMM()
    C = pd.read_parquet(PREP / "shards" / "clips.parquet")
    C = C[(C.kind == "isolated") & (C.view == "primary") & C.has_span & C.concept.isin(words) & C.clip_id.isin(S.row) & ~C.signer.isin(sealed_signers())]
    C = C.assign(signer=C.signer.fillna(C.dataset + ":unknown"))
    cf = read_json(PREP / "manifest" / "forms.json")["clip_form"]
    rng = np.random.RandomState(0)

    def load(r, aug=False):
        kp, sc, hw = S.get(S.row[r.clip_id])
        if aug:
            kp2, sc2 = augment_v11(kp, sc, hw, rng)
            from modules.parts import resample_len
            f = len(kp2) / len(kp)
            return kp2, sc2, int(r.span_f0 * f), max(int(r.span_f1 * f), int(r.span_f0 * f) + 4)
        kp, sc, _ = canonical_hands(kp, sc, hw)
        return to_iso(kp, hw), sc, int(r.span_f0), max(int(r.span_f1), int(r.span_f0) + 4)

    tr, va = C[C.role == "train"], C[C.role == "val"]
    log.info("verifier: %d groups, %d words, %d TRAIN clips, %d TUNE clips", len(groups), len(words), len(tr), len(va))
    feats = {}
    for r in tr.itertuples():
        kp, sc, f0, f1 = load(r)
        feats[r.clip_id] = segfeat(model, prep, kp, sc, f0, f1, dev)
    def build_ref(w, exclude_signer=None):
        ref = WordRef()
        g = tr[(tr.concept == w) & (tr.signer != exclude_signer)]
        for f, gg in g.groupby(g.clip_id.map(lambda c: cf.get(c, [None, "f0"])[1])):
            zs = [np.mean([feats[c]["z"] for c in g2.clip_id], 0) for _, g2 in gg.groupby("signer")]
            p = np.mean([z / (np.linalg.norm(z) + 1e-8) for z in zs], 0); ref.P.append(p / (np.linalg.norm(p) + 1e-8))
        for r in g.itertuples():
            fe = feats[r.clip_id]
            ref.E.append(fe["z"]); ref.S.append(r.signer); ref.D.append(fe["desc"]); ref.loc.append(fe["loc"]); ref.mov.append(fe["mov"]); ref.L.append(fe["len"])
        for _, g2 in g.groupby("signer"):
            ref.F.append(feats[g2.clip_id.iloc[0]]["Hs"])
        return ref
    refs = {w: build_ref(w) for w in words if (tr.concept == w).any()}
    # TUNE evidence where the decisive groups live: aligned words of the TSL51 TUNE sentences (role tagger_train, an unseen signer)
    A = read_json(ART / "runs" / "seq-v7" / "tsl51_alignment.json")
    t51_spans = []
    for cid, v in A.items():
        key = "t51s:" + cid.split(":", 1)[1]
        if v["role"] != "tagger_train" or key not in S.row:
            continue
        kp, sc, hw = S.get(S.row[key]); kp, sc, _ = canonical_hands(kp, sc, hw); kp = to_iso(kp, hw)
        for sg in v["segments"]:
            if sg["concept"] in words:
                t51_spans.append((kp, sc, int(sg["f0"]), max(int(sg["f1"]), int(sg["f0"]) + 4), sg["concept"]))
    log.info("TUNE check: %d isolated val clips + %d TSL51 sentence spans of group words", len(va), len(t51_spans))
    models, report = {}, {}
    for g in groups:
        g = [w for w in g if w in refs]
        if len(g) < 2:
            continue
        X, y, sg = [], [], []
        for r in tr[tr.concept.isin(g)].itertuples():
            loo = {w: build_ref(w, exclude_signer=r.signer) for w in g}          # the clip's own signer is never in the reference
            items = [feats[r.clip_id]]
            for _ in range(a.virtual):
                kp, sc, f0, f1 = load(r, aug=True)
                items.append(segfeat(model, prep, kp, sc, f0, f1, dev))
            for fe in items:
                for w in g:
                    if not loo[w].E:
                        continue
                    X.append(pair_features(fe, loo[w], n_sub)); y.append(int(w == r.concept)); sg.append(r.signer)
        if len(set(y)) < 2 or len(X) < 40:
            continue
        X, y, sg = np.array(X), np.array(y), np.array(sg)
        clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0).fit(X, y)
        # TUNE check: val clips of the group words, R@1 inside the group (encoder alone vs verifier)
        vv = va[va.concept.isin(g)]
        enc_ok = ver_ok = n = 0
        items = [load(r) + (r.concept,) for r in vv.itertuples()] + [x for x in t51_spans if x[4] in g]
        for kp, sc, f0, f1, concept in items:
            r = type("R", (), dict(concept=concept))
            fe = segfeat(model, prep, kp, sc, f0, f1, dev)
            P = np.array([pair_features(fe, refs[w], n_sub) for w in g])
            enc_ok += g[int(np.argmax(P[:, 0]))] == r.concept
            ver_ok += g[int(np.argmax(clf.predict_proba(P)[:, 1]))] == r.concept
            n += 1
        keep = n >= 5 and ver_ok > enc_ok
        report["|".join(g)] = dict(train_pairs=int(len(y)), tune_n=n, tune_R1_encoder=enc_ok / max(n, 1), tune_R1_verifier=ver_ok / max(n, 1), kept=bool(keep))
        log.info("group %s: TUNE R@1 in-group encoder %.3f → verifier %.3f (n %d) %s", g, enc_ok / max(n, 1), ver_ok / max(n, 1), n, "KEPT" if keep else "dropped")
        if keep:
            models[tuple(g)] = clf
    out = ART / "runs" / a.enc / "verifier.pkl"
    with open(out, "wb") as f:
        pickle.dump(dict(refs=refs, models=models, n_sub=n_sub, feats=FEATS, enc=a.enc), f)
    write_json(ART / "reports" / f"v11_verifier_{a.enc}.json", dict(groups=report, kept=len(models)))
    log.info("verifier: %d / %d groups kept → %s", len(models), len(report), out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--enc", required=True); f.add_argument("--virtual", type=int, default=2)
    a = ap.parse_args()
    dict(fit=cmd_fit)[a.cmd](a)


if __name__ == "__main__":
    main()
