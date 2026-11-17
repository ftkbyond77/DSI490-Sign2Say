"""v11 zero-enrolment signer adaptation — simulation on TUNE signers (prompt-opt-v11 §3E).

  python main.py v11-adapt sim --enc <run> [--beta 0.3] [--seeds 5]   → artifacts/reports/v11_adapt_<run>.json

Encoder frozen. For every TUNE (val-role) signer with ≥ 20 clips, the signer's isolated clips (vocabulary words AND Tail signs = unknown
inputs) are streamed in random order; after 0 / 30 / 60 / 120 s of signing, the not-yet-seen clips are read:
  base         TRAIN bank (top-2 per word), no adaptation
  normaliser   + session normaliser: running estimate of the user's shoulder width, signing-space centre / scale (wrist spread around the
               shoulder centre) and tempo from the clips seen so far → applied to the landmarks before features()
  pseudo       + pseudo-enrolment: a word with ≥ 2 KNOWN-SURE (accepted) readings of the user gets the session prototype
               (1 − β)·proto_word + β·mean(user spans) (never from AMBIGUOUS / UNKNOWN readings)
  explicit     explicit enrolment (oracle labels of the seen clips, mean of the user's clips per word) — the upper reference
Accept rule = score margin threshold fitted on the other TUNE signers at 95 % accepted precision (leave-one-signer-out).
Reported per signer and stage: R@1 on unseen in-vocabulary clips, accepted error, open-set false accept (Tail clips accepted as a word).
Safety: a variant may ship (opt-in flag) only if accepted error and open-set false accept never rise vs base at any stage; on by default
only if it is strictly better for every TUNE signer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch

from modules.utils import ART, PREP, get_logger, read_json, write_json

log = get_logger("v11_adapt")
NULL = "<null>"


def session_norm(kp, sc, stats, pop):
    """re-scale the arms / hands of one clip around the shoulder centre so the user's signing space matches the population's"""
    if not stats or stats["n"] < 3:
        return kp
    k = kp.copy()
    ok = (sc[:, 5] > 0.3) & (sc[:, 6] > 0.3)
    if not ok.any():
        return k
    mid = ((k[ok, 5] + k[ok, 6]) / 2).mean(0)
    f = float(np.clip(pop["space"] / max(stats["space"], 1e-3), 0.7, 1.4))
    idx = np.r_[7:11, 91:133]
    k[:, idx] = mid + (k[:, idx] - mid) * f
    return k


def clip_stats(kp, sc):
    ok = (sc[:, 5] > 0.3) & (sc[:, 6] > 0.3)
    if not ok.any():
        return None
    mid = ((kp[ok, 5] + kp[ok, 6]) / 2).mean(0)
    sw = float(np.median(np.linalg.norm(kp[ok, 5] - kp[ok, 6], axis=1)))
    w = np.r_[kp[(sc[:, 9] > 0.3), 9], kp[(sc[:, 10] > 0.3), 10]]
    space = float(np.median(np.linalg.norm(w - mid, axis=1)) / max(sw, 1e-3)) if len(w) else 1.0
    return dict(sw=sw, space=space, T=len(kp))


def cmd_sim(a):
    from train_encoder import PoseStoreMM, ensure_mm
    from modules.lmenc import embed, load_model, prep_for
    from modules.parts import to_iso
    from modules.schema import canonical_hands
    from modules.sealed import sealed_signers
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ART / "runs" / a.enc / "encoder_best.pt", dev)
    prep = prep_for(model.cfg)
    ensure_mm(); S = PoseStoreMM()
    C = pd.read_parquet(PREP / "shards" / "clips.parquet")
    C = C[(C.kind == "isolated") & (C.view == "primary") & C.has_span & C.concept.notna() & C.clip_id.isin(S.row) & ~C.signer.isin(sealed_signers())]
    vocab = set(read_json(ROOT / "models" / "model-v10" / "vocab_conversation.json")["concepts"])
    tr = C[(C.role == "train") & C.concept.isin(vocab)]
    va = C[(C.role == "val")]

    def load(r):
        kp, sc, hw = S.get(S.row[r.clip_id]); kp, sc, _ = canonical_hands(kp, sc, hw)
        kp = to_iso(kp, hw); f0, f1 = int(r.span_f0), max(int(r.span_f1), int(r.span_f0) + 4)
        return kp[f0:f1], sc[f0:f1]
    log.info("embedding %d TRAIN bank clips", len(tr))
    Zb = embed(model, [prep(*load(r)) for r in tr.itertuples()], device=dev)
    words = sorted(set(tr.concept)); widx = {w: i for i, w in enumerate(words)}
    cb = tr.concept.map(widx).values

    def protos(Z, c):
        P = np.zeros((len(words), Z.shape[1]), np.float32)
        for i in range(len(words)):
            m = c == i
            if m.any():
                v = Z[m].mean(0); P[i] = v / (np.linalg.norm(v) + 1e-8)
        return P
    P0 = protos(Zb, cb)
    pop = dict(space=float(np.median([s["space"] for s in (clip_stats(*load(r)) for r in tr.sample(min(800, len(tr)), random_state=0).itertuples()) if s])))

    def score(Q, P, user=None):
        Sx = Q @ P.T
        if user:
            for i, u in user.items():
                Sx[:, i] = np.maximum(Sx[:, i], Q @ u)
        o = np.argsort(-Sx, 1)
        return o[:, 0], Sx[np.arange(len(Q)), o[:, 0]] - Sx[np.arange(len(Q)), o[:, 1]]

    signers = [s for s, g in va.groupby("signer") if len(g) >= 20]
    pre = {}
    for s in signers:
        g = va[va.signer == s]
        clips = [load(r) for r in g.itertuples()]
        pre[s] = dict(g=g.reset_index(drop=True), clips=clips, Z=embed(model, [prep(k, c) for k, c in clips], device=dev))
    # + the TSL51 signer's aligned sentence words (TUNE sentences, role tagger_train): the one TUNE stream where words REPEAT, so
    #   pseudo / explicit enrolment can act (isolated TUNE data has one clip per signer × word)
    A = read_json(ART / "runs" / "seq-v7" / "tsl51_alignment.json")
    rows_, clips_ = [], []
    for cid, v in A.items():
        key = "t51s:" + cid.split(":", 1)[1]
        if v["role"] != "tagger_train" or key not in S.row:
            continue
        kp, sc, hw = S.get(S.row[key]); kp, sc, _ = canonical_hands(kp, sc, hw); kp = to_iso(kp, hw)
        for sg in v["segments"]:
            f0, f1 = int(sg["f0"]), max(int(sg["f1"]), int(sg["f0"]) + 4)
            rows_.append(dict(concept=sg["concept"], signer="S_51_user(sentences)")); clips_.append((kp[f0:f1], sc[f0:f1]))
    if rows_:
        signers.append("S_51_user(sentences)")
        pre["S_51_user(sentences)"] = dict(g=pd.DataFrame(rows_), clips=clips_, Z=embed(model, [prep(k, c) for k, c in clips_], device=dev))
    # accept threshold per held-out signer: margin at 95 % precision on the OTHER TUNE signers (base reading)
    def thr_without(s):
        ms, ok = [], []
        for t in signers:
            if t == s:
                continue
            g = pre[t]["g"]; iv = g.concept.isin(words).values
            pred, mg = score(pre[t]["Z"][iv], P0)
            ms += list(mg); ok += list(pred == g.concept[iv].map(widx).values)
        ms, ok = np.array(ms), np.array(ok)
        for th in np.unique(np.round(ms, 3)):
            sel = ms >= th
            if sel.sum() >= 20 and ok[sel].mean() >= 0.95:
                return float(th)
        return 1.0
    stages = (0, 30, 60, 120)
    res = {}
    for s in signers:
        th = thr_without(s)
        g = pre[s]["g"]; Z = pre[s]["Z"]; clips = pre[s]["clips"]
        rows = {v: {st: [] for st in stages} for v in ("base", "normaliser", "pseudo", "explicit")}
        for seed in range(a.seeds):
            order = np.random.RandomState(seed).permutation(len(g))
            secs = np.cumsum([len(clips[i][0]) / 25.0 for i in order])
            for st in stages:
                k = int(np.searchsorted(secs, st, side="right")) if st else 0
                seen, unseen = order[:k], order[k:]
                if len(unseen) < 5:
                    continue
                stats_l = [clip_stats(*clips[i]) for i in seen]; stats_l = [x for x in stats_l if x]
                stats = dict(n=len(stats_l), space=float(np.median([x["space"] for x in stats_l]))) if stats_l else None
                Zn = embed(model, [prep(session_norm(clips[i][0], clips[i][1], stats, pop), clips[i][1]) for i in unseen], device=dev) if stats and stats["n"] >= 3 else Z[unseen]
                # pseudo-enrolment from ACCEPTED readings of the seen clips (no labels)
                user = {}
                if len(seen):
                    pr, mg = score(Z[seen], P0)
                    acc = mg >= th
                    for w in np.unique(pr[acc]):
                        mm = acc & (pr == w)
                        if mm.sum() >= 2:
                            v = (1 - a.beta) * P0[w] + a.beta * Z[seen][mm].mean(0); user[int(w)] = v / (np.linalg.norm(v) + 1e-8)
                ex = {}
                gs = g.iloc[seen]
                for w, gg in gs[gs.concept.isin(words)].groupby("concept"):
                    v = Z[seen][(gs.concept == w).values].mean(0); ex[widx[w]] = v / (np.linalg.norm(v) + 1e-8)
                gu = g.iloc[unseen]
                iv = gu.concept.isin(words).values
                tgt = gu.concept.map(lambda w: widx.get(w, -1)).values
                for v, (Q, U) in dict(base=(Z[unseen], None), normaliser=(Zn, None), pseudo=(Zn, user), explicit=(Z[unseen], ex)).items():
                    pred, mg = score(Q, P0, U)
                    acc = mg >= th
                    r1 = float((pred[iv] == tgt[iv]).mean()) if iv.any() else np.nan
                    ae = float((pred[iv & acc] != tgt[iv & acc]).mean()) if (iv & acc).any() else 0.0
                    fa = float(acc[~iv].mean()) if (~iv).any() else np.nan
                    rows[v][st].append((r1, ae, fa))
        res[s] = {v: {str(st): dict(R1=float(np.nanmean([x[0] for x in L])) if L else None, accepted_error=float(np.nanmean([x[1] for x in L])) if L else None,
                                     openset_fa=float(np.nanmean([x[2] for x in L])) if L else None) for st, L in d.items()} for v, d in rows.items()}
        log.info("%s: %s", s, {v: [round(res[s][v][str(st)]["R1"] or 0, 3) for st in stages] for v in res[s]})
    # safety + default decision
    verdict = {}
    for v in ("normaliser", "pseudo"):
        safe = all((res[s][v][str(st)]["accepted_error"] or 0) <= (res[s]["base"][str(st)]["accepted_error"] or 0) + 1e-9 and
                   (res[s][v][str(st)]["openset_fa"] or 0) <= (res[s]["base"][str(st)]["openset_fa"] or 0) + 1e-9
                   for s in res for st in stages if res[s][v][str(st)]["R1"] is not None)
        better_all = all((res[s][v]["120"]["R1"] or 0) > (res[s]["base"]["120"]["R1"] or 0) for s in res if res[s][v]["120"]["R1"] is not None)
        verdict[v] = dict(safe=bool(safe), strictly_better_every_signer=bool(better_all), default="on" if (safe and better_all) else ("opt-in" if safe else "off"))
    write_json(ART / "reports" / f"v11_adapt_{a.enc}.json", dict(enc=a.enc, beta=a.beta, seeds=a.seeds, stages=stages, signers=res, verdict=verdict, pop=pop))
    log.info("verdict %s", verdict)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim"); s.add_argument("--enc", required=True); s.add_argument("--beta", type=float, default=0.3); s.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    dict(sim=cmd_sim)[a.cmd](a)


if __name__ == "__main__":
    main()
