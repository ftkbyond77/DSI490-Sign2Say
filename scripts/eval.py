"""ThaiSLM — local build / tuning / evaluation (v7) on top of a finished encoder run and the local sequence stage.

  python main.py eval build    --run <encoder_run_dir> --seq <sequence_run_dir>   # models/ (deployable) + artifacts/eval (train-only bank)
  python main.py eval grammar                                                      # TSL word order (TSL51) → models/grammar.json, roles.json
  python main.py eval spot     --seq <sequence_run_dir>                            # decoder, word prior, tempo, exact re-score, calibration,
                                                                                   #   transition / repeat filters — per vocabulary
  python main.py eval isolated --run <encoder_run_dir> [--v6]                      # held-out signers, bank depth, ASL/CSL prior bias, latent space
  python main.py test          [--tag final] [--vocab conversation|full] [--no_llm] # data_test end-to-end from the raw videos (+ scores)
  python main.py eval selftest [--make]                                            # self-made sentence videos (held-out signers) end-to-end
  python main.py eval vocab                                                        # vocab/ exports

Rules kept from v4–v6: nothing is tuned on data_test or on the self-made test videos. Decoder / calibration decisions use the TSL51
TUNE templates with out-of-fold tagger outputs + synthetic held-out-signer sentences (tune half); reports use the TSL51 TEST
templates, the synthetic test half, the self-made videos and data_test.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from modules.utils import ART, MODEL, PREP, get_logger, read_json, write_json

log = get_logger("eval")
import os
EVAL, REP, MAN = ART / "eval", ART / "reports", PREP / "manifest"
SHARDS = Path(os.environ.get("THAISLM_SHARDS", PREP / "shards"))
NULL = "<null>"
VERSION = "v7"
LENS = list(range(8, 50, 4))


def run_dir(p):
    d = Path(p)
    if not d.exists():
        d = ART / "runs" / p
    assert d.exists(), f"run not found: {p}"
    return d


# ----------------------------------------------------------------------------- build: models/ + artifacts/eval
def conversation_vocab(concepts, present):
    """Daily-conversation lexicon (vocab/daily_conversation.json) ∪ the TSL51 conversation vocabulary, restricted to concepts the
    bank can recognise."""
    cmap = read_json(MAN / "concepts.json")["concept"]
    daily = read_json(ROOT / "vocab" / "daily_conversation.json")
    words = {cmap.get(w, w) for k, v in daily.items() if not k.startswith("_") for w in v}
    clips = pd.read_parquet(SHARDS / "clips.parquet")
    words |= set(clips[(clips.dataset == "tsl51") & (clips.kind == "isolated")].concept.dropna())
    keep = sorted(w for w in words if w in concepts and concepts.index(w) in present)
    missing = sorted(w for w in words if w not in keep)
    return keep, missing


def cmd_build(a):
    rd, sd = run_dir(a.run), run_dir(a.seq)
    MODEL.mkdir(parents=True, exist_ok=True); EVAL.mkdir(parents=True, exist_ok=True)
    shutil.copy(rd / "encoder_best.pt", MODEL / "encoder.pt")
    for f in ("tagger.pt", "span_head.npz", "null_proto.npy"):
        shutil.copy(sd / f, MODEL / f)
    shutil.copy(ART / "pretrained" / "mt5_config.json", MODEL / "mt5_config.json")
    concepts = read_json(rd / "concept_index.json")
    write_json(MODEL / "concepts.json", concepts)
    cidx = {c: i for i, c in enumerate(concepts)}
    C = len(concepts)
    idx = pd.read_parquet(rd / "emb_index.parquet")
    Z = np.load(rd / "emb.npy").astype(np.float32)
    sign = (idx.concept != NULL).values & idx.c.notna().values

    def save_bank(path, mask, extra=None):
        Zb, cb = Z[mask], idx.c.values[mask].astype(np.int32)
        clip, ds, sig = list(idx.clip_id.values[mask]), list(idx.dataset.values[mask]), list(idx.signer.values[mask])
        if extra is not None and len(extra[0]):
            Zb = np.concatenate([Zb, extra[0]]); cb = np.concatenate([cb, extra[1]])
            clip += [m[0] for m in extra[2]]; ds += [m[1] for m in extra[2]]; sig += [m[2] for m in extra[2]]
        np.savez(path, Z=Zb.astype(np.float16), c=cb, clip=np.array([str(x) for x in clip]), dataset=np.array([str(x) for x in ds]),
                 signer=np.array([str(x) for x in sig]))
        return cb
    save_bank(EVAL / "bank_eval.npz", sign & (idx.role == "train").values)
    # deployment bank: every labelled clip (all roles — real users are new people anyway) + confidently aligned words of the
    # TSL51 sentence videos (continuous-signing examples of the 51 conversation words)
    from modules.encoder import embed_parts, load_encoder
    from modules.parts import prep, to_iso
    from modules.encoder import PoseStore
    align = read_json(sd / "tsl51_alignment.json")
    scores = [s["score"] for v in align.values() for s in v["segments"]]
    cut = float(np.quantile(scores, 0.25)) if scores else 1.0
    store = PoseStore(SHARDS / "pose_store.npz")
    enc = load_encoder(MODEL, MODEL / "encoder.pt", device="cuda")
    parts, meta, cs = [], [], []
    for cid, v in align.items():
        if cid not in store.row:
            continue
        kp, sc, hw = store.get(store.row[cid])
        kp = to_iso(kp, hw)
        for s in v["segments"]:
            if s["score"] >= cut and s["concept"] in cidx and s["f1"] - s["f0"] >= 6 and s["f1"] <= len(kp):
                parts.append(prep(kp[s["f0"]:s["f1"]], sc[s["f0"]:s["f1"]]))
                meta.append((f"{cid}@{s['f0']}", "tsl51_sentence_aligned", "S_51_user")); cs.append(cidx[s["concept"]])
    Zs = embed_parts(enc, parts, bs=64) if parts else np.zeros((0, 768), np.float32)
    cd = save_bank(MODEL / "bank.npz", sign, (Zs, np.array(cs, np.int32), meta))
    present = set(cd.tolist())
    conv, missing = conversation_vocab(concepts, present)
    depth = pd.DataFrame(dict(c=cd, s=np.load(MODEL / "bank.npz")["signer"])).groupby("c").s.nunique()
    write_json(MODEL / "vocab_conversation.json", dict(
        doc="Conversation-mode vocabulary: vocab/daily_conversation.json ∪ TSL51 words, restricted to concepts with ≥ 1 bank example. "
            "signers = distinct people / channels behind each word in models/bank.npz.",
        concepts=conv, signers={w: int(depth.get(cidx[w], 0)) for w in conv}, not_in_bank=missing))
    from modules.decode import frequency_prior
    try:
        from pythainlp.corpus import tnc
        groups = read_json(MAN / "concepts.json")["groups"]
        np.save(MODEL / "freq_prior.npy", frequency_prior(concepts, groups, dict(tnc.word_freqs())))
    except Exception as e:  # noqa: BLE001
        log.warning("no frequency prior: %s", e)
    for f in ("hub_deploy.npy", "depth_deploy.npy", "bank_deploy.npz", "onnx/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx",
              "onnx/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx", "onnx/yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx"):
        (MODEL / f).unlink(missing_ok=True)                    # v6 leftovers (RTMW extractor, switched-off hubness / depth priors)
    cfg = dict(version=VERSION, encoder_run=rd.name, sequence_run=sd.name, tagger_arch="gru", extractor="MediaPipe Holistic (tasks 1.0.1)",
               built=time.strftime("%Y-%m-%d %H:%M"), concepts=C, bank_eval=int(((idx.role == "train").values & sign).sum()), bank=int(len(cd)),
               bank_concepts=len(present), conversation_concepts=len(conv), aligned_sentence_words_added=len(meta), alignment_score_cut=cut)
    write_json(MODEL / "config.json", cfg)
    log.info("models/ built: %s", cfg)


# ----------------------------------------------------------------------------- grammar
def cmd_grammar(a):
    from modules.grammar import build_role_table, learn_patterns
    from modules.lexicon import thsl_tags, ttrs_pos
    lex = pd.read_parquet(MAN / "lexicon.parquet")
    role = build_role_table(lex, ttrs_pos(lex), thsl_tags(lex))
    s = pd.read_parquet(SHARDS / "clips.parquet")
    s = s[(s.kind == "sentence") & (s.view == "primary")]
    uniq = sorted({" ".join(x) for x in s.concept.str.split("|")})
    uniq_tune = sorted({" ".join(x) for x in s[s.role == "tagger_train"].concept.str.split("|")})
    EVAL.mkdir(parents=True, exist_ok=True)
    write_json(EVAL / "grammar_tune.json", learn_patterns([x.split() for x in uniq_tune], role))
    pat = learn_patterns([x.split() for x in uniq], role)
    thai = read_json(PREP / "params" / "tsl51_thai_sentences.json")
    pat["examples"] = [dict(tsl_order=g, thai=thai.get(g, ""), roles=" ".join(role.get(w, "OBJ") for w in g.split())) for g in uniq]
    write_json(MODEL / "grammar.json", pat)
    write_json(MODEL / "roles.json", role)
    from modules.grammar import language_roles
    write_json(MODEL / "roles_language.json", language_roles(role, read_json(ROOT / "vocab" / "daily_conversation.json")))
    log.info("grammar: %d TSL51 sentences, %d role templates, roles for %d concepts", len(uniq), len(pat["templates"]), len(role))


# ----------------------------------------------------------------------------- spotting simulation on CPU (same code as inference)
class Exact:
    """Exact (teacher) embeddings of individual windows, computed on demand on the GPU and cached — exactly what the production
    pipeline computes for the segments the decoder chose (modules.runtime.rescore)."""

    def __init__(self, model_dir=MODEL, device="cuda"):
        from modules.runtime import TorchBackend
        self.be = TorchBackend(model_dir, device)
        self.cache = {}

    def ensure(self, reqs):
        from modules.parts import prep
        todo, seen = [], set()
        for key, kp, sc, a, b in reqs:
            k = (key, a, b)
            if k not in self.cache and k not in seen:
                seen.add(k); todo.append((k, kp, sc, a, b))
        for i in range(0, len(todo), 2048):
            chunk = todo[i:i + 2048]
            Z = self.be.embed([prep(kp[a:b].astype(np.float32), sc[a:b].astype(np.float32)) for _, kp, sc, a, b in chunk])
            for (k, *_), z in zip(chunk, Z):
                self.cache[k] = z

    def get(self, key, a, b):
        return self.cache[(key, a, b)]


def _subset(t, keep):
    k = np.flatnonzero(keep)
    return dict(wins=[t["wins"][i] for i in k], top_idx=t["top_idx"][k], top_sc=t["top_sc"][k].astype(np.float32), mu=t["mu"][k], sd=t["sd"][k],
                null=t["null"][k])


def _grid_mask(t, cfg):
    W = np.asarray(t["wins"]) if len(t.get("wins", [])) else np.zeros((0, 2), int)
    if not len(W):
        return np.zeros(0, bool)
    return (W[:, 0] % cfg["stride"] == 0) & np.isin(W[:, 1] - W[:, 0], cfg["lens"])


def decode_item(it, cfg, vocab, concepts, prior):
    """Decoder part of modules.pipeline on stored tables: tempo normalisation → (student | teacher) table → semi-Markov decode.
    → (segments in the decoded timeline, tempo factor, pose source key + poses for exact re-scoring)."""
    from modules.decode import apply_csls, concept_prior, decode
    from modules.segment import decode_tagger
    tables, prob, f, kp, sc, key = it["tables"], it["prob"].astype(np.float32), 1.0, it["kp"], it["sc"], it["id"]
    tempo = cfg.get("tempo")
    if tempo and it.get("stretched"):
        segs_t = decode_tagger(prob)
        if len(segs_t) >= tempo.get("min_segments", 2):
            med = float(np.median([b - a for a, b in segs_t]))
            f_raw = float(np.clip(tempo["ref_len"] / max(med, 1.0), 1.0, tempo.get("max_factor", 2.5)))
            if f_raw >= tempo.get("min_factor", 1.2):
                f = min((1.5, 2.0), key=lambda x: abs(x - f_raw))
                st = it["stretched"][f]
                tables, prob, kp, sc, key = st["tables"], st["prob"].astype(np.float32), st["kp"], st["sc"], f"{it['id']}~{f}"
    src = "student" if cfg.get("one_pass", True) else "teacher"
    t = tables.get(f"{src}/{vocab}")
    if t is None or not len(t.get("wins", [])):
        return [], f, key, kp, sc
    t = _subset(t, _grid_mask(t, cfg))
    if not len(t["wins"]):
        return [], f, key, kp, sc
    wp = concept_prior(prior, None, cfg.get("freq_gamma", 0.0), 0.0)
    segs = decode(apply_csls(t, None, 0.0), prob, concepts, cfg, rank_table=apply_csls(t, None, 0.0, wp, 1.0))
    return segs, f, key, kp, sc


def evaluate(items, cfg, vocab, concepts, prior, ctx, vocab_set=None):
    """Decode every item like the production pipeline (incl. exact re-scoring of the chosen segments) and score it."""
    from modules.decode import aggregate, sentence_metrics, wer
    from modules.runtime import rescore, rescore_windows
    from modules.segment import gt_segments
    dec = [decode_item(it, cfg, vocab, concepts, prior) for it in items]
    exact = cfg.get("exact_rescore", True) and cfg.get("one_pass", True)
    if exact:
        ctx["exact"].ensure([(key, kp, sc, a, b) for segs, f, key, kp, sc in dec for ws in rescore_windows(segs) for a, b in ws])
    rows, segs_all = [], []
    for it, (segs, f, key, kp, sc) in zip(items, dec):
        if segs and exact:
            Zs = [np.stack([ctx["exact"].get(key, a, b) for a, b in ws]) for ws in rescore_windows(segs)]
            rescore(segs, Zs, ctx["banks"][vocab], ctx["null"], concepts, prior, cfg.get("freq_gamma", 0.0))
        if f > 1.0:
            for s in segs:
                s["f0"], s["f1"] = int(round(s["f0"] / f)), max(int(round(s["f1"] / f)), int(round(s["f0"] / f)) + 1)
        gt = gt_segments(it["y"])
        m = sentence_metrics(segs, gt, it["glosses"][:len(gt)], vocab_set)
        m["wer"] = wer(it["glosses"], [s["nearest"] for s in segs]); m["n_gt"] = len(it["glosses"])
        rows.append(m); segs_all.append((segs, m["correct"]))
    return aggregate(rows), segs_all


def _objective(res):
    return float(np.mean([r["spot_f1"] + 0.5 * r["seg_f1"] - 0.05 * r["count_abs_err"] for r in res.values()]))


def _sets(items):
    out = {}
    for it in items:
        key = f"{it['kind']}_{it['split']}" + (f"_x{it['speed']}" if it["kind"] == "real" else "")
        out.setdefault(key, []).append(it)
    return out


def cmd_spot(a):
    from modules.decode import FEATS, LogReg, postprocess, seg_features, status_of, wer
    from modules.grammar import rerank
    sd = run_dir(a.seq)
    D = pickle.load(open(sd / "seq_tables.pkl", "rb"))
    concepts, sets = D["concepts"], _sets(D["items"])
    prior = np.load(MODEL / "freq_prior.npy") if (MODEL / "freq_prior.npy").exists() else None
    from modules.runtime import Bank
    be = np.load(EVAL / "bank_eval.npz")
    ctx = dict(exact=Exact(), null=np.load(sd / "null_proto.npy").astype(np.float32),
               banks={"full": Bank(be["Z"].astype(np.float32), be["c"], len(concepts)),
                      "conversation": Bank(be["Z"].astype(np.float32), be["c"], len(concepts), allowed=set(D["conversation"]))})
    log.info("evaluation sets: %s", {k: len(v) for k, v in sets.items()})
    tune = [k for k in sets if "_tune" in k]
    test = [k for k in sets if "_test" in k]
    base = dict(lens=LENS, stride=2, min_inside=0.6, delta=0.0, act_thr=0.5, mu_null=5.0, csls_alpha=0.0, one_pass=True, exact_rescore=True)
    spotting, calibration, report = {}, {}, {}
    for vocab in ("conversation", "full"):
        t0 = time.time()
        conv_set = {concepts[c] for c in D["conversation"]} if vocab == "conversation" else None
        R = lambda cfg, keys: {k: evaluate(sets[k], cfg, vocab, concepts, prior, ctx, conv_set)[0] for k in keys}  # noqa: E731
        grid = []
        for lam, gap, min_len, beta in product([2.0, 4.0, 6.0, 8.0, 10.0], [0.5, 1.5], [10, 14], [3.0, 5.0]):
            cfg = dict(base, lam=lam, gap_cost=gap, min_len=min_len, beta=beta, freq_gamma=0.0)
            grid.append(dict(cfg=cfg, objective=_objective(R(cfg, tune)), stage="decoder"))
        best = max(grid, key=lambda g: g["objective"])["cfg"]
        # word prior γ · window grid (cost) · exact re-score · one-pass (student) vs teacher windows
        for gamma, stride, lens_name, exact in product([0.0, 0.25, 0.5], [2, 4], ["all", "sparse"], [True, False]):
            cfg = dict(best, freq_gamma=gamma, stride=stride, lens=LENS if lens_name == "all" else [8, 12, 16, 20, 28, 36, 48], exact_rescore=exact)
            grid.append(dict(cfg=cfg, objective=_objective(R(cfg, tune)), stage="prior/windows/exact"))
        cands = [g for g in grid if g["stage"] == "prior/windows/exact"]
        top = max(g["objective"] for g in cands)
        # accuracy first; among configs within 0.005 of the best, the cheapest (fewer windows, no exact re-score) wins
        cost = lambda c: (len(c["lens"]) / c["stride"]) + (3 if c["exact_rescore"] else 0)  # noqa: E731
        best = min((g for g in cands if g["objective"] >= top - 0.005), key=lambda g: cost(g["cfg"]))["cfg"]
        teacher = dict(best, one_pass=False)
        grid.append(dict(cfg=teacher, objective=_objective(R(teacher, tune)), stage="reference: teacher windows (slow)"))
        tres = []
        for opt in [None] + [dict(ref_len=r, min_factor=1.2, max_factor=2.5, min_segments=2) for r in (16, 20, 24, 28, 32)]:
            cfg = dict(best, tempo=opt)
            tres.append(dict(tempo=opt, objective=_objective(R(cfg, tune))))
        best["tempo"] = max(tres, key=lambda r: r["objective"])["tempo"]
        # calibration on the tune decodes (out-of-fold tagger), thresholds for precision ≥ accept_precision / ≥ 0.45
        segs_tune = {k: evaluate(sets[k], best, vocab, concepts, prior, ctx, conv_set)[1] for k in tune}
        X = np.concatenate([seg_features(s, FEATS) for k in tune for s, _ in segs_tune[k]] or [np.zeros((0, len(FEATS)))])
        y = np.concatenate([np.array(c, float) for k in tune for _, c in segs_tune[k]] or [np.zeros(0)])
        lr = LogReg().fit(X, y); p_fit = lr.predict(X)

        def tau_for(prec):
            for t in np.linspace(0.05, 0.95, 91):
                mm = p_fit >= t
                if mm.sum() >= 10 and y[mm].mean() >= prec:
                    return float(t)
            return 0.95
        cal = dict(model=lr.to_dict(FEATS), tau_accept=tau_for(a.accept_precision), tau_uncertain=tau_for(0.45), fitted_on=tune)
        # post-processing (transition filter + repeat penalty): chosen on tune by the precision of the words a sentence may use
        def shown(segs_):
            P = lr.predict(seg_features(segs_, FEATS)) if segs_ else np.zeros(0)
            for s_, p_ in zip(segs_, P):
                s_["p_correct"] = float(p_); s_["status"] = status_of(p_, cal)
            return segs_
        post_res = []
        for mf, ng, ps in product([8, 10, 12, 14], [0.0, 0.05, 0.1], [0.45, 0.55, 0.65]):
            pp = dict(min_frames=mf, null_gap_max=ng, p_sign_min=ps, repeat_penalty=True)
            ok = used = cnt = 0
            for k in tune:
                for it, (segs, corr) in zip(sets[k], segs_tune[k]):
                    ss = postprocess(shown([dict(s_) for s_ in segs]), dict(post=pp))
                    keep = [s_ for s_ in ss if s_["status"] in ("accepted", "uncertain")]
                    ok += sum(1 for s_ in keep if s_["nearest"] in it["glosses"]); used += len(keep)
                    cnt += abs(len([s_ for s_ in ss if s_["status"] != "transition"]) - len(it["glosses"]))
            n = sum(len(sets[k]) for k in tune)
            post_res.append(dict(post=pp, precision_used=ok / max(used, 1), words_used=used, count_err=cnt / max(n, 1),
                                 score=ok / max(used, 1) - 0.05 * cnt / max(n, 1)))
        best["post"] = max(post_res, key=lambda r: r["score"])["post"]
        # grammar re-rank: on only if it lowers WER on the tune sentences
        gr, role = read_json(EVAL / "grammar_tune.json"), read_json(MODEL / "roles.json")

        def rr_wer(weight):
            ws = []
            for k in tune:
                for it, (segs, _) in zip(sets[k], segs_tune[k]):
                    ss = shown([dict(s_) for s_ in segs])
                    hyp = rerank(ss, role, gr, weight=weight) if weight else [s_["nearest"] for s_ in ss]
                    ws.append(wer(it["glosses"], hyp))
            return float(np.mean(ws)) if ws else 1.0
        rer = {w: rr_wer(w) for w in (0.0, 0.5)}
        best["grammar_rerank_weight"] = 0.5 if rer[0.0] - rer[0.5] > 0.005 else 0.0
        spotting[vocab], calibration[vocab] = best, cal
        rep = dict(best_cfg=best, tempo_trials=tres, post_trials=sorted(post_res, key=lambda r: -r["score"])[:10], grammar_rerank_WER=rer,
                   top10=sorted(grid, key=lambda g: -g["objective"])[:10],
                   teacher_reference=[g for g in grid if g["stage"].startswith("reference")][0])
        for k in sets:
            res, segs_k = evaluate(sets[k], best, vocab, concepts, prior, ctx, conv_set)
            P = [lr.predict(seg_features(s_, FEATS)) for s_, _ in segs_k]
            tiers = {}
            for t_name in ("accepted", "uncertain", "unknown"):
                lo, hi = {"accepted": (cal["tau_accept"], 2), "uncertain": (cal["tau_uncertain"], cal["tau_accept"]), "unknown": (-1, cal["tau_uncertain"])}[t_name]
                c_ = [c for (s_, cc), pp in zip(segs_k, P) for c, p_ in zip(cc, pp) if lo <= p_ < hi]
                tiers[t_name] = dict(n=len(c_), precision=float(np.mean(c_)) if c_ else None)
            res["tiers"] = tiers
            rep[k] = res
            log.info("[%s] %-16s %s", vocab, k, {m: round(v, 3) for m, v in res.items() if isinstance(v, float)})
        rep["teacher_windows_on_test"] = {k: R(teacher, [k])[k] for k in test if all(f"teacher/{vocab}" in it["tables"] for it in sets[k])}
        report[vocab] = rep
        log.info("[%s] best %s (%.0fs)", vocab, {k: v for k, v in best.items() if k not in ("lens",)}, time.time() - t0)
    write_json(MODEL / "spotting.json", spotting)
    write_json(MODEL / "segment_calibration.json", calibration)
    REP.mkdir(parents=True, exist_ok=True)
    write_json(REP / "spotting.json", report)


# ----------------------------------------------------------------------------- isolated signs (held-out signers)
def _cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def cmd_isolated(a):
    from modules.encoder import retrieval_metrics
    from modules.runtime import Bank
    rd = run_dir(a.run)
    zs, ft = read_json(rd / "metrics_zero_shot.json"), read_json(rd / "metrics_best.json")
    rows = []
    for k in zs:
        if k in ft and isinstance(zs[k], dict):
            for m in ("R1", "R5", "MRR", "n", "auc", "same_clip", "random_clip"):
                if m in zs[k]:
                    rows.append(dict(protocol=k, metric=m, zero_shot=zs[k][m], v7=ft[k].get(m)))
    idx = pd.read_parquet(rd / "emb_index.parquet")
    Z0, Z1 = np.load(rd / "emb_zero_shot.npy").astype(np.float32), np.load(rd / "emb.npy").astype(np.float32)
    concepts = read_json(rd / "concept_index.json"); C = len(concepts)
    tr = (idx.role == "train").values & (idx.concept != NULL).values
    target = np.nan_to_num(idx.c.values.astype(float), nan=-1).astype(int)
    out = dict(table=rows)
    embs = {"zero_shot": Z0, "v7": Z1}
    if a.v6:              # the v6 encoder on the SAME MediaPipe clips and the same bank composition (artifacts/eval/emb_v6_encoder_mp.npz)
        z6 = np.load(EVAL / "emb_v6_encoder_mp.npz", allow_pickle=True)
        pos = {c: i for i, c in enumerate(z6["clip_id"])}
        ok6 = idx.clip_id.map(pos)
        if ok6.notna().all():
            embs["v6"] = z6["Z"].astype(np.float32)[ok6.astype(int).values]
        else:
            log.warning("v6 embeddings missing for %d clips — v6 comparison skipped", int(ok6.isna().sum()))
    daily = set(read_json(MODEL / "vocab_conversation.json")["concepts"]) if (MODEL / "vocab_conversation.json").exists() else set()
    dcols = [concepts.index(c) for c in daily if c in concepts]
    protos = {"TTRS_test": (idx.dataset == "ttrs") & (idx.role == "test"), "THSL_test": (idx.dataset == "th_sl") & (idx.role == "test"),
              "ONE_test": (idx.dataset == "tslone") & (idx.role == "test"), "AGENT_test (YouTube, new channels)": (idx.dataset == "agent") & (idx.role == "test"),
              "T51U (TSL51 researcher, video)": (idx.subset == "tsl51_user_sign") & (idx.concept != NULL)}
    comp = {}
    for name, Z in embs.items():
        bank = Bank(Z[tr], idx.c.values[tr], C)
        for pn, qm in protos.items():
            qm = qm.values
            if not qm.any():
                continue
            S = bank.scores(Z[qm]); S[:, concepts.index(NULL)] = -2
            comp.setdefault(pn, {})[name] = retrieval_metrics(S, target[qm])
            if dcols:
                md = np.isin(target[qm], dcols)
                if md.any():
                    comp.setdefault(pn + " · daily words, conversation vocab", {})[name] = retrieval_metrics(S[md], target[qm][md], allowed=dcols)
    out["comparison"] = comp
    # recall by bank depth (signers per concept among training clips) on all held-out test queries
    dep = idx[tr].groupby("c").signer.nunique()
    q = (idx.role == "test").values & (idx.concept != NULL).values & (target >= 0)
    bank1 = Bank(Z1[tr], idx.c.values[tr], C)
    S1 = bank1.scores(Z1[q]); S1[:, concepts.index(NULL)] = -2
    t = target[q]
    r1 = S1.argmax(1) == t
    d = np.array([dep.get(c, 0) for c in t])
    out["recall_by_depth"] = [dict(depth=lab, n=int(m.sum()), R1=float(r1[m].mean()) if m.any() else None)
                              for lab, m in (("0", d == 0), ("1", d == 1), ("2", d == 2), ("3-4", (d >= 3) & (d <= 4)), ("5-9", (d >= 5) & (d <= 9)), ("10+", d >= 10))]
    # ASL/CSL prior bias: of v7's errors, how many are the zero-shot (prior) model's same wrong answer?
    bank0 = Bank(Z0[tr], idx.c.values[tr], C)
    S0 = bank0.scores(Z0[q]); S0[:, concepts.index(NULL)] = -2
    p0, p1 = S0.argmax(1), S1.argmax(1)
    err1 = p1 != t
    out["prior_bias"] = dict(inherited_confusion_rate=float(((p0 == p1) & err1).sum() / max(err1.sum(), 1)),
                             zero_shot_errors_fixed=float(((p0 != t) & (p1 == t)).sum() / max((p0 != t).sum(), 1)),
                             new_errors_introduced=float(((p0 == t) & (p1 != t)).sum() / max((p0 == t).sum(), 1)),
                             cka_zero_shot_vs_v7=_cka(Z0[np.random.RandomState(0).choice(len(Z0), min(5000, len(Z0)), replace=False)],
                                                      Z1[np.random.RandomState(0).choice(len(Z0), min(5000, len(Z0)), replace=False)]))
    if "v6" in embs:
        bank6 = Bank(embs["v6"][tr], idx.c.values[tr], C)
        S6 = bank6.scores(embs["v6"][q]); S6[:, concepts.index(NULL)] = -2
        p6 = S6.argmax(1); err6 = p6 != t
        out["prior_bias"]["v6_inherited_confusion_rate_same_data"] = float(((p0 == p6) & err6).sum() / max(err6.sum(), 1))
    # latent-space neighbourhoods: is a clip's nearest neighbour the same word (good) or the same signer / dataset (bias)?
    from sklearn.neighbors import NearestNeighbors
    qm = idx.role.isin(["test", "val"]).values & (idx.concept != NULL).values
    lat = {}
    for name, Z in embs.items():
        nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(Z)
        _, nb = nn.kneighbors(Z[qm]); nb = nb[:, 1:]; qi = np.flatnonzero(qm)
        lat[name] = {f"knn5_same_{k}": float((idx[k].values[nb] == idx[k].values[qi][:, None]).mean()) for k in ("concept", "signer", "dataset")}
    out["latent"] = lat
    out["train_history"] = read_json(rd / "train_history.json")
    out["args"] = read_json(rd / "summary.json").get("args")
    write_json(REP / "isolated.json", out)
    log.info("comparison: %s", {k: {n: round(v.get("R1", 0), 3) for n, v in d.items()} for k, d in comp.items()})
    log.info("depth: %s", out["recall_by_depth"]); log.info("prior bias: %s", out["prior_bias"]); log.info("latent: %s", lat)


# ----------------------------------------------------------------------------- data_test end-to-end
def score_video(r, ref, canon, bank_vocab):
    from modules.decode import wer
    refc = [canon(w) for w in ref]
    segs = [s for s in r["segments"] if s["status"] != "transition"]
    near = [s["nearest"] for s in segs]
    acc = [s["nearest"] for s in segs if s["status"] == "accepted"]
    used = [w for w in (r.get("fusion") or {}).get("chosen", []) if w != "[?]"]
    rank = {}
    for w in refc:
        rs = [[c["word"] for c in s["candidates"]].index(w) + 1 for s in segs if w in [c["word"] for c in s["candidates"]]]
        rank[w] = min(rs) if rs else None
    f = r.get("fusion") or {}
    return dict(ref=refc, in_vocab={w: w in bank_vocab for w in refc}, n_segments=len(segs), n_ref=len(refc), top1=near, accepted=acc, used_in_sentence=used,
                segments=[dict(t0=s["t0"], t1=s["t1"], status=s["status"], p=round(s.get("p_correct", 0), 2), top5=[c["word"] for c in s["candidates"]])
                          for s in r["segments"]],
                recall_top1=float(np.mean([w in near for w in refc])), recall_top5=float(np.mean([rank[w] is not None for w in refc])),
                ref_used_in_sentence=float(np.mean([w in used for w in refc])),
                precision_used=float(np.mean([w in refc for w in used])) if used else None,
                precision_accepted=float(np.mean([w in refc for w in acc])) if acc else None,
                WER_top1=wer(refc, near), WER_used=wer(refc, used), best_rank=rank, sentence=f.get("sentence_th"), tentative=f.get("tentative_sentence_th"),
                evidence_gloss=f.get("evidence_gloss"), provider=f.get("provider"), face={k: (r.get("face") or {}).get(k) for k in ("question_yesno", "negation", "emotion")},
                timings_ms=r["timings_ms"], duration_s=r.get("duration_s"))


def summarise(videos):
    allref = sum(v["n_ref"] for v in videos.values())
    used = [w in v["ref"] for v in videos.values() for w in v["used_in_sentence"]]
    acc = [w in v["ref"] for v in videos.values() for w in v["accepted"]]
    return dict(videos=len(videos), ref_words=allref, segments=sum(v["n_segments"] for v in videos.values()),
                recall_top1=float(sum(v["recall_top1"] * v["n_ref"] for v in videos.values()) / max(allref, 1)),
                recall_top5=float(sum(v["recall_top5"] * v["n_ref"] for v in videos.values()) / max(allref, 1)),
                ref_words_in_sentence=float(sum(v["ref_used_in_sentence"] * v["n_ref"] for v in videos.values()) / max(allref, 1)),
                words_in_sentence=len(used), words_in_sentence_correct=int(sum(used)), accepted=len(acc), accepted_correct=int(sum(acc)),
                count_abs_err=float(np.mean([abs(v["n_segments"] - v["n_ref"]) for v in videos.values()])),
                mean_WER_top1=float(np.mean([v["WER_top1"] for v in videos.values()])), mean_WER_sentence=float(np.mean([v["WER_used"] for v in videos.values()])),
                mean_latency_ms=float(np.mean([v["timings_ms"].get("total_ms", sum(x for k, x in v["timings_ms"].items() if k.endswith("_ms"))) for v in videos.values()])))


def _canon():
    from modules.lexicon import TEST_SURFACE
    cm = read_json(MAN / "concepts.json")["concept"]
    return lambda w: cm.get(TEST_SURFACE.get(w, w), TEST_SURFACE.get(w, w))


def cmd_infer(a):
    from modules.lexicon import DATA_TEST_REFS
    from modules.mediapipe_pose import extract, load_raw, save_raw
    from modules.pipeline import SignPipeline
    canon = _canon()
    cache = ROOT / "cache" / "data_test_mp"; cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for vocab in (a.vocab.split(",") if a.vocab else ["conversation", "full"]):
        pipe = SignPipeline(backend=a.backend, device=a.device, vocab=vocab, use_llm=not a.no_llm)
        bank_vocab = {pipe.concepts[c] for c in pipe.bank.concepts_present}
        vids = {}
        for stem, ref in DATA_TEST_REFS.items():
            video = ROOT / "data_test" / f"{stem}.mp4"
            f = cache / f"{stem}.npz"
            t = time.perf_counter()
            if not f.exists():
                save_raw(f, extract(video))
            mp_ms = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            r = pipe.run_landmarks(load_raw(f), out_dir=ROOT / "result_reporting" / "inference" / vocab / stem, title=stem, tts=a.tts)
            r["timings_ms"]["total_ms"] = round((time.perf_counter() - t) * 1000, 1)
            r["timings_ms"]["mediapipe_ms"] = round(mp_ms, 1)
            vids[stem] = score_video(r, ref, canon, bank_vocab)
            v = vids[stem]
            log.info("[%s] %s: segs %d/%d top1=%s used=%s R@1 %.2f R@5 %.2f | sentence=%s", vocab, stem, v["n_segments"], v["n_ref"], v["top1"], v["used_in_sentence"],
                     v["recall_top1"], v["recall_top5"], v["sentence"] or f"(withheld) {v['tentative']}")
        out[vocab] = dict(summary=summarise(vids), videos=vids)
        log.info("[%s] data_test: %s", vocab, out[vocab]["summary"])
    write_json(REP / f"infer_{a.tag}.json", dict(results=out, model=read_json(MODEL / "config.json"), llm=not a.no_llm, backend=a.backend))


# ----------------------------------------------------------------------------- self-made sentence videos (held-out signers)
# A small TSL grammar (word order learned from TSL51 / interpreters: time → topic/object → subject → verb → question) with
# semantic slots, so every generated sentence is something a person would actually sign.
LEX = dict(
    SUBJ=["ฉัน", "คุณ", "แม่", "พ่อ", "พี่", "น้อง", "ยาย", "แฟน", "เพื่อน", "เรา", "น้องชาย", "น้องสาว", "ครู", "หมอ", "ลูก"],
    FOOD=["ข้าว", "ขนมปัง", "ไข่", "ส้มตำ", "ขนม", "ผลไม้", "ก๋วยเตี๋ยว"], DRINK=["น้ำ", "นม", "กาแฟ", "ชา"],
    PLACE=["ตลาด", "โรงเรียน", "บ้าน", "กรุงเทพ", "ร้าน", "โรงพยาบาล", "ห้องน้ำ"], TIME=["วันนี้", "พรุ่งนี้", "เช้า", "กลางคืน", "เมื่อวาน"],
    STATE=["เหนื่อย", "ง่วง", "เหงา", "โกรธ", "กลัว", "ร้อน", "หิว", "เบื่อ", "ดีใจ", "เสียใจ", "ป่วย", "เจ็บ", "สบายดี"],
    LIKE_OBJ=["แมว", "หมา", "ขนมปัง", "ภาษามือ", "ไข่", "ตลาด", "แฟน"], GO=["ไป", "เที่ยว"], LIKE=["ชอบ", "รัก"])
PATTERNS = [  # (TSL gloss order with slots, Thai meaning template)
    ("SUBJ FOOD กิน", "{SUBJ}กิน{FOOD}"), ("TIME SUBJ FOOD กิน", "{TIME}{SUBJ}กิน{FOOD}"), ("SUBJ DRINK ดื่ม", "{SUBJ}ดื่ม{DRINK}"),
    ("SUBJ PLACE GO", "{SUBJ}{GO}{PLACE}"), ("TIME SUBJ PLACE GO", "{TIME}{SUBJ}{GO}{PLACE}"), ("SUBJ STATE", "{SUBJ}{STATE}"),
    ("TIME SUBJ STATE", "{TIME}{SUBJ}{STATE}"), ("SUBJ LIKE_OBJ LIKE", "{SUBJ}{LIKE}{LIKE_OBJ}"), ("SUBJ ภาษามือ เรียน", "{SUBJ}เรียนภาษามือ"),
    ("SUBJ ทำงาน", "{SUBJ}ทำงาน"), ("คุณ ชื่อ อะไร", "คุณชื่ออะไร"), ("คุณ ที่ไหน ไป", "คุณไปไหน"), ("สวัสดี คุณ สบายดี", "สวัสดี คุณสบายดีไหม"),
    ("SUBJ STATE ทำไม", "ทำไม{SUBJ}{STATE}"), ("ขอบคุณ", "ขอบคุณ"), ("ขอโทษ", "ขอโทษ"), ("SUBJ อยู่บ้าน", "{SUBJ}อยู่บ้าน")]


def make_sentences(have, rng, n=12, exclude=()):
    """Every pattern instantiation whose words this signer has, minus `exclude` (e.g. the TSL51 sentence templates), sampled."""
    out = []
    for pat, thai in PATTERNS:
        slots = pat.split()
        opts = [[w for w in LEX[s] if w in have] if s in LEX else ([s] if s in have else []) for s in slots]
        if not all(opts):
            continue
        for _ in range(6):
            pick = [o[rng.randint(len(o))] for o in opts]
            if len(set(pick)) < len(pick):
                continue
            gloss = " ".join(pick)
            if gloss in exclude or any(g == gloss for g, _ in out):
                continue
            fill = {s: w for s, w in zip(slots, pick) if s in LEX}
            out.append((gloss, thai.format(**{k: fill.get(k, "") for k in LEX})))
    rng.shuffle(out)
    return out[:n]


def _h264(path):
    """OpenCV writes MPEG-4 part 2, which browsers cannot play; re-encode to H.264 (what phones record) when ffmpeg is present."""
    import shutil
    import subprocess
    if shutil.which("ffmpeg"):
        tmp = path.with_suffix(".h264.mp4")
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)])
        if r.returncode == 0:
            tmp.replace(path)


def cmd_selftest(a):
    """Sentence videos that no model has seen: real isolated-sign videos of HELD-OUT signers (test roles — never in encoder training;
    the evaluation uses the training-signer bank only), cut to their signing span and joined in TSL word order with a short
    cross-fade (the hand transition between signs), at natural (1.3×) and fast phone pace (1.8×). The full system then runs from the
    pixels: MediaPipe → schema → tagger → spotting → sentence. Sentences that are TSL51 templates are excluded (new combinations only)."""
    import cv2
    from modules.mediapipe_pose import extract, load_raw, save_raw
    from modules.pipeline import SignPipeline
    out_dir = ROOT / "cache" / "selftest"; out_dir.mkdir(parents=True, exist_ok=True)
    clips = pd.read_parquet(SHARDS / "clips.parquet")
    clips = clips[clips.view == "primary"]
    templates = {" ".join(c.split("|")) for c in clips[clips.kind == "sentence"].concept}
    vi = pd.read_csv(PREP / "params" / "video_items.csv")
    src = {r.clip_id: PREP.parent / "raw_data" / r.rel_path for r in vi.itertuples()}
    for r in vi[vi.dataset == "tsl51_user_sign"].itertuples():
        src["t51u:" + r.clip_id.split(":", 1)[1]] = PREP.parent / "raw_data" / r.rel_path
    ag = PREP.parent / "agent_data" / "youtube_words" / "videos.csv"
    if ag.exists():
        for r in pd.read_csv(ag).itertuples():
            src[r.clip_id] = PREP.parent / "agent_data" / "youtube_words" / r.video
    pool = clips[(clips.kind == "isolated") & (clips.role == "test") & (clips.subset.isin(["tsl51_user_sign"]) | clips.dataset.isin(["th_sl", "agent", "ttrs"]))]
    if a.make:
        rng = np.random.RandomState(7)
        meta = []
        for signer, g in pool.groupby("signer"):
            g = g[g.clip_id.isin(src)]
            byc = {c: list(gg.clip_id) for c, gg in g.groupby("concept")}
            for gloss, thai in make_sentences(set(byc), rng, n=a.per_signer, exclude=templates):
                frames, size = [], None
                for w in gloss.split():
                    cid = byc[w][rng.randint(len(byc[w]))]
                    r = g[g.clip_id == cid].iloc[0]
                    cap = cv2.VideoCapture(str(src[cid])); vf = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    f0 = max(0, int((r.span_f0 / 25.0 - 0.2) * vf)); f1 = int((r.span_f1 / 25.0 + 0.15) * vf)
                    seg, i = [], 0
                    while True:
                        ok, fr = cap.read()
                        if not ok or i > f1:
                            break
                        if i >= f0:
                            size = size or (480, int(round(480 * fr.shape[0] / fr.shape[1] / 2)) * 2)
                            seg.append(cv2.resize(fr, size))
                        i += 1
                    cap.release()
                    seg = [seg[j] for j in np.round(np.arange(0, len(seg), vf / 25.0)).astype(int) if j < len(seg)]
                    if frames and seg:                        # 4-frame cross-fade = the hand moving from one sign to the next
                        last = frames[-1]
                        for k in range(1, 5):
                            frames.append(cv2.addWeighted(last, 1 - k / 5, seg[0], k / 5, 0))
                    frames += seg
                if len(frames) < 20:
                    continue
                for speed in (1.3, 1.8):
                    name = f"{signer.replace(':', '_')}__{gloss.replace(' ', '_')}__x{speed}"
                    vw = cv2.VideoWriter(str(out_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, size)
                    for j in np.round(np.arange(0, len(frames), speed)).astype(int):
                        if j < len(frames):
                            vw.write(frames[j])
                    vw.release()
                    _h264(out_dir / f"{name}.mp4")
                    meta.append(dict(video=f"{name}.mp4", signer=signer, source=g.subset.iloc[0] if g.dataset.iloc[0] == "tsl51" else g.dataset.iloc[0],
                                     gloss=gloss, thai=thai, speed=speed, n_words=len(gloss.split())))
        pd.DataFrame(meta).to_csv(out_dir / "selftest.csv", index=False, encoding="utf-8")
        log.info("self-test videos: %d (%d sentences, %d signers)", len(meta), len(meta) // 2, len({m['signer'] for m in meta}))
        if a.make_only:
            return
    meta = pd.read_csv(out_dir / "selftest.csv")
    canon = _canon()
    res = {}
    for vocab in ("conversation", "full"):
        pipe = SignPipeline(backend=a.backend, device=a.device, vocab=vocab, use_llm=not a.no_llm, bank_file=EVAL / "bank_eval.npz")
        bank_vocab = {pipe.concepts[c] for c in pipe.bank.concepts_present}
        vids = {}
        for r in meta.itertuples():
            f = out_dir / (Path(r.video).stem + ".npz")
            if not f.exists():
                save_raw(f, extract(out_dir / r.video))
            t = time.perf_counter()
            o = pipe.run_landmarks(load_raw(f), title=r.video)
            o["timings_ms"]["total_ms"] = round((time.perf_counter() - t) * 1000, 1)
            v = score_video(o, r.gloss.split(), canon, bank_vocab)
            v.update(thai_reference=r.thai, speed=r.speed, signer=r.signer, source=r.source)
            vids[r.video] = v
        res[vocab] = dict(summary=summarise(vids), by_speed={str(s): summarise({k: v for k, v in vids.items() if v["speed"] == s}) for s in sorted(set(meta.speed))},
                          by_source={s: summarise({k: v for k, v in vids.items() if v["source"] == s}) for s in sorted(set(meta.source))}, videos=vids)
        log.info("[%s] self-test: %s", vocab, res[vocab]["summary"])
    write_json(REP / "selftest.json", res)


# ----------------------------------------------------------------------------- real phrases by an unseen person
def _phrase_glosses(core, concepts):
    """Thai phrase → reference concepts in Thai order (word tokenizer + synonym map); words outside the vocabulary are kept as
    out-of-vocabulary references (they count against recall: the system cannot know them)."""
    from pythainlp.tokenize import word_tokenize
    cmap = read_json(MAN / "concepts.json")["concept"]
    stop = {"ที่", "จะ", "ของ", "ๆ", "นี้", "นั้น", "ครับ", "ค่ะ", "คะ", "นะ", "ก็", "ให้", "กับ", "และ", "มา", "อยู่"}
    out = []
    for w in word_tokenize(core.replace(" ", ""), engine="newmm"):
        if w in stop or not w.strip():
            continue
        out.append(cmap.get(w, w))
    return out


def cmd_phrases(a):
    """Real continuous signing by a person in no training set: short YouTube TSL phrase lessons ('ภาษามือ[<phrase>]', the phrase signer is
    a test person — scripts/harvest_words.py). Scored with the training-signer bank: recall of the reference words (top-1 / top-5 of
    any segment), precision of the words the sentence used, and the sentence shown."""
    from modules.mediapipe_pose import load_raw
    from modules.pipeline import SignPipeline
    H = PREP.parent / "agent_data" / "youtube_words"
    ph = pd.read_csv(H / "phrases.csv")
    ph = ph[ph.kind == "phrase"]
    split = phrase_split(list(ph.id))
    res = {}
    for vocab in ("conversation", "full"):
        pipe = SignPipeline(backend=a.backend, device=a.device, vocab=vocab, use_llm=not a.no_llm, bank_file=EVAL / "bank_eval.npz")
        bank_vocab = {pipe.concepts[c] for c in pipe.bank.concepts_present}
        vids = {}
        for r in ph.itertuples():
            f = PREP / "pose_mp_raw" / "agent" / f"agent_yt_{r.id}.npz"
            if not f.exists():
                continue
            ref = _phrase_glosses(r.core, pipe.concepts)
            t = time.perf_counter()
            o = pipe.run_landmarks(load_raw(f), title=r.core)
            o["timings_ms"]["total_ms"] = round((time.perf_counter() - t) * 1000, 1)
            v = score_video(o, ref, lambda w: w, bank_vocab)
            v.update(phrase=r.core, n_ref_in_vocab=sum(w in bank_vocab for w in ref), split=split[r.id])
            vids[r.id] = v
        res[vocab] = dict(summary_test_half=summarise({k: v for k, v in vids.items() if v["split"] == "test"}),
                          summary_tune_half=summarise({k: v for k, v in vids.items() if v["split"] == "tune"}), summary=summarise(vids), videos=vids)
        log.info("[%s] phrases (unseen person, real sentences) TEST half: %s", vocab, res[vocab]["summary_test_half"])
    write_json(REP / "phrases.json", res)


def cmd_qparticle(a):
    """ไหม (yes/no question particle) is in no dictionary source. Short TSL phrase lessons ending in ไหม (scripts/harvest_words.py,
    kind 'question_particle') sign it last: the model segments each phrase, the LAST sign segment is cut out, and the cuts are kept only
    if they agree with each other (each cut closer to the other cuts than to the last signs of the statement phrases of the same
    person). Accepted cuts are appended to models/bank.npz as the concept ไหม — the append-only bank: a new word without retraining."""
    import cv2
    from modules.mediapipe_pose import load_raw
    from modules.parts import prep, to_iso
    from modules.pipeline import SignPipeline
    H = PREP.parent / "agent_data" / "youtube_words"
    ph = pd.read_csv(H / "phrases.csv")
    pipe = SignPipeline(backend="torch", device="cuda", vocab="full", use_llm=False, use_face=False)
    cuts = []
    for r in ph.itertuples():
        f = PREP / "pose_mp_raw" / "agent" / f"agent_yt_{r.id}.npz"
        if not f.exists():
            continue
        o = pipe.run_landmarks(load_raw(f), title=r.core)
        segs = [s for s in o["segments"] if s["status"] != "transition"]
        if not segs:
            continue
        kp, sc, hw, _, _ = pipe.standardise(load_raw(f))
        s = segs[-1]
        z = pipe.be.embed([prep(to_iso(kp, hw)[s["f0"]:s["f1"]], sc[s["f0"]:s["f1"]])])[0]
        cuts.append(dict(id=r.id, core=r.core, kind=r.kind, f0=s["f0"], f1=s["f1"], n_segments=len(segs), z=z))
    q = [c for c in cuts if c["kind"] == "question_particle"]
    o_ = [c for c in cuts if c["kind"] != "question_particle"]
    if len(q) < 2:
        log.warning("question particle: %d phrase cuts — nothing added", len(q)); return
    Zq, Zo = np.stack([c["z"] for c in q]), np.stack([c["z"] for c in o_]) if o_ else np.zeros((0, 768))
    rows = []
    for i, c in enumerate(q):
        same = float(np.mean([Zq[i] @ Zq[j] for j in range(len(q)) if j != i]))
        other = float(np.mean(Zo @ Zq[i])) if len(Zo) else -1.0
        c.update(sim_to_other_question_cuts=round(same, 3), sim_to_statement_last_signs=round(other, 3), accepted=same > other + 0.05)
        rows.append({k: v for k, v in c.items() if k != "z"})
    acc = [c for c in q if c["accepted"]]
    pd.DataFrame(rows).to_csv(H / "qparticle.csv", index=False, encoding="utf-8")
    log.info("question-particle cuts: %s", [(c["core"], c["sim_to_other_question_cuts"], c["sim_to_statement_last_signs"], c["accepted"]) for c in q])
    if len(acc) < 2:
        log.warning("cuts do not agree — ไหม NOT added (the face cue stays the only question signal)"); return
    concepts = read_json(MODEL / "concepts.json")
    if "ไหม" not in concepts:
        concepts.append("ไหม"); write_json(MODEL / "concepts.json", concepts)
        for f in ("freq_prior.npy",):
            if (MODEL / f).exists():
                v = np.load(MODEL / f); np.save(MODEL / f, np.r_[v, v.max()].astype(v.dtype))   # the particle is a very frequent word
    ci = concepts.index("ไหม")
    b = dict(np.load(MODEL / "bank.npz", allow_pickle=False))
    keep = ~np.isin(b["clip"], [f"agent:yt:{c['id']}@q" for c in acc])
    b = {k: v[keep] for k, v in b.items()}
    b["Z"] = np.concatenate([b["Z"], np.stack([c["z"] for c in acc]).astype(np.float16)])
    b["c"] = np.concatenate([b["c"], np.full(len(acc), ci, b["c"].dtype)])
    b["clip"] = np.concatenate([b["clip"], np.array([f"agent:yt:{c['id']}@q" for c in acc])])
    b["dataset"] = np.concatenate([b["dataset"], np.array(["agent_question_phrase"] * len(acc))])
    b["signer"] = np.concatenate([b["signer"], np.array(["yt_phrase_signer"] * len(acc))])
    np.savez(MODEL / "bank.npz", **b)
    vc = read_json(MODEL / "vocab_conversation.json")
    if "ไหม" not in vc["concepts"]:
        vc["concepts"].append("ไหม"); vc["signers"]["ไหม"] = 1
        vc["not_in_bank"] = [w for w in vc.get("not_in_bank", []) if w != "ไหม"]
        write_json(MODEL / "vocab_conversation.json", vc)
    # contact sheet: middle frame of every accepted cut
    tiles = []
    for c in acc:
        cap = cv2.VideoCapture(str(H / "videos" / f"{c['id']}.mp4")); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((c["f0"] + c["f1"]) / 2 / 25.0 * fps)); ok, fr = cap.read(); cap.release()
        if ok:
            tiles.append(cv2.resize(fr, (200, int(200 * fr.shape[0] / fr.shape[1]))))
    if tiles:
        h = min(t.shape[0] for t in tiles)
        cv2.imwrite(str(H / "qc" / "question_particle_cuts.jpg"), np.concatenate([t[:h] for t in tiles], 1))
    log.info("ไหม added to the bank from %d phrase cuts (1 signer)", len(acc))


def phrase_split(ids):
    """deterministic 50/50 split of the phrase clips: 'tune' (natural-signing decoder tuning) / 'test' (report only)"""
    import hashlib
    return {i: ("tune" if int(hashlib.md5(i.encode()).hexdigest(), 16) % 2 == 0 else "test") for i in ids}


def cmd_tune_natural(a):
    """Natural, fast signing (prompt_9 P1/P2): the TSL51 templates are slow, deliberate signing, so the decoder chosen on them
    under-segments real conversation (3 signs in 1.5 s → 1 segment). Half of the real phrase clips of the unseen YouTube person
    (by phrase, `phrase_split`) become a natural-signing tuning set; the segmentation settings (minimum sign length, window grid, λ,
    begin bonus, activity threshold, tempo normalisation) are re-chosen to maximise the phrase-tune score while the TSL51 tune
    objective may not drop by more than 0.02. Reports use the other phrase half, the TSL51 test templates, the self-made videos and
    data_test — none of them is used here."""
    from copy import deepcopy
    from modules.mediapipe_pose import load_raw
    from modules.pipeline import SignPipeline
    H = PREP.parent / "agent_data" / "youtube_words"
    ph = pd.read_csv(H / "phrases.csv")
    ph = ph[ph.kind == "phrase"]
    split = phrase_split(list(ph.id))
    tune = ph[ph.id.map(split) == "tune"]
    sd = run_dir(a.seq)
    D = pickle.load(open(sd / "seq_tables.pkl", "rb"))
    concepts, sets = D["concepts"], _sets(D["items"])
    prior = np.load(MODEL / "freq_prior.npy") if (MODEL / "freq_prior.npy").exists() else None
    from modules.runtime import Bank
    be = np.load(EVAL / "bank_eval.npz")
    ctx = dict(exact=Exact(), null=np.load(sd / "null_proto.npy").astype(np.float32),
               banks={"full": Bank(be["Z"].astype(np.float32), be["c"], len(concepts)),
                      "conversation": Bank(be["Z"].astype(np.float32), be["c"], len(concepts), allowed=set(D["conversation"]))})
    spot_all = read_json(MODEL / "spotting.json")
    cal_all = read_json(MODEL / "segment_calibration.json")
    report = {}
    for vocab in ("conversation", "full"):
        pipe = SignPipeline(backend="torch", device="cuda", vocab=vocab, use_llm=False, use_face=False, bank_file=EVAL / "bank_eval.npz")
        bank_vocab = {pipe.concepts[c] for c in pipe.bank.concepts_present}
        _tag, _cache = pipe._tag, {}

        def cached_tag(kpi, sc, present):              # the encoder + tagger pass does not depend on the decoder settings
            key = (kpi.shape, round(float(kpi.sum()), 3), round(float(sc.sum()), 3))
            if key not in _cache:
                _cache[key] = _tag(kpi, sc, present)
            return _cache[key]
        pipe._tag = cached_tag
        items = []
        for r in tune.itertuples():
            f = PREP / "pose_mp_raw" / "agent" / f"agent_yt_{r.id}.npz"
            if f.exists():
                items.append((load_raw(f), _phrase_glosses(r.core, pipe.concepts)))

        def phrase_score(cfg):
            pipe.spot = cfg
            hit = n_in = used_ok = used = cnt = 0
            for raw, ref in items:
                o = pipe.run_landmarks(raw)
                segs = [s for s in o["segments"] if s["status"] != "transition"]
                top = [s["nearest"] for s in segs]
                inv = [w for w in ref if w in bank_vocab]
                n_in += len(inv); hit += sum(w in top for w in inv)
                used += len(top); used_ok += sum(w in ref for w in top)
                cnt += abs(len(segs) - len(ref))
            rec, prec = hit / max(n_in, 1), used_ok / max(used, 1)
            return dict(recall=rec, precision=prec, count_err=cnt / max(len(items), 1), score=rec + 0.5 * prec - 0.1 * cnt / max(len(items), 1))
        tune51 = [k for k in sets if "_tune" in k]
        base = spot_all[vocab]
        b51 = _objective({k: evaluate(sets[k], base, vocab, concepts, prior, ctx)[0] for k in tune51})
        trials = [dict(cfg=base, phrase=phrase_score(base), tsl51=b51, change="tuned on TSL51 only")]
        tempos = [base.get("tempo")] + [dict(ref_len=r, min_factor=1.2, max_factor=2.5, min_segments=1) for r in (20, 28)]
        for min_len, stride, lam, tempo in product([8, 10, 12, 14], [2, 4], [base["lam"] - 2, base["lam"], base["lam"] + 2], tempos):
            cfg = dict(deepcopy(base), min_len=min_len, stride=stride, lam=lam, tempo=tempo)
            trials.append(dict(cfg=cfg, phrase=phrase_score(cfg), change=f"min_len {min_len} stride {stride} lam {lam} tempo {tempo}"))
        trials.sort(key=lambda t: -t["phrase"]["score"])
        chosen = None
        for t in trials:                                       # best natural-signing config that keeps the TSL51 tune objective
            if t["phrase"]["score"] <= trials[[x["change"] for x in trials].index("tuned on TSL51 only")]["phrase"]["score"]:
                break
            if "tsl51" not in t:
                t["tsl51"] = _objective({k: evaluate(sets[k], t["cfg"], vocab, concepts, prior, ctx)[0] for k in tune51})
            if t["tsl51"] >= b51 - 0.02:
                chosen = t; break
        chosen = chosen or [t for t in trials if t["change"] == "tuned on TSL51 only"][0]
        spot_all[vocab] = chosen["cfg"]
        # re-fit the calibration for the chosen decoder (same procedure as `spot`: TSL51 tune decodes, out-of-fold tagger)
        from modules.decode import FEATS, LogReg, seg_features
        segs_tune = {k: evaluate(sets[k], chosen["cfg"], vocab, concepts, prior, ctx)[1] for k in tune51}
        X = np.concatenate([seg_features(s, FEATS) for k in tune51 for s, _ in segs_tune[k]] or [np.zeros((0, len(FEATS)))])
        y = np.concatenate([np.array(c, float) for k in tune51 for _, c in segs_tune[k]] or [np.zeros(0)])
        lr = LogReg().fit(X, y); p_fit = lr.predict(X)

        def tau_for(prec):
            for t in np.linspace(0.05, 0.95, 91):
                mm = p_fit >= t
                if mm.sum() >= 10 and y[mm].mean() >= prec:
                    return float(t)
            return 0.95
        cal_all[vocab] = dict(model=lr.to_dict(FEATS), tau_accept=tau_for(0.85), tau_uncertain=tau_for(0.45), fitted_on=tune51,
                              note="re-fitted after natural-signing tuning")
        report[vocab] = dict(before=[t for t in trials if t["change"] == "tuned on TSL51 only"][0],
                             chosen={k: v for k, v in chosen.items()}, top=[{k: v for k, v in t.items() if k != "cfg"} for t in trials[:10]])
        log.info("[%s] natural-signing tuning: before %s → chosen %s (%s) TSL51 tune %.3f → %.3f", vocab, report[vocab]["before"]["phrase"], chosen["phrase"],
                 chosen["change"], b51, chosen.get("tsl51", b51))
    write_json(MODEL / "spotting.json", spot_all)
    write_json(MODEL / "segment_calibration.json", cal_all)
    write_json(REP / "natural_tuning.json", report)


# ----------------------------------------------------------------------------- vocab exports
def cmd_vocab(a):
    clips = pd.read_parquet(SHARDS / "clips.parquet")
    c = clips[(clips.view == "primary") & (clips.kind == "isolated") & clips.concept.notna() & (clips.concept != NULL)]
    role = read_json(MODEL / "roles.json") if (MODEL / "roles.json").exists() else {}
    g = c.groupby("concept").agg(instances=("clip_id", "size"), signers=("signer", "nunique"), datasets=("dataset", lambda s: " + ".join(sorted(set(s)))))
    g["role"] = [role.get(k) for k in g.index]
    conv = set(read_json(MODEL / "vocab_conversation.json")["concepts"])
    g["conversation_vocab"] = [k in conv for k in g.index]
    corpus = ROOT / "vocab"; corpus.mkdir(parents=True, exist_ok=True)
    v = g.sort_values(["signers", "instances"], ascending=False).reset_index()
    v.to_csv(corpus / "vocab.csv", index=False, encoding="utf-8-sig")
    daily = read_json(ROOT / "vocab" / "daily_conversation.json")
    cm = read_json(MAN / "concepts.json")["concept"]
    rows = []
    for cat, ws in daily.items():
        if cat.startswith("_"):
            continue
        for w in ws:
            k = cm.get(w, w)
            rows.append(dict(category=cat, word=w, concept=k, signers=int(g.signers.get(k, 0)), instances=int(g.instances.get(k, 0)),
                             datasets=g.datasets.get(k, ""), in_model=k in conv))
    pd.DataFrame(rows).to_csv(corpus / "daily_conversation_depth.csv", index=False, encoding="utf-8-sig")
    log.info("vocab: %d concepts; daily words with ≥ 3 signers: %d / %d", len(v), sum(r["signers"] >= 3 for r in rows), len(rows))


def _npz(path):
    """Read a whole .npz and close it (Windows cannot overwrite a file that is still open)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def cmd_bank_add(a):
    """Append new harvested word clips (agent_data/youtube_words/items.csv, status accepted, not yet in the bank) without retraining:
    MediaPipe Standard Schema → canonical hands → labelled sign span → encoder embedding (the recipe the bank was built with; checked:
    cosine ≥ 0.997 to the training job's own embeddings). models/bank.npz gets every role (real users are new people anyway);
    artifacts/eval/bank_eval.npz only training-role clips (held-out people stay out of the evaluation bank)."""
    from modules.parts import prep, to_iso
    from modules.runtime import make_backend
    from modules.schema import canonical_hands
    allit = pd.read_csv(PREP.parent / "agent_data" / "youtube_words" / "items.csv")
    # clips the latest labelling rejects as a data_test look-alike or a duplicate leave both banks (data_test stays unseen)
    drop = set(allit[(allit.status != "accepted") & allit.reason.fillna("").str.contains("data_test|duplicate")].clip_id)
    for path in (MODEL / "bank.npz", EVAL / "bank_eval.npz"):
        b = _npz(path)
        keep = ~np.isin(b["clip"], list(drop))
        if (~keep).any():
            np.savez(path, **{k: v[keep] for k, v in b.items()})
            log.info("bank_add: removed %d rejected clips from %s", int((~keep).sum()), path.name)
    it = allit[allit.status == "accepted"]
    have = set(_npz(MODEL / "bank.npz")["clip"].tolist())
    new = it[~it.clip_id.isin(have)]
    if not len(new):
        log.info("bank_add: nothing new"); return
    concepts = read_json(MODEL / "concepts.json")
    be = make_backend(MODEL, "onnx", "cpu")
    rows, Z = [], []
    for r in new.itertuples():
        f = PREP / "pose_mp" / "agent" / f"agent_yt_{r.clip_id.split(':')[-1]}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        hw = tuple(int(v) for v in z["hw"])
        kp, sc, _ = canonical_hands(z["kp"], z["sc"], hw)
        kpi = to_iso(kp, hw)
        a0, b0 = int(r.f0), int(r.f1)
        if b0 - a0 < 8:
            continue
        Z.append(be.embed([prep(kpi[a0:b0], sc[a0:b0])])[0]); rows.append(r)
    if not rows:
        log.info("bank_add: no usable clips"); return
    Z = np.stack(Z)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    added_concepts = []
    for r in rows:
        if r.concept not in concepts:
            concepts.append(r.concept); added_concepts.append(r.concept)
    if added_concepts:
        write_json(MODEL / "concepts.json", concepts)
        pf = MODEL / "freq_prior.npy"
        if pf.exists():
            v = np.load(pf).copy(); np.save(pf, np.r_[v, np.full(len(added_concepts), np.median(v), v.dtype)])
    cidx = {c: i for i, c in enumerate(concepts)}
    c_new = np.array([cidx[r.concept] for r in rows])

    def append(path, mask):
        b = _npz(path)
        if not mask.any():
            return 0
        b["Z"] = np.concatenate([b["Z"], Z[mask].astype(b["Z"].dtype)])
        b["c"] = np.concatenate([b["c"], c_new[mask].astype(b["c"].dtype)])
        for k, vals in (("clip", [r.clip_id for r in rows]), ("dataset", ["agent"] * len(rows)), ("signer", [r.signer for r in rows])):
            if k in b:
                b[k] = np.concatenate([b[k].astype(str), np.array(vals, dtype=str)[mask]])
        np.savez(path, **b)
        return int(mask.sum())
    n_dep = append(MODEL / "bank.npz", np.ones(len(rows), bool))
    n_eval = append(EVAL / "bank_eval.npz", np.array([r.role == "train" for r in rows]))
    fb = _npz(MODEL / "bank.npz")
    present = set(fb["c"].tolist())
    conv, missing = conversation_vocab(concepts, present)
    depth = pd.Series(fb["signer"]).groupby(fb["c"]).nunique()
    write_json(MODEL / "vocab_conversation.json", dict(
        doc="Conversation-mode vocabulary: vocab/daily_conversation.json ∪ TSL51 words, restricted to concepts with ≥ 1 bank example. "
            "signers = distinct people / channels behind each word in models/bank.npz.",
        concepts=conv, signers={w: int(depth.get(cidx[w], 0)) for w in conv}, not_in_bank=missing))
    cfg = read_json(MODEL / "config.json")
    cfg.update(bank=int(len(fb["c"])), concepts=len(concepts), conversation_concepts=len(conv),
               bank_added=dict(clips=n_dep, eval_clips=n_eval, new_concepts=added_concepts, built=time.strftime("%Y-%m-%d %H:%M")))
    write_json(MODEL / "config.json", cfg)
    log.info("bank_add: +%d clips (eval bank +%d), %d new concepts, conversation vocabulary %d words (not in bank: %s)", n_dep, n_eval,
             len(added_concepts), len(conv), missing)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--run", required=True); b.add_argument("--seq", required=True)
    sub.add_parser("grammar"); sub.add_parser("vocab"); sub.add_parser("qparticle"); sub.add_parser("bank_add")
    tn = sub.add_parser("tune_natural"); tn.add_argument("--seq", required=True)
    s = sub.add_parser("spot"); s.add_argument("--seq", required=True); s.add_argument("--accept_precision", type=float, default=0.85)
    i = sub.add_parser("isolated"); i.add_argument("--run", required=True); i.add_argument("--v6", action="store_true")
    for name in ("infer", "selftest", "phrases"):
        n = sub.add_parser(name); n.add_argument("--tag", default="final"); n.add_argument("--no_llm", action="store_true"); n.add_argument("--tts", action="store_true")
        n.add_argument("--backend", default="onnx"); n.add_argument("--device", default="cpu"); n.add_argument("--vocab", default="")
        if name == "selftest":
            n.add_argument("--make", action="store_true"); n.add_argument("--make_only", action="store_true"); n.add_argument("--per_signer", type=int, default=14)
    a = ap.parse_args()
    dict(build=cmd_build, grammar=cmd_grammar, spot=cmd_spot, isolated=cmd_isolated, infer=cmd_infer, selftest=cmd_selftest, phrases=cmd_phrases, qparticle=cmd_qparticle, tune_natural=cmd_tune_natural, vocab=cmd_vocab, bank_add=cmd_bank_add)[a.cmd](a)


if __name__ == "__main__":
    main()
