"""`python main.py diagnose v12` — the v12 harness (prompt-opt-v12 1H / C6 / C7): the v11 catalogue (scripts/diag_v11.py) on every set
+ the v12 additions:

  * NEWSIG sets (nsig_dev; nsig_blind12 only with THAISLM_OPEN_BLIND12=1): isolated top-1 / top-5 of never-seen (or word-held-out) signers
  * decoder-span R@1 (fusion): gold words whose best-overlapping decoder segment (IoU >= 0.3) is read as the right word
  * matched-precision coverage: the acceptance threshold moved until pooled accepted precision = the REFERENCE's (v10's, frozen in
    artifacts/baselines/v12/v10.json) → correct-accepted share there (never compare acceptance at a fixed tau)
  * sentence output: BLEU-4 / chrF++ (sacrebleu, pythainlp newmm tokens for BLEU) of the rule-composed Thai sentence (LLM off) vs the
    TSL51 template sentence; caption-bag recall / precision on natural sentences when a caption set is given
  * --cand <rung>: ladder flags (read.jitter, read.primary2, cal.v12 …) merged into the version's spotting settings

  python main.py diagnose v12 --tag v10 [--sets tune|devtest|a,b] [--cand '{"read": {"jitter": 5}}'] [--out name] [--freeze]
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIDE = ROOT / "cache" / "side_pkgs"                      # extra packages (sacrebleu) outside the 490-sign env
if SIDE.exists() and str(SIDE) not in sys.path:
    sys.path.append(str(SIDE))

import diag_v11  # noqa: E402
from decision import TEST_SETS, TUNE_SETS, REP, _iou, _p, cache_path, load_set  # noqa: E402
from modules.utils import ART, PREP, read_json, write_json  # noqa: E402

BASE = ART / "baselines" / "v12"
TUNE12 = TUNE_SETS + ["nsig_dev", "w_t51_tune", "w_iso_tune", "idle_tune", "w_idle_tune", "tail_tune", "w_tail_tune"]
DEVTEST12 = TEST_SETS + ["w_t51_test", "w_iso_test", "idle_test", "w_idle_test", "tail_test", "w_tail_test"]
LEGACY = ["t51_blind_x1.0", "t51_blind_x1.5", "t51_blind_x2.0", "iso_blind", "flu_blind", "tail_blind"]

_BLOCKS = {}
_ROWS = {}


def _capture():
    """wrap diag_v11.diag_set so the per-segment blocks (P, Y, status, segments) of every set are kept for the v12 metrics"""
    orig = diag_v11.diag_set

    def wrapped(h, name, rows, G, T):
        m, rec, blk = orig(h, name, rows, G, T)
        _BLOCKS[name] = blk; _ROWS[name] = rows
        m.update(_decoder_span_r1(rows, blk[5]))
        if name.startswith("t51") or name.startswith("w_t51"):
            m.update(_sentence_metrics(h, rows, blk[5]))
        if name.startswith("nsig"):
            m.update(_newsig_metrics(h, rows, blk[5], blk[6]))
        return m, rec, blk
    diag_v11.diag_set = wrapped


def _decoder_span_r1(rows, segs_all):
    """+ the boundary-error distribution (decoder start / end − gold start / end, frames at 25 fps) of segments matched to a gold word
    (IoU >= 0.3) — the 'decoder-cut span' noise of prompt 1E.2 / 2B"""
    hit = n = 0
    offs = []
    for r, segs in zip(rows, segs_all):
        live = [s for s in segs if s["status"] != "transition"]
        for w, g in zip(r["ref"], r.get("gold") or []):
            n += 1
            best = max(live, key=lambda s: _iou((s["f0"], s["f1"]), g), default=None)
            if best is not None and _iou((best["f0"], best["f1"]), g) >= 0.3:
                offs.append((int(best["f0"] - g[0]), int(best["f1"] - g[1])))
                hit += best["nearest"] == w
    o = np.array(offs) if offs else np.zeros((0, 2))
    bd = dict(n=len(o), start_pcts=np.percentile(o[:, 0], [10, 25, 50, 75, 90]).tolist() if len(o) else None,
              end_pcts=np.percentile(o[:, 1], [10, 25, 50, 75, 90]).tolist() if len(o) else None, pairs=o.astype(int).tolist())
    return dict(decoder_span_R1=hit / n if n else None, decoder_span_n=n, boundary_offsets=bd)


def _thai_tokens(s):
    from pythainlp.tokenize import word_tokenize
    return [t for t in word_tokenize(str(s), engine="newmm", keep_whitespace=False) if t.strip()]


def _sentence_metrics(h, rows, segs_all):
    """BLEU-4 / chrF++ of the rule-composed sentence (accepted words only, LLM off) vs the TSL51 template's Thai sentence"""
    try:
        import sacrebleu
    except ImportError:
        return dict(BLEU=None, chrF=None, sentence_note="sacrebleu not installed (cache/side_pkgs)")
    from modules.language import rules_compose
    ref_map = read_json(PREP / "params" / "tsl51_thai_sentences.json")
    hyps, refs = [], []
    for r, segs in zip(rows, segs_all):
        key = " ".join(sorted(r["ref"]))
        ref = ref_map.get(key) or ref_map.get(" ".join(r["ref"]))
        if not ref:
            continue
        live = [s for s in segs if s["status"] != "transition"]
        hyp = rules_compose(live, {}, roles=getattr(h.pipe, "lang_roles", None))["sentence_th"]
        hyps.append(" ".join(_thai_tokens(hyp))); refs.append(" ".join(_thai_tokens(ref)))
    if not hyps:
        return dict(BLEU=None, chrF=None, sentence_n=0)
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
    chrf = sacrebleu.corpus_chrf([h_.replace(" ", "") for h_ in hyps], [[r_.replace(" ", "") for r_ in refs]], word_order=2)
    sbleu = float(np.mean([sacrebleu.sentence_bleu(h_, [r_], tokenize="none", smooth_method="exp").score for h_, r_ in zip(hyps, refs)]))
    return dict(BLEU=round(bleu.score, 2), sentence_BLEU=round(sbleu, 2), chrF=round(chrf.score, 2), sentence_n=len(hyps), bleu_tokenizer="pythainlp newmm, sacrebleu tokenize=none; chrF++ on characters (spaces removed)",
                empty_sentence_share=float(np.mean([not x for x in hyps])))


def _newsig_metrics(h, rows, segs_all, sc_all):
    """isolated clip of a new signer: top-1 = the clip's best segment (longest live) names the word; top-5 = word among its candidates"""
    t1 = t5 = shown = 0
    for r, segs in zip(rows, segs_all):
        live = [s for s in segs if s["status"] != "transition"]
        w = r["ref"][0]
        if not live:
            continue
        s = max(live, key=lambda x: x["f1"] - x["f0"])
        t1 += s["nearest"] == w
        t5 += w in [c["word"] for c in s.get("candidates", [])[:5]] or s["nearest"] == w
        shown += any(x["nearest"] == w and x["status"] == "accepted" for x in live)
    n = max(len(rows), 1)
    return dict(newsig_top1=t1 / n, newsig_top5=t5 / n, newsig_correct_shown=shown / n, newsig_n=len(rows))


def matched_precision(blocks, sets, ref_precision=None):
    """pooled over `sets`, everything on ONE rule family (a threshold on P(correct)):
    own_P_precision = precision of the top-k segments by P, k = the number the version itself accepts (its operating point);
    correct_accepted_at_target = share of correct segments kept at the largest k whose precision >= ref_precision (the reference
    version's own_P_precision on the same sets) — the matched-precision comparison of C3 E3 / C5."""
    P = np.concatenate([np.asarray(blocks[s][0], float) for s in sets if s in blocks] or [np.zeros(0)])
    Y = np.concatenate([np.asarray(blocks[s][1], float) for s in sets if s in blocks] or [np.zeros(0)])
    ST = sum([list(blocks[s][2]) for s in sets if s in blocks], [])
    if not len(P):
        return {}
    P = np.nan_to_num(P, nan=0.0)
    acc = np.array([x == "accepted" for x in ST])
    order = np.argsort(-P, kind="stable")
    cy = np.cumsum(Y[order]); k = np.arange(1, len(P) + 1); prec = cy / k
    n_acc = int(acc.sum())
    out = dict(n_segments=len(P), n_correct=int(Y.sum()), own_accepted_precision=float(Y[acc].mean()) if n_acc else None,
               own_correct_accepted=float((Y * acc).sum() / max(Y.sum(), 1)),
               own_P_precision=float(prec[n_acc - 1]) if n_acc else None)
    target = ref_precision if ref_precision is not None else out["own_P_precision"]
    if target is not None:
        ok = np.flatnonzero(prec >= target - 1e-9)
        j = int(ok.max()) if len(ok) else -1
        out.update(target_precision=float(target), tau_at_target=float(P[order][j]) if j >= 0 else None,
                   correct_accepted_at_target=float(cy[j] / max(Y.sum(), 1)) if j >= 0 else 0.0,
                   coverage_at_target=float((j + 1) / len(P)) if j >= 0 else 0.0)
    return out


def cmd_v12(a):
    _capture()
    a.sets_name = a.sets
    names = {"tune": TUNE12, "devtest": DEVTEST12, "legacy": LEGACY}.get(a.sets, a.sets.split(","))
    missing = [n for n in names if not cache_path(n).exists()]
    if missing:
        _p("not cached (python main.py decision cache --sets %s)", ",".join(missing))
    a.sets = ",".join(n for n in names if n not in missing)
    if a.cand:
        a.spot_override = a.cand
    out = diag_v11.cmd_v11(a)
    refs = {}
    rf = BASE / "v10.json"
    if rf.exists() and a.tag != "v10":              # the reference = v10 on the SAME set group (tune / devtest / ...)
        mp10 = read_json(rf).get(a.sets_name, {}).get("matched_precision", {})
        refs = {k: v.get("own_P_precision") for k, v in mp10.items()}
    dec_sets = [n for n in _BLOCKS if not n.startswith(("idle", "w_idle", "tail", "w_tail"))]
    groups_ = dict(pooled=dec_sets, offline=[n for n in dec_sets if not n.startswith("w_")], web=[n for n in dec_sets if n.startswith("w_")])
    out["matched_precision"] = {g: matched_precision(_BLOCKS, ss, refs.get(g)) for g, ss in groups_.items()}
    sent = {n: {k: out["sets"][n].get(k) for k in ("BLEU", "chrF", "sentence_n", "decoder_span_R1")} for n in out["sets"] if "BLEU" in out["sets"][n]}
    out["sentence_output"] = sent
    out["cand"] = a.cand or None
    for n, m in out["sets"].items():
        if n.startswith("nsig"):
            _p("%-16s new-signer top-1 %.3f top-5 %.3f correct-shown %.3f (n %d)", n, m["newsig_top1"], m["newsig_top5"], m["newsig_correct_shown"], m["newsig_n"])
        if "decoder_span_R1" in m and m.get("decoder_span_R1") is not None:
            _p("%-16s decoder-span R1 %.3f%s", n, m["decoder_span_R1"], f" | BLEU {m['BLEU']} chrF {m['chrF']}" if m.get("BLEU") is not None else "")
    _p("matched precision: %s", {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in out["matched_precision"].items()})
    name = a.out or a.tag
    write_json(REP / f"diag_v12_{name}.json", out)
    if a.freeze:
        BASE.mkdir(parents=True, exist_ok=True)
        f = BASE / f"{a.tag}.json"
        old = read_json(f) if f.exists() else {}
        old[a.sets_name] = out
        write_json(f, old)
        _p("frozen → %s [%s]", f, a.sets_name)
    return out
