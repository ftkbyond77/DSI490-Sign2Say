"""v12 caption-constrained spotting (prompt-opt-v12 2C, round 1): co-articulated word spans in TRAIN-signer sentences / phrases whose
caption names the words (a BAG — Thai order is not sign order).

  python scripts/v12_spot.py run [--tag v10] [--pct 90] [--round 1]

Per clip: the version's front + decoder → segments; every segment scored for every vocabulary word by the primary reader (span head on the
bank) fused with the hand cue and the landmark encoder exactly as the reader does (modules.handshape.fuse, version weights). A segment is
labelled w when w is the best CAPTION word on it, the primary reader AND the landmark encoder both rank w in their vocabulary top-3, and the
fused score beats the per-sentence null (90th percentile of the scores that the 50 most frequent vocabulary words NOT in the caption reach
on the same segments). One span per caption word (its best segment).
Precision proxy (no human audit possible here): same-signer check — when the signer has an isolated clip of w in the bank, is that clip the
nearest of the signer's own isolated clips to the span (primary reader)? Reported, and spans are bank-eligible (`bank_ok`) only when the
proxy precision over the checkable spans is >= 0.85 AND the span itself passes the check.
Output: data_prep/manifest/pseudo_spans_v12.parquet (+ round column), never BLIND / NEWSIG-dev signers' sentences.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from modules.utils import PREP, read_json

OUT = PREP / "manifest" / "pseudo_spans_v12.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--tag", default="v10"); ap.add_argument("--pct", type=float, default=90); ap.add_argument("--round", type=int, default=1)
    a = ap.parse_args()
    from decision import Harness
    from modules.handshape import descriptor, fuse
    from modules.schema import load
    from modules.sealed import is_sealed
    N = pd.read_parquet(PREP / "manifest" / "clips_v12_new.parquet")
    q = N[N.use & N.extracted & N.role.eq("train") & N.kind.isin(["sentence", "compound"]) & N.caption_words.notna()]
    q = q[~q.signer.map(lambda s: is_sealed(signer=s))]
    h = Harness(); h.use(a.tag)
    bank = h.banks["eval"]
    concepts = h.pipe.concepts
    cidx = {c: i for i, c in enumerate(concepts)}
    vocab = [w for w in h.vocab_words if w in cidx]
    V = np.array([cidx[w] for w in vocab])
    freq = pd.read_parquet(PREP / "shards" / "clips.parquet", columns=["concept", "kind"])
    top50 = [w for w in freq[freq.kind == "isolated"].concept.value_counts().index if w in set(vocab)][:80]
    rd = h.pipe.spot.get("read", {}); hand_w = float(h.pipe.spot.get("hand_w", 0.0)); lm_w = float(rd.get("lm_w", 0.0))
    # same-signer isolated exemplars (for the precision proxy)
    iso = N[N.use & N.kind.eq("isolated") & N.in_vocab & N.role.isin(["train"])]
    emb = np.load(ROOT / "cache" / "v12" / "emb_v10_new.npz"); ez = {str(c): z.astype(np.float32) for c, z in zip(emb["clip"], emb["Z"])}
    own = collections.defaultdict(list)
    for r in iso.itertuples():
        if r.clip_id in ez:
            own[r.signer].append((r.concept, ez[r.clip_id]))
    rows = []
    for k, r in enumerate(q.itertuples()):
        d = load(PREP / r.pose_file)
        it = dict(kp=np.asarray(d["kp"], np.float32), sc=np.asarray(d["sc"], np.float32), hw=tuple(int(x) for x in d["hw"]))
        st = h.front(it)
        st32 = {kk: (v.astype(np.float32) if kk in ("H", "H2") and v is not None else v) for kk, v in st.items()}
        segs = [s for s in h.back(st, "eval") if s["status"] != "transition"]
        cap = [w for w in str(r.caption_words).split("|") if w in cidx]
        if not segs or not cap or st32.get("H") is None:
            continue
        spans = [(int(s["f0"]), int(max(s["f1"], s["f0"] + 1))) for s in segs]
        E = (getattr(bank, "reader", None) or bank).scores(h.pipe.span(st32["H"], spans))
        Hh = bank.hand.scores(np.stack([descriptor(st32["kpi"], st32["sc"], x, y) for x, y in spans])) if bank.hand is not None else np.full_like(E, -2)
        L = bank.lm.scores(h.pipe.lm.spans(st32["kpi"], st32["sc"], spans)) if (bank.lm is not None and h.pipe.lm is not None) else np.full_like(E, -2)
        Fz = fuse(fuse(E, Hh, hand_w), L, lm_w) if lm_w else fuse(E, Hh, hand_w)
        nulls = [cidx[w] for w in top50 if w not in cap][:50]
        null_t = float(np.percentile(Fz[:, nulls], a.pct)) if nulls else -np.inf
        Ev, Lv = E[:, V], L[:, V]
        top3e = [set(V[np.argsort(-Ev[i])[:3]]) for i in range(len(spans))]
        top3l = [set(V[np.argsort(-Lv[i])[:3]]) for i in range(len(spans))]
        used = set()
        for w in sorted(cap, key=lambda w_: -Fz[:, cidx[w_]].max()):
            c = cidx[w]
            order = np.argsort(-Fz[:, c])
            for i in order:
                if i in used:
                    continue
                best_cap = max(cap, key=lambda w_: Fz[i, cidx[w_]])
                ok = best_cap == w and c in top3e[i] and c in top3l[i] and Fz[i, c] >= null_t
                if ok:
                    used.add(i)
                    zspan = h.pipe.span(st32["H"], [spans[i]])[0]; zspan = zspan / (np.linalg.norm(zspan) + 1e-8)
                    ex = own.get(r.signer, [])
                    check = None
                    if any(cw == w for cw, _ in ex):
                        sims = [(float(zspan @ z), cw) for cw, z in ex]
                        check = max(sims)[1] == w
                    rows.append(dict(clip_id=r.clip_id, concept=w, f0=spans[i][0], f1=spans[i][1], signer=r.signer, dataset=r.dataset,
                                     pose_file=r.pose_file, role=r.role, score=float(Fz[i, c]), null90=null_t, kind_src=r.kind,
                                     same_signer_check=check, round=a.round))
                break
        if k % 25 == 0:
            print(k, len(q), "spans", len(rows), flush=True)
    S = pd.DataFrame(rows)
    chk = S.same_signer_check.dropna().astype(bool) if len(S) else pd.Series(dtype=bool)
    prec = float(chk.mean()) if len(chk) else None
    S["bank_ok"] = bool(prec is not None and prec >= 0.85) & S.same_signer_check.fillna(False).astype(bool)
    S.to_parquet(OUT, index=False)
    capw = sum(len(str(x).split("|")) for x in q.caption_words)
    rep = dict(clips=len(q), spans=len(S), distinct_words=int(S.concept.nunique()) if len(S) else 0, caption_words_total=capw,
               located_share=len(S) / max(capw, 1), proxy_checkable=int(len(chk)), proxy_precision=prec, bank_ok=int(S.bank_ok.sum()),
               words_first_coarticulated=int(len(set(S.concept)) if len(S) else 0))
    (ROOT / "artifacts" / "reports" / f"v12_spot_round{a.round}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
