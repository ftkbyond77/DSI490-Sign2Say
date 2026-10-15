"""pitty: why are TTRS / THSL isolated-sign retrieval so much worse than ONE? Uses the v7 eval embeddings (no GPU).

For every non-null val/test query: its open-vocab rank (same rule as job_entry protocols), and what might explain a miss —
how many training instances / signers its concept has (and from which dataset), whether its signer is in train, hand visibility,
clip length, and what the nearest training instance is (same signer? same dataset? same concept?).
Run from Main_demo/:  python pitty/scripts/analyze_failures.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from modules.encoder import NULL, PoseStore, concept_scores  # noqa: E402

RUN = ROOT / "artifacts/runs/enc-v7-europe-west4"
E = pd.read_parquet(RUN / "emb_index.parquet")
Z = np.load(RUN / "emb.npy").astype(np.float32)
Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
C = int(E.c.max()) + 1
null_c = E.loc[E.concept == NULL, "c"].iloc[0] if (E.concept == NULL).any() else -1

tr = (E.role == "train").values & (E.concept != NULL).values
Ztr, ctr = Z[tr], E.c.values[tr]
Etr = E[tr].reset_index(drop=True)
per_c = Etr.groupby("c").agg(n_tr=("clip_id", "size"), n_sig=("signer", "nunique"))
per_c_ds = Etr.groupby(["c", "dataset"]).size().unstack(fill_value=0)
train_signers = set(Etr.signer)

q = E[(E.role.isin(["val", "test"])) & (E.concept != NULL) & (E.dataset.isin(["ttrs", "th_sl", "tslone", "agent", "tsl51"]))].copy()
Q = Z[q.index.values]
S = concept_scores(Q, Ztr, ctr, C)
if null_c >= 0:
    S[:, null_c] = -2
t = q.c.values
in_bank = S[np.arange(len(t)), t] > -1.5
rank = (S > S[np.arange(len(t)), t][:, None]).sum(1)
q["in_bank"] = in_bank
q["rank"] = np.where(in_bank, rank, -1)
q["hit"] = in_bank & (rank == 0)
q["top1_c"] = S.argmax(1)

nn = (Q @ Ztr.T).argmax(1)
q["nn_same_signer"] = Etr.signer.values[nn] == q.signer.values
q["nn_same_ds"] = Etr.dataset.values[nn] == q.dataset.values
q["nn_same_c"] = Etr.c.values[nn] == t
q["n_tr"] = q.c.map(per_c.n_tr).fillna(0).astype(int)
q["n_sig"] = q.c.map(per_c.n_sig).fillna(0).astype(int)
q["n_tr_same_ds"] = [int(per_c_ds.at[c, d]) if c in per_c_ds.index and d in per_c_ds.columns else 0 for c, d in zip(q.c, q.dataset)]
q["signer_seen"] = q.signer.isin(train_signers)
q["len"] = q.f1 - q.f0

clips = pd.read_parquet(ROOT / "cloud_s3/data_prep/shards/clips.parquet")
store = PoseStore(ROOT / "cloud_s3/data_prep/shards/pose_store.npz")
hv = []
for r in q.itertuples():
    _, sc, _ = store.get(store.row[r.clip_id], int(r.f0), int(r.f1))
    L, R = (sc[:, 91:112] > 0).mean(1) > 0.5, (sc[:, 112:133] > 0).mean(1) > 0.5
    hv.append(((L | R).mean(), (L & R).mean()))
q["hand_any"], q["hand_both"] = np.array(hv).T

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
name = {"ttrs": "TTRS", "th_sl": "THSL", "tslone": "ONE", "agent": "AGENT", "tsl51": "T51U"}
q["set"] = q.dataset.map(name)
print("\n=== 1. per set: how many queries even have their concept in the train bank, and R1 among those ===")
g = q.groupby("set")
print(pd.DataFrame(dict(n=g.size(), in_bank=g.in_bank.mean(), R1_inbank=g.apply(lambda d: d.hit[d.in_bank].mean()),
                        R5_inbank=g.apply(lambda d: (d["rank"][d.in_bank] < 5).mean()),
                        signer_seen=g.signer_seen.mean(), n_signers=g.signer.nunique(),
                        med_tr_inst=g.n_tr.median(), med_tr_signers=g.n_sig.median(), frac_has_same_ds_train=g.apply(lambda d: (d.n_tr_same_ds > 0).mean()),
                        hand_any=g.hand_any.mean(), hand_both=g.hand_both.mean(), med_len=g.len.median())).round(3))

b = q[q.in_bank]
print("\n=== 2. nearest training instance: does the embedding cluster by SIGN or by SIGNER / DATASET? ===")
print(b.groupby("set")[["nn_same_c", "nn_same_signer", "nn_same_ds"]].mean().round(3))
print("   (nn_same_signer high + nn_same_c low ⇒ the embedding is grouping by who signs, not what is signed)")

print("\n=== 3. R1 vs whether the concept has training examples from the SAME dataset (in-bank queries) ===")
b = b.assign(same_ds_train=np.where(b.n_tr_same_ds > 0, "yes", "no"))
print(b.groupby(["set", "same_ds_train"]).hit.agg(["size", "mean"]).round(3))

print("\n=== 4. R1 vs number of training signers of the concept ===")
b = b.assign(sig_bin=pd.cut(b.n_sig, [0, 1, 2, 4, 100], labels=["1", "2", "3-4", "5+"]))
print(b.groupby(["set", "sig_bin"], observed=True).hit.agg(["size", "mean"]).round(3).unstack(0))

print("\n=== 5. R1 vs hand visibility (fraction of frames with ≥1 hand detected) ===")
b = b.assign(hand_bin=pd.cut(b.hand_any, [-0.01, 0.5, 0.8, 0.95, 1.0], labels=["<50%", "50-80%", "80-95%", "95%+"]))
print(b.groupby(["set", "hand_bin"], observed=True).hit.agg(["size", "mean"]).round(3).unstack(0))

print("\n=== 6. R1 vs whether the query's signer is in train ===")
print(b.groupby(["set", "signer_seen"]).hit.agg(["size", "mean"]).round(3))

print("\n=== 7. most frequent wrong top-1 concepts (hubs) in TTRS + THSL misses ===")
cname = E.drop_duplicates("c").set_index("c").concept
m = b[b.set.isin(["TTRS", "THSL"]) & ~b.hit]
top = m.top1_c.value_counts().head(12)
print(pd.DataFrame(dict(concept=cname.reindex(top.index).values, times_predicted=top.values, n_train=per_c.n_tr.reindex(top.index).values)))

out = ROOT / "cache/analysis"; out.mkdir(parents=True, exist_ok=True)
q.drop(columns=["top1_c"]).assign(top1=cname.reindex(q.top1_c).values).to_csv(out / "v7_query_failures.csv", index=False, encoding="utf-8-sig")
print(f"\nper-query table → {out / 'v7_query_failures.csv'}")
