"""Rebuild models/baseline-model/ from scratch (local GPU, ~15 min on an RTX 2050).

  python models/baseline-model/build.py            # all steps
  python models/baseline-model/build.py base       # only the frozen Uni-Sign weights
  python models/baseline-model/build.py embed      # embed the Thai clips with the frozen encoder + train the light adapter

1. base    Uni-Sign pose-only WLASL checkpoint (ZechengLi19/Uni-Sign, Hugging Face) → unisign_wlasl_enc.pt  (ST-GCN + mT5 encoder, fp16)
2. embed   every isolated Thai clip (data_prep shards) → frozen embedding Z0; adapter = residual MLP on Z0 trained with CosFace on the
           TRAIN signers only (epoch picked on VAL signers; test signers and data_test never touched) → Z1; both go into bank.npz
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
import torch.nn.functional as F

SEED = 0
BASE = HERE / "unisign_wlasl_enc.pt"


def step_base():
    from huggingface_hub import hf_hub_download
    from modules.unisign import UniSignEncoder
    hf_hub_download("ZechengLi19/Uni-Sign", "wlasl_pose_only_islr.pth")
    enc = UniSignEncoder("wlasl", use_mt5=True, device="cpu")
    sd = {k: (v.half() if v.is_floating_point() and k != "prefix" else v) for k, v in enc.state_dict().items()}
    torch.save(sd, BASE)
    (HERE / "mt5_config.json").write_bytes((ROOT / "artifacts" / "pretrained" / "mt5_config.json").read_bytes())
    print("wrote", BASE, f"{BASE.stat().st_size / 1e6:.0f} MB")


def step_embed(epochs=60, bs=512):
    import train_encoder as T
    from baseline import load_encoder, pooled
    from modules.encoder import collate, concept_scores, retrieval_metrics
    from modules.parts import prep, to_iso
    from concurrent.futures import ThreadPoolExecutor

    torch.manual_seed(SEED); np.random.seed(SEED)
    T.ensure_mm(); store = T.PoseStoreMM()
    clips, iso, concepts, cidx, daily = T.load_tables(store)
    iso = iso[iso.kind == "isolated"].reset_index(drop=True)
    iso = iso[iso.role.isin(["train", "val", "test", "deploy"])].reset_index(drop=True)
    concepts = [c for c in concepts if c != T.NULL]; cidx = {c: i for i, c in enumerate(concepts)}
    iso["c"] = iso.concept.map(cidx); iso = iso[iso.c.notna()].reset_index(drop=True); iso["c"] = iso.c.astype(int)
    span = int(np.median((iso.span_f1 - iso.span_f0)[iso.role == "train"]))
    (HERE / "config.json").write_text(json.dumps(dict(
        name="baseline-model", encoder="Uni-Sign pose-only (WLASL ISLR checkpoint): part-wise ST-GCN + mT5-base encoder, frozen",
        pooling="mean over frames of the mT5 encoder states, L2-normalised (768-d)", retrieval="mean of the top-2 instance cosines per concept",
        variants=dict(zeroshot="frozen encoder, bank of Thai clips", adapted="+ residual MLP adapter (768-1024-768, CosFace on train signers)"),
        median_sign_frames=span, seed=SEED, built=time.strftime("%Y-%m-%d %H:%M")), indent=1), encoding="utf-8")
    print(len(iso), "isolated clips,", len(concepts), "concepts;", iso.role.value_counts().to_dict())

    def feat(r):
        kp, sc, hw = store.get(int(r[0]), int(r[1]), int(r[2]))
        return prep(to_iso(kp, hw), sc)
    with ThreadPoolExecutor(6) as ex:
        feats = list(ex.map(feat, zip(iso.row, iso.f0, iso.f1), chunksize=64))

    enc = load_encoder(HERE, adapter=False)
    order = np.argsort([len(f["body"]) for f in feats]); Zr = np.zeros((len(feats), 768), np.float32)
    t0 = time.time()
    for i in range(0, len(order), 32):
        ix = order[i:i + 32]
        Zr[ix] = pooled(enc, [feats[j] for j in ix])
        if (i // 32) % 50 == 0:
            print(f"  embed {i}/{len(order)}  {time.time() - t0:.0f}s", flush=True)
    Z0 = Zr / np.linalg.norm(Zr, axis=1, keepdims=True)

    # ---- light adapter: residual MLP (the SignEncoder head, zero-init = identity) trained on frozen pooled features
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head = enc.head.to(dev).float().train()
    tr = (iso.role == "train").values; va = (iso.role == "val").values
    Xtr = torch.tensor(Zr[tr], device=dev); ytr = torch.tensor(iso.c.values[tr], device=dev)
    cls = sorted(set(iso.c.values[tr])); remap = {c: i for i, c in enumerate(cls)}
    ytr_m = torch.tensor([remap[c] for c in iso.c.values[tr]], device=dev)
    proto = torch.stack([F.normalize(Xtr[ytr_m == i].mean(0), dim=0) for i in range(len(cls))])
    W = torch.nn.Parameter(proto.clone())
    opt = torch.optim.AdamW([dict(params=head.parameters(), lr=5e-4), dict(params=[W], lr=5e-4)], weight_decay=1e-2)

    def emb(x):
        return F.normalize(x + head(x), dim=-1)

    def val_mrr():
        head.eval()
        with torch.no_grad():
            Z = emb(torch.tensor(Zr, device=dev)).cpu().numpy()
        head.train()
        S = concept_scores(Z[va], Z[tr], iso.c.values[tr], len(concepts)); return retrieval_metrics(S, iso.c.values[va])["MRR"], Z
    best, hist, best_sd = (val_mrr()[0], 0), [], {k: v.clone() for k, v in head.state_dict().items()}
    hist.append(dict(epoch=0, val_mrr=best[0]))
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(Xtr), device=dev); tot = 0
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            z = emb(F.dropout(Xtr[j], 0.1))
            logit = 30 * (z @ F.normalize(W, dim=-1).T)
            logit = logit - 30 * 0.25 * F.one_hot(ytr_m[j], len(cls))          # CosFace margin
            loss = F.cross_entropy(logit, ytr_m[j]); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(j)
        m, _ = val_mrr(); hist.append(dict(epoch=ep, loss=tot / len(perm), val_mrr=m))
        if m > best[0]:
            best, best_sd = (m, ep), {k: v.clone() for k, v in head.state_dict().items()}
        if ep % 10 == 0:
            print(f"  adapter epoch {ep} loss {tot / len(perm):.3f} val MRR {m:.3f} (best {best[0]:.3f} @ {best[1]})", flush=True)
    head.load_state_dict(best_sd); head.eval()
    with torch.no_grad():
        Z1 = emb(torch.tensor(Zr, device=dev)).cpu().numpy()
    torch.save({k: v.cpu() for k, v in head.state_dict().items()}, HERE / "adapter.pt")
    np.savez_compressed(HERE / "bank.npz", Z0=Z0.astype(np.float16), Z1=Z1.astype(np.float16), concept=iso.c.values.astype(np.int32),
                        concepts=np.array(concepts), role=iso.role.values.astype(str), dataset=iso.dataset.values.astype(str),
                        signer=iso.signer.values.astype(str), clip_id=iso.clip_id.values.astype(str),
                        daily=np.array(json.loads((ROOT / "models" / "model-v10" / "vocab_conversation.json").read_text(encoding="utf-8"))["concepts"]))
    (HERE / "results" / "adapter_history.json").write_text(json.dumps(dict(best_epoch=best[1], best_val_mrr=best[0], history=hist)), encoding="utf-8")
    print("adapter best epoch", best[1], "val MRR", round(best[0], 4))


if __name__ == "__main__":
    steps = sys.argv[1:] or ["base", "embed"]
    for s in steps:
        {"base": step_base, "embed": step_embed}[s]()
