"""v12 banks (prompt-opt-v12 2A / 3B): add new-source exemplars to the three reader banks of a candidate folder.

  python scripts/v12_bank.py build --out artifacts/v12/B1 [--base models/model-v10] [--enc <onnx dir or v10>] [--pseudo]

Writes <out>/bank.npz, hand_bank.npz, lm_bank.npz (production: every non-sealed new exemplar, TRAIN + NEWSIG-dev) and
<out>/eval/{bank_eval,hand_bank_eval,lm_bank_eval}.npz (evaluation: old TRAIN-signer rows + new TRAIN-role exemplars only), rows appended
in the SAME order to all three banks (the sealing masks of the harness index them together). Recipe = the v10 bank recipe:
canonical hands → iso → sign span (movement phases f0..f3) → primary encoder embedding (L2) / hand descriptor projected with the
EXISTING metric / landmark-encoder embedding. Only concepts the model already knows are added (new Tail concepts would change the
concept list — recorded, not done). BLIND-v12 rows are refused by modules.sealed.assert_clean.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from modules.utils import ART, PREP, read_json

V10 = ROOT / "models" / "model-v10"


def new_rows(role_eval=("train",), role_prod=("train", "newsig_dev"), pseudo=False, max_per=0, known_forms=False):
    N = pd.read_parquet(PREP / "manifest" / "clips_v12_new.parquet")
    q = N[N.use & N.exemplar & N.kind.eq("isolated") & N.role.isin(role_prod)].copy()
    if known_forms:                          # only clips that joined a KNOWN form of a vocabulary word (Tail clips keep their rule)
        q = q[~(q.in_vocab & q.new_form.fillna(False).astype(bool))]
    if max_per:                              # at most max_per exemplars per (word, signer), best quality first
        q = q.sort_values("quality", ascending=False).groupby(["concept", "signer"], group_keys=False).head(max_per)
    q["eval"] = q.role.isin(role_eval)
    rows = [dict(clip=r.clip_id, concept=r.concept, signer=r.signer, dataset=r.dataset, pose=r.pose_file, f0=int(r.phase_f0), f1=int(r.phase_f3) + 1,
                 eval=bool(r.eval)) for r in q.itertuples()]
    if pseudo:
        f = PREP / "manifest" / "pseudo_spans_v12.parquet"
        if f.exists():
            S = pd.read_parquet(f)
            S = S[S.bank_ok]
            for r in S.itertuples():
                rows.append(dict(clip=f"{r.clip_id}@{r.f0}", concept=r.concept, signer=r.signer, dataset=r.dataset, pose=r.pose_file,
                                 f0=int(r.f0), f1=int(r.f1), eval=r.role == "train"))
    return pd.DataFrame(rows)


def embed_rows(R, enc_dir=None, lm_onnx=None, device="cuda"):
    """→ Z primary [N,768], D hand descriptors [N,DIM], L lmenc [N,d], ok [N]"""
    from modules import lmenc
    from modules.handshape import descriptor
    from modules.parts import prep, to_iso
    from modules.runtime import make_backend
    from modules.schema import canonical_hands, load
    be = make_backend(enc_dir or V10, "torch", device) if not (enc_dir and (Path(enc_dir) / "encoder.onnx").exists()) else None
    if be is None:                                   # an ONNX-only candidate encoder
        from modules.runtime import OnnxBackend
        be = OnnxBackend(Path(enc_dir), "cpu")
    lm = lmenc.OnnxLM(lm_onnx or (V10 / "onnx" / "lmenc.onnx"))
    parts, D, F, ok = [], [], [], []
    for r in R.itertuples():
        d = load(PREP / r.pose)
        hw = tuple(int(x) for x in d["hw"])
        kp, sc, _ = canonical_hands(np.asarray(d["kp"], np.float32), np.asarray(d["sc"], np.float32), hw)
        kpi = to_iso(kp, hw)
        f0, f1 = max(0, r.f0), min(len(kp), r.f1)
        if f1 - f0 < 8:
            f0, f1 = 0, len(kp)
        parts.append(prep(kpi[f0:f1], sc[f0:f1])); D.append(descriptor(kpi, sc, f0, f1)); F.append(lm.prep_fn(kpi[f0:f1], sc[f0:f1])); ok.append(True)
    Z = be.embed(parts); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    L = lm.embed(F)
    return Z.astype(np.float32), np.stack(D).astype(np.float32), np.asarray(L, np.float32), np.array(ok)


def append_banks(out, R, Z, D, L, base=V10, eval_src=None):
    from modules.handshape import HandBank
    from modules.sealed import assert_clean
    assert_clean([c.split("@")[0] for c in R["clip"]], "v12 bank", signers=R.signer.tolist())
    concepts = read_json(base / "concepts.json")
    cidx = {c: i for i, c in enumerate(concepts)}
    known = R.concept.isin(cidx).values
    R, Z, D, L = R[known].reset_index(drop=True), Z[known], D[known], L[known]
    c_new = np.array([cidx[c] for c in R.concept], np.int32)
    out = Path(out); (out / "eval").mkdir(parents=True, exist_ok=True)
    ev = eval_src or dict(bank=ART / "eval" / "bank_eval.npz", hand=ART / "eval" / "hand_bank_eval.npz", lm=ART / "eval" / "lm_bank_eval.npz")
    pr = dict(bank=base / "bank.npz", hand=base / "hand_bank.npz", lm=base / "lm_bank.npz")
    for name, src, mask, dst in (("eval", ev, R["eval"].values, out / "eval"), ("prod", pr, np.ones(len(R), bool), out)):
        b = dict(np.load(src["bank"]))
        h = dict(np.load(src["hand"]))
        lz = dict(np.load(src["lm"]))
        assert len(b["c"]) == len(h["c"]) == len(lz["c"]), f"{name}: bank rows misaligned"
        tf = dict(mu=h["mu"], sd=h["sd"], A=h["A"])
        P = HandBank.project(D[mask], tf)
        nb = dict(Z=np.concatenate([b["Z"], Z[mask].astype(b["Z"].dtype)]), c=np.concatenate([b["c"], c_new[mask]]),
                  clip=np.concatenate([b["clip"].astype(str), R["clip"].values[mask].astype(str)]),
                  dataset=np.concatenate([b["dataset"].astype(str), R.dataset.values[mask].astype(str)]),
                  signer=np.concatenate([b["signer"].astype(str), R.signer.values[mask].astype(str)]))
        np.savez(dst / ("bank_eval.npz" if name == "eval" else "bank.npz"), **nb)
        np.savez(dst / ("hand_bank_eval.npz" if name == "eval" else "hand_bank.npz"), P=np.concatenate([h["P"], P.astype(h["P"].dtype)]),
                 c=np.concatenate([h["c"], c_new[mask]]), **tf)
        Lz = np.zeros((int(mask.sum()), lz["Z"].shape[1]), lz["Z"].dtype); Lz[:] = L[mask][:, :lz["Z"].shape[1]]
        np.savez(dst / ("lm_bank_eval.npz" if name == "eval" else "lm_bank.npz"), Z=np.concatenate([lz["Z"], Lz]),
                 c=np.concatenate([lz["c"], c_new[mask]]), ok=np.concatenate([lz["ok"], np.ones(int(mask.sum()), bool)]))
        print(f"{name}: +{int(mask.sum())} rows → {len(nb['c'])} ({dst})", flush=True)
    return R


def cmd_build(a):
    R = new_rows(pseudo=a.pseudo, max_per=a.max_per, known_forms=a.known_forms)
    print("new exemplar rows", len(R), R.groupby(["dataset", "eval"]).size().to_dict(), flush=True)
    Z, D, L, ok = embed_rows(R, a.enc or None, a.lm or None)
    R = append_banks(a.out, R, Z, D, L)
    Path(a.out, "bank_v12.json").write_text(json.dumps(dict(rows=len(R), eval_rows=int(R["eval"].sum()), enc=a.enc or "v10", lm=a.lm or "v10",
                                                           pseudo=a.pseudo, words=int(R.concept.nunique())), ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build"); p.add_argument("--out", required=True); p.add_argument("--enc", default=""); p.add_argument("--lm", default="")
    p.add_argument("--pseudo", action="store_true"); p.add_argument("--max_per", type=int, default=0); p.add_argument("--known_forms", action="store_true")
    a = ap.parse_args()
    cmd_build(a)
