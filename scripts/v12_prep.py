"""v12 data preparation (prompt-opt-v12 TPH 1): new sources → Standard Schema v12 + annotation layers + QC + splits.

  python main.py prep v12 ingest     # extracted new-source clips → rows (+ signer identity from faces, contact sheets in qc/v12/)
  python main.py prep v12 labels     # gloss → concept, kinds (compound / classifier / letter), captions → caption_words
  python main.py prep v12 annotate   # phases, handedness, quality, v10 embeddings, forms
  python main.py prep v12 qc         # extraction QC, duplicates, label audit, attractor audit, exemplar flag
  python main.py prep v12 splits     # roles of the new signers + BLIND-v12 sealing (artifacts/eval/blind_v12_manifest.json)
  python main.py prep v12 eda        # artifacts/reports/v12_eda.json + figures + coverage matrix
  python main.py prep v12 shards     # data_prep/shards_v12/: new clips packed like shards/ (training reads shards/ + shards_v12/)

Every step is re-runnable and idempotent; results live in data_prep/manifest/clips_v12.parquet (old rows + new rows + v12 columns).
Nothing is deleted: clips are excluded by `use = False` with `exclude_reason`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from modules.utils import ART, CACHE, PREP, read_json, safe_id, write_json

MAN = PREP / "manifest"
QC = PREP / "qc" / "v12"
NEW = MAN / "clips_v12_new.parquet"          # new-source rows (all v12 columns)
V12 = MAN / "clips_v12.parquet"              # old + new rows, v12 columns
ITEMS = PREP / "params" / "v12_video_items.csv"
RID_DIR, ETSL_DIR = ROOT / "cloud_s3" / "sign-ราชบัณฑิต", ROOT / "cloud_s3" / "sign-etsl-chula"
LIC = {"royal_institute": "Office of the Royal Society TSL dictionary (rights reserved) - research use only, do not redistribute videos",
       "etsl": "e-TSL Chulalongkorn University (rights reserved) - research use only, do not redistribute videos"}
EMB = CACHE / "v12" / "emb_v10_new.npz"


def _p(*a):
    print(*a, flush=True)


def tiers():
    return read_json(MAN / "tiers.json")


# ----------------------------------------------------------------------------- ingest
def cmd_ingest(a):
    """rows of every extracted new-source clip + signer identity (face embeddings → clusters per source → contact sheet)."""
    from modules.identity import embed_faces, face_crops
    from modules.mediapipe_pose import load_raw
    D = pd.read_csv(ITEMS)
    R = pd.read_csv(RID_DIR / "manifest.csv"); E = pd.read_csv(ETSL_DIR / "manifest.csv")
    meta = {}
    for r in R[R.file.notna()].itertuples():
        meta[f"rid:{Path(r.file).stem}"] = dict(caption=r.caption, headword=r.word, source_url=r.video, group=int(r.group), entry=int(r.entry),
                                               sub=None if pd.isna(r.sub) else int(r.sub), status=r.status, pos=None, meaning=None, n_signs=1,
                                               synonyms=None, wrd_id=None, sign_no=None)
    for r in E[E.file.notna()].itertuples():
        meta[f"etsl:{Path(r.file).stem}"] = dict(caption=r.word_th if r.type in ("phrase", "sentence") else None, headword=r.word_th,
                                                source_url=r.url, group=None, entry=None, sub=None, status=None, pos=r.pos_en,
                                                meaning=r.meaning_th, n_signs=int(r.n_signs), synonyms=r.synonyms, wrd_id=int(r.wrd_id),
                                                sign_no=int(r.sign_no))
    rows = []
    for r in D.itertuples():
        raw = PREP / "pose_mp_raw" / r.dataset / f"{r.out_name}.npz"
        std = PREP / "pose_mp" / r.dataset / f"{r.out_name}.npz"
        m = meta.get(r.clip_id, {})
        rows.append(dict(clip_id=r.clip_id, dataset=r.dataset, subset=r.dataset, src_kind=r.kind, gloss=r.word, video=r.video,
                         pose_file=str(std.relative_to(PREP)) if std.exists() else None, pose_raw_file=str(raw.relative_to(PREP)) if raw.exists() else None,
                         extracted=bool(std.exists() and raw.exists()), stage=int(r.stage), dialog=bool(r.dialog), dup_tsl51=bool(r.dup_tsl51),
                         licence=LIC[r.dataset], extractor="mp", **m))
    N = pd.DataFrame(rows)
    _p("rows", len(N), "extracted", int(N.extracted.sum()), N.groupby(["dataset", "extracted"]).size().to_dict())
    # ---- signer identity: DINOv2 face embedding (3 frames per clip), cached per clip
    cache_f = CACHE / "v12" / "face_emb_new.npz"; cache_f.parent.mkdir(parents=True, exist_ok=True)
    have = {}
    if cache_f.exists():
        z = np.load(cache_f, allow_pickle=True); have = {str(c): (v, int(f)) for c, v, f in zip(z["clip"], z["Z"], z["found"])}
    todo = N[N.extracted & ~N.clip_id.isin(list(have))]
    crops, owner = [], []
    t0 = time.time()
    for i, r in enumerate(todo.itertuples()):
        cs = face_crops(r.video, load_raw(PREP / r.pose_raw_file), (0.25, 0.5, 0.75))
        crops += cs; owner += [r.clip_id] * len(cs)
        if (i + 1) % 300 == 0:
            _p(f"  face crops {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)")
    if crops:
        Ef = embed_faces(crops)
        owner = np.array(owner)
        for c in todo.clip_id:
            mm = owner == c
            if mm.any():
                v = Ef[mm].mean(0); have[c] = (v / (np.linalg.norm(v) + 1e-9), int(mm.sum()))
            else:
                have[c] = (np.zeros(768, np.float32), 0)
    for c in todo.clip_id:
        have.setdefault(c, (np.zeros(768, np.float32), 0))
    cl = sorted(have)
    np.savez_compressed(cache_f, clip=np.array(cl), Z=np.stack([have[c][0] for c in cl]).astype(np.float32), found=np.array([have[c][1] for c in cl]))
    N["face_found"] = N.clip_id.map(lambda c: have.get(c, (None, 0))[1])
    # cluster per source (centred cosine, average linkage, small clusters folded into the nearest big one) — as th_sl (data_prep.cmd_signers)
    from sklearn.cluster import AgglomerativeClustering
    N["signer"] = None; N["session"] = None; N["id_margin"] = np.nan
    for ds, tag in (("royal_institute", "rid"), ("etsl", "etsl")):
        ids = [c for c in N[(N.dataset == ds) & (N.face_found > 0) & ~N.dialog].clip_id]
        if len(ids) < 5:
            continue
        Z = np.stack([have[c][0] for c in ids]); Z = Z - Z.mean(0); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
        thr = a.thr_rid if ds == "royal_institute" else a.thr
        lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=thr).fit(Z).labels_
        sizes = pd.Series(lab).value_counts()
        big = [c for c, n in sizes.items() if n >= a.min_size] or [sizes.index[0]]
        cent = np.stack([Z[lab == c].mean(0) / np.linalg.norm(Z[lab == c].mean(0)) for c in big])
        lab = np.array([l if l in big else big[int(np.argmax(cent @ z))] for l, z in zip(lab, Z)])
        order = {c: k for k, c in enumerate(sorted(big, key=lambda c: -(lab == c).sum()))}
        sig = {c: f"S_{tag}_{order[l]:02d}" for c, l in zip(ids, lab)}
        anchors = PREP / "params" / "v12_signer_anchors_rid.npz"
        if ds == "royal_institute":
            # identities checked by eye on qc/v12/signers_rid.jpg (2026-11-15, 291 clips): clusters 00+04 = man, red shirt, glasses;
            # 01+02 = young woman, grey shirt; 03 = older woman, pink shirt. Frozen as anchors; later runs assign the nearest anchor.
            if not anchors.exists() and a.rid_merge:
                groups = [[int(x) for x in g.split("+")] for g in a.rid_merge.split(",")]
                prev = read_json(PREP / "params" / "v12_rid_clusters.json")      # the clustering that was LOOKED AT (previous run)
                ids_p = [c for c in ids if c in prev]
                mu = np.stack([have[c][0] for c in ids_p]).mean(0)
                Zp = np.stack([have[c][0] for c in ids_p]) - mu; Zp /= np.linalg.norm(Zp, axis=1, keepdims=True) + 1e-9
                P = []
                for g in groups:
                    m = np.isin(np.array([int(prev[c][-2:]) for c in ids_p]), g)
                    v = Zp[m].mean(0); P.append(v / np.linalg.norm(v))
                np.savez(anchors, P=np.stack(P), mu=mu, names=np.array([f"S_rid_{k:02d}" for k in range(len(groups))]))
            if not anchors.exists():
                write_json(PREP / "params" / "v12_rid_clusters.json", sig)
            if anchors.exists():
                az = np.load(anchors)
                Zr = np.stack([have[c][0] for c in ids]) - az["mu"]; Zr /= np.linalg.norm(Zr, axis=1, keepdims=True) + 1e-9
                sim = Zr @ az["P"].T
                sig = {c: str(az["names"][k]) for c, k in zip(ids, sim.argmax(1))}
                N.loc[N.clip_id.isin(ids), "id_margin"] = pd.Series(np.sort(sim, 1)[:, -1] - np.sort(sim, 1)[:, -2], index=ids).reindex(N.clip_id[N.clip_id.isin(ids)]).values
        if ds == "etsl":
            # checked by eye on qc/v12/signers_etsl_sessions.jpg (2026-11-15): every cluster is the SAME woman in different recording
            # sessions (lighting / background) -> one signer; the clusters are kept as `session` (nuisance, not identity)
            N.loc[N.clip_id.isin(sig), "session"] = N.clip_id.map(sig).str.replace("S_etsl_", "etsl_session_")
            _sheet(N[N.clip_id.isin(sig)].assign(signer=N.clip_id.map(sig)), QC / "signers_etsl_sessions.jpg")
            sig = {c: "S_etsl_00" for c in sig}
        N.loc[N.clip_id.isin(sig), "signer"] = N.clip_id.map(sig)
        _p(ds, "signers", pd.Series(list(sig.values())).value_counts().to_dict())
        _sheet(N[N.clip_id.isin(sig)], QC / f"signers_{tag}.jpg")
    N.loc[N.dialog, "signer"] = "S_rid_dialog"          # two men alternate in the Q/A clips: per-clip identity is set by the lock QC
    N.loc[N.signer.isna() & N.extracted, "signer"] = N.dataset.map({"royal_institute": "S_rid_unknown", "etsl": "S_etsl_unknown"})
    _save_new(N, merge=("concept", "kind", "caption_words", "phase_f0", "phase_f1", "phase_f2", "phase_f3", "form_id", "handedness",
                        "n_hands", "symmetric", "quality", "quality_flags", "exemplar", "label_status", "role", "use", "exclude_reason",
                        "view", "lemma", "sense", "sign_variant", "pseudo", "sealed", "signer_manual"))


def _save_new(N, merge=()):
    """write NEW, keeping columns later steps filled (re-running ingest must not wipe annotation)"""
    if NEW.exists():
        old = pd.read_parquet(NEW)
        keep = [c for c in merge if c in old.columns]
        if keep:
            N = N.drop(columns=[c for c in keep if c in N.columns]).merge(old[["clip_id"] + keep], on="clip_id", how="left")
        if "signer_manual" in N.columns:                     # a manual signer correction always wins
            m = N.signer_manual.notna()
            N.loc[m, "signer"] = N.loc[m, "signer_manual"]
    N.to_parquet(NEW, index=False)
    _p("saved", NEW, N.shape)


def _sheet(N, out, per=8, size=(96, 120)):
    """contact sheet: one row per signer cluster, `per` face crops from different clips"""
    import cv2
    from modules.identity import face_crops
    from modules.mediapipe_pose import load_raw
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sg, g in N.groupby("signer"):
        g = g.sample(min(per, len(g)), random_state=0)
        ims = []
        for r in g.itertuples():
            cs = face_crops(r.video, load_raw(PREP / r.pose_raw_file), (0.5,), size=size)
            ims.append(cv2.cvtColor(cs[0], cv2.COLOR_RGB2BGR) if cs else np.zeros((size[1], size[0], 3), np.uint8))
        ims += [np.zeros((size[1], size[0], 3), np.uint8)] * (per - len(ims))
        row = np.hstack(ims)
        lab = np.zeros((size[1], 170, 3), np.uint8)
        cv2.putText(lab, f"{sg}", (4, 40), 0, 0.5, (255, 255, 255), 1); cv2.putText(lab, f"n={len(N[N.signer == sg])}", (4, 70), 0, 0.5, (255, 255, 255), 1)
        rows.append(np.hstack([lab, row]))
    if rows:
        cv2.imwrite(str(out), np.vstack(rows))
        _p("sheet", out)


# ----------------------------------------------------------------------------- labels (prompt 1C)
def cmd_labels(a):
    """gloss → lemma → concept (existing concept map; new lemmas are new concepts), kind, sense / variant, compounds, captions."""
    from pythainlp.tokenize import word_tokenize
    from modules.lexicon import thsl_lemma
    from modules.v12data import caption_words
    N = pd.read_parquet(NEW)
    cmap = read_json(MAN / "concepts.json")["concept"]
    W = set(tiers()["words"])
    allc = set(cmap.values())
    E = pd.read_csv(ETSL_DIR / "manifest.csv")
    n_wrd = E[E.type.isin(["word", "phrase"])].groupby("word_th").wrd_id.nunique()
    out = []
    for r in N.itertuples():
        try:
            lem, var, sense, aliases = thsl_lemma(r.gloss)
        except KeyError:                                            # numbers above 999,999 ("1000000"): keep the gloss
            lem, var, sense, aliases = str(r.gloss).strip(), None, None, []
        concept = cmap.get(lem, lem)
        kind, cw, sv = None, None, None
        if r.dataset == "royal_institute":
            kind = {"word": "isolated", "sentence": "sentence", "classifier": "classifier"}[r.src_kind]
            if kind == "classifier":
                concept = f"clf:{lem}"                              # never a positive of the headword (hard negative / Tail)
        else:
            if r.src_kind == "letter":
                kind, concept = "letter", f"letter:{lem}"
            elif r.src_kind == "sentence":
                kind = "sentence"
            elif (r.n_signs or 1) > 1:
                kind = "compound"
                parts = [cmap.get(t, t) for t in word_tokenize(lem, engine="newmm", keep_whitespace=False) if t.strip()]
                if len(parts) == int(r.n_signs) and all(p in allc for p in parts):
                    concept = "|".join(parts)                       # a two-word mini-sentence with a known gloss sequence
            else:
                kind = "isolated"
            if r.wrd_id is not None and not pd.isna(r.wrd_id):
                sv = f"etsl:{int(r.wrd_id)}.{int(r.sign_no)}"       # documented variant (word id . sign number)
                if n_wrd.get(r.gloss, 1) > 1:
                    sense = sense or f"etsl_wrd{int(r.wrd_id)}"     # homograph entries: kept apart until the forms step decides
        if kind in ("sentence", "compound") or (r.dataset == "royal_institute" and kind == "sentence"):
            cw = caption_words(r.caption if isinstance(r.caption, str) else r.gloss, W, cmap)
        out.append(dict(clip_id=r.clip_id, lemma=lem, concept=concept, kind=kind, sense=sense, sign_variant=sv or var,
                        caption_words="|".join(cw) if cw else None, aliases="|".join(aliases) or None))
    L = pd.DataFrame(out)
    N = N.drop(columns=[c for c in L.columns if c != "clip_id" and c in N.columns]).merge(L, on="clip_id", how="left")
    # synonym candidates (e-TSL `synonyms`): decided in `annotate` by embeddings (prompt 1C rule), recorded here
    cand = []
    for r in N[N.dataset == "etsl"].itertuples():
        if isinstance(r.synonyms, str) and r.synonyms.strip() not in ("", "-"):
            for s_ in str(r.synonyms).replace(",", "|").split("|"):
                s_ = s_.strip()
                if s_:
                    cand.append(dict(a=r.concept, b=cmap.get(s_, s_), clip=r.clip_id))
    pd.DataFrame(cand).drop_duplicates(["a", "b"]).to_csv(PREP / "params" / "v12_synonym_candidates.csv", index=False, encoding="utf-8")
    N["in_vocab"] = N.concept.isin(W)
    _save_new(N)
    _p(N.groupby(["dataset", "kind"]).agg(clips=("clip_id", "size"), in_vocab=("in_vocab", "sum"), concepts=("concept", "nunique")).to_string())
    s = N[N.kind == "sentence"]
    nw = s.caption_words.fillna("").map(lambda x: len([t for t in x.split("|") if t]))
    _p("sentences", len(s), "with >=1 / >=2 vocabulary words", int((nw >= 1).sum()), int((nw >= 2).sum()),
       "distinct vocabulary words in captions", len({t for x in s.caption_words.dropna() for t in x.split("|")}))
    c = N[N.kind == "compound"]
    _p("compounds", len(c), "decomposed into known concepts", int(c.concept.str.contains("|", regex=False).sum()),
       "| synonym candidates", len(cand), "| new concepts (not in corpus)", int((~N.concept.isin(allc) & N.kind.isin(["isolated"])).sum()))


# ----------------------------------------------------------------------------- annotate (prompt 1D)
def _load_std(rel):
    from modules.schema import load
    d = load(PREP / rel)
    return np.asarray(d["kp"], np.float32), np.asarray(d["sc"], np.float32), tuple(int(x) for x in d["hw"])


def cmd_annotate(a):
    """phases · handedness · quality (all new clips) → v10 primary-reader embedding of the sign span (bank recipe: canonical hands →
    iso → parts.prep → encoder, L2-normalised; cached) → forms of the 328 vocabulary words (forms.json rule) + synonym decisions."""
    from modules import v12data as V
    from modules.parts import prep, to_iso
    from modules.schema import canonical_hands
    N = pd.read_parquet(NEW)
    ann = []
    for r in N[N.extracted].itertuples():
        kp, sc, hw = _load_std(r.pose_file)
        dom, nh, sym = V.handedness(sc, kp, hw)
        kc, scc, mir = canonical_hands(kp, sc, hw)
        ph = V.phases(kc, scc, hw, "right")                       # canonical = right-dominant (as training / inference)
        lock_ok = not r.dialog
        a0, a1 = (ph["f0"], ph["f3"] + 1) if ph["found"] else (0, len(kp))     # quality inside the signing span (hands may leave the frame at rest)
        q, flags, qd = V.quality(sc[a0:a1], kp[a0:a1], hw, lock_ok=lock_ok)
        ann.append(dict(clip_id=r.clip_id, n_frames=len(kp), handedness=dom, n_hands=nh, symmetric=sym, mirrored=bool(mir),
                        phase_f0=ph["f0"], phase_f1=ph["f1"], phase_f2=ph["f2"], phase_f3=ph["f3"], phases_found=ph["found"],
                        quality=q, quality_flags="|".join(flags) or None, hand_frac=qd["hand_frac"], hand_loss=qd["hand_loss"],
                        sw_frac=qd["sw_frac"], view="turned" if r.dialog else "frontal", width=hw[1], height=hw[0]))
    A = pd.DataFrame(ann)
    N = N.drop(columns=[c for c in A.columns if c != "clip_id" and c in N.columns]).merge(A, on="clip_id", how="left")
    _p("annotated", len(A), "| phases found", round(float(A.phases_found.mean()), 3), "| quality median", round(float(A.quality.median()), 3),
       "| left-dominant", int((A.handedness == "left").sum()), "| two-handed", int((A.n_hands == 2).sum()))
    # ---- v10 primary-reader embeddings of the sign span (isolated / compound / classifier clips), cached by clip
    EMB.parent.mkdir(parents=True, exist_ok=True)
    have = {}
    if EMB.exists():
        z = np.load(EMB); have = {str(c): v for c, v in zip(z["clip"], z["Z"])}
    todo = N[N.extracted & N.kind.isin(["isolated", "compound", "classifier"]) & ~N.clip_id.isin(list(have))]
    if len(todo):
        from modules.runtime import make_backend
        be = make_backend(ROOT / "models" / "model-v10", "torch", "cuda")
        parts, ids = [], []
        for r in todo.itertuples():
            kp, sc, hw = _load_std(r.pose_file)
            kc, scc, _ = canonical_hands(kp, sc, hw)
            f0, f1 = (int(r.phase_f0), int(r.phase_f3) + 1) if r.phases_found else (0, len(kp))
            if f1 - f0 < 8:
                f0, f1 = 0, len(kp)
            parts.append(prep(to_iso(kc, hw)[f0:f1], scc[f0:f1])); ids.append(r.clip_id)
        Z = be.embed(parts)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        have.update(dict(zip(ids, Z)))
        cl = sorted(have)
        np.savez(EMB, clip=np.array(cl), Z=np.stack([have[c] for c in cl]).astype(np.float16))
        _p("embedded", len(ids), "new clips (v10 primary reader, GPU)")
    # ---- forms (forms.json rule, threshold = median same-label cross-signer cosine)
    F = read_json(MAN / "forms.json"); thr = float(F["threshold"])
    b = np.load(ROOT / "models" / "model-v10" / "bank.npz")
    bz = {str(c): z for c, z in zip(b["clip"], b["Z"].astype(np.float32))}
    bsig = {str(c): str(s) for c, s in zip(b["clip"], b["signer"])}
    proto = {}
    for w, wd in F["words"].items():
        for fid, fd in wd["forms"].items():
            per = {}
            for c in fd["clips"]:
                if c in bz:
                    per.setdefault(bsig[c], []).append(bz[c] / (np.linalg.norm(bz[c]) + 1e-8))
            if per:
                v = np.mean([np.mean(x, 0) for x in per.values()], 0); proto[(w, fid)] = v / np.linalg.norm(v)
    form_of, new_forms, cos_best = {}, {}, {}
    for r in N[N.in_vocab & N.kind.eq("isolated") & N.clip_id.isin(list(have))].itertuples():
        z = have[r.clip_id].astype(np.float32)
        cands = [(float(z @ v), fid) for (w, fid), v in proto.items() if w == r.concept]
        best = max(cands) if cands else (-1.0, None)
        cos_best[r.clip_id] = best[0]
        if best[0] >= thr:
            form_of[r.clip_id] = best[1]
        else:                                                      # a different sign, not a noisy sample: a new form per documented label
            key = (r.concept, r.dataset, r.sign_variant or "unlabelled")
            if key not in new_forms:
                k = len(F["words"].get(r.concept, {}).get("forms", {})) + sum(1 for x in new_forms if x[0] == r.concept)
                new_forms[key] = f"f{k}"
            form_of[r.clip_id] = new_forms[key]
    N["form_id"] = N.clip_id.map(form_of)
    N["form_cos"] = N.clip_id.map(cos_best)
    N["new_form"] = N.clip_id.map(lambda c: c in form_of and form_of[c] not in {fid for (w, fid) in proto if w == N.loc[N.clip_id == c, "concept"].iloc[0]})
    nf = N[N.form_id.notna()]
    _p("forms: vocabulary isolated clips", len(nf), "join an existing form", int((~nf.new_form).sum()), "open a new form", int(nf.new_form.sum()),
       "(", len(new_forms), "new forms over", len({k[0] for k in new_forms}), "words )")
    # ---- synonym candidates: accepted only when both sides have prototypes closer than the threshold (prompt 1C rule b)
    S = pd.read_csv(PREP / "params" / "v12_synonym_candidates.csv")
    def cproto(cn):
        vs = [v for (w, f), v in proto.items() if w == cn]
        if not vs:
            zs = [have[c].astype(np.float32) for c in N[(N.concept == cn) & N.clip_id.isin(list(have))].clip_id]
            vs = zs
        if not vs:
            return None
        v = np.mean(vs, 0); return v / np.linalg.norm(v)
    dec = []
    for r in S.itertuples():
        pa, pb = cproto(r.a), cproto(r.b)
        cs = float(pa @ pb) if pa is not None and pb is not None else None
        dec.append(dict(a=r.a, b=r.b, cos=cs, same=bool(r.a == r.b or (cs is not None and cs >= thr)), rule="lexicon" if r.a == r.b else "embedding"))
    Dd = pd.DataFrame(dec); Dd.to_parquet(MAN / "synonym_decisions_v12.parquet", index=False)
    _p("synonym candidates", len(Dd), "decided same", int(Dd.same.sum()), "(lexicon", int(((Dd.rule == "lexicon") & Dd.same).sum()), ") no evidence", int(Dd.cos.isna().sum()))
    _save_new(N)


# ----------------------------------------------------------------------------- QC (prompt 1F)
def _corpus_vocab_bank():
    """v10 production bank restricted to the 328 vocabulary concepts (old clips, every role, BLIND-v11 included only for auditing labels)
    + the new-source vocabulary clips (v10 primary-reader embeddings) → Z, concept, signer, clip, is_new, role"""
    W = sorted(tiers()["words"])
    concepts = read_json(ROOT / "models" / "model-v10" / "concepts.json")
    b = np.load(ROOT / "models" / "model-v10" / "bank.npz")
    cname = np.array([concepts[i] for i in b["c"]])
    m = np.isin(cname, W)
    Z = b["Z"][m].astype(np.float32); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    roles = pd.read_parquet(PREP / "shards" / "clips.parquet", columns=["clip_id", "role"]).drop_duplicates("clip_id").set_index("clip_id").role
    clip = b["clip"][m].astype(str)
    old = pd.DataFrame(dict(clip=clip, concept=cname[m], signer=b["signer"][m].astype(str), is_new=False,
                            role=[roles.get(c.split("@")[0], "train") for c in clip]))
    N = pd.read_parquet(NEW)
    e = np.load(EMB); ez = {str(c): v for c, v in zip(e["clip"], e["Z"].astype(np.float32))}
    n = N[N.in_vocab & N.kind.eq("isolated") & N.clip_id.isin(list(ez))]
    new = pd.DataFrame(dict(clip=n.clip_id.values, concept=n.concept.values, signer=n.signer.values, is_new=True, role="new"))
    Zn = np.stack([ez[c] for c in n.clip_id]) if len(n) else np.zeros((0, Z.shape[1]), np.float32)
    return np.concatenate([Z, Zn]), pd.concat([old, new], ignore_index=True), W


def _concept_scores(Z, meta, W, q_idx, exclude_same_signer=True, bank_mask=None):
    """for queries q_idx: per vocabulary concept the max cosine over bank rows (other signers only) → [len(q), len(W)]"""
    cid = pd.Categorical(meta.concept, categories=W).codes
    bm = np.ones(len(meta), bool) if bank_mask is None else bank_mask
    sig = meta.signer.values
    out = np.full((len(q_idx), len(W)), -1.0, np.float32)
    for s0 in range(0, len(q_idx), 512):
        qi = q_idx[s0:s0 + 512]
        S = Z[qi] @ Z.T
        S[:, ~bm] = -2
        if exclude_same_signer:
            S[sig[qi][:, None] == sig[None, :]] = -2
        S[np.arange(len(qi)), qi] = -2
        for k in range(len(W)):
            cols = np.flatnonzero(cid == k)
            if len(cols):
                out[s0:s0 + len(qi), k] = S[:, cols].max(1)
    return out


def cmd_qc(a):
    N = pd.read_parquet(NEW)
    reason = pd.Series([None] * len(N), index=N.index, dtype=object)
    reason[~N.extracted] = "not_extracted_yet"
    e = N.extracted
    reason[e & (N.n_frames < 8)] = "too_short"
    reason[e & reason.isna() & N.kind.isin(["isolated", "compound", "classifier"]) & (N.hand_frac < 0.25)] = "hands_lt_25pct"
    reason[e & reason.isna() & N.kind.isin(["isolated", "compound"]) & ~N.phases_found.fillna(False).astype(bool)] = "no_signing_motion"
    reason[e & reason.isna() & N.dup_tsl51] = "duplicate_of_tsl51_thdict"
    reason[e & reason.isna() & N.dialog] = "two_person_lock_unverified"      # kept for the caption-bag test only (signer lock not verified)
    N["exclude_reason"] = reason; N["use"] = reason.isna()
    # ---- duplicates across sources: a new clip whose v10 embedding is ≥ 0.98 to an old clip of the same concept = same video
    Z, meta, W = _corpus_vocab_bank()
    newi = np.flatnonzero(meta.is_new.values); oldm = ~meta.is_new.values
    dup = {}
    if len(newi):
        S = Z[newi] @ Z[oldm].T
        same = meta.concept.values[newi][:, None] == meta.concept.values[oldm][None, :]
        S[~same] = -1
        j = S.argmax(1)
        for k, i in enumerate(newi):
            if S[k, j[k]] >= 0.98:
                dup[meta.clip.iloc[i]] = meta.clip.values[oldm][j[k]]
    N["duplicate_of"] = N.clip_id.map(dup)
    N.loc[N.duplicate_of.notna() & N.use, ["use", "exclude_reason"]] = [False, "duplicate_of_old_clip"]
    _p("duplicates (cos >= 0.98, same concept):", len(dup))
    # ---- label audit (confident-learning style): leave-signer-out bank prediction over every vocabulary clip (old + new)
    q = np.arange(len(meta))
    P = _concept_scores(Z, meta, W, q)
    own = pd.Categorical(meta.concept, categories=W).codes
    own_s = P[np.arange(len(q)), own]
    rank = (P > own_s[:, None]).sum(1) + 1
    self_med = pd.Series(own_s).groupby(own).median()
    top = P.argmax(1)
    beats = P[np.arange(len(q)), top] > self_med.reindex(top).fillna(9).values
    flag = (rank > 5) & beats & (own_s > -1.5)          # a word with no other-signer exemplar cannot be audited (rank meaningless)
    meta["rank"], meta["top"], meta["flag"] = rank, np.array(W)[top], flag
    meta.to_parquet(QC / "label_audit.parquet", index=False)
    _p("label audit:", int(flag.sum()), "flagged of", len(meta), f"({flag.mean():.3f}) | old", int(flag[~meta.is_new.values].sum()),
       "| new", int(flag[meta.is_new.values].sum()))
    fl = meta[meta.flag & meta.is_new]
    N["label_status"] = "ok"
    nf = N.clip_id.isin(fl["clip"]) & N.new_form.fillna(False).astype(bool)
    N.loc[nf, "label_status"] = "other_form"             # far from every known form AND read as another word: a different sign (form)
    N.loc[N.clip_id.isin(fl["clip"]) & ~nf, "label_status"] = "suspect"
    # ---- attractor audit: TUNE (val-role) queries against TRAIN-role bank rows, top-1 captures of OTHER words
    tune = np.flatnonzero((meta.role == "val").values)
    trainm = (meta.role == "train").values
    Pt = _concept_scores(Z, meta, W, tune, bank_mask=trainm)
    t1 = np.array(W)[Pt.argmax(1)]
    cap = pd.Series(t1[t1 != meta.concept.values[tune]]).value_counts()
    nclips = meta[trainm].groupby("concept").size()
    nsig = meta[trainm].groupby("concept").signer.nunique()
    att = pd.DataFrame(dict(captures=cap, train_clips=nclips.reindex(cap.index), train_signers=nsig.reindex(cap.index)))
    att["victims"] = [pd.Series(meta.concept.values[tune][(t1 == w) & (t1 != meta.concept.values[tune])]).value_counts().head(3).to_dict() for w in att.index]
    att.to_csv(QC / "attractors.csv", encoding="utf-8-sig")
    _p("attractor audit (TUNE val queries", len(tune), "): top-1 accuracy", round(float((t1 == meta.concept.values[tune]).mean()), 3))
    _p(att.head(12).to_string())
    # ---- exemplars
    N["exemplar"] = N.use & N.kind.eq("isolated") & (N.quality >= 0.6) & N.label_status.eq("ok") & N.phases_found.fillna(False).astype(bool)
    _p("use", int(N.use.sum()), "| exemplars", int(N.exemplar.sum()), "| label_status", N.label_status.value_counts().to_dict())
    _p(N[N.extracted].exclude_reason.value_counts().to_dict())
    _save_new(N)


# ----------------------------------------------------------------------------- splits + BLIND-v12 sealing (prompt 1G)
BLIND_SIGNERS = ["S_rid_00", "S_rid_02", "S_rid_dialog"]      # Royal Institute man (red shirt) + older woman (pink) + every 2-person Q/A clip
TRAIN_NEW = ["S_etsl_00", "S_rid_01"]


def _hash_pct(key, seed=12):
    import hashlib
    return int(hashlib.md5(f"{key}|{seed}".encode("utf-8")).hexdigest(), 16) % 100


def cmd_splits(a):
    """Roles of the new clips. No new signer has >= 40 in-vocabulary isolated clips (prompt 1G (a)); the e-TSL signer has the largest
    in-vocabulary coverage -> TRAIN. BLIND-v12 = two whole Royal Institute people (fewest clips shared with training would be none) + all
    two-person dialogue clips; NEWSIG-dev = a deterministic 15 % word-level hold-out of each TRAIN new signer's in-vocabulary words
    (hash of word, seed 12); 20 % of the TRAIN Royal Institute signer's sentences go to BLIND (caption-bag metric only)."""
    N = pd.read_parquet(NEW)
    role = pd.Series("train", index=N.index, dtype=object)
    role[N.signer.isin(BLIND_SIGNERS) | N.dialog] = "blind_v12"
    iso = N.kind.isin(["isolated", "compound"]) & N.in_vocab & N.signer.isin(TRAIN_NEW)
    dev_word = N.concept.map(lambda w: _hash_pct(w) < 15)
    role[iso & dev_word] = "newsig_dev"
    sent = N.kind.eq("sentence") & N.signer.eq("S_rid_01")
    role[sent & N.clip_id.map(lambda c: _hash_pct(c, 21) < 20)] = "blind_v12"
    role[N.signer.isna() | N.signer.str.endswith("unknown").fillna(False)] = "unassigned"   # not extracted / identity not known yet
    N["role"] = role
    N["sealed"] = N.role.eq("blind_v12")
    _save_new(N)
    _p(N.groupby(["role", "signer", "kind"]).size().to_string())
    # ---- seal (written once; later runs only ADD clips of the sealed people / sentences, never remove)
    f = ART / "eval" / "blind_v12_manifest.json"
    old = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    ids = sorted(N[N.sealed].clip_id)
    ea = json.load(open(ROOT / "artifacts/eval/ea_v12/ood_sel.json", encoding="utf-8"))
    eav = {str(Path(s["video"])).lower() for s in ea}
    ea_overlap = sorted(N[N.sealed & N.video.str.lower().isin(eav)].clip_id)
    m = dict(old or {}, version="BLIND-v12", seed=12, created=(old or {}).get("created", time.strftime("%Y-%m-%d %H:%M")),
             rule=cmd_splits.__doc__.strip(), signers=BLIND_SIGNERS, clips=sorted(set((old or {}).get("clips", [])) | set(ids)),
             parts=dict(a_people=["S_rid_00", "S_rid_02"], a_dialog="S_rid_dialog (every Q/A clip, signer lock unverified)",
                        c_sentences="20 % of S_rid_01 sentences (hash seed 21), caption-bag metric only", b_live="LIVE-L test sessions (added when recorded)"),
             ea_v12_overlap=dict(n=len(ea_overlap), note="these clips were scored ONCE by v10 / v11 in EA-v12 (analysis, no selection)", clips=ea_overlap),
             openings=(old or {}).get("openings", []))
    f.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    _p("sealed", f, "clips", len(m["clips"]), "| EA-v12 overlap", len(ea_overlap))


# ----------------------------------------------------------------------------- EDA (prompt 1A)
def _clip_stats(kp, sc, hw):
    from modules import v12data as V
    from modules.schema import canonical_hands
    hp = V.hand_present(sc)
    kc, scc, mir = canonical_hands(kp, sc, hw)
    ph = V.phases(kc, scc, hw, "right")
    dom, nh, sym = V.handedness(sc, kp, hw)
    mid, sw, ok = V.shoulder_geometry(kp, sc, hw)
    k = V._iso(kp, hw)
    hs = []
    for sl in (V.LH, V.RH):
        m = sc[:, sl.start] > 0
        if m.any() and np.isfinite(sw):
            hs.append(float(np.median(np.linalg.norm(k[m][:, sl.start + 9] - k[m][:, sl.start], axis=1)) / sw))   # wrist → middle-finger MCP
    runs_ = V.runs(~hp.any(1)) / 25.0
    bone = []
    if ok.sum() > 5:
        for a_, b_ in ((5, 7), (7, 9), (6, 8), (8, 10)):
            m = (sc[:, a_] > 0.3) & (sc[:, b_] > 0.3)
            if m.sum() > 5:
                L = np.linalg.norm(k[m, a_] - k[m, b_], axis=1); bone.append(float(L.std() / (L.mean() + 1e-9)))
    T = len(kp)
    return dict(frames=T, any_hand=float(hp.any(1).mean()), both_hands=float(hp.all(1).mean()), hand_loss_share=V.hand_loss_share(sc),
                loss_runs_gt03=int((runs_ > 0.3).sum()), hand_size_sw=float(np.mean(hs)) if hs else None,
                sw_frac=float(sw * hw[0] / hw[1]) if np.isfinite(sw) else None,
                bone_cv=float(np.mean(bone)) if bone else None, span=ph["f3"] - ph["f0"] if ph["found"] else None,
                stroke=ph["f2"] - ph["f1"] if ph["found"] else None, rest_before=ph["f0"] if ph["found"] else None,
                rest_after=T - 1 - ph["f3"] if ph["found"] else None, two_handed=nh == 2, left=dom == "left", mirrored=bool(mir))


def cmd_eda(a):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from decision import _npz_memmap
    from modules.encoder import SLOTS
    C = pd.read_parquet(PREP / "shards" / "clips.parquet"); C = C[C.view == "primary"]
    M = pd.read_parquet(MAN / "clips.parquet", columns=["clip_id", "width", "height", "dataset"]).drop_duplicates("clip_id").set_index("clip_id")
    N = pd.read_parquet(NEW)
    S = _npz_memmap(PREP / "shards" / "pose_store.npz")
    row = {str(c): i for i, c in enumerate(S["clip_id"])}
    rep = dict(created=time.strftime("%Y-%m-%d %H:%M"), sources={})
    stats = []
    for src, g in C.groupby("source"):
        vol = dict(clips=len(g), signers=int(g.signer.nunique()), concepts=int(g.concept.nunique()), hours=float(g.n_frames.sum() / 25 / 3600),
                   by_kind=g.kind.value_counts().to_dict(), by_role=g.role.value_counts().to_dict())
        wh = M.reindex(g.clip_id.str.split("#").str[0])
        vol["resolution"] = (wh.width.astype(str) + "x" + wh.height.astype(str)).value_counts().head(3).to_dict() if wh.width.notna().any() else {}
        iso = g[g.kind.eq("isolated")]
        smp = iso.sample(min(200, len(iso)), random_state=12)
        for r in smp.itertuples():
            i = row.get(r.clip_id)
            if i is None:
                continue
            s0, e0 = int(S["off"][i]), int(S["off"][i + 1])
            kp = np.zeros((e0 - s0, 133, 2), np.float32); sc = np.zeros((e0 - s0, 133), np.float32)
            kp[:, SLOTS] = S["KP"][s0:e0]; sc[:, SLOTS] = S["SC"][s0:e0]
            stats.append(dict(_clip_stats(kp, sc, tuple(int(x) for x in S["hw"][i])), source=src, new=False))
        rep["sources"][src] = dict(volume=vol)
    for ds, g in N[N.extracted].groupby("dataset"):
        named = g[~g.signer.fillna("unknown").str.contains("unknown|dialog")]
        vol = dict(clips_total=int((N.dataset == ds).sum()), extracted=len(g), signers=int(named.signer.nunique()),
                   concepts=int(g.concept.nunique()), hours=float(g.n_frames.sum() / 25 / 3600), by_kind=g.kind.value_counts().to_dict(),
                   by_role=g.role.value_counts().to_dict() if "role" in g else {},
                   resolution=(g.width.fillna(0).astype(int).astype(str) + "x" + g.height.fillna(0).astype(int).astype(str)).value_counts().head(3).to_dict(),
                   two_person_clips=int(g.dialog.sum()), views=g.view.value_counts().to_dict())
        iso = g[g.kind.eq("isolated")]
        for r in iso.sample(min(300, len(iso)), random_state=12).itertuples():
            kp, sc, hw = _load_std(r.pose_file)
            stats.append(dict(_clip_stats(kp, sc, hw), source=ds, new=True))
        rep["sources"][ds] = dict(volume=vol)
    St = pd.DataFrame(stats)
    agg = St.groupby("source").agg(n=("frames", "size"), frames=("frames", "median"), any_hand=("any_hand", "median"), both_hands=("both_hands", "mean"),
                                   hand_loss_share=("hand_loss_share", "mean"), loss_runs_gt03=("loss_runs_gt03", "mean"), hand_size_sw=("hand_size_sw", "median"),
                                   sw_frac=("sw_frac", "median"), bone_cv=("bone_cv", "median"), span=("span", "median"), stroke=("stroke", "median"),
                                   rest_before=("rest_before", "median"), rest_after=("rest_after", "median"), two_handed=("two_handed", "mean"), left=("left", "mean"))
    for src, r in agg.iterrows():
        rep["sources"].setdefault(src, {})["landmarks"] = {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in r.items()}
    rep["LOG_reference"] = dict(note="EA-v12 section 3: saved live requests vs dataset video (hand-loss runs > 0.3 s 32 % vs 14 %; flips 1.43 vs 0.67 /s; "
                                     "hand / shoulder 0.68 vs 0.61)", file="artifacts/eval/ea_v12/live_handloss.json")
    _p(agg.round(3).to_string())
    W = sorted(tiers()["words"])
    have = {src: set(g.concept) & set(W) for src, g in C[C.kind.eq("isolated")].groupby("source")}
    for ds, g in N[N.in_vocab & N.kind.eq("isolated")].groupby("dataset"):
        have[ds] = set(g.concept)
    srcs = sorted(have)
    rep["vocab_overlap"] = {a_: {b_: len(have[a_] & have[b_]) for b_ in srcs} for a_ in srcs}
    rep["vocab_per_source"] = {s_: len(v) for s_, v in have.items()}
    F = read_json(MAN / "forms.json")
    TW = tiers()["words"]
    old = C[C.kind.eq("isolated") & C.role.eq("train") & C.concept.isin(W)]
    new = N[N.kind.eq("isolated") & N.role.eq("train") & N.use & N.concept.isin(W)]
    rows = []
    for w in W:
        o, n_ = old[old.concept == w], new[new.concept == w]
        fo = len(F["words"].get(w, {}).get("forms", {}))
        nf = n_[n_.new_form.fillna(False).astype(bool)].form_id.nunique()
        rows.append(dict(word=w, tier=TW[w]["tier"], signers_before=o.signer.nunique(), signers_after=len(set(o.signer) | set(n_.signer)),
                         clips_before=len(o), clips_after=len(o) + len(n_), forms_before=fo, forms_after=fo + int(nf),
                         new_sources="|".join(sorted(set(n_.dataset))) or None, eff_signers=TW[w].get("eff_signers")))
    D = pd.DataFrame(rows)
    out_csv = ROOT / "result_reporting" / "data_coverage_matrix_v12.csv"
    D.to_csv(out_csv, index=False, encoding="utf-8-sig")
    bins = [-1, 1, 2, 4, 9, 999]; lab = ["<=1", "2", "3-4", "5-9", ">=10"]
    rep["signer_depth"] = dict(before=pd.cut(D.signers_before, bins, labels=lab).value_counts().sort_index().to_dict(),
                               after=pd.cut(D.signers_after, bins, labels=lab).value_counts().sort_index().to_dict(),
                               words_gaining=int((D.signers_after > D.signers_before).sum()),
                               thin_words_gaining=int(((D.signers_before <= 2) & (D.signers_after > D.signers_before)).sum()),
                               note="TRAIN role only; new signers counted once extracted + QC-passed (extraction may still run)")
    _p("signer depth before", rep["signer_depth"]["before"], "after", rep["signer_depth"]["after"], "| words gaining", rep["signer_depth"]["words_gaining"],
       "| thin words gaining", rep["signer_depth"]["thin_words_gaining"])
    coll = D[D.signers_after < 5].sort_values(["tier", "signers_after"])
    coll.to_csv(ROOT / "result_reporting" / "collection_list_v12.csv", index=False, encoding="utf-8-sig")
    rep["collection_list"] = dict(words_below_5_signers=len(coll), file="result_reporting/collection_list_v12.csv")
    fd = ROOT / "result_reporting" / "figures_v12"; fd.mkdir(parents=True, exist_ok=True)
    order = list(agg.index)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a_, col, tt in zip(ax, ("hand_loss_share", "hand_size_sw", "span"), ("hand-loss share (runs > 0.3 s)", "hand size / shoulder width", "sign span (frames, 25 fps)")):
        a_.boxplot([St[St.source == s_][col].dropna() for s_ in order], labels=order, vert=False, showfliers=False); a_.set_title(tt)
    plt.tight_layout(); plt.savefig(fd / "eda_landmarks.png", dpi=110); plt.close()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(lab))
    ax.bar(x - 0.2, [rep["signer_depth"]["before"].get(l, 0) for l in lab], 0.4, label="before (old TRAIN)")
    ax.bar(x + 0.2, [rep["signer_depth"]["after"].get(l, 0) for l in lab], 0.4, label="after (+ new TRAIN signers)")
    ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_xlabel("signers of the word (TRAIN)"); ax.set_ylabel("vocabulary words (328)"); ax.legend()
    plt.tight_layout(); plt.savefig(fd / "eda_signer_depth.png", dpi=110); plt.close()
    write_json(ART / "reports" / "v12_eda.json", rep)
    _p("eda ->", ART / "reports" / "v12_eda.json", "| coverage", out_csv)


# ----------------------------------------------------------------------------- shards (training input of TPH 2)
def cmd_shards(a):
    """data_prep/shards_v12/: the usable new-source clips of TRAIN + NEWSIG-dev roles packed exactly like shards/ (USED slots, float16,
    canonical handedness for isolated / compound clips, span = movement phases f0..f3) + clips.parquet with the shards/ columns and the v12
    annotation columns. Training reads shards/ + shards_v12/ and filters role == train; BLIND-v12 rows are never packed (assert_clean)."""
    from modules.schema import USED, canonical_hands
    from modules.sealed import assert_clean
    N = pd.read_parquet(NEW)
    q = N[N.use & N.extracted & N.role.isin(["train", "newsig_dev"])].copy()
    assert_clean(q.clip_id.tolist(), "shards_v12", signers=q.signer.tolist())
    out = PREP / "shards_v12"; out.mkdir(parents=True, exist_ok=True)
    KP, SC, off, hw, ids, meta = [], [], [0], [], [], []
    for r in q.itertuples():
        kp, sc, hw_ = _load_std(r.pose_file)
        mir = False
        if r.kind in ("isolated", "compound", "classifier"):
            kp, sc, mir = canonical_hands(kp, sc, hw_)
        sp = (int(r.phase_f0), int(r.phase_f3) + 1) if r.phases_found else (0, len(kp))
        KP.append(kp[:, USED].astype(np.float16)); SC.append(sc[:, USED].astype(np.float16)); off.append(off[-1] + len(kp)); hw.append(hw_)
        ids.append(r.clip_id)
        meta.append(dict(clip_id=r.clip_id, source=r.dataset, dataset=r.dataset, subset=r.dataset, kind=r.kind, signer=r.signer, gloss=r.gloss,
                         concept=r.concept, role=r.role, extractor="mp", pair_id=None, recording_variation=r.session if isinstance(r.session, str) else None,
                         view="primary", aux_clip=None, n_frames=len(kp), span_f0=sp[0], span_f1=sp[1], has_span=bool(r.phases_found), mirrored=bool(mir),
                         stroke_f0=int(r.phase_f1) if r.phases_found else None, stroke_f1=int(r.phase_f2) if r.phases_found else None,
                         form_id=r.form_id, quality=float(r.quality) if pd.notna(r.quality) else None, exemplar=bool(r.exemplar),
                         label_status=r.label_status, handedness=r.handedness, caption_words=r.caption_words, in_vocab=bool(r.in_vocab),
                         licence=r.licence, source_url=r.source_url))
    if not ids:
        _p("nothing to pack"); return
    np.savez(out / "pose_store.npz", KP=np.concatenate(KP), SC=np.concatenate(SC), off=np.array(off, np.int64), hw=np.array(hw, np.int32), clip_id=np.array(ids))
    C = pd.DataFrame(meta)
    old = pd.read_parquet(PREP / "shards" / "clips.parquet", columns=["concept", "signer", "kind", "view", "role"])
    iso = pd.concat([old[(old.kind == "isolated") & (old.view == "primary") & (old.role == "train")][["concept", "signer"]],
                     C[(C.kind == "isolated") & (C.role == "train")][["concept", "signer"]]])
    C["concept_signers"] = C.concept.map(iso.groupby("concept").signer.nunique()).fillna(0).astype(int)
    C.to_parquet(out / "clips.parquet", index=False)
    _p("shards_v12:", len(C), "clips", C.groupby(["role", "kind"]).size().to_dict(), "frames", off[-1], "->", out,
       f"({sum(f.stat().st_size for f in out.iterdir()) / 1e6:.0f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest"); p.add_argument("--thr", type=float, default=0.7); p.add_argument("--min_size", type=int, default=8)
    p.add_argument("--thr_rid", type=float, default=0.5); p.add_argument("--rid_merge", default="")
    for c in ("labels", "annotate", "qc", "splits", "eda", "shards"):
        sub.add_parser(c)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
