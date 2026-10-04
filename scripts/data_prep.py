"""ThaiSLM data preparation: raw_data/ (S3 mirror) → data_prep/ (one standard schema for every source).

  python main.py prep pose_items      # every source video → data_prep/params/video_items.csv
  python scripts/extract_mp.py        # MediaPipe Holistic (the production extractor) on every video → data_prep/pose_mp(_raw)/
  python main.py prep ingest          # landmark-only sources (TSL-ONE-S, TSL51 csv) + v6 RTMW poses → data_prep/pose/<dataset>/
  python main.py prep face_emb        # th_sl face embeddings → signer identities (signers)
  python main.py prep signers         # th_sl signer clusters (contact sheet in data_prep/qc) → held-out roles
  python main.py prep lexicon         # gloss → lemma → concept for every source (rules + synonym groups + review)
  python main.py prep manifest        # data_prep/manifest/clips.parquet (+ QC, roles, concepts, summary)
  python main.py prep shards          # v7 training input: MediaPipe views + RTMW views + harvested words → data_prep/shards/
  python main.py prep docs            # data_prep/SCHEMA.md + README (what every file / column means)

Everything reads from utils.RAW (cloud_s3/raw_data) and writes to utils.PREP (cloud_s3/data_prep); nothing here goes to git.
Harvested words (cloud_s3/agent_data/) come from scripts/harvest_words.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from modules.utils import ART, CACHE, DATA, PREP, RAW, get_logger, read_json, safe_id, write_json

log = get_logger("data_prep")
POSE = PREP / "pose"                    # Standard Schema poses, one npz per clip / video
POSE_RAW = PREP / "pose_raw"            # RTMW-X output before harmonisation (real face points kept; for re-harmonising / face models)
LEGACY = CACHE / "legacy_v5"            # v5 intermediates (harmonisation parameters, YouTube title/mined clips)


def ttrs_meta() -> pd.DataFrame:
    """TTRS clips: uid (ttrs:<page id>) ↔ file, gloss, signer fields — from the harvest catalogue / sidecars."""
    mm = ROOT / "metadata" / "metadata_master.csv"
    if mm.exists():
        m = pd.read_csv(mm)
        m = m[m.source == "ttrs_dictionary"].copy()
        m["rel_path"] = m.rel_path.str.replace("^raw_data/", "", regex=True)
        return m
    rows = []                                    # collaborators without metadata/: rebuild from the .json sidecars on S3
    for j in sorted((RAW / "word_level" / "ttrs_dictionary").glob("ttrs_*.json")):
        d = json.loads(j.read_text(encoding="utf-8"))
        rows.append(dict(uid="ttrs:" + d["source_page"].rstrip("/").split("/")[-1], rel_path=f"word_level/ttrs_dictionary/{j.stem}.mp4",
                         label=d.get("gloss_th"), category_th=d.get("category_th"), parts_of_speech_th=d.get("parts_of_speech_th"),
                         synonyms_th=d.get("synonyms_th"), signer_creator=d.get("signer_creator"), signer_gender_th=d.get("signer_gender_th"),
                         reference_book_th=d.get("reference_book_th"), definition_th=d.get("definition_th")))
    return pd.DataFrame(rows)


def youtube_meta() -> pd.DataFrame:
    m = pd.read_csv(ROOT / "metadata" / "metadata_master.csv")
    m = m[m.source == "youtube"].copy()
    m["rel_path"] = m.rel_path.str.replace("^raw_data/", "", regex=True)
    return m


def thsl_meta() -> pd.DataFrame:
    m = pd.read_csv(RAW / "word_level" / "th_sl_dictionary" / "metadata" / "th_sl_metadata.csv")
    m["rel_path"] = "word_level/th_sl_dictionary/videos/th_sl/" + m.clip_file
    return m


# ----------------------------------------------------------------------------- pose job input
def cmd_pose_items(a):
    """Every source video (clip id → raw video → output name) for the MediaPipe extractor (scripts/extract_mp.py)."""
    rows = []
    for r in thsl_meta().itertuples():
        rows.append(("thsl:" + r.clip_id, r.rel_path, f"th_sl/{r.clip_id}.npz", "th_sl"))
    for r in ttrs_meta().itertuples():
        rows.append((r.uid, r.rel_path, f"ttrs/{safe_id(r.uid)}.npz", "ttrs"))
    for meta in ("user_sign_metadata.csv", "sentence_metadata.csv"):
        m = pd.read_csv(RAW / "TSL51" / "metadata" / meta)
        sub = "user_sign" if "sign" in meta else "user_sentence"
        for r in m.itertuples():
            rows.append((f"tsl51_{sub}:{r.video_id}", f"TSL51/{r.video_path}", f"tsl51_{sub}/{safe_id(r.video_id)}.npz", f"tsl51_{sub}"))
    for r in youtube_meta().itertuples():
        if r.level in ("continuous", "sentence"):
            ds = r.rel_path.split("/")[1]
            rows.append((r.uid, r.rel_path, f"{ds}/{safe_id(r.uid)}.npz", ds))
    it = pd.DataFrame(rows, columns=["clip_id", "rel_path", "out_rel", "dataset"])
    miss = [p for p in it.rel_path if not (RAW / p).exists()]
    assert not miss, f"{len(miss)} videos missing locally, e.g. {miss[:3]}"
    it["bytes"] = [(RAW / p).stat().st_size for p in it.rel_path]
    out = PREP / "params" / "video_items.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    it.to_csv(out, index=False)
    log.info("video items: %s (%.1f GB) → %s  (next: python scripts/extract_mp.py)", it.groupby("dataset").size().to_dict(), it.bytes.sum() / 1e9, out)


# ----------------------------------------------------------------------------- ingest: every source → Standard Schema
def pose_path(dataset, clip_id):
    return POSE / dataset / f"{safe_id(clip_id)}.npz"


def _ingest_rtmw(args):
    from modules.schema import from_rtmw, save
    cid, src, dataset, force = args
    out = pose_path(dataset, cid)
    if out.exists() and not force:
        return cid, "skip"
    if not Path(src).exists():
        return cid, "missing"
    try:
        d = np.load(src)
        n = len(d["kp"])
    except Exception:  # noqa: BLE001  (truncated / still uploading)
        return cid, "unreadable"
    if n < 2:
        return cid, "empty"
    save(out, from_rtmw(d["kp"].astype(np.float32), d["sc"].astype(np.float32), tuple(int(x) for x in d["hw"]), fps=float(d["fps"]), src=dataset))
    return cid, len(d["kp"])


def _ingest_mp75(args):
    from modules.schema import from_mp75, save
    cid, src, aspect, force = args
    out = pose_path("tslone", cid)
    if out.exists() and not force:
        return cid, "skip"
    save(out, from_mp75(np.load(src), hw=(1000, int(round(1000 * aspect))), fps=29.97, src="tslone"))
    return cid, "ok"


def _ingest_mp54(args):
    import io
    from modules.schema import from_mp54, save
    cid, dataset, name, blob, hw, fps, force = args
    out = pose_path(dataset, cid)
    if out.exists() and not force:
        return cid, "skip"
    df = pd.read_csv(io.BytesIO(blob)) if blob is not None else pd.read_csv(name)
    if len(df) < 2:
        return cid, "empty"
    save(out, from_mp54(df, hw, fps, src=dataset))
    return cid, "ok"


def _pool(fn, items, workers):
    from concurrent.futures import ProcessPoolExecutor
    res, t0 = [], time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(fn, items, chunksize=8), 1):
            res.append(r)
            if i % 1000 == 0:
                log.info("  %d/%d (%.0fs)", i, len(items), time.time() - t0)
    bad = [r for r in res if r[1] in ("missing", "empty", "unreadable")]
    return len(res), bad


def cmd_ingest(a):
    import zipfile
    todo = set(a.src.split(",")) if a.src else {"rtmw", "tslone", "tsl51", "youtube_legacy"}
    report = {}
    if "rtmw" in todo:       # th_sl, TTRS, TSL51 user videos, YouTube continuous/sentence — all from the cloud RTMW job
        it = pd.read_csv(PREP / "params" / "video_items.csv")
        items = [(r.clip_id, str(POSE_RAW / r.out_rel), r.dataset + ("_rtmw" if r.dataset.startswith("tsl51") else ""), a.force) for r in it.itertuples()]
        report["rtmw"] = _pool(_ingest_rtmw, items, a.workers)
    if "tslone" in todo:     # TSL-ONE-S MediaPipe 75, per-clip frame aspect estimated from body geometry (v5 finding, parameters reused)
        asp = read_json(LEGACY / "tslone_aspect.json")
        (PREP / "params").mkdir(parents=True, exist_ok=True)
        write_json(PREP / "params" / "tslone_aspect.json", asp)
        files = sorted((RAW / "TSL-ONE-S" / "TSL-ONE-Pose").rglob("*.npy"))
        items = [(p.stem, str(p), asp["per_clip_aspect"][p.stem], a.force) for p in files]
        report["tslone"] = _pool(_ingest_mp75, items, a.workers)
    if "tsl51" in todo:      # TSL51 MediaPipe csv: expert originals (augmented copies skipped) + researcher signs / sentences
        T51 = RAW / "TSL51"
        ex = pd.read_csv(T51 / "metadata" / "expert_metadata.csv")
        orig = ex[~ex.is_augmented.astype(bool)]
        zips = {z: zipfile.ZipFile(T51 / "landmarks" / f"{z}.zip") for z in ("expert_primary_02", "expert_primary_03", "expert_scraped")}
        index = {}
        for z, zf in zips.items():
            for n in zf.namelist():
                index.setdefault(Path(n).name, []).append((z, n))
        items = []
        for r in orig.itertuples():
            hits = index.get(Path(r.landmark_path).name, [])
            if hits:
                z, n = hits[0]
                hw = (int(r.height_extracted), int(r.width_extracted)) if not pd.isna(r.height_extracted) else (720, 1280)
                items.append((f"t51x:{r.video_id}", "tsl51_expert", n, zips[z].read(n), hw, float(r.fps_extracted), a.force))
        for meta, ds, tag in (("user_sign_metadata.csv", "tsl51_user_sign", "t51u"), ("sentence_metadata.csv", "tsl51_user_sentence", "t51s")):
            m = pd.read_csv(T51 / "metadata" / meta)
            items += [(f"{tag}:{r.video_id}", ds, str(T51 / r.landmark_path), None, (int(r.height_extracted), int(r.width_extracted)),
                       float(r.fps_extracted), a.force) for r in m.itertuples()]
        report["tsl51"] = _pool(_ingest_mp54, items, a.workers)
    if "youtube_legacy" in todo:   # v4 YouTube word instances (title clips / speech-mined): already Standard Schema in v5, copied as-is
        import shutil
        for sub, ds in (("yt_title", "youtube_title"), ("yt_mined", "youtube_mined")):
            (POSE / ds).mkdir(parents=True, exist_ok=True)
            for f in (LEGACY / "pose" / sub).glob("*.npz"):
                if not (POSE / ds / f.name).exists():
                    shutil.copyfile(f, POSE / ds / f.name)
        report["youtube_legacy"] = {ds: len(list((POSE / ds).glob("*.npz"))) for ds in ("youtube_title", "youtube_mined")}
    for k, v in report.items():
        log.info("ingest %s: %s", k, v if isinstance(v, dict) else f"{v[0]} items, problems {len(v[1])}: {v[1][:5]}")


# ----------------------------------------------------------------------------- lexicon (gloss → lemma → concept)
def cmd_lexicon(a):
    """v5's reviewed table (TTRS / TSL51 / TSL-ONE-S, incl. its LLM + manual decisions) is kept; th_sl entries are cleaned by rule
    (modules.lexicon.thsl_lemma) and only pairs that involve a NEW lemma go through embedding neighbours + LLM verification.
    Dictionary slash entries ('แต่งงาน/สมรส' = one sign) are synonyms by construction."""
    from modules.lexicon import DATA_TEST_REFS, NULL, TEST_SURFACE, apply_manual_review, build_concepts, thsl_lemma
    man = PREP / "manifest"; man.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(LEGACY / "lexicon.parquet").drop(columns=["concept"], errors="ignore")
    old = apply_manual_review(old)
    rows, slash = [], []
    for g in sorted(set(thsl_meta().word)):
        lem, var, sense, aliases = thsl_lemma(g)
        rows.append(dict(dataset="th_sl", gloss=g, lemma=lem, sign_variant=var, sense=sense, aliases="|".join(aliases) or None,
                         lemma_source="rule" if (var or sense or aliases or lem != g) else "gloss", confidence=1.0))
        slash += [(lem, al, "th_sl_same_entry") for al in aliases]
    th = pd.DataFrame(rows)
    lex = pd.concat([old, th], ignore_index=True)
    for al in {al for _, al, _ in slash}:            # alias spellings must exist as lemmas to be joined
        if al not in set(lex.lemma):
            lex = pd.concat([lex, pd.DataFrame([dict(dataset="th_sl", gloss=al, lemma=al, lemma_source="th_sl_alias", confidence=1.0)])], ignore_index=True)
    prior = pd.read_parquet(LEGACY / "synonym_decisions.parquet")
    if (man / "synonym_decisions.parquet").exists():          # re-runs keep every decision already made (no repeated LLM calls)
        prior = pd.concat([prior, pd.read_parquet(man / "synonym_decisions.parquet")]).drop_duplicates(["a", "b"], keep="last")
    new = sorted(set(lex.lemma) - set(old.lemma) - {NULL})
    test_words = sorted({w for ws in DATA_TEST_REFS.values() for w in ws} | set(TEST_SURFACE))
    concept, decisions, groups = build_concepts(lex, extra_words=test_words, use_llm=not a.no_llm, prior=prior, new_lemmas=new, same_pairs=slash)
    lex["concept"] = lex.lemma.map(concept)
    lex.to_parquet(man / "lexicon.parquet", index=False)
    decisions.to_parquet(man / "synonym_decisions.parquet", index=False)
    write_json(man / "concepts.json", dict(concept=concept, groups=groups))
    n_same = int((decisions.label == "same").sum())
    log.info("lexicon: %d rows (%d new th_sl lemmas); decisions %d (same %d); multi-member concepts %d; concepts %d",
             len(lex), len(new), len(decisions), n_same, len(groups), len(set(concept.values())))


# ----------------------------------------------------------------------------- signer identity (th_sl has no signer column)
def face_embeddings(videos, raws, out_npz, fracs=(0.25, 0.5, 0.75)):
    """Mean DINOv2-base CLS of face crops (box from the MediaPipe face points, modules/identity.py) at 3 frames per video."""
    from modules.identity import embed_faces, face_crops
    from modules.mediapipe_pose import load_raw
    crops, owner = [], []
    t0 = time.time()
    for vi, (v, r) in enumerate(zip(videos, raws)):
        if Path(r).exists():
            cs = face_crops(v, load_raw(r), fracs)
            crops += cs; owner += [vi] * len(cs)
        if (vi + 1) % 1000 == 0:
            log.info("face crops %d/%d videos (%.0fs)", vi + 1, len(videos), time.time() - t0)
    E = embed_faces(crops) if crops else np.zeros((0, 768), np.float32)
    owner = np.array(owner)
    Z = np.zeros((len(videos), 768), np.float32); found = np.zeros(len(videos), np.int32)
    for vi in range(len(videos)):
        m = owner == vi
        if m.any():
            z = E[m].mean(0); Z[vi] = z / (np.linalg.norm(z) + 1e-9); found[vi] = m.sum()
    np.savez_compressed(out_npz, Z=Z, found=found, crops=np.stack(crops)[np.unique(owner, return_index=True)[1]] if crops else np.zeros(0))
    log.info("face embeddings: %d/%d videos with a face (%.0fs)", (found > 0).sum(), len(videos), time.time() - t0)


def cmd_face_emb(a):
    m = thsl_meta()
    face_embeddings([RAW / p for p in m.rel_path], [PREP / "pose_mp_raw" / "th_sl" / f"{c}.npz" for c in m.clip_id], CACHE / "thsl_face_emb.npz")


# ----------------------------------------------------------------------------- manifest + quality control
QC = dict(min_frames=8, max_frames_isolated=25 * 20, min_body_frac=0.6, min_hand_frac=0.25)
LICENCE = dict(ttrs="TTRS dictionary - public research use, rights with TTRS",
               th_sl="th-sl.com (NADT) - rights reserved (ai-train=no); research use approved by the project owner",
               tslone="TSL-ONE-S - dataset licence (see raw_data/TSL-ONE-S)",
               tsl51="TSL51 (Hugging Face Namonpas/thai-sign-language-tsl51) - dataset licence",
               youtube="YouTube - publisher's rights, research use only")


def licence_of(dataset):
    for k, v in LICENCE.items():
        if dataset.startswith(k):
            return v
    return ""


def _clip_stats(args):
    """Standard-schema npz -> frames, sign span, hand / body presence; raw RTMW npz -> max persons, signer-lock rate."""
    from modules.schema import activity, load, sign_span
    pose_file, raw_file, kind = args
    p = PREP / pose_file
    if not p.exists():
        return dict(pose_file=pose_file, missing=True)
    d = load(p)
    sc = d["sc"]
    hands = (sc[:, 91:112].mean(1) > 0.3) | (sc[:, 112:133].mean(1) > 0.3)
    body = (sc[:, 5] > 0.5) & (sc[:, 6] > 0.5)
    sp = sign_span(d["kp"], d["sc"], d["hw"]) if kind == "isolated" else None
    act = activity(d["kp"], d["sc"], d["hw"])
    hh = act["height"][act["height"] > -2.5]
    out = dict(pose_file=pose_file, hand_height_max=float(hh.max()) if len(hh) else -3.0, wrist_speed_p95=float(np.percentile(act["speed"], 95)), n_frames=len(sc), width=int(d["hw"][1]), height=int(d["hw"][0]), hand_frac=float(hands.mean()),
               body_frac=float(body.mean()), span_f0=int(sp[0]) if sp else 0, span_f1=int(sp[1]) if sp else len(sc), has_span=sp is not None,
               missing=False)
    if isinstance(raw_file, str) and (PREP / raw_file).exists():
        r = np.load(PREP / raw_file)
        if "n_persons" in r:
            out.update(n_persons_max=int(r["n_persons"].max()), multi_person_frac=float((r["n_persons"] > 1).mean()),
                       lock_frac=float(r["body_found"].mean()))
    return out


def qc_reasons(r):
    """Why a clip must not be used for training / banks (empty = keep). Only hard evidence of noise, never 'hard example'."""
    if r.get("missing") is True:
        return ["pose_missing"]
    why = []
    if r["n_frames"] < QC["min_frames"]:
        why.append("too_short")
    if r["kind"] == "isolated" and r["n_frames"] > QC["max_frames_isolated"]:
        why.append("too_long_for_a_word")
    if r["kind"] in ("isolated", "null") and r["body_frac"] < QC["min_body_frac"]:
        why.append("no_signer_in_frame")          # e.g. close-up of a body part, title card
    if r["kind"] == "isolated" and r["hand_frac"] < QC["min_hand_frac"]:
        why.append("hands_not_visible")
    if r["kind"] == "isolated" and r["hand_height_max"] < -1.0 and r["wrist_speed_p95"] < 0.8:
        why.append("no_signing_motion")          # hands never above the waist AND barely moving (a tight clip that is all signing is fine)
    if r["kind"] == "isolated" and not isinstance(r.get("concept"), str):
        why.append("no_label")
    return why


V5_POSE = {  # v5 clip-id prefix -> (dataset, extractor, raw source, how the pose file is named)
    "tslone": ("tslone", "mp75", "raw_data/TSL-ONE-S/TSL-ONE-Pose"),
    "t51x": ("tsl51_expert", "mp54", "raw_data/TSL51/landmarks/expert_*.zip"),
    "t51u": ("tsl51_user_sign", "mp54", "raw_data/TSL51/landmarks/user_sign"),
    "t51s": ("tsl51_user_sentence", "mp54", "raw_data/TSL51/landmarks/user_sentence"),
    "t51ur": ("tsl51_user_sign_rtmw", "rtmw", "raw_data/TSL51/videos/user_sign"),
    "t51sr": ("tsl51_user_sentence_rtmw", "rtmw", "raw_data/TSL51/videos/user_sentence"),
    "yttitle": ("youtube_title", "rtmw", "raw_data/word_level/youtube_word"),
    "ytmined": ("youtube_mined", "rtmw", "raw_data/word_level/youtube_word"),
}


def cmd_manifest(a):
    """One row per clip / video, every source, one schema. Roles: v5 clips keep their v5 role (comparable test sets); th_sl
    roles come from held-out signer clusters; continuous YouTube is 'unlabeled' (tagger pseudo-labels / hubness reference only)."""
    from modules.lexicon import NULL, thsl_lemma, tsl51_lemma
    man = PREP / "manifest"
    (PREP / "qc").mkdir(parents=True, exist_ok=True)
    lex = pd.read_parquet(man / "lexicon.parquet")
    cmap = {(r.dataset, r.gloss): r for r in lex.itertuples()}
    concept_of = read_json(man / "concepts.json")["concept"]
    v5 = pd.read_parquet(LEGACY / "clips_v5.parquet").set_index("clip_id")
    rows = []

    def add(**kw):
        kw.setdefault("licence", licence_of(kw["dataset"]))
        rows.append(kw)

    # TTRS: RTMW 25 fps from the cloud job; v4 signer ids + signer-unit splits carried by the v5 table
    for r in ttrs_meta().itertuples():
        o = v5.loc[r.uid] if r.uid in v5.index else None
        L = cmap.get(("ttrs", r.label))
        add(clip_id=r.uid, dataset="ttrs", subset="ttrs", kind="isolated", source_file=f"raw_data/{r.rel_path}", extractor="rtmw",
            pose_file=f"pose/ttrs/{safe_id(r.uid)}.npz", pose_raw_file=f"pose_raw/ttrs/{safe_id(r.uid)}.npz",
            signer=o.signer if o is not None else "S_ttrs_unknown", gloss=r.label, lemma=L.lemma if L is not None else r.label,
            sign_variant=(L.sign_variant if L is not None else None) or (o.sign_variant if o is not None else None),
            category=r.category_th, role=o.role if o is not None else "train")
    # th_sl (new): signer = face cluster (scripts/data_prep.py signers), role from held-out signer clusters
    sgp = PREP / "params" / "thsl_signers.csv"
    sg = pd.read_csv(sgp).set_index("clip_id") if sgp.exists() else None
    for r in thsl_meta().itertuples():
        lem, var, sense, _ = thsl_lemma(r.word)
        cid = "thsl:" + r.clip_id
        known = sg is not None and cid in sg.index
        add(clip_id=cid, dataset="th_sl", subset="th_sl", kind="isolated", source_file=f"raw_data/{r.rel_path}", extractor="rtmw",
            pose_file=f"pose/th_sl/{safe_id(cid)}.npz", pose_raw_file=f"pose_raw/th_sl/{r.clip_id}.npz",
            signer=sg.loc[cid, "signer"] if known else "S_thsl_unknown", gloss=r.word, lemma=lem, sign_variant=var, sense=sense,
            category=r.tags, role=sg.loc[cid, "role"] if known else "train", source_url=r.video_url)
    # TSL-ONE-S / TSL51 / YouTube word instances: v5 rows (roles, signers, labels) + the new pose files
    th_files = set(thsl_meta().video_url.str.split("/").str[-1])
    ex = pd.read_csv(RAW / "TSL51" / "metadata" / "expert_metadata.csv").set_index("video_id")
    for cid, o in v5.iterrows():
        pre = cid.split(":")[0]
        if pre not in V5_POSE:
            continue
        key = cid.split(":", 1)[1]
        ds, ext, src = V5_POSE[pre]
        if pre in ("yttitle", "ytmined"):
            pf = f"pose/{ds}/{Path(o.pose_path).name}"
        elif pre in ("t51ur", "t51sr"):
            sub = "tsl51_user_sign" if pre == "t51ur" else "tsl51_user_sentence"
            pf = f"pose/{ds}/{safe_id(sub + ':' + key)}.npz"
        else:
            pf = f"pose/{ds}/{safe_id(key if pre == 'tslone' else cid)}.npz"
        rf = f"pose_raw/{'tsl51_user_sign' if pre == 't51ur' else 'tsl51_user_sentence'}/{safe_id(key)}.npz" if pre in ("t51ur", "t51sr") else None
        kw = dict(clip_id=cid, dataset=ds, subset=o.subset, kind=o.kind, source_file=src, extractor=ext, pose_file=pf, pose_raw_file=rf,
                  signer=o.signer, gloss=o.gloss, lemma=o.lemma, sign_variant=o.sign_variant, category=o.category, role=o.role,
                  pair_id=o.pair_id, recording_variation=o.recording_variation, official_split=o.official_split)
        if o.kind == "sentence":
            kw["concept"] = "|".join(concept_of.get(tsl51_lemma(g)[0], tsl51_lemma(g)[0]) for g in str(o.gloss).split("__"))
        if pre == "t51x" and key in ex.index and ex.loc[key, "source"] == "thsl" \
                and str(ex.loc[key, "file_name"]).replace("_original", "") in th_files:
            kw["duplicate_of"] = "th_sl"
        add(**kw)
    # continuous / sentence-level YouTube (no per-word labels)
    for r in youtube_meta().itertuples():
        if r.level in ("continuous", "sentence"):
            ds = r.rel_path.split("/")[1]
            add(clip_id=r.uid, dataset=ds, subset=ds, kind="continuous", source_file=f"raw_data/{r.rel_path}", extractor="rtmw",
                pose_file=f"pose/{ds}/{safe_id(r.uid)}.npz", pose_raw_file=f"pose_raw/{ds}/{safe_id(r.uid)}.npz",
                signer=f"S_yt_{r.channel_id}", gloss=r.label, role="unlabeled")
    m = pd.DataFrame(rows)
    if "duplicate_of" not in m:
        m["duplicate_of"] = None
    if "concept" not in m:
        m["concept"] = None
    iso = m.kind.isin(["isolated", "null"])
    need = iso & m.concept.isna()
    m.loc[need, "concept"] = m.loc[need, "lemma"].map(lambda x: concept_of.get(x, x) if isinstance(x, str) else None)
    m.loc[m.kind == "null", "concept"] = NULL
    stats = _pool_rows(_clip_stats, [(r.pose_file, r.pose_raw_file, r.kind) for r in m.itertuples()], a.workers)
    m = m.merge(pd.DataFrame(stats), on="pose_file", how="left")
    reasons = [qc_reasons(r) for r in m.to_dict("records")]
    m["exclude_reason"] = ["|".join(x) if x else None for x in reasons]
    m.loc[m.duplicate_of.notna(), "exclude_reason"] = "duplicate_of_th_sl"
    m["use"] = m.exclude_reason.isna()
    sc = m[m.use & m.kind.eq("isolated")].groupby("concept").signer.nunique()
    m["concept_signers"] = m.concept.map(sc).fillna(0).astype(int)
    m.to_parquet(man / "clips.parquet", index=False)
    summ = m.groupby(["dataset", "role", "use"]).agg(clips=("clip_id", "size"), signers=("signer", "nunique"),
                                                     concepts=("concept", "nunique"), frames=("n_frames", "sum")).reset_index()
    summ.to_csv(man / "summary.csv", index=False, encoding="utf-8-sig")
    exs = m[~m.use].groupby(["dataset", "exclude_reason"]).size().reset_index(name="clips")
    exs.to_csv(PREP / "qc" / "excluded_summary.csv", index=False, encoding="utf-8-sig")
    m[~m.use][["clip_id", "dataset", "gloss", "exclude_reason", "source_file"]].to_csv(PREP / "qc" / "excluded_clips.csv", index=False,
                                                                                      encoding="utf-8-sig")
    log.info("manifest: %d rows\n%s\nexcluded:\n%s", len(m), summ.to_string(), exs.to_string())


def _pool_rows(fn, items, workers):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, items, chunksize=16))


def cmd_signers(a):
    """th_sl signer ids: average-linkage clustering of face embeddings (cosine), small clusters folded into their nearest large
    cluster, then a contact sheet per cluster (data_prep/qc/thsl_signers.jpg) for visual verification. Held-out roles: whole
    signer clusters (val ≈ 8 %, test ≈ 12 % of clips), so every th_sl test clip is from a person the model never trained on."""
    import cv2
    from sklearn.cluster import AgglomerativeClustering
    m = thsl_meta()
    z = np.load(CACHE / "thsl_face_emb.npz")
    Z, found = z["Z"], z["found"] > 0
    ids = np.flatnonzero(found)
    Z = Z.copy(); Z[ids] -= Z[ids].mean(0)                 # centre: remove the component every face shares, keep identity
    Z[ids] /= np.linalg.norm(Z[ids], axis=1, keepdims=True)
    cl = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=a.thr).fit(Z[ids])
    lab = np.full(len(m), -1); lab[ids] = cl.labels_
    sizes = pd.Series(lab[ids]).value_counts()
    big = [c for c, n in sizes.items() if n >= a.min_size]
    cent = {c: Z[lab == c].mean(0) for c in big}
    cent = {c: v / np.linalg.norm(v) for c, v in cent.items()}
    C = np.stack([cent[c] for c in big])
    for i in ids:                                          # fold small clusters into the nearest large one
        if lab[i] not in big:
            lab[i] = big[int(np.argmax(C @ Z[i]))]
    order = {c: k for k, c in enumerate(sorted(big, key=lambda c: -(lab == c).sum()))}
    signer = np.array([f"S_thsl_{order[l]:02d}" if l >= 0 else "S_thsl_unknown" for l in lab])
    # held-out clusters: pick whole signers whose clip share fills the target fractions (deterministic, mid-sized signers)
    cnt = pd.Series(signer[signer != "S_thsl_unknown"]).value_counts()
    rng = np.random.RandomState(13)
    pool = [s_ for s_ in cnt.index if cnt[s_] < 0.25 * cnt.sum()]
    rng.shuffle(pool)
    role, tot, test, val = {}, cnt.sum(), 0, 0
    if a.test_clusters or a.val_clusters:      # chosen after looking at the contact sheet: clusters that are visibly ONE person
        role.update({f"S_thsl_{int(c):02d}": "test" for c in a.test_clusters.split(",") if c})
        role.update({f"S_thsl_{int(c):02d}": "val" for c in a.val_clusters.split(",") if c})
        pool = []
    for s_ in pool:
        if test < 0.12 * tot:
            role[s_] = "test"; test += cnt[s_]
        elif val < 0.08 * tot:
            role[s_] = "val"; val += cnt[s_]
    out = pd.DataFrame(dict(clip_id="thsl:" + m.clip_id, signer=signer, role=[role.get(x, "train") for x in signer], face_found=found))
    (PREP / "params").mkdir(parents=True, exist_ok=True); (PREP / "qc").mkdir(parents=True, exist_ok=True)
    out.to_csv(PREP / "params" / "thsl_signers.csv", index=False)
    # contact sheet: 8 random face crops per signer cluster (one row each)
    crops = z["crops"]
    crop_of = {v: k for k, v in enumerate(ids)}
    rows = []
    for sname in sorted(set(signer) - {"S_thsl_unknown"}):
        mem = [i for i in np.flatnonzero(signer == sname) if i in crop_of]
        pick = rng.choice(mem, min(8, len(mem)), replace=False)
        tiles = [cv2.cvtColor(crops[crop_of[i]], cv2.COLOR_RGB2BGR) for i in pick] + [np.zeros((140, 112, 3), np.uint8)] * (8 - len(pick))
        label = np.zeros((140, 150, 3), np.uint8)
        cv2.putText(label, sname[-2:], (5, 60), 0, 1.2, (255, 255, 255), 2)
        cv2.putText(label, f"n={int((signer == sname).sum())}", (5, 95), 0, 0.6, (255, 255, 255), 1)
        cv2.putText(label, role.get(sname, "train"), (5, 125), 0, 0.6, (0, 255, 255), 1)
        rows.append(np.concatenate([label] + tiles, 1))
    cv2.imwrite(str(PREP / "qc" / (a.sheet or "thsl_signers.jpg")), np.concatenate(rows, 0))
    log.info("th_sl signers: %d clusters (thr %.2f, min %d); no face %d; roles %s", len(big), a.thr, a.min_size, int((~found).sum()),
             out.groupby("role").signer.agg(["nunique", "size"]).to_dict())


FAMILY = {"ttrs": "ttrs", "th_sl": "th_sl", "tslone": "tslone", "tsl51_expert": "tsl51", "tsl51_user_sign": "tsl51", "tsl51_user_sentence": "tsl51",
          "youtube_bigsign": "youtube_cont", "youtube_continuous": "youtube_cont", "youtube_sentence": "youtube_cont", "agent": "agent"}
# v6 dataset → (MediaPipe output folder, how the file stem is derived from the clip id). Video sources are re-extracted with the
# production extractor; the v6 RTMW Standard-Schema file of the same video becomes the auxiliary view.
MP_SOURCE = {"ttrs": "ttrs", "th_sl": "th_sl", "tsl51_user_sign_rtmw": "tsl51_user_sign", "tsl51_user_sentence_rtmw": "tsl51_user_sentence",
             "youtube_bigsign": "youtube_bigsign", "youtube_continuous": "youtube_continuous", "youtube_sentence": "youtube_sentence"}
RENAME = {"tsl51_user_sign_rtmw": ("tsl51_user_sign", "t51ur:", "t51u:"), "tsl51_user_sentence_rtmw": ("tsl51_user_sentence", "t51sr:", "t51s:")}
POSE_MP = PREP / "pose_mp"


def daily_concepts():
    daily = read_json(ROOT / "vocab" / "daily_conversation.json")
    cmap = read_json(PREP / "manifest" / "concepts.json")["concept"]
    return sorted({cmap.get(w, w) for k, v in daily.items() if not k.startswith("_") for w in v})


def _pack_one(args):
    """primary (MediaPipe) + optional aux (RTMW) file → canonical handedness (same decision for both views), sign span, USED slots."""
    from modules.schema import USED, canonical_hands, load, mirror, sign_span
    prim, aux, kind, span = args
    if not Path(prim).exists():
        return None
    d = load(prim)
    kp, sc, hw = d["kp"], d["sc"], d["hw"]
    if len(kp) < 4:
        return None
    mirrored = False
    if kind == "isolated":
        kp, sc, mirrored = canonical_hands(kp, sc, hw)
        sp = tuple(int(x) for x in span) if span is not None else sign_span(kp, sc, hw)
    else:
        sp = None
    out = dict(kp=kp[:, USED].astype(np.float16), sc=sc[:, USED].astype(np.float16), hw=hw, n=len(kp), mirrored=mirrored,
               span=sp if sp else (0, len(kp)), has_span=sp is not None)
    if aux and Path(aux).exists():
        a = load(aux)
        ka, sa = (mirror(a["kp"], a["sc"]) if mirrored else (a["kp"], a["sc"]))
        out["aux"] = dict(kp=ka[:, USED].astype(np.float16), sc=sa[:, USED].astype(np.float16), hw=a["hw"], n=len(ka))
    return out


def cmd_shards(a):
    """v7 training input (MediaPipe everywhere): pose_store.npz (primary MediaPipe views + RTMW views of the same videos, USED slots,
    float16, one offset table) + clips.parquet + concepts.json + transitions.parquet + daily_concepts.json. data_test is never packed."""
    from multiprocessing import Pool
    out = Path(a.out) if a.out else PREP / "shards"; out.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(PREP / "manifest" / "clips.parquet")
    m = m[m.use & m.role.ne("data_test")].copy()
    it = pd.read_csv(PREP / "params" / "video_items.csv")
    stem_of = {r.clip_id: Path(r.out_rel).stem for r in it.itertuples()}
    rows = []
    for r in m.itertuples():
        ds = r.dataset
        if ds in ("tsl51_user_sign", "tsl51_user_sentence", "youtube_title", "youtube_mined"):
            continue                              # csv duplicates of re-extracted videos · v4 RTMW-only YouTube segments (replaced by agent data)
        if ds in MP_SOURCE:
            if ds in RENAME:
                new_ds, old, new = RENAME[ds]
                name = r.clip_id[len(old):]
                prim, cid, src = POSE_MP / MP_SOURCE[ds] / f"{name}.npz", new + name, new_ds
            else:
                stem = stem_of.get(r.clip_id) or Path(r.source_file).stem
                prim, cid, src = POSE_MP / MP_SOURCE[ds] / f"{stem}.npz", r.clip_id, ds
            aux = PREP / r.pose_file
            ext = "mp"
        else:
            prim, cid, src, aux, ext = PREP / r.pose_file, r.clip_id, ds, None, r.extractor
        rows.append(dict(clip_id=cid, source=src, dataset=FAMILY.get(src, src), subset=src if src.startswith("tsl51_user") else r.subset,
                         kind=r.kind, signer=r.signer, gloss=r.gloss, concept=r.concept, role=r.role, extractor=ext, pair_id=r.pair_id,
                         recording_variation=r.recording_variation, prim=str(prim), aux=str(aux) if aux is not None else None, span=None))
    ag = DATA / "agent_data" / "youtube_words" / "items.csv"
    if ag.exists():
        g = pd.read_csv(ag)
        g = g[g.status == "accepted"]
        for r in g.itertuples():
            rows.append(dict(clip_id=r.clip_id, source="agent", dataset="agent", subset="agent_youtube_words", kind="isolated", signer=r.signer,
                             gloss=r.word, concept=r.concept, role=r.role, extractor="mp", pair_id=None, recording_variation=None,
                             prim=str(POSE_MP / "agent" / f"{safe_id(r.clip_id)}.npz"), aux=None, span=(int(r.f0), int(r.f1))))
    R = pd.DataFrame(rows)
    log.info("shards: %d primary clips to pack (%s)", len(R), R.source.value_counts().to_dict())
    KP, SC, off, hw, ids = [], [], [0], [], []
    meta = []
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (r, res) in enumerate(zip(R.itertuples(), pool.imap(_pack_one, [(x.prim, x.aux, x.kind, x.span if isinstance(x.span, tuple) else None) for x in R.itertuples()], chunksize=16)), 1):
            if res is None:
                continue
            for view, d, cid in (("primary", res, r.clip_id), ("rtmw", res.get("aux"), f"{r.clip_id}#rtmw")):
                if d is None:
                    continue
                KP.append(d["kp"]); SC.append(d["sc"]); off.append(off[-1] + d["n"]); hw.append(d["hw"]); ids.append(cid)
                meta.append(dict(r._asdict(), clip_id=cid, view=view, aux_clip=f"{r.clip_id}#rtmw" if (view == "primary" and res.get("aux")) else None,
                                 n_frames=d["n"], span_f0=int(res["span"][0]), span_f1=int(res["span"][1]), has_span=res["has_span"],
                                 mirrored=res["mirrored"], extractor=r.extractor if view == "primary" else "rtmw"))
            if i % 4000 == 0:
                log.info("  packed %d/%d (%.0fs)", i, len(R), time.time() - t0)
    np.savez(out / "pose_store.npz", KP=np.concatenate(KP), SC=np.concatenate(SC), off=np.array(off, np.int64), hw=np.array(hw, np.int32), clip_id=np.array(ids))
    C = pd.DataFrame(meta).drop(columns=["Index", "prim", "aux", "span"], errors="ignore")
    iso = C[(C.kind == "isolated") & (C.view == "primary")]
    C["concept_signers"] = C.concept.map(iso.groupby("concept").signer.nunique()).fillna(0).astype(int)
    C.to_parquet(out / "clips.parquet", index=False)
    (out / "concepts.json").write_bytes((PREP / "manifest" / "concepts.json").read_bytes())
    write_json(out / "daily_concepts.json", daily_concepts())
    # real between-sign movements: gaps between consecutive aligned words of the TSL51 tune sentences (v6 forced alignment, kept in params/)
    al = PREP / "params" / "tsl51_alignment_v6.json"
    tr = []
    if al.exists():
        for cid, v in read_json(al).items():
            if not cid.startswith("t51sr:") or v.get("role") != "tagger_train":
                continue
            seg = v["segments"]
            for s0, s1 in zip(seg[:-1], seg[1:]):
                if s1["f0"] - s0["f1"] >= 4:
                    tr.append(dict(clip_id="t51s:" + cid[len("t51sr:"):], f0=int(s0["f1"]), f1=int(s1["f0"])))
    pd.DataFrame(tr, columns=["clip_id", "f0", "f1"]).to_parquet(out / "transitions.parquet", index=False)
    log.info("shards: %d rows (%d primary, %d RTMW views), %d frames, %d transitions → %s (%.0f MB)", len(ids), int((C.view == "primary").sum()),
             int((C.view == "rtmw").sum()), off[-1], len(tr), out, sum(f.stat().st_size for f in out.iterdir()) / 1e6)


SCHEMA_DOC = """# data_prep/ — ThaiSLM Standard Schema (one format for every source)

`s3://dsi490-lake-signdata/data_prep/` (local mirror: `cloud_s3/data_prep/`). Built from `raw_data/` by `scripts/data_prep.py`
(entry point `python main.py prep <step>`). Every clip from every source — dictionary videos, MediaPipe landmark
releases, researcher recordings, continuous broadcasts — ends up as the same kind of file with the same meaning of
every number, so a model trained on it cannot tell the sources apart by format, only by the signing.

## Layout

```
data_prep/
  SCHEMA.md                 this file (generated)
  manifest/
    clips.parquet           ONE ROW PER CLIP / VIDEO, every source (columns below) — start here
    lexicon.parquet         every source gloss → Thai lemma → concept (+ how it was decided)
    concepts.json           {"concept": {lemma: concept}, "groups": {concept: [synonyms]}}
    synonym_decisions.parquet  every synonym pair that was checked (a, b, label same/different, source rule|llm|manual)
    summary.csv             clips / signers / concepts / frames per dataset × role × use
  pose/<dataset>/<clip>.npz Standard Schema keypoints (below) — what training and banks read
  pose_raw/<dataset>/<clip>.npz  RTMW-X output before harmonisation (real face points, per-frame QC) — for re-harmonising
  params/                   harmonisation parameters (TSL-ONE-S per-clip aspect, th_sl signer clusters, TSL51 Thai sentences)
  qc/                       excluded_clips.csv (clip, reason), excluded_summary.csv, thsl_signers.jpg (signer-cluster contact sheet)
  shards/                   packed training input for GPU jobs: pose_store.npz + clips.parquet + concepts.json
```

## Keypoint file (`pose/<dataset>/<clip>.npz`)

| key | shape / type | meaning |
|---|---|---|
| `kp` | [T, 133, 2] float16 | x / W, y / H of the 133 COCO-WholeBody points (RTMW layout). Only the 69 slots the encoder reads are filled (body 9 · left hand 21 · right hand 21 · face 18); others are 0 |
| `sc` | [T, 133] float16 | confidence in [0, 1]; 0 = missing. MediaPipe detections are 1.0 |
| `fps` | float | always **25** — every source is resampled on its own timestamps |
| `hw` | (H, W) | frame size the coordinates are normalised by; `x·W/H` gives isotropic units |
| `src`, `extractor` | str | dataset name; `rtmw` (RTMW-X from video) · `mp75` (TSL-ONE-S MediaPipe) · `mp54` (TSL51 MediaPipe csv) |
| `src_fps`, `imputed` | float, str[] | original frame rate; which parts were imputed (`face_template`, `nose`, `eyes`, `ears`) |

Load with `modules.schema.load(path)`; `modules.unisign.prepare_parts` turns it into encoder input.

## Harmonisation (why every source looks the same)

| issue | evidence | rule |
|---|---|---|
| topology | RTMW 133 pts · TSL-ONE-S MediaPipe 75 (x,y) · TSL51 MediaPipe 54 (x,y,z) | map into the 133-slot layout; hand-21 joint order is identical in MediaPipe and COCO-WholeBody |
| handedness | left hand root ↔ left wrist in 100 % of sampled frames in all sources | 1:1 mapping, no mirroring |
| face points | MediaPipe sources lack jaw / inner mouth; TSL51 lacks nose / eyes / ears | for EVERY source the 18 face points are a mean face template (`models/face_template.npz`) placed per frame by a smoothed similarity on the anchors the source has; real mouth corners kept |
| time axis | TSL51 csv `t_ms` is extraction wall-clock (clips stretched up to 3×) | time = row / metadata fps; everything resampled to 25 fps |
| TSL-ONE-S frame | coordinates not normalised to the stated 1280×720 (hands 1.9× too wide) | per-clip aspect from body geometry (`params/tslone_aspect.json`) |
| confidence | RTMW 0–1.1, MediaPipe presence only, MediaPipe drops hands | clip to [0,1]; hand gaps ≤ 0.2 s interpolated for all sources |
| extractor for video | inference always runs RTMW-X on video | every source that has video (TTRS, th_sl, TSL51 researcher, YouTube) is extracted with **the same code as inference** (`modules/wholebody.py`: YOLOX signer lock + RTMW-X on the upper-body box, 25 fps) |
| labels | "ขอโทษ(ท่าที่ 2)", "หยิก (แขน)", "แต่งงาน/สมรส", "20", English TSL-ONE-S glosses, synonyms across sources | `modules/lexicon.py`: variant / sense split by rule, slash = same sign, numbers → Thai words, synonym groups verified by LLM + manual review → `concept` |
| signer identity | th_sl has no signer column | face-embedding clustering (`params/thsl_signers.csv`, contact sheet in `qc/`); held-out roles are whole signer clusters |
| duplicates | TSL51 "web-scraped" clips are copies of th_sl / TTRS videos | dropped from TSL51 (`duplicate_of_th_sl`); v5 already dropped the TTRS copies |

## `manifest/clips.parquet` columns

| column | meaning |
|---|---|
| `clip_id` | stable id: `ttrs:<page id>` · `thsl:th_sl_#####` · `tslone:<video_id>` · `t51x:` (TSL51 expert) · `t51u:`/`t51ur:` (TSL51 researcher sign, MediaPipe / RTMW) · `t51s:`/`t51sr:` (researcher sentence) · `yttitle:`/`ytmined:` · `yt:<id>` (continuous) |
| `dataset` | ttrs · th_sl · tslone · tsl51_expert · tsl51_user_sign(_rtmw) · tsl51_user_sentence(_rtmw) · youtube_title · youtube_mined · youtube_bigsign · youtube_sentence · youtube_continuous |
| `kind` | isolated (one word) · null (not a sign: rest, fidgeting) · sentence (known gloss sequence) · continuous (no per-word labels) |
| `source_file`, `source_url` | where it came from in `raw_data/` (and upstream URL) |
| `pose_file`, `pose_raw_file` | paths inside `data_prep/` |
| `extractor` | rtmw · mp75 · mp54 |
| `signer` | signer id (never shared across datasets unless the same person) |
| `gloss`, `lemma`, `sign_variant`, `sense`, `concept` | original label → cleaned Thai lemma → variant / sense note → concept (synonym group) used for training and scoring; sentences: `concept` = `c1|c2|…` in TSL order |
| `category` | source category / tag (TTRS category, th_sl tag, TSL-ONE-S category) |
| `role` | train · val · test (held-out **signers**) · tagger_train / sentence_test (TSL51 sentence templates) · deploy (bank only) · unlabeled |
| `n_frames`, `width`, `height` | length at 25 fps, frame size |
| `span_f0`, `span_f1` | signing span inside an isolated clip (hands raised / moving) |
| `hand_frac`, `body_frac` | share of frames with a visible hand / both shoulders |
| `n_persons_max`, `multi_person_frac`, `lock_frac` | RTMW sources: people detected, signer-lock rate |
| `concept_signers` | number of distinct signers of this concept among usable isolated clips (bank depth) |
| `exclude_reason`, `use` | why a clip is not used (see QC) — `use = exclude_reason is null` |
| `licence` | usage terms of the source |
| `pair_id`, `recording_variation`, `official_split` | source-specific extras (TSL51 extractor pairs / recording style, TSL-ONE-S official split) |

## Quality control (only hard evidence of noise is excluded)

__QC__

## Contents

__SUMMARY__

## Licences

__LICENCE__
"""


def cmd_docs(a):
    m = pd.read_parquet(PREP / "manifest" / "clips.parquet")
    qc = "\n".join([
        f"- rules: fewer than {QC['min_frames']} frames · isolated word longer than {QC['max_frames_isolated'] // 25} s · both shoulders visible in "
        f"< {QC['min_body_frac']:.0%} of frames (no signer in frame, e.g. a close-up of a hand / foot) · a hand visible in < {QC['min_hand_frac']:.0%} "
        "of frames · no signing motion (hands never above the waist AND 95th-percentile wrist speed < 0.8 shoulder-widths/s) · no label · "
        "duplicate of another source's video · pose missing (video could not be decoded)",
        "- flagged clips stay in the manifest (`use = False`, `exclude_reason`) and in `qc/excluded_clips.csv`; nothing is deleted",
        "- the sources turned out clean: QC removes only a few dozen clips, most of them duplicates that must go anyway (leakage / double-counted "
        "signers); keeping the rest changes the training set by ~0.1 % — too little for a training run to measure, so no 'with noise' run was "
        "trained. The data ablation instead measures whole sources (th_sl; MediaPipe-only sources) — see result_v6.md",
        "", "| dataset | reason | clips |", "|---|---|---|"] + [f"| {r.dataset} | {r.exclude_reason} | {r.n} |" for r in
                                                           m[~m.use].groupby(["dataset", "exclude_reason"]).size().reset_index(name="n").itertuples()])
    g = m.groupby("dataset").agg(clips=("clip_id", "size"), used=("use", "sum"), signers=("signer", "nunique"),
                                 concepts=("concept", "nunique"), hours=("n_frames", lambda x: x.sum() / 25 / 3600), extractor=("extractor", "first"))
    summ = "| dataset | clips | used | signers | concepts | hours | extractor |\n|---|---|---|---|---|---|---|\n" + "\n".join(
        f"| {k} | {int(r.clips)} | {int(r.used)} | {r.signers} | {r.concepts} | {r.hours:.2f} | {r.extractor} |" for k, r in g.iterrows())
    roles = m[m.use].groupby(["role"]).size().to_dict()
    summ += f"\n\nUsed clips by role: {roles}. Concepts with ≥ 2 signers: {int((m[m.use & m.kind.eq('isolated')].groupby('concept').signer.nunique() >= 2).sum())}."
    lic = "\n".join(f"- **{k}**: {v}" for k, v in LICENCE.items())
    doc = SCHEMA_DOC.replace("__QC__", qc).replace("__SUMMARY__", summ).replace("__LICENCE__", lic)
    (PREP / "SCHEMA.md").write_text(doc, encoding="utf-8")
    (PREP / "README.md").write_text("See SCHEMA.md. Built by `python main.py prep <step>` from `raw_data/`; pull with `python main.py data pull data_prep`.\n",
                                    encoding="utf-8")
    log.info("docs → %s", PREP / "SCHEMA.md")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pose_items"); sub.add_parser("face_emb")
    p = sub.add_parser("ingest"); p.add_argument("--src", default=""); p.add_argument("--workers", type=int, default=6); p.add_argument("--force", action="store_true")
    p = sub.add_parser("lexicon"); p.add_argument("--no_llm", action="store_true")
    p = sub.add_parser("manifest"); p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("signers"); p.add_argument("--thr", type=float, default=0.7); p.add_argument("--min_size", type=int, default=25)
    p.add_argument("--sheet", default=""); p.add_argument("--test_clusters", default="2"); p.add_argument("--val_clusters", default="3")
    p = sub.add_parser("shards"); p.add_argument("--workers", type=int, default=6); p.add_argument("--out", default="")
    sub.add_parser("docs")
    a = ap.parse_args()
    dict(docs=cmd_docs, shards=cmd_shards, signers=cmd_signers, manifest=cmd_manifest, lexicon=cmd_lexicon, pose_items=cmd_pose_items, face_emb=cmd_face_emb, ingest=cmd_ingest)[a.cmd](a)


if __name__ == "__main__":
    main()
