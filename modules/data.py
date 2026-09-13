"""Adapters → canonical manifests (ClipRecord / LexemeRecord), text normalisation,
video decoding at 12.5 fps, VTT parsing, and leakage-safe splits."""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ART, ROOT, get_logger, load_cfg, write_json

log = get_logger("data")
MAN = ART / "manifests"

# ----------------------------------------------------------------------------- text
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_ZW = re.compile(r"[​‌‍﻿]")
_PAREN_TAIL = re.compile(r"\s*\(([^()]*)\)\s*$")
_VARIANT_SLUGS = {"เศรษฐเสถียร": "setsatian", "ราชสุดา": "ratchasuda", "พจนานุกรมสารสนเทศ": "infodict"}


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFC", s)
    s = _ZW.sub("", s).translate(_THAI_DIGITS)
    s = re.sub(r"\s+", " ", s).strip()
    try:
        from pythainlp.util import normalize as th_norm
        s = th_norm(s)
    except Exception:
        pass
    return s


def split_variant(label: str) -> tuple[str, str | None]:
    """'คำ (ภาษามือโรงเรียนเศรษฐเสถียรฯ)' → ('คำ', 'setsatian')."""
    s = normalize_text(label)
    m = _PAREN_TAIL.search(s)
    if not m:
        return s, None
    lemma = s[: m.start()].strip()
    tag = m.group(1)
    slug = next((v for k, v in _VARIANT_SLUGS.items() if k in tag), "other")
    return (lemma or s), slug


_TITLES = ("นางสาว", "นาย", "นาง", "น.ส.")


def normalize_signer(name) -> str | None:
    if not isinstance(name, str) or not name.strip():
        return None
    s = normalize_text(name)
    for t in _TITLES:
        if s.startswith(t):
            s = s[len(t):]
    s = s.replace(" ", "")
    if s in ("ทดสอบ", ""):
        return None
    return s


# ----------------------------------------------------------------------------- manifests
def build_manifests(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    MAN.mkdir(parents=True, exist_ok=True)
    if (MAN / "clips.parquet").exists() and not force:
        return pd.read_parquet(MAN / "clips.parquet"), pd.read_parquet(MAN / "lexicon.parquet")
    cfg = load_cfg("data")
    df = pd.read_csv(ROOT / "metadata" / "metadata_master.csv")
    rows = []
    # --- TTRS adapter
    t = df[df.source == "ttrs_dictionary"].copy()
    names = t.signer_creator.map(normalize_signer)
    freq = names.value_counts()
    major = [n for n, c in freq.items() if c >= 10]
    def canon(n):
        if n is None:
            return None
        if n in major:
            return n
        m = difflib.get_close_matches(n, major, n=1, cutoff=0.85)
        return m[0] if m else n
    names = names.map(canon)
    sid_map = {n: f"S_ttrs_{i:02d}" for i, n in enumerate(sorted(set(names.dropna())))}
    for (_, r), nm in zip(t.iterrows(), names):
        lemma, tag = split_variant(r.label)
        rows.append(dict(
            clip_id=r.uid, source="ttrs", subset="ttrs", level="word", media_path=r.rel_path,
            media_sha256=r.sha256, duration_s=float(r.duration_sec), fps_native=float(r.fps),
            width=int(r.width), height=int(r.height), layout="full", signer_zone=None,
            mask_regions=json.dumps([[0.0, cfg["ttrs"]["caption_band_frac"], 1.0, 1.0], cfg["ttrs"]["logo_box"]]),
            chroma=True, label=normalize_text(r.label), lemma=lemma, variant_tag=tag,
            category=normalize_text(r.category_th), pos=normalize_text(r.parts_of_speech_th),
            synonyms=normalize_text(r.synonyms_th), signer_name=nm, signer_id=sid_map.get(nm),
            signer_gender=r.signer_gender_th if isinstance(r.signer_gender_th, str) else None,
            channel="ttrs", domain_tags="studio", license_status="research_only", title=normalize_text(r.label),
        ))
    # --- YouTube adapter
    y = df[df.source == "youtube"]
    for _, r in y.iterrows():
        subset = {"word": "youtube_word", "sentence": "youtube_sentence", "continuous": "bigsign"}[r.level]
        if "bigsign" not in str(r.rel_path) and r.level == "continuous":
            subset = "youtube_continuous"
        vertical = int(r.height) > int(r.width)
        title = normalize_text(r.label)
        tags = ["broadcast" if r.level == "continuous" else "lesson"]
        if "เพลง" in title:
            tags.append("song")
        rows.append(dict(
            clip_id=r.uid, source="youtube", subset=subset, level=r.level, media_path=r.rel_path,
            media_sha256=r.sha256, duration_s=float(r.duration_sec), fps_native=float(r.fps),
            width=int(r.width), height=int(r.height), layout="vertical_split" if vertical else "full",
            signer_zone=json.dumps(cfg["layouts"]["bigsign_vertical"]["signer_zone"]) if vertical else None,
            mask_regions=json.dumps([]), chroma=False, label=title, lemma=None, variant_tag=None,
            category=None, pos=None, synonyms=None, signer_name=None, signer_id=None, signer_gender=None,
            channel=str(r.channel) if isinstance(r.channel, str) else "unknown", domain_tags=",".join(tags),
            license_status="research_only", title=title,
        ))
    clips = pd.DataFrame(rows)
    # --- Lexicon (LexemeRecord)
    tt = clips[clips.source == "ttrs"]
    lemmas = sorted(tt.lemma.unique())
    lid = {l: f"L{i:06d}" for i, l in enumerate(lemmas)}
    clips["lexeme_id"] = clips.lemma.map(lid)
    clips["sign_variant_id"] = [
        (f"{l}#{v}" if v else l) if isinstance(l, str) else None for l, v in zip(clips.lexeme_id, clips.variant_tag)
    ]
    lex = (tt.assign(lexeme_id=tt.lemma.map(lid))
             .groupby("lexeme_id")
             .agg(lemma_th=("lemma", "first"), surface_forms=("label", lambda x: sorted(set(x))),
                  category_th=("category", "first"), pos_th=("pos", "first"), synonyms_th=("synonyms", "first"),
                  n_instances=("clip_id", "size"), n_signers=("signer_id", lambda x: x.nunique()))
             .reset_index())
    lex["is_fingerspelling"] = lex.lemma_th.str.len() == 1
    clips["schema_version"] = cfg["schema_version"]
    clips.to_parquet(MAN / "clips.parquet", index=False)
    lex.to_parquet(MAN / "lexicon.parquet", index=False)
    log.info("manifests: %d clips, %d lexemes, %d named TTRS signers", len(clips), len(lex), len(sid_map))
    return clips, lex


# ----------------------------------------------------------------------------- video decode
def probe(path: str | Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return dict(width=int(s["width"]), height=int(s["height"]), fps=float(num) / float(den),
                duration=float(j["format"]["duration"]))


def decode_frames(path: str | Path, fps: float = 12.5, start_s: float = 0.0, max_s: float | None = None,
                  max_long_side: int | None = None, chunk: int = 64):
    """Yield uint8 BGR frame chunks [N,H,W,3] resampled to `fps` by timestamp (in-process OpenCV decode;
    spawning ffmpeg per clip cost more than decoding for the 2–8 s TTRS clips)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        yield from decode_frames_ffmpeg(path, fps, start_s, max_s, max_long_side, chunk)
        return
    nfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    i, next_t, buf = 0, 0.0, []
    t_end = max_s if max_s else float("inf")
    try:
        while True:
            t = i / nfps
            if t > t_end:
                break
            if t + 0.5 / nfps < next_t:
                if not cap.grab():
                    break
                i += 1
                continue
            ok, fr = cap.read()
            if not ok:
                break
            i += 1
            next_t += 1.0 / fps
            if max_long_side and max(fr.shape[:2]) > max_long_side:
                s = max_long_side / max(fr.shape[:2])
                fr = cv2.resize(fr, (int(fr.shape[1] * s), int(fr.shape[0] * s)), interpolation=cv2.INTER_AREA)
            buf.append(fr)
            if len(buf) == chunk:
                yield np.stack(buf)
                buf = []
        if buf:
            yield np.stack(buf)
    finally:
        cap.release()


def decode_frames_ffmpeg(path: str | Path, fps: float = 12.5, start_s: float = 0.0, max_s: float | None = None,
                         max_long_side: int | None = None, chunk: int = 64):
    """Yield uint8 BGR frame chunks [N,H,W,3] sampled at `fps` via an ffmpeg pipe."""
    info = probe(path)
    w, h = info["width"], info["height"]
    if max_long_side and max(w, h) > max_long_side:
        s = max_long_side / max(w, h)
        w, h = int(round(w * s / 2) * 2), int(round(h * s / 2) * 2)
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(path)]
    if max_s:
        cmd += ["-t", f"{max_s:.3f}"]
    cmd += ["-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
    fsz = w * h * 3
    buf = []
    try:
        while True:
            raw = p.stdout.read(fsz)
            if len(raw) < fsz:
                break
            buf.append(np.frombuffer(raw, np.uint8).reshape(h, w, 3))
            if len(buf) == chunk:
                yield np.stack(buf)
                buf = []
        if buf:
            yield np.stack(buf)
    finally:
        p.stdout.close()
        p.wait()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- VTT
_TS = r"(\d+):(\d{2}):(\d{2})\.(\d{3})"


def _ts(m) -> float:
    return int(m[0]) * 3600 + int(m[1]) * 60 + int(m[2]) + int(m[3]) / 1000


def parse_vtt(path: str | Path) -> list[dict]:
    """Parse a YouTube VTT into cues [{t0,t1,text}] with rolling duplicates removed."""
    cues, seen_last = [], ""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    for block in re.split(r"\n\s*\n", text):
        m = re.search(_TS + r"\s*-->\s*" + _TS, block)
        if not m:
            continue
        g = m.groups()
        t0, t1 = _ts(g[:4]), _ts(g[4:])
        lines = [re.sub(r"<[^>]+>", "", l).strip() for l in block[m.end():].strip().splitlines()]
        lines = [normalize_text(l) for l in lines if l.strip()]
        # rolling auto-captions re-print the previous line first → keep only new lines
        new = [l for l in lines if l != seen_last]
        if not new:
            continue
        seen_last = lines[-1]
        cues.append(dict(t0=t0, t1=t1, text=" ".join(new)))
    return cues


# ----------------------------------------------------------------------------- signers (E8)
def assign_signers(clips: pd.DataFrame, face_feat: dict, min_prob: float = 0.95, cluster_thr: float = 0.25,
                   min_cluster: int = 8, seed: int = 42):
    """Fill signer ids for unnamed TTRS clips from face-crop embeddings.
    face_feat: clip_id → mean face-stream [CLS] (384-d). Named clips train a classifier (reported CV accuracy);
    unnamed clips get a name when p ≥ min_prob, the rest are clustered into new signer ids (clusters ≥ min_cluster)."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    c = clips.copy()
    c["signer_src"] = np.where(c.signer_id.notna(), "name", None)
    t = c[(c.source == "ttrs") & c.clip_id.isin(face_feat)]
    named = t[t.signer_id.notna()]
    vc = named.signer_id.value_counts()
    trainable = named[named.signer_id.isin(vc[vc >= 10].index)]
    report = dict(named_clips=len(named), named_signers_ge10=int(len(vc[vc >= 10])))
    if trainable.signer_id.nunique() >= 2:  # sanity: is face-crop identity separable at all?
        X = np.stack([face_feat[i] for i in trainable.clip_id]); y = trainable.signer_id.values
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        report["named_cv_accuracy"] = float(cross_val_score(clf, X, y, cv=5).mean())
    # open-set: cluster everything, then name clusters by majority vote of their named members
    X = np.stack([face_feat[i] for i in t.clip_id])
    X = X - X.mean(0)  # raw DINOv2 cosine distances between faces are all tiny; centre first
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    lab =AgglomerativeClustering(n_clusters=None, distance_threshold=cluster_thr, metric="cosine",
                                  linkage="average").fit_predict(X)
    t = t.assign(cluster=lab)
    agree = total_named = 0
    new_id = 0
    for l, g in t.groupby("cluster"):
        nm = g.signer_id.dropna()
        if len(nm) >= 3:
            top, share = nm.value_counts().index[0], nm.value_counts().iloc[0] / len(nm)
            agree += int((nm == top).sum()); total_named += len(nm)
            if share < 0.9:
                continue  # mixed cluster → leave unnamed members unknown
            sid, src = top, "face_cluster_named"
        elif len(nm) == 0 and len(g) >= min_cluster:
            sid, src = f"S_ttrs_u{new_id:02d}", "face_cluster_new"; new_id += 1
        else:
            continue
        ix = g.index[g.signer_id.isna()]
        c.loc[ix, "signer_id"] = sid
        c.loc[ix, "signer_src"] = src
    report.update(n_clusters=int(lab.max() + 1), cluster_name_purity=agree / max(total_named, 1), new_signers=new_id,
                  by_src=c[c.source == "ttrs"].signer_src.value_counts(dropna=False).to_dict(),
                  still_unknown=int(c[(c.source == "ttrs")].signer_id.isna().sum()))
    return c, report


# ----------------------------------------------------------------------------- splits
def ttrs_pairs(clips: pd.DataFrame, unit: str = "sign_variant_id") -> pd.DataFrame:
    """Groups of TTRS clips of the same sign performed by ≥2 different signers."""
    t = clips[(clips.source == "ttrs") & clips.signer_id.notna()]
    g = t.groupby(unit).signer_id.nunique()
    keep = g[g >= 2].index
    return t[t[unit].isin(keep)]


def make_splits(clips: pd.DataFrame, seed: int = 42, unit: str = "sign_variant_id",
                fracs: tuple = (0.2, 0.1, 0.1, 0.3, 0.3)) -> dict:
    """Leakage-safe split. Cross-signer signs are divided train/val/test by `fracs` (blueprint v2 put them all in
    val/test, which leaves no cross-signer positive pair to learn signer invariance from). Every clip of a val/test
    sign is removed from SSL + ISLR training. Real-world files (data_test/) are sha256-blocked from every pool."""
    """fracs over cross-signer signs: (train pairs, E2 unseen val, E2 unseen test, E1 seen val, E1 seen test)."""
    if len(fracs) == 3:
        fracs = (fracs[0], fracs[1], fracs[2], 0.0, 0.0)
    rng = np.random.RandomState(seed)
    pairs = ttrs_pairs(clips, unit)
    units = sorted(pairs[unit].unique())
    rng.shuffle(units)
    cut = np.round(np.cumsum(fracs) / np.sum(fracs) * len(units)).astype(int)
    groups = np.split(np.array(units, dtype=object), cut[:-1])
    val_u, test_u, sval_u, stest_u = (sorted(g.tolist()) for g in groups[1:5])
    t = clips[clips.source == "ttrs"]
    val_clips = t[t[unit].isin(val_u)].clip_id.tolist()
    test_clips = t[t[unit].isin(test_u)].clip_id.tolist()

    def query_side(us):
        """E1: hold out one signer's clips per sign (the query signer); the other signer's clips stay in training."""
        q = []
        for u in us:
            g = t[(t[unit] == u) & t.signer_id.notna()]
            sig = sorted(g.signer_id.unique())
            s = sig[rng.randint(len(sig))]
            q += g[g.signer_id == s].clip_id.tolist()
        return q
    sval_q, stest_q = query_side(sval_u), query_side(stest_u)
    block = set(sha256_file(p) for p in sorted((ROOT / "data_test").glob("*.mp4")))
    held = set(val_clips) | set(test_clips) | set(sval_q) | set(stest_q)
    leak = clips[clips.media_sha256.isin(block)].clip_id.tolist()
    train_islr = t[~t.clip_id.isin(held) & ~t.clip_id.isin(leak)].clip_id.tolist()
    ssl_pool = clips[~clips.clip_id.isin(held | set(leak))].clip_id.tolist()
    sp = dict(unit=unit, seed=seed, fracs=list(fracs), val_units=val_u, test_units=test_u, val_clips=val_clips,
              test_clips=test_clips, seen_val_units=sval_u, seen_test_units=stest_u, seen_val_queries=sval_q,
              seen_test_queries=stest_q, train_islr=train_islr, ssl_pool=ssl_pool,
              realworld_sha256_blocklist=sorted(block), leaked_realworld_in_corpus=leak)
    write_json(MAN / "splits.json", sp)
    log.info("splits: E2 unseen val %d units (%d clips) / test %d (%d) · E1 seen val %d units (%d queries) / test %d (%d) · "
             "%d ISLR train, %d SSL pool, leak=%d", len(val_u), len(val_clips), len(test_u), len(test_clips), len(sval_u),
             len(sval_q), len(stest_u), len(stest_q), len(train_islr), len(ssl_pool), len(leak))
    return sp
