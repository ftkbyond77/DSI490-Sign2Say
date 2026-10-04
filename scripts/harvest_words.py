"""Harvest more signers for daily-conversation words from public Thai sign-language videos (YouTube single-word lessons).

  python scripts/harvest_words.py discover [--max_depth 4] [--per_query 25]   # search + channel crawl → agent_data/youtube_words/candidates.csv
  python scripts/harvest_words.py download [--limit N]                        # ≤480p mp4 of accepted candidates (+ data_test leakage check)
  python scripts/harvest_words.py videos                                      # videos.csv → python scripts/extract_mp.py --datasets agent
  python scripts/harvest_words.py signers                                     # faces: channels → people, data_test look-alikes excluded
  python scripts/harvest_words.py label                                       # sign span + model consistency → items.csv (annotations)

Why: recognition depends on how many different people the bank has seen signing a word (result_v6 §4: R@1 ≈ 0.14 with 1–2
signers, ≈ 0.88 with ≥ 10). Daily-conversation words have a median of 2 signers. Short lesson clips whose title names exactly
one word are a reliable label source (the title IS the gloss); every clip is still checked (span found, not a duplicate of a
data_test video, consistent with the other examples of that word) before it may enter training.

Output: cloud_s3/agent_data/youtube_words/ (videos/, meta/, candidates.csv, items.csv, qc/). Licence: YouTube publisher's
rights, research use only — same terms as the v4–v6 YouTube data.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from modules.utils import DATA, PREP, get_logger, write_json

log = get_logger("harvest")
OUT = DATA / "agent_data" / "youtube_words"
YTDLP = "yt-dlp"
MARKERS = re.compile(r"ภาษามือ|thsl|tsl|sign\s*language|ภาษามือไทย", re.I)
# marker phrases (removed wherever they occur, with a lesson verb glued in front: "สอนภาษามือ", "ท่าภาษามือ")
MARK_SUB = re.compile(r"(?:สอน|เรียน|ท่า|ทำ)?(?:ภาษามือไทย|ภาษามือ)(?:ไทย)?|คำว่า|คำศัพท์|ในชีวิตประจำวัน|ขั้นพื้นฐาน|thsl|tsl|sign\s*language|(?<!\S)#\S+", re.I)
SPELLING = re.compile(r"สะกด")                    # fingerspelling clips show letters, not the lexical sign → never a label for the word
QUESTION = re.compile(r"(ไหม|มั้ย|หรือยัง|หรือเปล่า)$")
GENERIC = {"หมวด", "ตอน", "เพลง", "บทเรียน", "บทที่", "แนะนำตัว", "ทักทาย", "พื้นฐาน", "คลิป", "วิดีโอ", "ภาษามือ", "ประโยค"}
# whole tokens that are not the word (only removed when they stand alone — Thai words are not space separated: "ทำงาน" ≠ "ทำ")
NOISE_TOK = {"สอน", "เรียน", "วิธี", "ท่า", "ทำ", "ยังไง", "ง่ายๆ", "ไทย", "thai", "shorts", "short", "ค่ะ", "ครับ", "นะคะ", "นะครับ", "พื้นฐาน",
             "ep", "คำ", "ศัพท์", "ภาษา", "มือ"}
PUNCT = re.compile(r"[\"'“”‘’()\[\]{}<>|:;,.!?~\-–—_/\\*+=@&%^$•·…]+")


def run(cmd, timeout=120):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")        # yt-dlp prints titles in the console code page otherwise
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)
    return r.stdout


def title_core(title):
    """Single-word lesson titles → the word; e.g. 'ภาษามือ คำว่า รัก' → 'รัก', 'ภาษามือไทยในชีวิตประจำวัน/หมวดร่างกาย/สมอง' → 'สมอง'."""
    t = title.strip()
    t = re.sub(r"^\s*(?:ep\.?\s*)?\d+\s*[#.)\-:]?\s*", "", t, flags=re.I)          # "02#พับผ้าห่ม", "5 เพื่อนปลูกข้าว", "6.ภาษามือ:โชคดี"
    if "/" in t:                                   # "<series>/หมวด<category>/<word>"
        last = t.split("/")[-1].strip()
        if last and not MARKERS.search(last):
            return PUNCT.sub(" ", last).strip()
    t = MARK_SUB.sub(" ", t)
    t = PUNCT.sub(" ", t)
    toks = [x for x in t.split() if x.lower() not in NOISE_TOK and not re.fullmatch(r"(ep)?\d+", x.lower())]
    return " ".join(toks).strip()


def concepts_index():
    c = json.loads((PREP / "manifest" / "concepts.json").read_text(encoding="utf-8"))
    return c["concept"]


def search(query, n):
    out = run([YTDLP, "--flat-playlist", "--print", "%(id)s\t%(duration)s\t%(channel_id)s\t%(channel)s\t%(title)s", f"ytsearch{n}:{query}"])
    rows = []
    for line in out.splitlines():
        p = line.split("\t")
        if len(p) == 5:
            rows.append(dict(id=p[0], duration=float(p[1]) if p[1] not in ("NA", "None", "") else np.nan, channel_id=p[2], channel=p[3], title=p[4], query=query))
    return rows


def channel_uploads(channel_id, n=600):
    rows = []
    for tab in ("videos", "shorts"):
        out = run([YTDLP, "--flat-playlist", "--playlist-end", str(n), "--print",
                   "%(id)s\t%(duration)s\t%(channel_id)s\t%(channel)s\t%(title)s", f"https://www.youtube.com/channel/{channel_id}/{tab}"], timeout=300)
        for line in out.splitlines():
            p = line.split("\t")
            if len(p) == 5:
                rows.append(dict(id=p[0], duration=float(p[1]) if p[1] not in ("NA", "None", "") else np.nan, channel_id=p[2] if p[2] != "NA" else channel_id,
                                 channel=p[3], title=p[4], query=f"channel:{channel_id}/{tab}"))
    return rows


def cmd_discover(a):
    OUT.mkdir(parents=True, exist_ok=True)
    daily = json.loads((ROOT / "vocab" / "daily_conversation.json").read_text(encoding="utf-8"))
    words = list(dict.fromkeys(w for k, v in daily.items() if not k.startswith("_") for w in v))
    cidx = concepts_index()
    dep = pd.read_csv(ROOT / "cache" / "daily_depth_v6.csv").set_index("word").signers.to_dict() if (ROOT / "cache" / "daily_depth_v6.csv").exists() else {}
    targets = [w for w in words if dep.get(w, 0) <= a.max_depth]
    templates = ("ภาษามือ คำว่า {w}", "ภาษามือไทย {w}")
    if a.words:                     # targeted round: these words only, with more query wordings (new signers for thin words)
        targets = [w for w in a.words.split(",") if w]
        templates = ("ภาษามือ คำว่า {w}", "ภาษามือไทย {w}", "ภาษามือ {w}", "{w} ภาษามือ", "สอนภาษามือ {w}", "ท่าภาษามือ {w}")
    log.info("discover: %d target words (≤ %d signers)", len(targets), a.max_depth)
    cand_f = OUT / "search_raw.jsonl"
    seen_q = set()
    if cand_f.exists():
        for line in cand_f.read_text(encoding="utf-8").splitlines():
            seen_q.add(json.loads(line)["query"])
    with open(cand_f, "a", encoding="utf-8") as f:
        for i, w in enumerate(targets):
            for q in (tpl.format(w=w) for tpl in templates):
                if q in seen_q:
                    continue
                for r in search(q, a.per_query):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                seen_q.add(q)
            if (i + 1) % 20 == 0:
                log.info("  searched %d/%d words", i + 1, len(targets))
    raw = label_rows(pd.DataFrame([json.loads(x) for x in cand_f.read_text(encoding="utf-8").splitlines()]), cidx, crawled=False)
    good = raw[raw.marker & raw.concept.notna() & raw.duration.between(1.5, 45)]
    # channels that publish single-word lessons → crawl all their uploads (one channel ≈ one or a few signers)
    ch = good.groupby("channel_id").id.nunique().sort_values(ascending=False)
    crawl = [c for c, n in ch.items() if n >= a.min_channel_hits][:a.max_channels]
    crawled_f = OUT / "channels_raw.jsonl"
    done_ch = set()
    if crawled_f.exists():
        done_ch = {json.loads(x)["query"].split(":")[1].split("/")[0] for x in crawled_f.read_text(encoding="utf-8").splitlines()}
    with open(crawled_f, "a", encoding="utf-8") as f:
        for c in crawl:
            if c in done_ch:
                continue
            rows = channel_uploads(c)
            log.info("  channel %s: %d uploads", c, len(rows))
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cmd_candidates(a)


def _norm(t):
    from modules.lexicon import TEST_SURFACE
    t = re.sub(r"\s+", "", t)
    for k, v in TEST_SURFACE.items():
        t = t.replace(k, v)
    return t


_WORDS = None


def _one_word(core):
    """A single dictionary word (Thai is written without spaces: 'เพื่อนปลูกข้าว' is a phrase, 'ล้างหน้า' a word)."""
    global _WORDS
    from pythainlp.tokenize import word_tokenize
    if _WORDS is None:
        from pythainlp.corpus import thai_words
        _WORDS = set(thai_words())
    return core in _WORDS or len(word_tokenize(core, engine="newmm")) == 1


def label_rows(df, cidx, crawled):
    """title → (core text, concept, kind): kind = word (one known or new single word) | question_phrase (ends in ไหม / มั้ย /
    หรือยัง / หรือเปล่า — used to cut out the question sign) | spelling / other (not used)."""
    from modules.lexicon import DATA_TEST_REFS
    vals = set(cidx.values())
    test_sent = {_norm("".join(r)) for r in DATA_TEST_REFS.values()} | {_norm(k) for k in DATA_TEST_REFS}
    df = df.copy()
    df["core"] = df.title.map(title_core)
    df["marker"] = True if crawled else df.title.str.contains(MARKERS)

    def concept_of(core, title):
        if not core or SPELLING.search(title):
            return None, "spelling" if core else "other"
        if _norm(core) in test_sent:
            return None, "data_test_sentence"           # the same sentence as a data_test video → never used (whoever signs it)
        if core in cidx or core in vals:
            return cidx.get(core, core), "word"
        if QUESTION.search(core) and len(core) <= 30 and " " not in core:
            return core, "question_phrase"
        if re.fullmatch(r"[฀-๿]{2,15}", core) and core not in GENERIC and _one_word(core):
            return core, "new_word"                      # a Thai word no source had — a new concept
        return None, "other"
    ck = [concept_of(c, t) for c, t in zip(df.core, df.title)]
    df["concept"] = [c for c, _ in ck]
    df["kind"] = [k for _, k in ck]
    return df


def cmd_candidates(a):
    """Re-label the cached search + crawl results (no network): candidates.csv (single words, known or new) and phrases.csv."""
    cidx = concepts_index()
    parts = [label_rows(pd.DataFrame([json.loads(x) for x in (OUT / "search_raw.jsonl").read_text(encoding="utf-8").splitlines()]), cidx, False)]
    crawled_f = OUT / "channels_raw.jsonl"
    if crawled_f.exists() and crawled_f.stat().st_size:
        parts.append(label_rows(pd.DataFrame([json.loads(x) for x in crawled_f.read_text(encoding="utf-8").splitlines()]), cidx, True))
    allr = pd.concat(parts, ignore_index=True)
    allr["channel_id"] = allr.channel_id.where(allr.channel_id.notna() & (allr.channel_id != "NA"), allr["query"].str.extract(r"channel:([^/]+)/")[0])
    ok = allr.marker & allr.duration.between(1.5, 45)
    cand = allr[ok & allr.kind.isin(["word", "new_word"])].drop_duplicates("id")
    cand = cand[["id", "duration", "channel_id", "channel", "title", "core", "concept", "kind", "query"]]
    cand.to_csv(OUT / "candidates.csv", index=False)
    # short multi-word phrases from lesson channels: question phrases ending in ไหม/มั้ย (their last sign = the particle) and every
    # other phrase as a REAL continuous-signing test set (never trained on); data_test sentences are excluded by label_rows
    br = allr.title.str.fullmatch(r"\s*ภาษามือ\s*\[[^\]]+\]\s*")                # the phrase-lesson format "ภาษามือ[<Thai phrase>]"
    other = allr[br & allr.duration.between(1.5, 10) & allr.kind.isin(["other", "new_word", "question_phrase"])].copy()
    other = other[[re.fullmatch(r"[฀-๿ ]{4,30}", c or "") is not None and not _one_word(c.replace(" ", "")) for c in other.core.fillna("")]]
    other["kind"] = "phrase"
    qp = allr[ok & (allr.kind == "question_phrase")].copy()
    qp["kind"] = np.where(qp.core.str.contains("ไหม|มั้ย"), "question_particle", "phrase")
    ph = pd.concat([qp, other], ignore_index=True).drop_duplicates("id")
    ph[["id", "duration", "channel_id", "channel", "title", "core", "kind"]].to_csv(OUT / "phrases.csv", index=False)
    daily = json.loads((ROOT / "vocab" / "daily_conversation.json").read_text(encoding="utf-8"))
    words = {cidx.get(w, w) for k, v in daily.items() if not k.startswith("_") for w in v}
    log.info("candidates: %d clips (%d known-word, %d new-word), %d concepts, %d channels; daily words covered %d; question phrases %d; "
             "excluded: spelling %d, data_test sentence %d", len(cand), int((cand.kind == "word").sum()), int((cand.kind == "new_word").sum()),
             cand.concept.nunique(), cand.channel_id.nunique(), len(set(cand.concept) & words), int((ok & (allr.kind == "question_phrase")).sum()),
             int((allr.kind == "spelling").sum()), int((allr.kind == "data_test_sentence").sum()))


def thumbs(video, step=5, size=12):
    import cv2
    cap = cv2.VideoCapture(str(video)); out = []; i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (size, size), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            out.append((g - g.mean()) / (g.std() + 1e-3))
        i += 1
    cap.release()
    return np.array(out, np.float32)


def is_duplicate_of(test_thumbs, video, thr=0.15):
    """True when ≥ 30 % of a data_test video's sampled frames have a near-identical frame in `video` (re-upload / same recording)."""
    v = thumbs(video)
    if not len(v):
        return None
    for name, tt in test_thumbs.items():
        d = ((tt[:, None, :] - v[None, :, :]) ** 2).mean(-1)
        if (d.min(1) < thr).mean() >= 0.3:
            return name
    return None


def cmd_download(a):
    cand = pd.read_csv(OUT / "candidates.csv")
    (OUT / "videos").mkdir(parents=True, exist_ok=True); (OUT / "meta").mkdir(parents=True, exist_ok=True)
    test_thumbs = {p.stem: thumbs(p) for p in sorted((ROOT / "data_test").glob("*.mp4"))}
    # per concept keep at most `per_concept` clips, preferring distinct channels (= distinct signers)
    cand = cand.sample(frac=1.0, random_state=0).sort_values("concept")
    cand["k"] = cand.groupby(["concept", "channel_id"]).cumcount()
    cand = cand[cand.k < a.per_channel_concept]
    cand["kc"] = cand.groupby("concept").cumcount()
    cand = cand[cand.kc < a.per_concept]
    if (OUT / "phrases.csv").exists():                       # question phrases: their last sign is the question particle (ไหม)
        cand = pd.concat([cand, pd.read_csv(OUT / "phrases.csv")], ignore_index=True).drop_duplicates("id")
    if a.limit:
        cand = cand.head(a.limit)
    log.info("download: %d clips", len(cand))

    def fetch(vid):
        mp4 = OUT / "videos" / f"{vid}.mp4"
        if mp4.exists():
            return vid
        run([YTDLP, "-q", "--no-warnings", "-f", "bv*[height<=480][ext=mp4]/b[height<=480][ext=mp4]/bv*[height<=480]/b",
             "--remux-video", "mp4", "-o", str(OUT / "videos" / "%(id)s.%(ext)s"), "--write-info-json", "--no-write-comments",
             f"https://www.youtube.com/watch?v={vid}"], timeout=180)
        for j in (OUT / "videos").glob(f"{vid}*.info.json"):
            info = json.loads(j.read_text(encoding="utf-8"))
            keep = {k: info.get(k) for k in ("id", "title", "description", "channel", "channel_id", "uploader", "upload_date", "duration",
                                             "webpage_url", "license", "width", "height", "fps")}
            (OUT / "meta" / f"{vid}.json").write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
            j.unlink()
        return vid
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as ex:                    # 4 polite parallel downloads (yt-dlp start-up dominates)
        for i, _ in enumerate(ex.map(fetch, list(cand.id)), 1):
            if i % 50 == 0:
                log.info("  %d/%d downloaded", i, len(cand))
    status = []
    for r in cand.itertuples():
        mp4 = OUT / "videos" / f"{r.id}.mp4"
        if not mp4.exists():
            status.append(dict(id=r.id, status="download_failed")); continue
        dup = is_duplicate_of(test_thumbs, mp4)
        if dup:
            mp4.unlink()
            status.append(dict(id=r.id, status=f"duplicate_of_data_test:{dup}")); continue
        status.append(dict(id=r.id, status="ok"))
    st = pd.DataFrame(status)
    st.to_csv(OUT / "download_status.csv", index=False)
    log.info("download status: %s", st.status.value_counts().to_dict())


def cmd_videos(a):
    """videos.csv for the MediaPipe extractor (scripts/extract_mp.py): one row per downloaded, non-duplicate clip."""
    st = pd.read_csv(OUT / "download_status.csv")
    ok = st[st.status == "ok"]
    rows = [dict(clip_id=f"agent:yt:{i}", video=f"videos/{i}.mp4") for i in ok.id if (OUT / "videos" / f"{i}.mp4").exists()]
    pd.DataFrame(rows).to_csv(OUT / "videos.csv", index=False)
    log.info("videos.csv: %d clips", len(rows))


def bursts(kp, sc, hw, min_gap=12, min_len=8):
    """Signing bursts of a lesson clip: frames where a hand is raised above its rest level or moving, merged over gaps < min_gap."""
    from modules.schema import activity
    act = activity(kp, sc, hw)
    h = act["height"]
    base = np.percentile(h[h > -2.5], 10) if (h > -2.5).any() else -2.0
    on = (h > max(base + 0.45, -1.6)) | ((act["speed"] > 1.2) & (h > base + 0.2))
    on &= act["body_seen"]
    segs, t, T = [], 0, len(on)
    while t < T:
        if not on[t]:
            t += 1; continue
        s = t
        while t < T and on[t]:
            t += 1
        if segs and s - segs[-1][1] < min_gap:
            segs[-1] = (segs[-1][0], t)
        else:
            segs.append((s, t))
    return [(max(0, s - 2), min(T, e + 2)) for s, e in segs if e - s >= min_len]


def cmd_signers(a):
    """Signer ids for harvested clips + data_test leakage check (faces, modules/identity.py).

    channel → person: channels whose mean face embeddings are near-identical are one person (e.g. the same teacher posting on two
    channels) → one signer id. Any harvested channel that looks like a data_test signer is excluded from training and from every
    bank (data_test must stay a set of unseen people). A contact sheet (qc/agent_signers.jpg) shows one row per signer, data_test last."""
    import cv2
    from modules.identity import embed_faces, face_crops
    from modules.mediapipe_pose import load_raw
    vids = pd.read_csv(OUT / "videos.csv")
    cand = pd.read_csv(OUT / "candidates.csv").drop_duplicates("id").set_index("id")
    crops, owner = [], []
    for r in vids.itertuples():
        f = PREP / "pose_mp_raw" / "agent" / f"{r.clip_id.replace(':', '_')}.npz"
        if not f.exists():
            continue
        cs = face_crops(OUT / r.video, load_raw(f), fracs=(0.5,))
        crops += cs; owner += [r.clip_id] * len(cs)
    tests = {}
    for p in sorted((ROOT / "data_test").glob("*.mp4")):
        f = ROOT / "cache" / "data_test_mp" / f"{p.stem}.npz"
        if f.exists():
            cs = face_crops(p, load_raw(f), fracs=(0.3, 0.5, 0.7))
            tests[p.stem] = cs
    E = embed_faces(crops + [c for cs in tests.values() for c in cs])
    mu = E[:len(crops)].mean(0)
    E = E - mu; E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    Ea, Et = E[:len(crops)], E[len(crops):]
    ch_of = {r.clip_id: cand.loc[r.clip_id.split(":")[-1], "channel_id"] for r in vids.itertuples() if r.clip_id.split(":")[-1] in cand.index}
    owner = np.array(owner); chans = np.array([ch_of.get(o, "?") for o in owner])
    uniq = sorted(set(chans))
    M = np.stack([Ea[chans == c].mean(0) for c in uniq]); M /= np.linalg.norm(M, axis=1, keepdims=True)
    # within-channel similarity sets the scale: two channels are one person when their means are as close as a channel to itself
    self_sim = np.array([float(np.mean(Ea[chans == c] @ M[i])) for i, c in enumerate(uniq)])
    S = M @ M.T
    parent = list(range(len(uniq)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    merges = []
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            if S[i, j] >= a.same_person * min(self_sim[i], self_sim[j]):
                parent[find(i)] = find(j); merges.append((uniq[i], uniq[j], round(float(S[i, j]), 3)))
    signer = {c: f"yt_person_{find(i):03d}" for i, c in enumerate(uniq)}
    # data_test signers vs every channel
    leak = {}
    k = 0
    for stem, cs in tests.items():
        et = Et[k:k + len(cs)].mean(0); k += len(cs)
        et /= np.linalg.norm(et) + 1e-9
        sims = M @ et
        j = int(np.argmax(sims))
        leak[stem] = dict(best_channel=uniq[j], sim=round(float(sims[j]), 3), channel_self_sim=round(float(self_sim[j]), 3),
                          same_person=bool(sims[j] >= a.same_person * self_sim[j]))
    bad_channels = {v["best_channel"] for v in leak.values() if v["same_person"]}
    # the same person as a HELD-OUT corpus signer (th_sl val/test clusters, the TSL51 researcher) → the harvested clips inherit that
    # role, so no held-out person ever trains the model through a second source
    held = {}
    thsl = pd.read_csv(PREP / "params" / "thsl_signers.csv")
    vi = pd.read_csv(PREP / "params" / "video_items.csv")
    rng_ = np.random.RandomState(0)
    for (sname, role), g in thsl[thsl.role.isin(["val", "test"])].groupby(["signer", "role"]):
        ids = rng_.choice(g.clip_id.values, min(12, len(g)), replace=False)
        held[(sname, role)] = [(DATA / "raw_data" / vi.set_index("clip_id").loc[c, "rel_path"], PREP / "pose_mp_raw" / "th_sl" / f"{c.split(':')[1]}.npz") for c in ids]
    t51 = vi[vi.dataset == "tsl51_user_sign"].sample(12, random_state=0)
    held[("TSL51 researcher", "test")] = [(DATA / "raw_data" / r.rel_path, PREP / "pose_mp_raw" / "tsl51_user_sign" / f"{Path(r.out_rel).stem}.npz")
                                         for r in t51.itertuples()]
    ref_crops, ref_owner = [], []
    for key, lst in held.items():
        for v, f in lst:
            if f.exists():
                cs = face_crops(v, load_raw(f), fracs=(0.5,)); ref_crops += cs; ref_owner += [key] * len(cs)
    inherit = {}
    if ref_crops:
        Er = embed_faces(ref_crops) - mu
        Er /= np.linalg.norm(Er, axis=1, keepdims=True) + 1e-9
        for key in held:
            m_ = [i for i, o in enumerate(ref_owner) if o == key]
            if not m_:
                continue
            c_ = Er[m_].mean(0); c_ /= np.linalg.norm(c_) + 1e-9
            ref_self = float(np.mean(Er[m_] @ c_))
            sims = M @ c_
            for i in np.flatnonzero(sims >= a.same_person * np.minimum(self_sim, ref_self)):
                inherit[uniq[i]] = dict(person=key[0], role=key[1], sim=round(float(sims[i]), 3))
    log.info("harvested channels that are a held-out corpus signer: %s", inherit or "none")
    items = pd.read_csv(OUT / "items.csv") if (OUT / "items.csv").exists() else None
    sig = pd.DataFrame(dict(channel_id=uniq, signer=[signer[c] for c in uniq], self_sim=self_sim.round(3),
                            n_clips=[int((chans == c).sum()) for c in uniq], datatest_lookalike=[c in bad_channels for c in uniq],
                            same_as_heldout=[inherit.get(c, {}).get("person") for c in uniq], inherited_role=[inherit.get(c, {}).get("role") for c in uniq]))
    sig.to_csv(OUT / "signers.csv", index=False)
    (OUT / "qc").mkdir(parents=True, exist_ok=True)
    write_json(OUT / "qc" / "identity_report.json", dict(merges=merges, data_test=leak, excluded_channels=sorted(bad_channels), heldout_corpus_signers=inherit))
    # contact sheet
    (OUT / "qc").mkdir(parents=True, exist_ok=True)
    rows = []
    for person, g in sig.groupby("signer"):
        idx = [i for i, c in enumerate(chans) if signer.get(c) == person][:8]
        tiles = [cv2.cvtColor(crops[i], cv2.COLOR_RGB2BGR) for i in idx] + [np.zeros((140, 112, 3), np.uint8)] * (8 - len(idx))
        lab = np.zeros((140, 170, 3), np.uint8)
        cv2.putText(lab, person[-3:], (5, 50), 0, 1.0, (255, 255, 255), 2)
        cv2.putText(lab, f"{len(g)} ch / {int(g.n_clips.sum())}", (5, 90), 0, 0.5, (255, 255, 255), 1)
        if g.datatest_lookalike.any():
            cv2.putText(lab, "DATA_TEST?", (5, 125), 0, 0.5, (0, 0, 255), 2)
        rows.append(np.concatenate([lab] + tiles, 1))
    for stem, cs in tests.items():
        tiles = [cv2.cvtColor(c, cv2.COLOR_RGB2BGR) for c in cs[:8]] + [np.zeros((140, 112, 3), np.uint8)] * (8 - min(8, len(cs)))
        lab = np.zeros((140, 170, 3), np.uint8); cv2.putText(lab, "data_test", (5, 70), 0, 0.6, (0, 255, 255), 2)
        rows.append(np.concatenate([lab] + tiles, 1))
    cv2.imwrite(str(OUT / "qc" / "agent_signers.jpg"), np.concatenate(rows, 0))
    log.info("signers: %d channels → %d people (%d merges); data_test look-alikes: %s", len(uniq), sig.signer.nunique(), len(merges),
             {k: (v["sim"], v["channel_self_sim"], v["same_person"]) for k, v in leak.items()})


def _frames(path, fps=12.5):
    """grey frames at `fps`: 16×16 (appearance) and 32×32 frame differences of the moving part (motion)."""
    import cv2
    cap = cv2.VideoCapture(str(path)); vf = cap.get(cv2.CAP_PROP_FPS) or 25.0; small, big = [], []; i, nxt = 0, 0.0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / vf; i += 1
        if t + 1e-6 < nxt:
            continue
        nxt = t + 1.0 / fps
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        s16 = cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        small.append((s16 - s16.mean()) / (s16.std() + 1e-3))
        big.append(cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))
    cap.release()
    if len(big) < 3:
        return None
    B = np.array(big); Dm = np.abs(np.diff(B, axis=0)).reshape(len(B) - 1, -1)
    Dm = Dm[Dm.mean(1) > np.percentile(Dm.mean(1), 40)]
    return np.array(small), (Dm - Dm.mean(1, keepdims=True)) / (Dm.std(1, keepdims=True) + 1e-3)


def _aligned_dist(A, B, min_overlap):
    best, n = 9.0, min(len(A), len(B))
    for off in range(-len(B) + 1, len(A)):
        a0, b0 = max(0, off), max(0, -off); L = min(len(A) - a0, len(B) - b0)
        if L >= max(2, min_overlap * n):
            best = min(best, float(((A[a0:a0 + L] - B[b0:b0 + L]) ** 2).mean()))
    return best


def is_copy(fa, fb, thr_look=0.05, thr_motion=1.0):
    """Same recording (re-upload / re-encode / resize)? Both the look of the frames AND the motion must match at one time offset —
    calibrated: a half-resolution, re-timed, re-compressed copy scores (0.044, 0.91); ten recordings of one word by one person in one
    room score (0.04-0.06, 1.08-1.19); different words (0.18-0.47, 1.5-1.6)."""
    if fa is None or fb is None:
        return False, (9.0, 9.0)
    d1 = _aligned_dist(fa[0], fb[0], 0.7)
    if d1 >= thr_look:
        return False, (d1, 9.0)
    d2 = _aligned_dist(fa[1], fb[1], 0.5)
    return d2 < thr_motion, (d1, d2)


def corpus_videos_by_concept():
    """concept → raw videos of the existing corpus (TTRS, th_sl, TSL51 researcher) — where a re-upload would come from."""
    m = pd.read_parquet(PREP / "manifest" / "clips.parquet")
    m = m[m.kind.eq("isolated") & m.source_file.str.endswith(".mp4", na=False)]
    out = {}
    for c, f in zip(m.concept, m.source_file):
        out.setdefault(c, []).append(DATA / f)
    vi = pd.read_csv(PREP / "params" / "video_items.csv")
    for r in vi[vi.dataset == "tsl51_user_sign"].itertuples():
        w = r.clip_id.split(":", 1)[1].split("_var_")[0]
        out.setdefault(w, []).append(DATA / "raw_data" / r.rel_path)
    return out


def sha256_of(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_label(a):
    """Sign span + model consistency for every harvested clip → items.csv (the annotation file) and qc/ contact sheets.

    status accepted when: the person is visible, 1–3 signing bursts (a lesson shows the word once or repeats it), the chosen burst
    lasts 0.3–4 s, and the clip is not an outlier of its own word (when ≥ 2 channels show the word, the clip must be closer to the
    other channels' clips than to random clips of other words) — titles are the label, the model only rejects broken segments."""
    import torch
    from modules.encoder import embed_parts, load_encoder
    from modules.parts import prep, to_iso
    from modules.schema import load
    meta_rows = []
    cand = pd.read_csv(OUT / "candidates.csv").drop_duplicates("id").set_index("id")
    sg = pd.read_csv(OUT / "signers.csv") if (OUT / "signers.csv").exists() else pd.DataFrame(columns=["channel_id", "signer", "datatest_lookalike"])
    person = dict(zip(sg.channel_id, sg.signer))
    lookalike = set(sg[sg.datatest_lookalike.astype(bool)].channel_id)
    vids = pd.read_csv(OUT / "videos.csv")
    enc = load_encoder(ROOT / "models", ROOT / "models" / "encoder.pt", device="cuda" if torch.cuda.is_available() else "cpu")
    parts, keys = [], []
    v4_ids = {p.stem[3:] for p in (DATA / "raw_data" / "word_level" / "youtube_word").glob("yt_*.mp4")}
    seen_hash, fp_rows = {}, []
    for r in vids.itertuples():
        vid = r.clip_id.split(":")[-1]
        f = PREP / "pose_mp" / "agent" / f"{r.clip_id.replace(':', '_')}.npz"
        info = cand.loc[vid] if vid in cand.index else None
        if info is None:
            continue
        row = dict(clip_id=r.clip_id, video=r.video, word=info.core, concept=info.concept, channel=info.channel, channel_id=info.channel_id,
                   signer=person.get(info.channel_id, f"yt_ch:{info.channel_id}"), title=info.title, source_url=f"https://www.youtube.com/watch?v={vid}",
                   licence="YouTube - publisher's rights, research use only")
        if not f.exists():
            meta_rows.append(dict(row, status="rejected", reason="no_pose")); continue
        if info.channel_id in lookalike:
            meta_rows.append(dict(row, status="rejected", reason="signer_looks_like_a_data_test_person")); continue
        d = load(f)
        kp, sc, hw = d["kp"], d["sc"], d["hw"]
        body = (sc[:, [5, 6]].min(1) > 0).mean()
        b = bursts(kp, sc, hw)
        row.update(n_frames=len(kp), body_frac=round(float(body), 2), n_bursts=len(b))
        if body < 0.5 or not b:
            meta_rows.append(dict(row, status="rejected", reason="no_signer" if body < 0.5 else "no_signing")); continue
        if len(b) > 3:
            meta_rows.append(dict(row, status="rejected", reason="many_bursts(phrase or several words)")); continue
        f0, f1 = max(b, key=lambda x: x[1] - x[0])                     # the main demonstration
        L = (f1 - f0) / 25.0
        row.update(f0=int(f0), f1=int(f1), span_s=round(L, 2))
        if not 0.3 <= L <= 4.0:
            meta_rows.append(dict(row, status="rejected", reason=f"span_{L:.1f}s")); continue
        if vid in v4_ids:
            meta_rows.append(dict(row, status="rejected", reason="duplicate:same_video_as_v4_youtube_word")); continue
        h = sha256_of(OUT / r.video)
        if h in seen_hash:
            meta_rows.append(dict(row, status="rejected", reason=f"duplicate:same_file_as_{seen_hash[h]}")); continue
        seen_hash[h] = r.clip_id
        fp_rows.append((len(meta_rows), r.video, info.concept))
        parts.append(prep(to_iso(kp[f0:f1], hw), sc[f0:f1])); keys.append(len(meta_rows))
        meta_rows.append(dict(row, status="pending", reason=""))
    M = pd.DataFrame(meta_rows)
    # ---- duplicates by content: a re-upload of a corpus video (or of another harvested clip) under the same word
    by_c = corpus_videos_by_concept()
    cache_fr = {}

    def fr(p):
        p = str(p)
        if p not in cache_fr:
            cache_fr[p] = _frames(p)
        return cache_fr[p]
    need = {str(OUT / v) for _, v, _ in fp_rows} | {str(p) for _, _, c in fp_rows for p in by_c.get(c, [])[:40]}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:                       # OpenCV decoding releases the GIL
        for p_, f_ in zip(need, ex.map(_frames, need)):
            cache_fr[p_] = f_
    log.info("copy check: %d harvested clips vs %d corpus / harvest videos of the same words", len(fp_rows), len(need))
    dup_log = []
    for k, (mi, video, concept) in enumerate(fp_rows):
        fa = fr(OUT / video)
        hit = None
        for pv in by_c.get(concept, [])[:40]:
            ok, d = is_copy(fa, fr(pv))
            if ok:
                hit = (str(Path(pv).name), d); break
        if hit is None:
            for mj, vj, cj in fp_rows[:k]:
                if cj == concept and M.loc[mj, "status"] == "pending":
                    ok, d = is_copy(fa, fr(OUT / vj))
                    if ok:
                        hit = (M.loc[mj, "clip_id"], d); break
        if hit:
            M.loc[mi, ["status", "reason"]] = ["rejected", f"duplicate:copy_of_{hit[0]}"]
            dup_log.append(dict(clip=M.loc[mi, "clip_id"], match=hit[0], look=round(hit[1][0], 4), motion=round(hit[1][1], 4)))
    (OUT / "qc").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dup_log, columns=["clip", "match", "look", "motion"]).to_csv(OUT / "qc" / "duplicates.csv", index=False)
    log.info("duplicates: %d by content (re-uploads), %s by id / file hash", len(dup_log),
             int(M.reason.fillna("").str.startswith("duplicate:same").sum()))
    Z = embed_parts(enc, parts, bs=64, device="cuda" if torch.cuda.is_available() else "cpu") if parts else np.zeros((0, 768))
    M["emb_i"] = -1
    M.loc[keys, "emb_i"] = np.arange(len(keys))
    # consistency: same-word clips from OTHER channels vs clips of other words
    pend = M[M.status == "pending"]
    rng = np.random.RandomState(0)
    for i, r in pend.iterrows():
        z = Z[int(r.emb_i)]
        same = pend[(pend.concept == r.concept) & (pend.channel_id != r.channel_id)]
        if len(same) == 0:
            M.loc[i, ["status", "reason", "consistency"]] = ["accepted", "only_channel_for_word", np.nan]
            continue
        s_same = float(np.mean(Z[same.emb_i.astype(int).values] @ z))
        other = pend[pend.concept != r.concept].sample(min(200, int((pend.concept != r.concept).sum())), random_state=int(rng.randint(1 << 30)))
        s_oth = Z[other.emb_i.astype(int).values] @ z
        pct = float((s_oth < s_same).mean())
        M.loc[i, "consistency"] = round(pct, 3)
        M.loc[i, ["status", "reason"]] = ["accepted", ""] if pct >= 0.5 else ["rejected", f"inconsistent_with_other_channels(pct={pct:.2f})"]
    # roles by channel: whole channels (≈ signers) are held out, so AGENT_val / AGENT_test measure NEW signers of daily words
    people = sorted(M.signer.dropna().unique())
    h = {c: (int.from_bytes(c.encode("utf-8")[-4:], "little") * 2654435761) % 100 for c in people}
    M["role"] = M.signer.map(lambda c: "test" if h.get(c, 50) < 15 else ("val" if h.get(c, 50) < 25 else "train"))
    inh = {c: r for c, r in zip(sg.channel_id, sg.get("inherited_role", pd.Series([None] * len(sg)))) if isinstance(r, str)}
    M.loc[M.channel_id.isin(inh), "role"] = M.channel_id.map(inh)
    # the person who signs the phrase lessons is the real-sentence TEST signer → every clip of that person is test (never trained)
    ph = pd.read_csv(OUT / "phrases.csv") if (OUT / "phrases.csv").exists() else pd.DataFrame(columns=["channel_id"])
    ph_people = {person.get(c) for c in ph.channel_id.dropna()} - {None}
    M.loc[M.signer.isin(ph_people), "role"] = "test"
    M.drop(columns=["emb_i"]).to_csv(OUT / "items.csv", index=False, encoding="utf-8")
    acc = M[M.status == "accepted"]
    log.info("label: %d clips → accepted %d (%d concepts, %d channels) · rejected %s", len(M), len(acc), acc.concept.nunique(), acc.channel_id.nunique(),
             M[M.status == "rejected"].reason.str.split("(").str[0].value_counts().to_dict())
    log.info("roles (accepted): %s", acc.role.value_counts().to_dict())


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover"); d.add_argument("--max_depth", type=int, default=4); d.add_argument("--per_query", type=int, default=25)
    d.add_argument("--words", default="", help="comma-separated words for a targeted round")
    d.add_argument("--min_channel_hits", type=int, default=2); d.add_argument("--max_channels", type=int, default=40)
    d = sub.add_parser("download"); d.add_argument("--limit", type=int, default=0); d.add_argument("--per_concept", type=int, default=8)
    d.add_argument("--per_channel_concept", type=int, default=1)
    sub.add_parser("videos"); sub.add_parser("candidates")
    sub.add_parser("label")
    d = sub.add_parser("signers"); d.add_argument("--same_person", type=float, default=0.9)
    a = p.parse_args()
    {"discover": cmd_discover, "download": cmd_download, "videos": cmd_videos, "signers": cmd_signers, "label": cmd_label, "candidates": cmd_candidates}[a.cmd](a)


if __name__ == "__main__":
    main()
