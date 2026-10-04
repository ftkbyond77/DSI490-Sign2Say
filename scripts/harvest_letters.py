"""Fingerspelling data: YouTube alphabet lessons → letter clips with labels (cloud_s3/agent_data/en_alpha, th_alpha).

English letters in Thai Sign Language are spelled with the one-handed American manual alphabet (the Thai schools for the deaf teach
it as "แบบสะกดนิ้วมือตัวอักษร A-Z"), so English data = A–Z alphabet lessons of Thai deaf-school teachers + ASL teachers (same
handshapes; recorded in `source`). Thai consonants come from TSL-ONE-S (25 signers × 42 letters, landmarks) and Thai alphabet lessons.

  python scripts/harvest_letters.py discover            # yt-dlp search → <set>/discover.csv
  python scripts/harvest_letters.py select              # title rules → <set>/candidates.csv
  python scripts/harvest_letters.py download            # 480p mp4 + metadata
  python scripts/harvest_letters.py extract             # MediaPipe Holistic → <set>/pose_mp_raw/<id>.npz
  python scripts/harvest_letters.py label               # holds of the signing hand → letters in alphabet order → <set>/items.csv

A lesson shows the letters in order, each held still for a moment. `label` finds the holds and keeps a video only when its holds
can be matched to the alphabet in order (dynamic programming over hold-handshape similarity, seeded by videos whose number of holds
equals the alphabet length), so labels never come from guessing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from modules.utils import DATA, get_logger

log = get_logger("letters")
AGENT = DATA / "agent_data"
SETS = {"en": AGENT / "en_alpha", "th": AGENT / "th_alpha"}
EN = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
TH = list("กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")
YTDLP = "yt-dlp"

QUERIES = {
    "en": ["ASL alphabet", "ASL alphabet A to Z", "ASL alphabet slow", "learn the ASL alphabet", "sign language alphabet A-Z",
           "ASL fingerspelling alphabet", "ASL alphabet for beginners", "ภาษามือ ตัวอักษรภาษาอังกฤษ", "ภาษามือ A-Z", "สะกดนิ้วมือ ภาษาอังกฤษ",
           "แบบสะกดนิ้วมือตัวอักษร A-Z", "สะกดนิ้วมือ A-Z", "ภาษามือ ABC"],
    "th": ["ภาษามือ ก-ฮ", "ภาษามือไทย พยัญชนะ", "สะกดนิ้วมือไทย", "แบบสะกดนิ้วมือตัวอักษร ก-ฮ", "สะกดนิ้วมือ ก-ฮ", "ภาษามือ พยัญชนะไทย 44 ตัว",
           "ตัวอักษรภาษามือไทย", "Thai fingerspelling", "ภาษามือ สระ", "สะกดนิ้วมือ สระ"],
}
EN_OK = re.compile(r"A\s*[-–~]?\s*Z|ABC|alphabet|ASL|อังกฤษ|English|American", re.I)       # an English-letter marker is required
EN_THAI = re.compile(r"ก\s*[-–~]\s*ฮ|พยัญชนะ|สะกดนิ้วมือไทย|นิ้วมือภาษาไทย|ภาษาไทย|คำศัพท์สะกด|Bebefinn|เพลง")  # Thai alphabet, words, songs
EN_BAD = re.compile(r"BSL|British|Auslan|Korean|French|NZ\b|New Zealand|SASL|GSL|Filipino|Deafblind|Japanese|Lao|Finnish|quiz|word|names|"
                    r"phonics|tips|rules|double|dos and|mistakes|memory|practice|exercise|story|song|chant|dance|rap|nato|1\s*-\s*10|ตัวเลข", re.I)
TH_OK = re.compile(r"ก\s*[-–~]\s*ฮ|พยัญชนะ|สะกดนิ้ว|fingerspell|ตัวอักษร.*ไทย|สระ", re.I)
TH_BAD = re.compile(r"A\s*[-–]\s*Z|ASL|English|อังกฤษ|ตัวเลข|เพลง|song|ร้อง|Lao|ลาว|Japanese|การ์ตูน|cartoon|KidsMeSong|ฝึกอ่าน|ฝึกเขียน|อ่าน", re.I)


def run(cmd, timeout=180):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")          # yt-dlp prints titles in the console code page otherwise
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)
    return r.stdout


def search(query, n=40):
    out = run([YTDLP, "--flat-playlist", "--print", "%(id)s\t%(duration)s\t%(channel_id)s\t%(channel)s\t%(title)s", f"ytsearch{n}:{query}"])
    rows = []
    for line in out.splitlines():
        p = line.split("\t")
        if len(p) == 5:
            rows.append(dict(id=p[0], duration=float(p[1]) if p[1] not in ("NA", "None", "") else np.nan, channel_id=p[2], channel=p[3],
                             title=p[4], query=query))
    return rows


def cmd_discover(a):
    for s, qs in QUERIES.items():
        rows = [r for q in qs for r in search(q, a.n)]
        df = pd.DataFrame(rows).drop_duplicates("id")
        SETS[s].mkdir(parents=True, exist_ok=True)
        df.to_csv(SETS[s] / "discover.csv", index=False, encoding="utf-8")
        log.info("[%s] %d videos, %d channels", s, len(df), df.channel_id.nunique())


def cmd_select(a):
    for s in SETS:
        df = pd.read_csv(SETS[s] / "discover.csv").fillna({"title": "", "channel": ""})
        ok, bad = (EN_OK, EN_BAD) if s == "en" else (TH_OK, TH_BAD)
        keep = df[df.title.str.contains(ok) & ~df.title.str.contains(bad) & df.duration.between(20, 900)]
        if s == "en":
            keep = keep[~keep.title.str.contains(EN_THAI)]
        keep = keep.assign(source=np.where(keep.title.str.contains(r"[ก-๙]") | keep.channel.str.contains(r"[ก-๙]"), "thai_channel", "asl_channel"))
        keep.to_csv(SETS[s] / "candidates.csv", index=False, encoding="utf-8")
        log.info("[%s] %d candidates (%d channels): %s", s, len(keep), keep.channel_id.nunique(), keep.source.value_counts().to_dict())


def cmd_download(a):
    from concurrent.futures import ThreadPoolExecutor
    for s in SETS:
        d = SETS[s]
        cand = pd.read_csv(d / "candidates.csv")
        if (d / "single_letter_candidates.csv").exists():       # "ASL letter X" clips: one letter, labelled by the title
            cand = pd.concat([cand, pd.read_csv(d / "single_letter_candidates.csv")], ignore_index=True).drop_duplicates("id")
        (d / "videos").mkdir(exist_ok=True); (d / "meta").mkdir(exist_ok=True)

        def fetch(vid):
            mp4 = d / "videos" / f"{vid}.mp4"
            if mp4.exists():
                return
            run([YTDLP, "-q", "--no-warnings", "-f", "bv*[height<=480][ext=mp4]/b[height<=480][ext=mp4]/bv*[height<=480]/b", "--remux-video", "mp4",
                 "-o", str(d / "videos" / "%(id)s.%(ext)s"), "--write-info-json", "--no-write-comments", f"https://www.youtube.com/watch?v={vid}"], timeout=600)
            for j in (d / "videos").glob(f"{vid}*.info.json"):
                info = json.loads(j.read_text(encoding="utf-8"))
                keep = {k: info.get(k) for k in ("id", "title", "channel", "channel_id", "upload_date", "duration", "webpage_url", "license", "width", "height", "fps")}
                (d / "meta" / f"{vid}.json").write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
                j.unlink()
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(fetch, list(cand.id)))
        log.info("[%s] %d / %d videos on disk", s, len(list((d / "videos").glob("*.mp4"))), len(cand))


def _extract_one(args):
    video, out = args
    from modules.mediapipe_pose import extract, save_raw
    if out.exists():
        return out.name, "cached"
    try:
        raw = extract(video, fps=12.5, max_side=640)          # letters are held ~1 s: 12.5 fps keeps every hold, 3–4× faster
        if raw is None:
            return out.name, "no frames"
        save_raw(out, raw)
        return out.name, "ok"
    except Exception as e:  # noqa: BLE001
        return out.name, f"error {type(e).__name__}"


def cmd_extract(a):
    from multiprocessing import Pool
    jobs = []
    for s in SETS:
        d = SETS[s]; (d / "pose_mp_raw").mkdir(exist_ok=True)
        jobs += [(v, d / "pose_mp_raw" / f"{v.stem}.npz") for v in sorted((d / "videos").glob("*.mp4"))]
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for i, (name, st) in enumerate(pool.imap_unordered(_extract_one, jobs), 1):
            if st != "cached":
                log.info("  %d/%d %s %s", i, len(jobs), name, st)


# ---------------------------------------------------------------------------------------------------------------- labelling
def standard_from_raw(raw):
    """MediaPipe raw → Standard Schema (25 fps) with the active hand canonicalised to the right → (kpi, sc)."""
    from modules.mediapipe_pose import to_standard
    from modules.parts import to_iso
    from modules.schema import canonical_hands
    d = to_standard(raw)
    kp, sc, hw = d["kp"], d["sc"], tuple(int(x) for x in d["hw"])
    kp, sc, _ = canonical_hands(kp, sc, hw)
    return to_iso(kp, hw).astype(np.float32), sc.astype(np.float32)


def holds(kpi, sc, v_thr=0.07, min_len=4, merge_gap=2):
    """Still moments of the dominant (right) hand while it is up: [(f0, f1)] at 25 fps."""
    from modules.schema import RH
    dom = kpi[:, RH]
    ok = sc[:, RH].mean(1) > 0.3
    palm = np.linalg.norm(dom[:, 9] - dom[:, 0], axis=1) + 1e-6
    v = np.r_[9.0, np.linalg.norm(np.diff(dom, axis=0), axis=2).mean(1) / palm[1:]]
    v[~ok] = 9.0
    v = np.convolve(v, np.ones(3) / 3, "same")
    sh_y = (kpi[:, 5, 1] + kpi[:, 6, 1]) / 2
    sw = np.linalg.norm(kpi[:, 5] - kpi[:, 6], axis=1) + 1e-6
    up = dom[:, 0, 1] < sh_y + 1.6 * sw                                    # wrist above the lower chest
    still = ok & up & (v < v_thr)
    segs, i, T = [], 0, len(still)
    while i < T:
        if still[i]:
            j = i
            while j + 1 < T and still[j + 1]:
                j += 1
            segs.append([i, j + 1])
            i = j + 1
        else:
            i += 1
    merged = []
    for a_, b_ in segs:
        if merged and a_ - merged[-1][1] <= merge_gap:
            merged[-1][1] = b_
        else:
            merged.append([a_, b_])
    return [(a_, b_) for a_, b_ in merged if b_ - a_ >= min_len]


def hold_vectors(kpi, sc, hs):
    """One feature vector per hold (window features over the hold, padded by 2 frames)."""
    from modules.fingerspell import window_features
    out = []
    for f0, f1 in hs:
        a_, b_ = max(0, f0 - 2), min(len(kpi), f1 + 2)
        X, _, keep = window_features(kpi[a_:b_], sc[a_:b_], starts=np.array([0, max(0, (b_ - a_) - 8)]))
        out.append(X[keep].mean(0) if keep.any() else None)
    return out


def align(S, gap_hold=-0.4, gap_letter=-1.2):
    """Monotonic alignment of holds (rows) to the alphabet (columns) maximising the summed log-score: a hold may be skipped (an
    intro, a repeated letter), a letter may be skipped (J / Z drawn without a pause). → [(hold index, letter index)], score."""
    M, K = S.shape
    D = np.full((M + 1, K + 1), -1e9)
    D[0, :] = np.arange(K + 1) * gap_letter
    D[:, 0] = np.arange(M + 1) * gap_hold
    B = np.zeros((M + 1, K + 1), np.int8)
    for i in range(1, M + 1):
        for j in range(1, K + 1):
            c = [D[i - 1, j - 1] + S[i - 1, j - 1], D[i - 1, j] + gap_hold, D[i, j - 1] + gap_letter]
            k = int(np.argmax(c))
            D[i, j] = c[k]
            B[i, j] = k
    i, j, pairs = M, K, []
    while i > 0 and j > 0:
        if B[i, j] == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif B[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1], float(D[M, K])


def tslone_letter_vectors(letters, roles=("train", "val", "test")):
    """TSL-ONE-S letter clips (25 signers) → one vector per clip (the middle of the clip) and the letter index."""
    from modules.fingerspell import window_features
    from modules.parts import to_iso
    m = pd.read_parquet(DATA / "data_prep" / "manifest" / "clips.parquet")
    m = m[(m.dataset == "tslone") & m.concept.isin(letters) & m.role.isin(roles)]
    X, y = [], []
    for r in m.itertuples():
        z = np.load(DATA / "data_prep" / r.pose_file)
        kpi = to_iso(z["kp"], tuple(int(v) for v in z["hw"])).astype(np.float32)
        sc = z["sc"].astype(np.float32)
        T = len(kpi)
        Xw, _, keep = window_features(kpi, sc, starts=np.array([max(0, T // 2 - 8), max(0, T // 2 - 4)]))
        if keep.any():
            X.append(Xw[keep].mean(0))
            y.append(letters.index(r.concept))
    return np.stack(X), np.array(y)


def _fit_hold_model(s, vids, labels, letters):
    """Hold vector → P(letter) [M,K]: logistic regression on the labelled holds (+ TSL-ONE-S letters for Thai)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = [], []
    for k, lab in labels.items():
        for x, j in zip(vids[k]["X"], lab):
            if j >= 0:
                X.append(x)
                y.append(j)
    if s == "th":
        Xo, yo = tslone_letter_vectors(letters)
        X += list(Xo)
        y += list(yo)
    X, y = np.stack(X), np.array(y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)).fit(X, y)
    cols = list(clf.classes_)

    def model(Xq):
        P = np.full((len(Xq), len(letters)), 1e-4)
        P[:, cols] = clf.predict_proba(Xq)
        return P / P.sum(1, keepdims=True)
    return model


# The seed lesson's holds checked by eye against the letter shown on screen (index = hold, value = letter, "" = not a letter:
# a repeated A, a movement between D and E). H and Q have no hold in this lesson — order alone would shift every later label.
SEED_EN = {"gOwZE2kapS8": ["A", "", "B", "C", "D", "", "E", "F", "G", "I", "J", "K", "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W",
                           "X", "Y", "Z"],
           # a Thai teacher's A–Z lesson with every letter on screen (J drawn over two holds, P held twice)
           "_T04Jsak7vo": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "J", "K", "L", "M", "N", "O", "P", "P", "Q", "R", "S", "T", "U", "V",
                           "W", "X", "Y", "Z"]}


def _label_by_prototypes(vids, seed, K=26, rounds=4, min_pairs=22, min_agree=0.5, min_cos=0.45):
    """English bootstrap: letter templates (mean standardised hold vector per letter) start from the seed videos (exactly K holds,
    labelled in alphabet order); every other lesson is aligned to them in alphabet order (cosine similarity) and kept when ≥ min_pairs
    letters are matched, the matched holds look like their letters (mean cosine ≥ min_cos) and the best template agrees with the
    order on ≥ min_agree of them; kept videos join the templates, repeated `rounds` times."""
    allX = np.concatenate([v["X"] for v in vids.values()])
    mu, sd = allX.mean(0), allX.std(0) + 1e-4

    def nz(X):
        Y = (X - mu) / sd
        return Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    labels = dict(seed)
    seed = list(seed)
    for _ in range(rounds):
        T, cnt = np.zeros((K, allX.shape[1])), np.zeros(K)
        for k, lab in labels.items():
            for x, j in zip(nz(vids[k]["X"]), lab):
                if j >= 0:
                    T[j] += x; cnt[j] += 1
        T = T / np.maximum(cnt, 1)[:, None]
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-8
        new = {k: labels[k] for k in seed}
        for k, v in vids.items():
            if k in seed:
                continue
            C = nz(v["X"]) @ T.T
            pairs, _ = align((C - 0.45) * 6.0)
            agree = float(np.mean([C[i].argmax() == j for i, j in pairs])) if pairs else 0.0
            cos = float(np.mean([C[i, j] for i, j in pairs])) if pairs else 0.0
            v["agree"], v["matched"], v["cos"] = agree, len(pairs), cos
            if len(pairs) >= min_pairs and agree >= min_agree and cos >= min_cos:
                lab = [-1] * len(v["holds"])
                for i, j in pairs:
                    lab[i] = j
                new[k] = lab
        labels = new
    # clean-up: a non-seed hold keeps its letter only if that letter is among its 2 most similar templates (a hold shifted onto its
    # neighbour's letter fails this); a lesson where < 70 % of the holds pass is dropped (checked by eye on per-letter sheets)
    out = {k: labels[k] for k in seed}
    for k, lab in labels.items():
        if k in seed:
            continue
        C = nz(vids[k]["X"]) @ T.T
        rank = np.argsort(-C, axis=1)
        new_lab = [j if j >= 0 and j in rank[i, :2] else -1 for i, j in enumerate(lab)]
        if sum(j >= 0 for j in new_lab) >= 0.7 * sum(j >= 0 for j in lab):
            out[k] = new_lab
    return out


def cmd_label(a):
    """holds → letters. Thai: holds scored by the TSL-ONE-S letter model (25 signers). English: bootstrap — videos with exactly 26
    holds are labelled in order; a model trained on them scores the others, which are aligned and kept when >= 80 % of the alphabet
    is matched and the model agrees on >= 70 % of the matched holds; two rounds."""
    import hashlib
    from modules.mediapipe_pose import load_raw
    from modules.fingerspell import EN_LETTERS, TH_LETTERS
    for s in (a.sets.split(",") if a.sets else SETS):
        d = SETS[s]
        letters = EN_LETTERS if s == "en" else [c for c in TH_LETTERS if not c.startswith("สระ")]
        cand = pd.read_csv(d / "candidates.csv").set_index("id")
        vids = {}
        for f in sorted((d / "pose_mp_raw").glob("*.npz")):
            if f.stem not in cand.index:
                continue
            kpi, sc = standard_from_raw(load_raw(f))
            hs = holds(kpi, sc)
            if len(hs) < len(letters) * 0.6:
                continue
            V = hold_vectors(kpi, sc, hs)
            ok = [i for i, v in enumerate(V) if v is not None]
            if len(ok) >= len(letters) * 0.6:
                vids[f.stem] = dict(holds=[hs[i] for i in ok], X=np.stack([V[i] for i in ok]))
        log.info("[%s] %d videos with enough holds", s, len(vids))
        labels, singles = {}, {}
        sf = d / "single_letter_candidates.csv"
        if sf.exists():                 # single-letter clips: the longest hold of the hand is the letter named in the title
            sl = pd.read_csv(sf).drop_duplicates("id").set_index("id")
            for vid, r in sl.iterrows():
                f = d / "pose_mp_raw" / f"{vid}.npz"
                if not f.exists() or r["letter"] not in letters:
                    continue
                kpi, sc = standard_from_raw(load_raw(f))
                hs = holds(kpi, sc)
                if not hs:
                    continue
                h = max(hs, key=lambda x: x[1] - x[0])
                V = hold_vectors(kpi, sc, [h])
                if V[0] is not None:
                    singles[vid] = dict(holds=[h], X=V[0][None], letter=letters.index(r["letter"]))
            cand = pd.concat([cand, sl[["channel_id", "channel"]].assign(source="asl_channel")])
            cand = cand[~cand.index.duplicated()]
            log.info("[%s] %d single-letter clips (%d channels)", s, len(singles), sl.loc[list(singles)].channel_id.nunique() if singles else 0)
        if s == "en":
            labels = {k: [letters.index(c) if c else -1 for c in lab] for k, lab in SEED_EN.items()
                      if k in vids and len(vids[k]["holds"]) == len(lab)}
            log.info("[en] seed: %d checked lesson(s) + %d single-letter clips", len(labels), len(singles))
        fixed = {k: [v["letter"]] for k, v in singles.items()}
        vids.update(singles)
        if s == "en":
            labels = _label_by_prototypes(vids, labels, K=len(letters)) if labels else {}
            log.info("[en] %d lesson videos labelled by prototype alignment", len(labels))
        for r in range(0 if s == "en" else 1):
            if s == "en" and not labels and not fixed:
                log.warning("[en] no seed — no labels")
                break
            model = _fit_hold_model(s, vids, {**labels, **fixed}, letters)
            new = {}
            for k, v in vids.items():
                if k in fixed:
                    continue
                P = model(v["X"])
                pairs, _ = align(np.log(P + 1e-4) + 1.5)          # +1.5: matching a hold to its letter beats skipping both
                agree = float(np.mean([P[i].argmax() == j for i, j in pairs])) if pairs else 0.0
                v["agree"], v["matched"] = agree, len(pairs)
                if len(pairs) >= 0.8 * len(letters) and agree >= (0.6 if (s == "en" and r == 0) else 0.7):
                    lab = [-1] * len(v["holds"])
                    for i, j in pairs:
                        lab[i] = j
                    new[k] = lab
            if s == "en":                                    # the seed videos (exact hold count) stay labelled
                new = {**{k: v for k, v in labels.items() if len(vids[k]["holds"]) == len(letters)}, **new}
            labels = new
            log.info("[%s] round %d: %d lesson videos labelled (+%d single-letter clips)", s, r + 1, len(labels), len(fixed))
        labels = {**labels, **fixed}
        rows = []
        for k, lab in labels.items():
            meta = cand.loc[k] if k in cand.index else None
            for (f0, f1), j in zip(vids[k]["holds"], lab):
                if j >= 0:
                    rows.append(dict(video=k, letter=letters[j], f0=int(f0), f1=int(f1),
                                     channel_id=str(meta["channel_id"]) if meta is not None else "", channel=str(meta["channel"]) if meta is not None else "",
                                     source=str(meta["source"]) if meta is not None else "", agree=round(vids[k].get("agree", 1.0), 3)))
        items = pd.DataFrame(rows)
        if len(items):     # roles by channel (one channel ~ one signer): 1 in 5 channels is held out for testing
            items["role"] = items.channel_id.map(lambda c: "test" if int(hashlib.md5(str(c).encode()).hexdigest(), 16) % 5 == 0 else "train")
        items.to_csv(d / "items.csv", index=False, encoding="utf-8")
        pd.DataFrame([dict(video=k, holds=len(v["holds"]), matched=v.get("matched"), agree=v.get("agree"), labelled=k in labels)
                      for k, v in vids.items()]).to_csv(d / "label_report.csv", index=False)
        if len(items):
            log.info("[%s] items: %d letter holds, %d videos, %d channels (test %d); fewest per letter %d", s, len(items), items.video.nunique(),
                     items.channel_id.nunique(), items[items.role == "test"].channel_id.nunique(), items.letter.value_counts().min())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("discover"); p.add_argument("--n", type=int, default=40)
    sub.add_parser("select")
    sub.add_parser("download")
    p = sub.add_parser("extract"); p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("label"); p.add_argument("--sets", default="")
    a = ap.parse_args()
    dict(discover=cmd_discover, select=cmd_select, download=cmd_download, extract=cmd_extract, label=cmd_label)[a.cmd](a)


if __name__ == "__main__":
    main()
