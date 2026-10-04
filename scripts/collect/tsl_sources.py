# -*- coding: utf-8 -*-
"""
tsl_sources.py -- Source adapters for the Thai Sign Language raw-video harvester.

Every adapter implements two phases:

    discover(ctx)        -> yields dict rows describing candidate videos
    fetch(ctx, row)      -> downloads one video, returns a FetchOutcome

Discovery is cheap and idempotent; fetching is expensive and resumable. The
orchestrator keeps them apart so an interrupted run never loses the catalogue.

Linguistic level taxonomy used throughout:
    word        -- an isolated lexical sign (dictionary entry, fingerspelling)
    sentence    -- a single utterance / short phrase, one gloss per clip
    continuous  -- running signing (news, songs, interpreted speech)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from tsl_common import (
    LOG,
    RATE,
    SHUTDOWN,
    DownloadResult,
    Http,
    Store,
    download_file,
    human_bytes,
    probe_media,
    safe_slug,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "raw_data"

LEVEL_DIRS = {
    "word": RAW / "word_level",
    "sentence": RAW / "sentence_level",
    "continuous": RAW / "continuous",
}
QUARANTINE = RAW / "_quarantine"


@dataclass
class Ctx:
    http: Http
    store: Store
    run_id: str
    limit: Optional[int] = None          # per-source cap on NEW downloads
    discover_only: bool = False
    max_video_seconds: int = 5400        # skip absurdly long streams
    youtube_max_height: int = 720


@dataclass
class FetchOutcome:
    ok: bool
    rel_path: Optional[str] = None
    bytes: int = 0
    sha256: Optional[str] = None
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    vcodec: Optional[str] = None
    acodec: Optional[str] = None
    error: Optional[str] = None
    terminal: bool = False               # do not retry on later runs
    skipped: bool = False
    breaker_open: bool = False           # never attempted -- leave the row untouched


def target_dir(level: str, subdir: str) -> Path:
    base = LEVEL_DIRS.get(level, RAW / "continuous")
    d = base / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_sidecar(path: Path, payload: Dict[str, Any]) -> None:
    """One JSON sidecar per video, holding the full upstream record. The CSV is
    the flat view; this is the lossless one."""
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("sidecar write failed for %s: %s", path.name, exc)


# ==========================================================================
#  Source 1 -- TTRS sign-language dictionary  (dic.ttrs.or.th)
# ==========================================================================
class TTRSDictionary:
    """Thai Telecommunication Relay Service sign-language vocabulary bank.

    Public JSON API behind the Angular front-end at https://dic.ttrs.or.th:
        GET  /catagory
        POST /video/search/page/{page}/rows/{rows}      body {"word": ""}
        GET  /video/search/catagory/{id}/page/{p}/rows/{r}

    Every record ships a gloss (`name`), a Thai definition, part of speech,
    signer gender, source book, and direct MP4s at 240/360/480/720p -- which
    is exactly the "already labelled" material we want.
    """

    name = "ttrs_dictionary"
    level_default = "word"
    subdir = "ttrs_dictionary"
    api = "https://nodedic.ttrs.or.th"
    referer = "https://dic.ttrs.or.th/"
    page_rows = 100
    # Above this, a clip is no longer plausibly a single lexical sign.
    WORD_MAX_SECONDS = 10.0

    # Upstream `content_type` -> our level taxonomy.
    CONTENT_TYPE_LEVEL = {
        "คำศัพท์": "word",
        "ประโยค": "sentence",
        "บทสนทนา": "continuous",
        "เรื่องเล่า": "continuous",
        "นิทาน": "continuous",
        "ข่าว": "continuous",
        "สารคดี": "continuous",
    }

    def __init__(self) -> None:
        RATE.configure("nodedic.ttrs.or.th", 0.8)
        RATE.configure("apidic.ttrs.or.th", 0.45)

    # -- discovery --------------------------------------------------------
    def discover(self, ctx: Ctx) -> Iterator[Dict[str, Any]]:
        categories = self._categories(ctx)
        LOG.info("[ttrs] %d categories in the vocabulary bank", len(categories))

        page = 1
        seen: set[str] = set()
        total_reported: Optional[int] = None

        while not SHUTDOWN.is_set():
            url = "%s/video/search/page/%d/rows/%d" % (self.api, page, self.page_rows)
            try:
                resp = ctx.http.post_json(
                    url, {"word": ""}, headers={"Referer": self.referer}
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                LOG.error("[ttrs] page %d failed permanently: %s", page, exc)
                break

            batch = data.get("result") or []
            if total_reported is None:
                total_reported = data.get("length")
                LOG.info("[ttrs] server reports %s total videos", total_reported)
            if not batch:
                break

            for rec in batch:
                vid = rec.get("id") or rec.get("vid")
                if not vid or vid in seen:
                    continue
                seen.add(vid)
                row = self._to_row(rec)
                if row:
                    yield row

            LOG.info("[ttrs] discovered %d / %s", len(seen), total_reported or "?")
            if total_reported and len(seen) >= total_reported:
                break
            if len(batch) < self.page_rows:
                break
            page += 1

    def _categories(self, ctx: Ctx) -> List[Dict[str, Any]]:
        try:
            data = ctx.http.get_json(
                "%s/catagory" % self.api, headers={"Referer": self.referer}
            )
            return data.get("result") or []
        except Exception as exc:  # noqa: BLE001
            LOG.warning("[ttrs] category listing failed (non-fatal): %s", exc)
            return []

    def _to_row(self, rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        gloss = (rec.get("name") or "").strip()
        if not gloss:
            # Unlabelled clip -- explicitly out of scope for this collection.
            return None
        url = self._best_url(rec)
        if not url:
            return None
        content_type = (rec.get("content_type") or "").strip()
        level = self.CONTENT_TYPE_LEVEL.get(content_type, self.level_default)

        # A handful of entries are tagged with a content type they clearly are
        # not -- a 3.7s clip glossed "สีเขียว" is filed as "news". Duration and
        # a single-token gloss are the more trustworthy signal, so a very short
        # clip with a one-word label is a lexical sign whatever the tag says.
        if level != "word":
            try:
                dur = float(rec.get("duration") or 0)
            except (TypeError, ValueError):
                dur = 0.0
            if 0 < dur <= self.WORD_MAX_SECONDS and len(gloss.split()) == 1:
                level = "word"

        synonyms = rec.get("synonyms")
        if isinstance(synonyms, list):
            synonyms = "; ".join(
                s.get("name", "") for s in synonyms if isinstance(s, dict)
            )

        meta = {
            "gloss_th": gloss,
            "definition_th": (rec.get("description") or "").strip(),
            "category_th": (rec.get("category") or "").strip(),
            "category_id": rec.get("catagory_id"),
            "content_type_th": content_type,
            "parts_of_speech_th": (rec.get("parts_of_speech") or "").strip(),
            "synonyms_th": synonyms or "",
            "signer_gender_th": (rec.get("gender") or "").strip(),
            "signer_creator": (rec.get("creator") or "").strip(),
            "reference_book_th": (rec.get("reference") or "").strip(),
            "translator": (rec.get("translator") or "").strip(),
            "upstream_duration": rec.get("duration"),
            "upstream_created_at": rec.get("createdAt"),
            "views": rec.get("view"),
            "vid": rec.get("vid"),
            "chosen_url": url,
            "variants": {
                k: rec.get(k) for k in
                ("urlmp4_720", "urlmp4_480", "urlmp4_360", "urlmp4_240",
                 "url_mp4", "url_m3u8", "thumbnail")
                if rec.get(k)
            },
            "source_page": "https://dic.ttrs.or.th/video/view/%s" % rec.get("id"),
        }
        return {
            "uid": "ttrs:%s" % rec.get("id"),
            "source": self.name,
            "level": level,
            "native_id": rec.get("id"),
            "url": url,
            "label": gloss,
            "label_type": "gloss",
            "meta_json": meta,
        }

    # Quality ladder, best first. 720p leads; the untranscoded original sits
    # third because it is often 1080p+ and 5-10x the bytes for no gain once
    # frames are resized for pre-training.
    URL_LADDER = ("urlmp4_720", "urlmp4_480", "url_mp4", "urlmp4_360", "urlmp4_240")

    @classmethod
    def _url_candidates(cls, rec: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        for key in cls.URL_LADDER:
            u = rec.get(key)
            if u and isinstance(u, str) and u.startswith("http") and u not in out:
                out.append(u)
        return out

    @classmethod
    def _best_url(cls, rec: Dict[str, Any]) -> Optional[str]:
        cands = cls._url_candidates(rec)
        return cands[0] if cands else None

    # -- fetch ------------------------------------------------------------
    def fetch(self, ctx: Ctx, row: Dict[str, Any]) -> FetchOutcome:
        meta = json.loads(row["meta_json"]) if row.get("meta_json") else {}
        level = row["level"]
        vid = meta.get("vid") or row["native_id"]
        stem = "ttrs_%s" % vid
        outdir = target_dir(level, self.subdir)
        dest = outdir / ("%s.mp4" % stem)

        if dest.exists() and dest.stat().st_size > 2048:
            info = probe_media(dest)
            if info.ok:
                return self._outcome_from_existing(dest, info)

        # The API advertises every rung of the quality ladder whether or not the
        # transcode actually exists -- roughly two thirds of entries 404 on
        # 720MP4 while 480MP4 serves fine. So walk the ladder at fetch time
        # instead of trusting the catalogue.
        variants = meta.get("variants") or {}
        candidates = [variants[k] for k in self.URL_LADDER if variants.get(k)]
        if row.get("url") and row["url"] not in candidates:
            candidates.insert(0, row["url"])
        if not candidates:
            return FetchOutcome(ok=False, error="no media URL in record", terminal=True)

        res: Optional[DownloadResult] = None
        errors: List[str] = []
        for url in candidates:
            if SHUTDOWN.is_set():
                return FetchOutcome(ok=False, error="shutdown")
            res = download_file(ctx.http, url, dest, referer=self.referer)
            if res.ok:
                meta["fetched_url"] = url
                meta["fetched_variant"] = next(
                    (k for k in self.URL_LADDER if variants.get(k) == url), "unknown"
                )
                break
            errors.append("%s -> %s" % (url.rsplit("/", 2)[-2:][0], res.error))
            # Only a missing rung is worth stepping down for; a transport error
            # means the whole host is unhappy, and retrying the run handles it.
            if not (res.error and ("HTTP 404" in res.error or "not media" in res.error)):
                break

        if res is None or not res.ok:
            all_404 = all("404" in e or "not media" in e for e in errors) and errors
            return FetchOutcome(
                ok=False, error=" | ".join(errors)[:400], terminal=bool(all_404)
            )

        meta["file"] = dest.name
        meta["sha256"] = res.sha256
        write_sidecar(outdir / ("%s.json" % stem), meta)

        m = res.media
        return FetchOutcome(
            ok=True,
            rel_path=str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            bytes=res.bytes, sha256=res.sha256,
            duration=m.duration if m else None,
            width=m.width if m else None, height=m.height if m else None,
            fps=m.fps if m else None, vcodec=m.vcodec if m else None,
            acodec=m.acodec if m else None,
        )

    @staticmethod
    def _outcome_from_existing(dest: Path, info) -> FetchOutcome:
        return FetchOutcome(
            ok=True,
            rel_path=str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            bytes=dest.stat().st_size, sha256=sha256_file(dest),
            duration=info.duration, width=info.width, height=info.height,
            fps=info.fps, vcodec=info.vcodec, acodec=info.acodec,
        )


# ==========================================================================
#  Source 2 -- YouTube targets (channels, playlists, keyword searches)
# ==========================================================================
@dataclass
class YtTarget:
    key: str                 # short id used in uid + folder naming
    url: str                 # channel/playlist URL, or "ytsearchN:query"
    level: str               # word | sentence | continuous
    subdir: str
    note: str = ""
    max_items: Optional[int] = None
    min_seconds: int = 2
    max_seconds: int = 5400
    # A curated channel where *every* upload is sign language. The per-title
    # relevance filter is meant for open keyword searches; applied to a channel
    # like Big Sign it throws away the majority of a good corpus, because those
    # titles name the topic ("Easy Agriculture") rather than the medium.
    trust_channel: bool = False


class YouTubeSource:
    """Harvests labelled Thai Sign Language video from YouTube via yt-dlp.

    Only targets whose titles carry the gloss or the topic are included -- the
    brief was explicitly "no bare signing with nothing telling me what it is".
    Subtitles are pulled alongside (`th`/`en`, manual and automatic) because for
    continuous material the caption track *is* the sentence-level annotation.
    """

    name = "youtube"

    TARGETS: List[YtTarget] = [
        # ---- continuous / sentence: full-screen interpreted programming ----
        YtTarget(
            key="thaipbs_bigsign",
            url="https://www.youtube.com/channel/UCFpFQ_j7AbBjD0Dpc_ZJQqQ/videos",
            level="continuous", subdir="youtube_bigsign",
            note="Thai PBS 'Big Sign' -- full-frame Thai Sign Language programming",
            trust_channel=True,
        ),
        YtTarget(
            key="ttrs_channel",
            url="https://www.youtube.com/@TTRSTHAILAND/videos",
            level="continuous", subdir="youtube_continuous",
            note="TTRS's own channel -- service explainers signed throughout",
            trust_channel=True,
        ),
        # ---- word level: labelled vocabulary series -----------------------
        YtTarget(
            key="kw_vocab",
            url="ytsearch120:ภาษามือไทย คำศัพท์",
            level="word", subdir="youtube_word",
            note="keyword harvest: Thai Sign Language vocabulary lessons",
            max_seconds=1200,
        ),
        YtTarget(
            key="kw_vocab_category",
            url="ytsearch120:ภาษามือไทย หมวดคำศัพท์ สอน",
            level="word", subdir="youtube_word",
            note="keyword harvest: vocabulary-by-category lessons",
            max_seconds=1200,
        ),
        YtTarget(
            key="kw_fingerspell",
            url="ytsearch80:ภาษามือ สะกดนิ้วมือ พยัญชนะไทย",
            level="word", subdir="youtube_word",
            note="keyword harvest: Thai fingerspelling / manual alphabet",
            max_seconds=1200,
        ),
        # ---- sentence level: phrase / conversation lessons ----------------
        YtTarget(
            key="kw_sentence",
            url="ytsearch120:ภาษามือไทย ประโยค สนทนา สอน",
            level="sentence", subdir="youtube_sentence",
            note="keyword harvest: sentence and conversation lessons",
            max_seconds=1800,
        ),
        YtTarget(
            key="kw_daily",
            url="ytsearch80:ภาษามือ ทักทาย ชีวิตประจำวัน",
            level="sentence", subdir="youtube_sentence",
            note="keyword harvest: everyday phrases",
            max_seconds=1800,
        ),
        # ---- continuous: interpreted news & public information ------------
        YtTarget(
            key="kw_news_interpreter",
            url="ytsearch100:ล่ามภาษามือ ข่าว",
            level="continuous", subdir="youtube_continuous",
            note="keyword harvest: sign-interpreted news bulletins",
            max_seconds=3600,
        ),
        YtTarget(
            key="kw_deaf_assoc",
            url="ytsearch80:สมาคมคนหูหนวกแห่งประเทศไทย ภาษามือ",
            level="continuous", subdir="youtube_continuous",
            note="keyword harvest: National Association of the Deaf in Thailand",
            max_seconds=3600,
        ),
        YtTarget(
            key="kw_gov_sign",
            url="ytsearch80:ภาษามือ ราชการ รัฐบาล แถลง",
            level="continuous", subdir="parliament",
            note="keyword harvest: government / parliamentary briefings with interpreter",
            max_seconds=3600,
        ),

        # ---- second wave: breadth for self-supervised pre-training --------
        # SSL wants volume and signer/topic diversity more than clean labels,
        # so these widen the net across genres while still requiring that the
        # title says what the clip is.
        YtTarget(
            key="kw_school",
            url="ytsearch100:ภาษามือ โรงเรียนโสตศึกษา นักเรียนหูหนวก",
            level="sentence", subdir="youtube_sentence",
            note="keyword harvest: deaf schools, classroom signing",
            max_seconds=2400,
        ),
        YtTarget(
            key="kw_story",
            url="ytsearch100:ภาษามือ นิทาน เล่าเรื่อง",
            level="continuous", subdir="youtube_continuous",
            note="keyword harvest: narrative / storytelling in sign",
            max_seconds=3600,
        ),
        YtTarget(
            key="kw_song",
            url="ytsearch100:ภาษามือ เพลง ประกอบเพลง",
            level="continuous", subdir="youtube_continuous",
            note="keyword harvest: song interpretation",
            max_seconds=1800,
        ),
        YtTarget(
            key="kw_health",
            url="ytsearch100:ภาษามือ สุขภาพ โรงพยาบาล ผู้พิการทางการได้ยิน",
            level="continuous", subdir="youtube_continuous",
            note="keyword harvest: health and public-service information",
            max_seconds=3600,
        ),
        YtTarget(
            key="kw_lesson_series",
            url="ytsearch100:สอนภาษามือไทย บทเรียน EP",
            level="word", subdir="youtube_word",
            note="keyword harvest: numbered teaching series",
            max_seconds=1800,
        ),
        YtTarget(
            key="kw_number_time",
            url="ytsearch80:ภาษามือ ตัวเลข วันเดือนปี เวลา",
            level="word", subdir="youtube_word",
            note="keyword harvest: numerals, dates, time expressions",
            max_seconds=1800,
        ),
        YtTarget(
            key="kw_en_thsl",
            url="ytsearch80:Thai sign language lesson ThSL",
            level="word", subdir="youtube_word",
            note="keyword harvest: English-titled Thai Sign Language material",
            max_seconds=1800,
        ),
        YtTarget(
            key="kw_deaf_vlog",
            url="ytsearch100:คนหูหนวก vlog ภาษามือ ชีวิต",
            level="continuous", subdir="youtube_continuous",
            note="keyword harvest: native deaf signers vlogging -- natural register",
            max_seconds=3600,
        ),
        YtTarget(
            key="kw_covid_briefing",
            url="ytsearch80:ล่ามภาษามือ แถลงข่าว ศบค กระทรวง",
            level="continuous", subdir="parliament",
            note="keyword harvest: ministry / task-force briefings with interpreter",
            max_seconds=3600,
        ),
        YtTarget(
            key="kw_religion",
            url="ytsearch60:ภาษามือ ธรรมะ ศาสนา เทศน์",
            level="continuous", subdir="youtube_continuous",
            note="keyword harvest: religious discourse in sign",
            max_seconds=3600,
        ),
    ]

    # YouTube answers sustained downloading from one IP with "Sign in to
    # confirm you're not a bot" on *every* video, regardless of player client.
    # Without a breaker the pool burns the whole queue in minutes, spending an
    # attempt on each item for nothing. Trip after this many in a row.
    BOT_BLOCK_THRESHOLD = 8
    BOT_BLOCK_MARKERS = ("not a bot",)

    def __init__(
        self,
        only_keys: Optional[List[str]] = None,
        cookies_from_browser: Optional[str] = None,
        cookies_file: Optional[str] = None,
    ) -> None:
        self.only_keys = set(only_keys) if only_keys else None
        self.ytdlp = shutil.which("yt-dlp") or "yt-dlp"
        self.cookies_from_browser = cookies_from_browser
        self.cookies_file = cookies_file
        self._consecutive_bot_blocks = 0
        self._blocked = threading.Event()
        RATE.configure("www.youtube.com", 1.2)

    def _note_result(self, stderr: str, ok: bool) -> None:
        """Track consecutive bot-blocks and trip the breaker."""
        low = (stderr or "").lower()
        if ok:
            self._consecutive_bot_blocks = 0
            return
        if any(m in low for m in self.BOT_BLOCK_MARKERS):
            self._consecutive_bot_blocks += 1
            if (self._consecutive_bot_blocks >= self.BOT_BLOCK_THRESHOLD
                    and not self._blocked.is_set()):
                self._blocked.set()
                LOG.error(
                    "YouTube is refusing this IP as a bot (%d consecutive). "
                    "Stopping the YouTube source so the queue is not burned. "
                    "Items stay pending and resume on a later run; wait for the "
                    "block to lapse, lower --workers, or pass "
                    "--yt-cookies-from-browser.",
                    self._consecutive_bot_blocks,
                )
        else:
            self._consecutive_bot_blocks = 0

    def targets(self) -> List[YtTarget]:
        if self.only_keys is None:
            return self.TARGETS
        return [t for t in self.TARGETS if t.key in self.only_keys]

    # -- discovery --------------------------------------------------------
    def discover(self, ctx: Ctx) -> Iterator[Dict[str, Any]]:
        for tgt in self.targets():
            if SHUTDOWN.is_set():
                return
            LOG.info("[yt:%s] enumerating %s", tgt.key, tgt.url)
            entries = self._flat_list(tgt)
            LOG.info("[yt:%s] %d entries returned", tgt.key, len(entries))
            kept = 0
            for e in entries:
                row = self._to_row(tgt, e, ctx)
                if row is not None:
                    kept += 1
                    yield row
            LOG.info("[yt:%s] %d/%d kept after filtering", tgt.key, kept, len(entries))

    def _flat_list(self, tgt: YtTarget) -> List[Dict[str, Any]]:
        cmd = [
            self.ytdlp, "--flat-playlist", "--dump-single-json",
            "--ignore-errors", "--no-warnings",
            "--socket-timeout", "30",
            "--retries", "10", "--extractor-retries", "5",
            "--sleep-requests", "1.0",
        ]
        if tgt.max_items:
            cmd += ["--playlist-end", str(tgt.max_items)]
        cmd += ["--", tgt.url]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=900)
        except subprocess.TimeoutExpired:
            LOG.error("[yt:%s] enumeration timed out", tgt.key)
            return []
        if not proc.stdout.strip():
            LOG.error(
                "[yt:%s] enumeration produced nothing: %s",
                tgt.key, proc.stderr.decode("utf-8", "replace")[:300],
            )
            return []
        try:
            data = json.loads(proc.stdout.decode("utf-8", "replace"))
        except json.JSONDecodeError as exc:
            LOG.error("[yt:%s] bad JSON from yt-dlp: %s", tgt.key, exc)
            return []
        return [e for e in (data.get("entries") or []) if e]

    def _to_row(
        self, tgt: YtTarget, e: Dict[str, Any], ctx: Ctx
    ) -> Optional[Dict[str, Any]]:
        vid = e.get("id")
        title = (e.get("title") or "").strip()
        if not vid or not title:
            return None
        if title.lower() in ("[private video]", "[deleted video]", "[unavailable video]"):
            return None

        # The brief: keep only material that says what is being signed. On an
        # open keyword search a title with no sign-language marker is not
        # usable; on a curated all-signing channel the marker is redundant.
        if not tgt.trust_channel and not self._is_relevant(title, e):
            return None

        dur = e.get("duration")
        if dur is not None:
            if dur < tgt.min_seconds:
                return None
            if dur > min(tgt.max_seconds, ctx.max_video_seconds):
                return None
        if e.get("live_status") in ("is_live", "is_upcoming"):
            return None

        meta = {
            "title": title,
            "description_snippet": (e.get("description") or "")[:1500],
            "channel": e.get("channel") or e.get("uploader"),
            "channel_id": e.get("channel_id") or e.get("uploader_id"),
            "upstream_duration": dur,
            "view_count": e.get("view_count"),
            "target_key": tgt.key,
            "target_note": tgt.note,
            "_subdir": tgt.subdir,
            "source_page": "https://www.youtube.com/watch?v=%s" % vid,
        }
        return {
            "uid": "yt:%s" % vid,
            "source": self.name,
            "level": tgt.level,
            "native_id": vid,
            "url": "https://www.youtube.com/watch?v=%s" % vid,
            "label": title,
            "label_type": "title",
            "meta_json": meta,
        }

    SIGN_MARKERS = (
        "ภาษามือ", "ล่าม", "คนหูหนวก", "หูหนวก", "sign language",
        "signlanguage", "big sign", "bigsign", "thsl", "asl", "ภาษาใบ้",
        "บกพร่องทางการได้ยิน",
    )

    def _is_relevant(self, title: str, e: Dict[str, Any]) -> bool:
        hay = (title + " " + (e.get("channel") or "") + " " +
               (e.get("description") or "")[:400]).lower()
        return any(m in hay for m in self.SIGN_MARKERS)

    # -- fetch ------------------------------------------------------------
    def fetch(self, ctx: Ctx, row: Dict[str, Any]) -> FetchOutcome:
        if self._blocked.is_set():
            # Neither "failed" (would burn a retry attempt on a request never
            # made) nor "skipped" (a dead-end status pending() never revisits)
            # fits here -- breaker_open leaves the row untouched so a later
            # run sees it exactly as before this one started.
            return FetchOutcome(ok=False, breaker_open=True)
        meta = json.loads(row["meta_json"]) if row.get("meta_json") else {}
        subdir = self._subdir_for_level(row["level"], meta.get("target_key"))
        outdir = target_dir(row["level"], subdir)
        vid = row["native_id"]
        stem = "yt_%s" % vid
        dest = outdir / ("%s.mp4" % stem)

        if dest.exists() and dest.stat().st_size > 2048:
            info = probe_media(dest)
            if info.ok:
                return TTRSDictionary._outcome_from_existing(dest, info)

        h = ctx.youtube_max_height
        fmt = (
            "bv*[height<=%d][ext=mp4]+ba[ext=m4a]/"
            "b[height<=%d][ext=mp4]/"
            "bv*[height<=%d]+ba/b[height<=%d]/b"
        ) % (h, h, h, h)

        cmd = [
            self.ytdlp,
            "-f", fmt,
            "--merge-output-format", "mp4",
            "--no-playlist", "--no-warnings", "--no-progress", "--no-part",
            "--write-info-json",
            "--write-subs", "--write-auto-subs",
            "--sub-langs", "th.*,en.*", "--sub-format", "vtt/best",
            # Captions are a bonus, not the payload. YouTube rate-limits the
            # timedtext endpoint far harder than media, and without this a 429
            # on a subtitle track aborts the whole item before the video is
            # even attempted -- which is exactly what happened on the first
            # pass. Keep going, and space the caption requests out.
            "--ignore-errors", "--sleep-subtitles", "3",
            "--socket-timeout", "30",
            "--retries", "10", "--fragment-retries", "10",
            "--extractor-retries", "5", "--file-access-retries", "5",
            # Politeness: yt-dlp's own spacing, on top of our host limiter.
            "--sleep-requests", "2.0",
            "--sleep-interval", "3", "--max-sleep-interval", "15",
            # Deno (installed alongside this harvester) solves YouTube's JS
            # challenges; without this flag yt-dlp won't fetch the solver
            # script and silently mis-reports many playable videos as
            # "not available" / "Please sign in". See yt-dlp wiki/EJS.
            "--remote-components", "ejs:github",
            "--match-filter", "!is_live",
            "-o", str(outdir / (stem + ".%(ext)s")),
        ]
        # Opt-in only. Authenticating as a real account is the user's call to
        # make, not the collector's -- bulk downloading while signed in puts
        # that account at risk.
        if self.cookies_from_browser:
            cmd += ["--cookies-from-browser", self.cookies_from_browser]
        elif self.cookies_file:
            cmd += ["--cookies", self.cookies_file]
        cmd += ["--", row["url"]]

        RATE.acquire("https://www.youtube.com/")
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=3600)
        except subprocess.TimeoutExpired:
            self._cleanup_partials(outdir, stem)
            self._note_result("", False)
            return FetchOutcome(ok=False, error="yt-dlp timeout after 3600s")

        stderr = proc.stderr.decode("utf-8", "replace")
        produced = self._find_video(outdir, stem)
        self._note_result(stderr, produced is not None)

        if produced is None:
            self._cleanup_partials(outdir, stem)
            return FetchOutcome(
                ok=False,
                error=(stderr[-400:] or "yt-dlp produced no file"),
                terminal=self._is_terminal(stderr),
            )

        info = probe_media(produced)
        if not info.ok:
            QUARANTINE.mkdir(parents=True, exist_ok=True)
            with_suffix = QUARANTINE / produced.name
            try:
                shutil.move(str(produced), str(with_suffix))
            except OSError:
                pass
            return FetchOutcome(
                ok=False, error="quarantined, unreadable media: %s" % info.error,
                terminal=True,
            )

        # Fold yt-dlp's own info.json into our sidecar, then drop the bulky original.
        self._merge_info_json(outdir, stem, meta, produced)

        digest = sha256_file(produced)
        return FetchOutcome(
            ok=True,
            rel_path=str(produced.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            bytes=produced.stat().st_size, sha256=digest,
            duration=info.duration, width=info.width, height=info.height,
            fps=info.fps, vcodec=info.vcodec, acodec=info.acodec,
        )

    # Subdirectories that are meaningful provenance labels within `continuous`.
    CONTINUOUS_SUBDIRS = {"youtube_bigsign", "parliament", "youtube_continuous"}

    def _subdir_for_level(self, level: str, key: Optional[str]) -> str:
        """Keep the folder name consistent with the level it sits under.

        A video can be discovered by more than one target (a clip found by both
        a word-level and a sentence-level query, say). The catalogue keeps the
        level from first discovery, but the target key follows the most recent
        one -- which produced paths like `word_level/youtube_sentence/`. The
        level directory was always right; only the label was stale. Derive the
        label from the level so the two can never disagree again.
        """
        if level == "word":
            return "youtube_word"
        if level == "sentence":
            return "youtube_sentence"
        target = next((t.subdir for t in self.TARGETS if t.key == key), None)
        if target in self.CONTINUOUS_SUBDIRS:
            return target
        return "youtube_continuous"

    @staticmethod
    def _find_video(outdir: Path, stem: str) -> Optional[Path]:
        for ext in (".mp4", ".mkv", ".webm"):
            p = outdir / (stem + ext)
            if p.exists() and p.stat().st_size > 2048:
                return p
        return None

    @staticmethod
    def _cleanup_partials(outdir: Path, stem: str) -> None:
        for p in outdir.glob(stem + "*"):
            if p.suffix in (".part", ".ytdl", ".temp") or p.name.endswith(".part"):
                try:
                    p.unlink()
                except OSError:
                    pass

    TERMINAL_PATTERNS = (
        "video unavailable", "private video", "removed by the uploader",
        "account associated with this video has been terminated",
        "this video is not available", "members-only", "sign in to confirm your age",
        "has been removed", "video has been removed", "unavailable in your country",
        "copyright", "no video formats found",
    )

    def _is_terminal(self, stderr: str) -> bool:
        low = stderr.lower()
        return any(p in low for p in self.TERMINAL_PATTERNS)

    @staticmethod
    def _merge_info_json(
        outdir: Path, stem: str, meta: Dict[str, Any], video: Path
    ) -> None:
        info_path = outdir / (stem + ".info.json")
        merged = dict(meta)
        merged["file"] = video.name
        if info_path.exists():
            try:
                raw = json.loads(info_path.read_text(encoding="utf-8"))
                keep = (
                    "id", "title", "description", "channel", "channel_id",
                    "uploader", "upload_date", "duration", "view_count",
                    "like_count", "categories", "tags", "language",
                    "webpage_url", "fps", "width", "height", "license",
                )
                merged["youtube"] = {k: raw.get(k) for k in keep if raw.get(k) is not None}
                subs = sorted(
                    set(list((raw.get("subtitles") or {}).keys()))
                )
                autos = sorted(
                    set(list((raw.get("automatic_captions") or {}).keys()))
                )
                merged["subtitles_manual"] = subs
                merged["subtitles_auto_available"] = [
                    a for a in autos if a.startswith(("th", "en"))
                ]
            except Exception:  # noqa: BLE001
                pass
            finally:
                try:
                    info_path.unlink()
                except OSError:
                    pass

        merged["subtitle_files"] = sorted(
            p.name for p in outdir.glob(stem + "*.vtt")
        )
        write_sidecar(outdir / (stem + ".json"), merged)


# ==========================================================================
#  Registry
# ==========================================================================
def build_sources(names: Optional[List[str]] = None,
                  yt_keys: Optional[List[str]] = None,
                  yt_cookies_from_browser: Optional[str] = None,
                  yt_cookies_file: Optional[str] = None) -> List[Any]:
    all_sources = {
        "ttrs_dictionary": TTRSDictionary,
        "youtube": lambda: YouTubeSource(
            only_keys=yt_keys,
            cookies_from_browser=yt_cookies_from_browser,
            cookies_file=yt_cookies_file,
        ),
    }
    chosen = names or list(all_sources)
    out = []
    for n in chosen:
        factory = all_sources.get(n)
        if factory is None:
            LOG.warning("unknown source '%s' -- skipping", n)
            continue
        out.append(factory())
    return out
