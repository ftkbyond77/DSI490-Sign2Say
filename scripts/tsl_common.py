# -*- coding: utf-8 -*-
"""
tsl_common.py -- Shared infrastructure for the Thai Sign Language raw-video harvester.

Provides:
  * UTF-8 safe console / file IO on Windows (cp874 default breaks Thai output)
  * A thread-safe, per-host token-bucket rate limiter with an adaptive brake
  * An HTTP session with exponential backoff + full jitter, Retry-After aware
  * A SQLite-backed job/state store giving idempotent resume across runs
  * Atomic file download with integrity verification (sha256 + ffprobe)
  * Structured logging to both console and a rotating log file
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, urlsplit, urlunsplit


# --------------------------------------------------------------------------
# 0. Windows / Thai text safety.  Must run before any print().
# --------------------------------------------------------------------------
def force_utf8() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            with contextlib.suppress(Exception):
                stream.reconfigure(encoding="utf-8", errors="replace")


force_utf8()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# 1. Logging
# --------------------------------------------------------------------------
_LOG_CONFIGURED = False


def get_logger(name: str = "tsl", logfile: Optional[Path] = None) -> logging.Logger:
    global _LOG_CONFIGURED
    logger = logging.getLogger("tsl")
    if not _LOG_CONFIGURED:
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(threadName)-12s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(fmt)
        logger.addHandler(console)

        logfile = logfile or (PROJECT_ROOT / "logs" / "harvest.log")
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fileh = RotatingFileHandler(
            logfile, maxBytes=25 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        fileh.setLevel(logging.DEBUG)
        fileh.setFormatter(fmt)
        logger.addHandler(fileh)
        _LOG_CONFIGURED = True
    return logger.getChild(name) if name != "tsl" else logger


LOG = get_logger()


# --------------------------------------------------------------------------
# 2. Cooperative shutdown
# --------------------------------------------------------------------------
SHUTDOWN = threading.Event()


def install_signal_handlers() -> None:
    import signal

    def _handler(signum, frame):  # noqa: ARG001
        if SHUTDOWN.is_set():
            LOG.warning("Second interrupt -- exiting hard.")
            os._exit(130)
        LOG.warning(
            "Interrupt received. Finishing in-flight work, then stopping. "
            "State is checkpointed; re-run to resume."
        )
        SHUTDOWN.set()

    for sig in ("SIGINT", "SIGTERM", "SIGBREAK"):
        s = getattr(signal, sig, None)
        if s is not None:
            with contextlib.suppress(Exception):
                signal.signal(s, _handler)


# --------------------------------------------------------------------------
# 3. Per-host rate limiting
# --------------------------------------------------------------------------
class HostRateLimiter:
    """Serialises requests per host to at most one every `min_interval` seconds
    plus uniform jitter, so a burst of worker threads still looks polite.

    A host that answers 429/503 gets an extra cooldown that decays
    geometrically on success -- an adaptive brake on top of the static spacing.
    """

    def __init__(self, default_interval: float = 0.6, jitter: float = 0.35):
        self.default_interval = default_interval
        self.jitter = jitter
        self._lock = threading.Lock()
        self._next_ok: Dict[str, float] = {}
        self._intervals: Dict[str, float] = {}
        self._penalty: Dict[str, float] = {}

    def configure(self, host: str, interval: float) -> None:
        with self._lock:
            self._intervals[host.lower()] = interval

    def _interval(self, host: str) -> float:
        return self._intervals.get(host, self.default_interval)

    def acquire(self, url: str) -> None:
        host = urlsplit(url).netloc.lower()
        while True:
            with self._lock:
                now = time.monotonic()
                ready = self._next_ok.get(host, 0.0)
                if now >= ready:
                    wait = self._interval(host) + random.uniform(0, self.jitter)
                    wait += self._penalty.get(host, 0.0)
                    self._next_ok[host] = now + wait
                    return
                sleep_for = ready - now
            # Sleep outside the lock, in slices, so shutdown stays responsive.
            slept = 0.0
            while slept < sleep_for:
                if SHUTDOWN.is_set():
                    return
                chunk = min(0.25, sleep_for - slept)
                time.sleep(chunk)
                slept += chunk

    def penalise(self, url: str, seconds: float) -> None:
        host = urlsplit(url).netloc.lower()
        with self._lock:
            cur = self._penalty.get(host, 0.0)
            self._penalty[host] = min(30.0, max(cur, seconds))
            self._next_ok[host] = time.monotonic() + seconds
        LOG.warning("Rate-limit brake on %s: +%.1fs cooldown", host, seconds)

    def relax(self, url: str) -> None:
        host = urlsplit(url).netloc.lower()
        with self._lock:
            if host in self._penalty:
                self._penalty[host] *= 0.5
                if self._penalty[host] < 0.05:
                    del self._penalty[host]


RATE = HostRateLimiter()


# --------------------------------------------------------------------------
# 4. HTTP with exponential backoff
# --------------------------------------------------------------------------
import requests  # noqa: E402
from requests.adapters import HTTPAdapter  # noqa: E402

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 524}


class Http:
    """Thin requests wrapper: rate limited, retried with exponential backoff
    and full jitter, honouring Retry-After when the server sends one."""

    def __init__(
        self,
        max_attempts: int = 6,
        base_delay: float = 1.5,
        max_delay: float = 120.0,
        timeout: Tuple[float, float] = (15.0, 180.0),
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self._local = threading.local()

    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update(
                {"User-Agent": USER_AGENT, "Accept-Language": "th,en;q=0.8"}
            )
            # urllib3-level retries are off: we do our own, with logging.
            adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            self._local.session = s
        return s

    def _backoff(self, attempt: int, retry_after: Optional[float]) -> float:
        if retry_after is not None:
            return min(self.max_delay, max(1.0, retry_after))
        window = min(self.max_delay, self.base_delay * (2 ** attempt))
        return random.uniform(0.0, window)  # full jitter

    def request(self, method: str, url: str, **kw) -> requests.Response:
        kw.setdefault("timeout", self.timeout)
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_attempts):
            if SHUTDOWN.is_set():
                raise RuntimeError("shutdown requested")
            RATE.acquire(url)
            try:
                resp = self.session.request(method, url, **kw)
            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.TooManyRedirects,
            ) as exc:
                last_exc = exc
                delay = self._backoff(attempt, None)
                LOG.warning(
                    "%s %s -> %s (attempt %d/%d); retry in %.1fs",
                    method, _short(url), type(exc).__name__,
                    attempt + 1, self.max_attempts, delay,
                )
                _interruptible_sleep(delay)
                continue

            if resp.status_code in RETRYABLE_STATUS:
                ra = _parse_retry_after(resp.headers.get("Retry-After"))
                delay = self._backoff(attempt, ra)
                if resp.status_code in (429, 503):
                    RATE.penalise(url, min(30.0, delay))
                LOG.warning(
                    "%s %s -> HTTP %d (attempt %d/%d); retry in %.1fs",
                    method, _short(url), resp.status_code,
                    attempt + 1, self.max_attempts, delay,
                )
                resp.close()
                _interruptible_sleep(delay)
                continue

            RATE.relax(url)
            return resp

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("exhausted %d attempts for %s" % (self.max_attempts, url))

    def get(self, url: str, **kw) -> requests.Response:
        return self.request("GET", url, **kw)

    def post_json(self, url: str, payload: Any, **kw) -> requests.Response:
        headers = dict(kw.pop("headers", {}) or {})
        headers.setdefault("Content-Type", "application/json")
        return self.request(
            "POST", url, data=json.dumps(payload).encode("utf-8"), headers=headers, **kw
        )

    def get_json(self, url: str, **kw) -> Any:
        r = self.get(url, **kw)
        r.raise_for_status()
        return r.json()


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    with contextlib.suppress(ValueError):
        return float(value)
    with contextlib.suppress(Exception):
        import datetime as _dt
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(value)
        return max(0.0, (dt - _dt.datetime.now(dt.tzinfo)).total_seconds())
    return None


def _interruptible_sleep(seconds: float) -> None:
    if SHUTDOWN.wait(timeout=seconds):
        raise RuntimeError("shutdown requested")


def _short(url: str, n: int = 90) -> str:
    return url if len(url) <= n else url[: n - 3] + "..."


def encode_url_path(url: str) -> str:
    """Percent-encode non-ASCII path segments (TTRS serves Thai filenames)."""
    parts = urlsplit(url)
    path = quote(parts.path, safe="/%:@&=+$,~")
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


# --------------------------------------------------------------------------
# 5. Filename hygiene
# --------------------------------------------------------------------------
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *("COM%d" % i for i in range(1, 10)),
    *("LPT%d" % i for i in range(1, 10)),
}


def safe_slug(text: str, max_len: int = 60) -> str:
    """ASCII-safe, filesystem-safe slug.

    Thai is deliberately dropped from the *filename*: the human-readable label
    lives in the CSV/JSON sidecar, so filenames stay portable across Linux
    training boxes, tarballs, and object stores. A pure-Thai label degrades to
    a stable short digest rather than to an empty string.
    """
    if not text:
        return "untitled"
    text = unicodedata.normalize("NFKC", text)
    ascii_only = text.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", ascii_only).strip("-._")
    if not slug:
        slug = "u" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    slug = slug[:max_len].strip("-._")
    if slug.upper() in _WINDOWS_RESERVED:
        slug = "_" + slug
    return slug or "untitled"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# 6. Media probing / validation
# --------------------------------------------------------------------------
@dataclass
class MediaInfo:
    ok: bool
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    vcodec: Optional[str] = None
    acodec: Optional[str] = None
    nb_frames: Optional[int] = None
    error: Optional[str] = None


_FFPROBE = shutil.which("ffprobe")


def probe_media(path: Path) -> MediaInfo:
    """Validate that a downloaded file really is decodable video.

    Files that fail here are quarantined rather than silently poisoning the
    pre-training set -- a truncated MP4 that still opens is the classic way a
    self-supervised run ends up learning from garbage.
    """
    if _FFPROBE is None:
        return MediaInfo(ok=True, error="ffprobe-unavailable")
    try:
        proc = subprocess.run(
            [_FFPROBE, "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, timeout=180,
        )
        if proc.returncode != 0:
            return MediaInfo(ok=False, error=proc.stderr.decode("utf-8", "replace")[:300])
        data = json.loads(proc.stdout.decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001
        return MediaInfo(ok=False, error=("%s: %s" % (type(exc).__name__, exc))[:300])

    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None:
        return MediaInfo(ok=False, error="no video stream")

    fps = None
    rate = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/0"
    with contextlib.suppress(Exception):
        num, _, den = rate.partition("/")
        if float(den or 0):
            fps = round(float(num) / float(den), 4)

    dur = None
    with contextlib.suppress(Exception):
        dur = float(data.get("format", {}).get("duration") or v.get("duration"))

    nbf = None
    with contextlib.suppress(Exception):
        nbf = int(v.get("nb_frames"))

    if not dur or dur <= 0.04:
        return MediaInfo(ok=False, error="implausible duration: %s" % dur)

    return MediaInfo(
        ok=True, duration=dur, width=v.get("width"), height=v.get("height"),
        fps=fps, vcodec=v.get("codec_name"),
        acodec=(a or {}).get("codec_name"), nb_frames=nbf,
    )


# --------------------------------------------------------------------------
# 7. Atomic download
# --------------------------------------------------------------------------
@dataclass
class DownloadResult:
    ok: bool
    path: Optional[Path] = None
    bytes: int = 0
    sha256: Optional[str] = None
    media: Optional[MediaInfo] = None
    error: Optional[str] = None


def download_file(
    http: Http,
    url: str,
    dest: Path,
    *,
    referer: Optional[str] = None,
    min_bytes: int = 2048,
    validate_media: bool = True,
) -> DownloadResult:
    """Stream to `dest.part`, verify, then atomically rename into place, so a
    partially written file can never be mistaken for a complete one."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"Referer": referer} if referer else {}
    try:
        with http.get(encode_url_path(url), headers=headers, stream=True) as resp:
            if resp.status_code != 200:
                return DownloadResult(ok=False, error="HTTP %d" % resp.status_code)
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" in ctype:
                return DownloadResult(
                    ok=False, error="server returned HTML, not media (%s)" % ctype
                )
            expected = resp.headers.get("Content-Length")
            expected = int(expected) if expected and expected.isdigit() else None

            written = 0
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 18):
                    if SHUTDOWN.is_set():
                        raise RuntimeError("shutdown requested")
                    if chunk:
                        fh.write(chunk)
                        written += len(chunk)

        if expected is not None and written != expected:
            raise IOError("truncated: got %d of %d bytes" % (written, expected))
        if written < min_bytes:
            raise IOError("suspiciously small: %d bytes" % written)

        media = probe_media(tmp) if validate_media else MediaInfo(ok=True)
        if not media.ok:
            raise IOError("media validation failed: %s" % media.error)

        digest = sha256_file(tmp)
        os.replace(tmp, dest)
        return DownloadResult(
            ok=True, path=dest, bytes=written, sha256=digest, media=media
        )

    except Exception as exc:  # noqa: BLE001
        with contextlib.suppress(OSError):
            if tmp.exists():
                tmp.unlink()
        return DownloadResult(ok=False, error=("%s: %s" % (type(exc).__name__, exc))[:400])


# --------------------------------------------------------------------------
# 8. State store (SQLite) -- gives idempotent resume
# --------------------------------------------------------------------------
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS items (
    uid           TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    level         TEXT NOT NULL,
    native_id     TEXT,
    url           TEXT,
    label         TEXT,
    label_type    TEXT,
    meta_json     TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',
    attempts      INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT,
    rel_path      TEXT,
    bytes         INTEGER,
    sha256        TEXT,
    duration      REAL,
    width         INTEGER,
    height        INTEGER,
    fps           REAL,
    vcodec        TEXT,
    acodec        TEXT,
    discovered_at TEXT,
    completed_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_items_status ON items(status);
CREATE INDEX IF NOT EXISTS idx_items_source ON items(source);
CREATE INDEX IF NOT EXISTS idx_items_sha    ON items(sha256);

CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    started_at TEXT,
    ended_at   TEXT,
    argv       TEXT,
    notes      TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT,
    run_id    TEXT,
    source    TEXT,
    kind      TEXT,
    detail    TEXT
);
"""


class Store:
    """All writes go through one lock and one connection. SQLite in WAL mode
    handles read concurrency; serialising writes avoids 'database is locked'
    storms from the worker pool."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._con = sqlite3.connect(str(path), check_same_thread=False, timeout=60.0)
        self._con.row_factory = sqlite3.Row
        with self._lock:
            self._con.executescript(SCHEMA)
            self._con.commit()

    def close(self) -> None:
        with self._lock:
            with contextlib.suppress(Exception):
                self._con.commit()
                self._con.close()

    # -- writes -----------------------------------------------------------
    def upsert_pending(self, rows: Iterable[Dict[str, Any]]) -> int:
        """Insert newly discovered items. Existing rows keep their status, so a
        re-discovery never re-downloads what is already on disk."""
        sql = """
        INSERT INTO items (uid, source, level, native_id, url, label, label_type,
                           meta_json, status, discovered_at)
        VALUES (:uid, :source, :level, :native_id, :url, :label, :label_type,
                :meta_json, 'pending', :discovered_at)
        ON CONFLICT(uid) DO UPDATE SET
            label      = COALESCE(excluded.label, items.label),
            label_type = COALESCE(excluded.label_type, items.label_type),
            meta_json  = COALESCE(excluded.meta_json, items.meta_json),
            url        = COALESCE(excluded.url, items.url)
        """
        now = utcnow()
        payload: List[Dict[str, Any]] = []
        for r in rows:
            r = dict(r)
            r.setdefault("discovered_at", now)
            for k in ("native_id", "label", "label_type", "url"):
                r.setdefault(k, None)
            if isinstance(r.get("meta_json"), (dict, list)):
                r["meta_json"] = json.dumps(r["meta_json"], ensure_ascii=False)
            payload.append(r)
        if not payload:
            return 0
        with self._lock:
            before = self._con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            self._con.executemany(sql, payload)
            self._con.commit()
            after = self._con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        return after - before

    def mark_done(self, uid: str, **fields: Any) -> None:
        fields["status"] = "done"
        fields["completed_at"] = utcnow()
        fields["last_error"] = None
        self._update(uid, fields)

    def mark_failed(self, uid: str, error: str, terminal: bool = False) -> None:
        with self._lock:
            self._con.execute(
                "UPDATE items SET status=?, attempts=attempts+1, last_error=? WHERE uid=?",
                ("failed" if terminal else "pending", error[:500], uid),
            )
            self._con.commit()

    def mark_status(self, uid: str, status: str, error: Optional[str] = None) -> None:
        self._update(uid, {"status": status, "last_error": error or None})

    def _update(self, uid: str, fields: Dict[str, Any]) -> None:
        if not fields:
            return
        cols = ", ".join("%s=?" % k for k in fields)
        with self._lock:
            self._con.execute(
                "UPDATE items SET %s WHERE uid=?" % cols, (*fields.values(), uid)
            )
            self._con.commit()

    def log_event(self, run_id: str, source: str, kind: str, detail: str) -> None:
        with self._lock:
            self._con.execute(
                "INSERT INTO events (ts, run_id, source, kind, detail) VALUES (?,?,?,?,?)",
                (utcnow(), run_id, source, kind, detail[:2000]),
            )
            self._con.commit()

    def start_run(self, run_id: str, argv: str, notes: str = "") -> None:
        with self._lock:
            self._con.execute(
                "INSERT OR REPLACE INTO runs (run_id, started_at, argv, notes) "
                "VALUES (?,?,?,?)",
                (run_id, utcnow(), argv, notes),
            )
            self._con.commit()

    def end_run(self, run_id: str) -> None:
        with self._lock:
            self._con.execute(
                "UPDATE runs SET ended_at=? WHERE run_id=?", (utcnow(), run_id)
            )
            self._con.commit()

    # -- reads ------------------------------------------------------------
    def pending(
        self, source: str, limit: Optional[int] = None, max_attempts: int = 4
    ) -> List[sqlite3.Row]:
        sql = (
            "SELECT * FROM items WHERE source=? AND status='pending' "
            "AND attempts < ? ORDER BY attempts ASC, rowid ASC"
        )
        args: List[Any] = [source, max_attempts]
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        with self._lock:
            return self._con.execute(sql, args).fetchall()

    def count(self, **where: Any) -> int:
        clause = " AND ".join("%s=?" % k for k in where) or "1=1"
        with self._lock:
            return self._con.execute(
                "SELECT COUNT(*) FROM items WHERE %s" % clause, tuple(where.values())
            ).fetchone()[0]

    def sha_exists(self, digest: str, exclude_uid: str) -> Optional[str]:
        with self._lock:
            row = self._con.execute(
                "SELECT uid FROM items WHERE sha256=? AND uid<>? AND status='done' LIMIT 1",
                (digest, exclude_uid),
            ).fetchone()
        return row["uid"] if row else None

    def query(self, sql: str, args: Tuple = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._con.execute(sql, args).fetchall()


def utcnow() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def human_bytes(n: Optional[float]) -> str:
    if not n:
        return "0 B"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return "%d B" % int(n) if unit == "B" else "%.1f %s" % (n, unit)
        n /= 1024.0
    return "%.1f TB" % n


def human_duration(seconds: Optional[float]) -> str:
    if not seconds:
        return "0s"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "%dh %dm %ds" % (h, m, s)
    if m:
        return "%dm %ds" % (m, s)
    return "%ds" % s
