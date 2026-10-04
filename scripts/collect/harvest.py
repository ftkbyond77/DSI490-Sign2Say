# -*- coding: utf-8 -*-
"""
harvest.py -- Orchestrator for the Thai Sign Language raw-video collection.

Two phases, deliberately decoupled:

  1. DISCOVER  cheap, idempotent catalogue building -> SQLite
  2. FETCH     expensive, resumable downloading      -> raw_data/<level>/<source>/

Everything is checkpointed in state/harvest.db, so Ctrl-C (or a crash, or a
dead network) costs at most the in-flight downloads. Re-running the same
command resumes exactly where it stopped and re-downloads nothing.

Usage
-----
  python harvest.py --discover-only
  python harvest.py --sources ttrs_dictionary --workers 4
  python harvest.py --sources youtube --yt-keys thaipbs_bigsign --limit 50
  python harvest.py --status
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tsl_common import (  # noqa: E402
    LOG,
    PROJECT_ROOT,
    SHUTDOWN,
    Http,
    Store,
    human_bytes,
    human_duration,
    install_signal_handlers,
    utcnow,
)
from tsl_sources import Ctx, build_sources  # noqa: E402

STATE_DB = PROJECT_ROOT / "state" / "harvest.db"


class Counters:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.done = 0
        self.failed = 0
        self.skipped = 0
        self.breaker_open = 0
        self.dupes = 0
        self.bytes = 0
        self.seconds = 0.0

    def add(self, **kw: Any) -> None:
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return dict(
                done=self.done, failed=self.failed, skipped=self.skipped,
                breaker_open=self.breaker_open,
                dupes=self.dupes, bytes=self.bytes, seconds=self.seconds,
            )


# --------------------------------------------------------------------------
# Phase 1: discovery
# --------------------------------------------------------------------------
def run_discovery(sources: List[Any], ctx: Ctx) -> Dict[str, int]:
    added: Dict[str, int] = {}
    for src in sources:
        if SHUTDOWN.is_set():
            break
        LOG.info("=" * 72)
        LOG.info("DISCOVER  source=%s", src.name)
        LOG.info("=" * 72)
        batch: List[Dict[str, Any]] = []
        n_new = 0
        n_seen = 0
        try:
            for row in src.discover(ctx):
                n_seen += 1
                batch.append(row)
                if len(batch) >= 200:
                    n_new += ctx.store.upsert_pending(batch)
                    batch.clear()
        except Exception as exc:  # noqa: BLE001
            LOG.exception("discovery for %s aborted: %s", src.name, exc)
            ctx.store.log_event(ctx.run_id, src.name, "discover_error", repr(exc))
        if batch:
            n_new += ctx.store.upsert_pending(batch)
        added[src.name] = n_new
        LOG.info("[%s] discovery complete: %d candidates seen, %d new to the catalogue",
                 src.name, n_seen, n_new)
        ctx.store.log_event(
            ctx.run_id, src.name, "discover_done",
            json.dumps({"seen": n_seen, "new": n_new}),
        )
    return added


# --------------------------------------------------------------------------
# Phase 2: fetching
# --------------------------------------------------------------------------
def run_fetch(sources: List[Any], ctx: Ctx, workers: int) -> Counters:
    counters = Counters()
    for src in sources:
        if SHUTDOWN.is_set():
            break
        rows = ctx.store.pending(src.name, limit=ctx.limit)
        if not rows:
            LOG.info("[%s] nothing pending -- already complete.", src.name)
            continue

        total_done_before = ctx.store.count(source=src.name, status="done")
        LOG.info("=" * 72)
        LOG.info("FETCH     source=%s  pending=%d  already_done=%d  workers=%d",
                 src.name, len(rows), total_done_before, workers)
        LOG.info("=" * 72)

        started = time.time()
        processed = 0
        # yt-dlp spawns its own processes and is heavier; keep its pool smaller.
        pool_size = max(1, workers if src.name != "youtube" else max(1, workers // 2))

        with ThreadPoolExecutor(max_workers=pool_size,
                                thread_name_prefix="fetch") as pool:
            futures = {
                pool.submit(_fetch_one, src, ctx, dict(r), counters): r["uid"]
                for r in rows
            }
            for fut in as_completed(futures):
                processed += 1
                uid = futures[fut]
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    LOG.exception("worker crashed on %s: %s", uid, exc)
                    ctx.store.mark_failed(uid, "worker crash: %r" % exc)
                    counters.add(failed=1)

                if processed % 25 == 0 or processed == len(rows):
                    snap = counters.snapshot()
                    elapsed = max(1e-6, time.time() - started)
                    rate = processed / elapsed
                    eta = (len(rows) - processed) / rate if rate > 0 else 0
                    LOG.info(
                        "[%s] %d/%d | ok=%d fail=%d skip=%d dup=%d | %s | %.2f items/s | ETA %s",
                        src.name, processed, len(rows), snap["done"], snap["failed"],
                        snap["breaker_open"], snap["dupes"], human_bytes(snap["bytes"]),
                        rate, human_duration(eta),
                    )
                if SHUTDOWN.is_set() or getattr(getattr(src, "_blocked", None), "is_set", lambda: False)():
                    # Stop handing out new work; in-flight futures still finish.
                    # (A tripped per-source breaker, e.g. YouTube's bot-block
                    # detector, cancels the same way a global shutdown does --
                    # the remaining futures resolve as instant breaker_open
                    # no-ops rather than real requests, so this is a courtesy
                    # to stop logging them, not a correctness requirement.)
                    for f in futures:
                        f.cancel()

        LOG.info("[%s] fetch phase finished in %s",
                 src.name, human_duration(time.time() - started))
    return counters


def _fetch_one(src: Any, ctx: Ctx, row: Dict[str, Any], counters: Counters) -> None:
    if SHUTDOWN.is_set():
        return
    uid = row["uid"]
    try:
        out = src.fetch(ctx, row)
    except Exception as exc:  # noqa: BLE001
        ctx.store.mark_failed(uid, "%s: %s" % (type(exc).__name__, exc))
        counters.add(failed=1)
        LOG.debug("fetch raised for %s", uid, exc_info=True)
        return

    if getattr(out, "breaker_open", False):
        # Never attempted -- leave status/attempts exactly as they were so a
        # later run (once the block clears) sees this item as fresh, not one
        # attempt closer to giving up on it.
        counters.add(breaker_open=1)
        return

    if out.skipped:
        ctx.store.mark_status(uid, "skipped", out.error)
        counters.add(skipped=1)
        return

    if not out.ok:
        ctx.store.mark_failed(uid, out.error or "unknown error", terminal=out.terminal)
        counters.add(failed=1)
        LOG.warning("FAIL %s :: %s", uid, (out.error or "")[:200])
        return

    # Content-level dedup: identical bytes from two different listings.
    if out.sha256:
        twin = ctx.store.sha_exists(out.sha256, uid)
        if twin:
            counters.add(dupes=1)
            ctx.store.mark_done(
                uid, rel_path=out.rel_path, bytes=out.bytes, sha256=out.sha256,
                duration=out.duration, width=out.width, height=out.height,
                fps=out.fps, vcodec=out.vcodec, acodec=out.acodec,
            )
            ctx.store.log_event(ctx.run_id, src.name, "duplicate",
                                "%s == %s" % (uid, twin))
            counters.add(done=1, bytes=out.bytes, seconds=out.duration or 0.0)
            return

    ctx.store.mark_done(
        uid, rel_path=out.rel_path, bytes=out.bytes, sha256=out.sha256,
        duration=out.duration, width=out.width, height=out.height,
        fps=out.fps, vcodec=out.vcodec, acodec=out.acodec,
    )
    counters.add(done=1, bytes=out.bytes, seconds=out.duration or 0.0)


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------
def print_status(store: Store) -> None:
    rows = store.query(
        "SELECT source, level, status, COUNT(*) n, "
        "       COALESCE(SUM(bytes),0) b, COALESCE(SUM(duration),0) d "
        "FROM items GROUP BY source, level, status ORDER BY source, level, status"
    )
    if not rows:
        print("catalogue is empty -- run discovery first")
        return
    print("\n%-18s %-11s %-10s %8s %12s %12s" %
          ("SOURCE", "LEVEL", "STATUS", "COUNT", "BYTES", "DURATION"))
    print("-" * 78)
    for r in rows:
        print("%-18s %-11s %-10s %8d %12s %12s" % (
            r["source"], r["level"], r["status"], r["n"],
            human_bytes(r["b"]), human_duration(r["d"]),
        ))
    tot = store.query(
        "SELECT COUNT(*) n, COALESCE(SUM(bytes),0) b, COALESCE(SUM(duration),0) d "
        "FROM items WHERE status='done'"
    )[0]
    print("-" * 78)
    print("%-41s %8d %12s %12s" % (
        "TOTAL DOWNLOADED", tot["n"], human_bytes(tot["b"]), human_duration(tot["d"])
    ))
    print()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Harvest raw Thai Sign Language video with labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--sources", nargs="*", default=None,
                    help="subset of sources: ttrs_dictionary youtube")
    ap.add_argument("--yt-keys", nargs="*", default=None,
                    help="subset of YouTube target keys")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on NEW downloads per source this run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-height", type=int, default=720,
                    help="cap YouTube video height")
    ap.add_argument("--max-seconds", type=int, default=5400,
                    help="skip videos longer than this")
    ap.add_argument("--discover-only", action="store_true")
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--yt-cookies-from-browser", default=None,
                    metavar="BROWSER",
                    help="pass browser cookies to yt-dlp (chrome, firefox, edge...). "
                         "Opt-in: this authenticates as your real YouTube account, "
                         "which carries account risk when downloading in bulk.")
    ap.add_argument("--yt-cookies-file", default=None,
                    help="path to a Netscape-format cookies.txt for yt-dlp")
    ap.add_argument("--host-interval", type=float, default=None,
                    help="override the default per-host request spacing (s)")
    args = ap.parse_args(argv)

    install_signal_handlers()
    store = Store(STATE_DB)

    if args.status:
        print_status(store)
        store.close()
        return 0

    if args.host_interval:
        from tsl_common import RATE
        RATE.default_interval = args.host_interval

    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    store.start_run(run_id, " ".join(sys.argv))
    LOG.info("run_id=%s  project=%s", run_id, PROJECT_ROOT)

    http = Http()
    ctx = Ctx(
        http=http, store=store, run_id=run_id, limit=args.limit,
        discover_only=args.discover_only, max_video_seconds=args.max_seconds,
        youtube_max_height=args.max_height,
    )
    sources = build_sources(
        args.sources, args.yt_keys,
        yt_cookies_from_browser=args.yt_cookies_from_browser,
        yt_cookies_file=args.yt_cookies_file,
    )
    if not sources:
        LOG.error("no valid sources selected")
        store.close()
        return 2

    t0 = time.time()
    try:
        if not args.fetch_only:
            run_discovery(sources, ctx)
        if not args.discover_only:
            counters = run_fetch(sources, ctx, args.workers)
            snap = counters.snapshot()
            LOG.info("-" * 72)
            LOG.info(
                "RUN SUMMARY  ok=%d failed=%d skipped=%d breaker_open=%d duplicates=%d",
                snap["done"], snap["failed"], snap["skipped"],
                snap["breaker_open"], snap["dupes"],
            )
            LOG.info("             downloaded=%s  video=%s  wall=%s",
                     human_bytes(snap["bytes"]), human_duration(snap["seconds"]),
                     human_duration(time.time() - t0))
    except KeyboardInterrupt:
        SHUTDOWN.set()
        LOG.warning("interrupted")
    finally:
        store.end_run(run_id)
        print_status(store)
        store.close()

    return 130 if SHUTDOWN.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
