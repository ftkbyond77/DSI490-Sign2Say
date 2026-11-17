"""v11 sealed BLIND split (prompt-opt-v11 C4).

BLIND-v11 is fixed by `python main.py v11 seal` BEFORE any v11 training, from existing data only, with a seeded script:
2 TSL-ONE-S signers · 1 th_sl signer · 1 TTRS signer · 3 YouTube people · 40 % of the TSL51 `sentence_test` templates (and every clip of
those people, Tail signs included — the open-set part). The manifest is artifacts/eval/blind_v11_manifest.json.

Rules enforced here:
  * training tables, banks and TUNE sets must never contain a sealed clip → `assert_clean(ids, where)` raises SealedError;
  * dev-test sets (built before the seal) drop sealed rows on load (`drop_sealed_rows`);
  * BLIND sets are only built / scored with THAISLM_OPEN_BLIND=1 (TPH 4, opened once; every opening is appended to the manifest log).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "eval" / "blind_v11_manifest.json"
# v12 (prompt-opt-v12 1G): BLIND-v12 = whole new-source people + held-out sentences; enforced together with BLIND-v11 (union)
MANIFEST_V12 = ROOT / "artifacts" / "eval" / "blind_v12_manifest.json"


class SealedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def manifest():
    """BLIND-v11 ∪ BLIND-v12 (signers, clips) — every loader keeps BOTH out of training, banks and TUNE"""
    m = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else dict(signers=[], clips=[], t51_sentences=[], templates=[])
    m = dict(m)
    if MANIFEST_V12.exists():
        m12 = json.loads(MANIFEST_V12.read_text(encoding="utf-8"))
        m["signers"] = list(m.get("signers", [])) + list(m12.get("signers", []))
        m["clips"] = list(m.get("clips", [])) + list(m12.get("clips", []))
    return m


@lru_cache(maxsize=1)
def manifest_v12():
    return json.loads(MANIFEST_V12.read_text(encoding="utf-8")) if MANIFEST_V12.exists() else dict(signers=[], clips=[])


def sealed_signers():
    return set(manifest()["signers"])


def sealed_clips():
    return set(manifest()["clips"]) | set(manifest()["t51_sentences"])


def is_sealed(clip_id=None, signer=None):
    m = manifest()
    if signer is not None and signer in set(m["signers"]):
        return True
    if clip_id is None:
        return False
    cid = str(clip_id).split("@")[0].split("#")[0]
    return cid in sealed_clips() or ("t51s:" + cid.split(":", 1)[-1]) in set(m["t51_sentences"])


def assert_clean(ids, where="", signers=None):
    """Raise when any id (clip id, "<clip>@<f0>", "<clip>#rtmw") or signer is sealed."""
    bad = [i for i in ids if is_sealed(i)]
    if signers is not None:
        bad += [f"signer:{s}" for s in set(signers) if s in sealed_signers()]
    if bad:
        raise SealedError(f"{len(bad)} sealed BLIND-v11 item(s) in {where or 'a loader'}: {bad[:5]}")


def drop_sealed_rows(rows):
    """dev-test rows (decision cache) → the rows that are not BLIND (signer / clip / TSL51 sentence)."""
    out = []
    for r in rows:
        rid = str(r.get("id", "")).split("@")[0]
        if r.get("signer") in sealed_signers() or is_sealed(rid):
            continue
        out.append(r)
    return out


def blind_open():
    return os.environ.get("THAISLM_OPEN_BLIND") == "1"


def blind12_open():
    """BLIND-v12 is opened once, in TPH 4 (THAISLM_OPEN_BLIND12=1); every opening is logged in its manifest"""
    return os.environ.get("THAISLM_OPEN_BLIND12") == "1"


def log_opening(note, which="v11"):
    path = MANIFEST_V12 if which == "v12" else MANIFEST
    m = json.loads(path.read_text(encoding="utf-8"))
    from datetime import datetime
    m.setdefault("openings", []).append(dict(time=datetime.now().isoformat(timespec="seconds"), note=note))
    path.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest.cache_clear(); manifest_v12.cache_clear()
