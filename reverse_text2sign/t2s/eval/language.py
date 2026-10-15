"""Language regression + held-out evaluation (v2, result_v2.md §4).

tests/language_heldout.tsv — 73 Thai + 12 English sentences written BEFORE any v2 language change and never used to develop a
rule (by category: WH / yes-no questions, negation, time, numbers, names, homographs, long sentences, dropped subjects). Each
row lists checks that do not need a Deaf expert's gold gloss sequence (there is none outside TSL51):
  type:T          sentence type (alternatives a|b)        has:G / not:G   a sign expressing G (gloss or source word) is / is not signed
  first:G last:G  position of that sign (TSL51 mode only)  spell:W         W is fingerspelled (names)        neg   a negation sign
A sign "expresses G" when its gloss or one of its source words equals G. Word ORDER checks test that the declared TSL51 rules
are applied consistently to new sentences — not that the rules are right (that needs the experts). Also reported: words signed
by recorded signs (coverage). Works on an IR dict, so the same checks run on v1's code (a git worktree) for the comparison.
"""
from __future__ import annotations

import csv

import pandas as pd

from .. import paths

HELDOUT = paths.HERE / "tests" / "language_heldout.tsv"


def load():
    return list(csv.DictReader(open(HELDOUT, encoding="utf-8"), delimiter="\t"))


def _expresses(u, g):
    return u.get("gloss") == g or g in (u.get("source_words") or [])


def check(ir, checks, mode="tsl51"):
    """→ list of (check, passed) for one IR dict."""
    units = [u for u in ir["units"]]
    signed = [u for u in units if u["kind"] in ("sign", "phrase", "fingerspell")]
    types = [c["type"] for c in ir["clauses"]]
    out = []
    for raw in checks.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        alts = raw.split("|")
        prefix = alts[0].split(":", 1)[0] if ":" in alts[0] else None
        ok = False
        for a in alts:
            if ":" not in a and a != "neg" and prefix:
                a = f"{prefix}:{a}"
            if a == "neg":
                ok |= any(u.get("role") == "neg" or (u.get("gloss") or "").startswith("ไม่") for u in signed)
                continue
            k, v = a.split(":", 1)
            if k == "type":
                ok |= v in types
            elif k == "has":
                ok |= any(_expresses(u, v) for u in signed if u["kind"] != "fingerspell" or u.get("gloss") == v)
            elif k == "not":
                ok |= not any(_expresses(u, v) and u["kind"] in ("sign", "phrase") and u.get("gloss") == v for u in units)
            elif k == "spell":
                ok |= any(u["kind"] == "fingerspell" and u.get("gloss") == v for u in units)
            elif k in ("first", "last"):
                if mode != "tsl51":
                    ok = None
                    break
                seq = [u for u in signed if u.get("role") != "greeting"]
                if seq:
                    u = seq[0] if k == "first" else seq[-1]
                    ok |= _expresses(u, v)
        if ok is not None:
            out.append((raw, bool(ok)))
    return out


def evaluate(translate, modes=("tsl51", "thai")):
    """translate(text, lang, mode) → IR dict. Returns per-sentence rows and a per-category summary."""
    rows = []
    for r in load():
        lang = "en" if r["category"] == "en" else "th"
        for mode in modes:
            try:
                ir = translate(r["text"], lang, mode)
            except Exception as e:  # noqa: BLE001
                rows.append(dict(id=r["id"], category=r["category"], mode=mode, text=r["text"], error=str(e)[:120], passed=0, n=1,
                                 all_pass=False))
                continue
            res = check(ir, r["checks"], mode)
            content = [u for u in ir["units"] if u.get("role") != "greeting"]
            rec = sum(u["kind"] in ("sign", "phrase") for u in content)
            rows.append(dict(id=r["id"], category=r["category"], mode=mode, text=r["text"], thai=ir.get("thai"),
                             glosses=" ".join(u["gloss"] for u in ir["units"]), passed=sum(ok for _, ok in res), n=len(res),
                             all_pass=all(ok for _, ok in res), failed="; ".join(c for c, ok in res if not ok),
                             units=len(content), recorded=rec, spelled=sum(u["kind"] == "fingerspell" for u in content),
                             text_cards=sum(u["kind"] == "text" for u in content)))
    d = pd.DataFrame(rows)
    s = d.groupby(["mode", "category"]).agg(sentences=("id", "count"), all_pass=("all_pass", "mean"), checks=("n", "sum"),
                                             passed=("passed", "sum"), units=("units", "sum"), recorded=("recorded", "sum"))
    s["check_rate"] = (s.passed / s.checks).round(3)
    s["coverage"] = (s.recorded / s.units).round(3)
    return d, s


def regression(translate):
    """v1 regression: TSL51 gold order (dev set) and the 149 everyday sentences of tests/conversation.tsv (TSL51 mode)."""
    import json
    pairs = json.loads((paths.PREP / "params" / "tsl51_thai_sentences.json").read_text(encoding="utf-8"))
    ok = sum([u["gloss"] for u in translate(th, "th", "tsl51", phrasebook=False)["units"]] == gold.split() for gold, th in pairs.items())
    conv = list(csv.DictReader(open(paths.HERE / "tests" / "conversation.tsv", encoding="utf-8"), delimiter="\t"))
    out = []
    for r in conv:
        ir = translate(r["text"], r["lang"], "tsl51")
        out.append(dict(lang=r["lang"], text=r["text"], glosses=" ".join(u["gloss"] for u in ir["units"]),
                        kinds=" ".join(u["kind"][0] for u in ir["units"])))
    return dict(tsl51_exact=ok, tsl51_n=len(pairs)), pd.DataFrame(out)
