"""Conversational coverage (Thai + English everyday sentences, tests/conversation.tsv): how much of what people actually say
becomes recorded signs, and how much falls back to fingerspelling / text. No gold TSL exists for these sentences, so the
output (data/eval/conversation.csv) is also the review sheet for the Deaf experts (translation level)."""
from __future__ import annotations

import pandas as pd

from .. import paths


def run(engine=None):
    from ..pipeline import Engine
    e = engine or Engine()
    t = pd.read_csv(paths.HERE / "tests" / "conversation.tsv", sep="\t")
    rows = []
    for r in t.itertuples():
        try:
            ir = e.translate(r.text, lang=r.lang)
        except RuntimeError as x:
            rows.append(dict(lang=r.lang, topic=r.topic, text=r.text, error=str(x))); continue
        u = ir.units
        content = [x for x in u if x.role not in ("greeting",)] or u
        rows.append(dict(lang=r.lang, topic=r.topic, text=r.text, thai=ir.thai, tsl=" ".join(x.gloss for x in u),
                         kinds=" ".join(x.kind[0] for x in u), n=len(u), signs=sum(x.kind in ("sign", "phrase") for x in u),
                         phrases=sum(x.kind == "phrase" for x in u),
                         spelled=sum(x.kind == "fingerspell" for x in u), text_cards=sum(x.kind == "text" for x in u),
                         all_signed=all(x.kind in ("sign", "phrase") for x in u),
                         sentence_type="|".join(c["type"] for c in ir.clauses),
                         fallbacks="; ".join(x.gloss + ":" + x.kind for x in u if x.kind not in ("sign", "phrase")),
                         translation_conf=ir.confidence.get("translation"), motion_conf=ir.confidence.get("motion")))
    d = pd.DataFrame(rows)
    (paths.WORK / "eval").mkdir(parents=True, exist_ok=True)
    d.to_csv(paths.WORK / "eval" / "conversation.csv", index=False, encoding="utf-8-sig")
    s = d.groupby("lang").agg(sentences=("text", "size"), all_signed=("all_signed", "mean"), units=("n", "sum"),
                              signs=("signs", "sum"), spelled=("spelled", "sum"), text_cards=("text_cards", "sum"),
                              phrases=("phrases", "sum"))
    s["sign_rate"] = s.signs / s.units
    return d, s
