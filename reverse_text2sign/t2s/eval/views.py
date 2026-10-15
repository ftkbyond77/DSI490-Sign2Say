"""Figures for the two contact fixes and the five characters (v2, result_v2.md §5–6), rendered by the real web app.

  contact_fixes()   v1 vs v2 library, the same moment of the sign, front and side view: ขอบคุณ (palm-to-palm wai) and
                    ห้องน้ำ (pinch at the nose) — the side view shows what the front camera hides (depth)
  characters()      every character in vrm/ signing the same sentence at the same moments (fitted to its body)
Motion payloads are computed in-process from either library (v1 = the git worktree's models) and loaded into the page.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import paths
from .render import Renderer, _label

OUT = paths.REPORT / "image"


def _payload(engine, concept):
    ir, mo = engine.render_ir_dict(dict(units=[dict(gloss=concept, kind="sign", concept=concept, source_words=[concept], match="lemma",
                                                    confidence={"translation": 1.0})], thai=concept), nmm=False)
    return dict(ir=ir.to_dict(), motion=mo)


def contact_fixes(v1_engine, v2_engine, base_url="http://localhost:8081", size=300):
    R = Renderer(base_url, size=(560, 560))
    rows = []
    try:
        for concept, rel in (("ขอบคุณ", 0.33), ("ห้องน้ำ", 0.5)):
            tiles = []
            for tag, eng in (("v1", v1_engine), ("v2", v2_engine)):
                p = _payload(eng, concept)
                dur = R.load(p)
                seg = next(s for s in p["motion"]["segments"] if s["kind"] == "sign")
                t = (seg["start"] + rel * (seg["end"] - seg["start"])) / 25.0
                for deg, view in ((0, "front"), (90, "side")):
                    R.page.evaluate("d => window.__t2s.view(d)", deg)
                    tiles.append(_label(cv2.resize(R.frame(t), (size, size)), f"{concept} · {tag} · {view}"))
                R.page.evaluate("d => window.__t2s.view(d)", 0)
            rows.append(np.hstack(tiles))
    finally:
        R.close()
    img = np.vstack(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", img)[1].tofile(str(OUT / "contact_fixes.png"))
    return OUT / "contact_fixes.png"


def characters(text="ห้องน้ำอยู่ที่ไหน", base_url="http://localhost:8081", size=230, n=4):
    R = Renderer(base_url, size=(560, 560))
    rows = []
    try:
        avs = R.api("/api/avatars")["avatars"]
        for a in avs:
            res = R.api("/api/translate", {"text": text, "avatar": a["id"], "phrasebook": False})
            R.page.evaluate("id => window.__t2s.pick(id)", a["id"])
            R.load(res)
            segs = [s for s in res["motion"]["segments"] if s["kind"] == "sign"]
            tiles = []
            for s in segs[:n]:
                t = (s["start"] + s["end"]) / 2 / 25.0
                tiles.append(_label(cv2.resize(R.frame(t), (size, size)), f"{a['label']} · {s['label']}"))
            rows.append(np.hstack(tiles))
    finally:
        R.close()
    w = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3), 255, np.uint8)]) if r.shape[1] < w else r for r in rows]
    img = np.vstack(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", img)[1].tofile(str(OUT / "characters.png"))
    return OUT / "characters.png"
