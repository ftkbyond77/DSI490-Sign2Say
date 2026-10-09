"""Visual comparison renders (motion level, for the eye and for the Deaf reviewers): the real web app in headless Chromium renders
the avatar; frames are paired with the source dictionary video at the SAME moment of the sign (single signs) or laid out as a
timeline next to a real person signing the same Thai sentence (sentences).

  render_signs(concepts)      → data/eval/renders/sign_<concept>.jpg   (rows: video | avatar, 6 moments of the stroke)
  render_sentences(items)     → data/eval/renders/sent_<k>.jpg        (top: our avatar · bottom: the real signer's video)
Needs the web app (npm run dev, :5181) and the API (:8091) running — or the docker stack on :8081 (base_url).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from .. import paths

OUT = paths.WORK / "eval" / "renders"


def _decode(data_url):
    b = base64.b64decode(data_url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


class Renderer:
    def __init__(self, base_url="http://localhost:5181", size=(520, 520)):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        self.page = self.browser.new_page(viewport={"width": 1280, "height": 800})
        self.page.goto(base_url)
        self.page.wait_for_function("window.__t2s && window.__t2s.ready()", timeout=120_000)
        self.page.evaluate("document.querySelector('.input-panel').style.display='none';"
                           "document.querySelector('.layout').style.gridTemplateColumns='1fr'")
        self.page.set_viewport_size({"width": size[0] + 40, "height": size[1] + 200})
        self.page.wait_for_timeout(300)

    def close(self):
        self.browser.close(); self._pw.stop()

    def load(self, result):
        self.page.evaluate("r => window.__t2s.load(r)", result)
        return self.page.evaluate("window.__t2s.duration()")

    def frame(self, t):
        return _decode(self.page.evaluate("t => window.__t2s.capture(t)", t))

    def api(self, path, body=None):
        js = ("async ([p, b]) => { const r = await fetch(p, b ? {method: 'POST', headers: {'Content-Type': 'application/json'}, "
              "body: JSON.stringify(b)} : {}); return await r.json(); }")
        return self.page.evaluate(js, [path, body])


def _video_frame(path, t, size):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return np.zeros((size, size, 3), np.uint8)
    h, w = fr.shape[:2]
    s = min(h, w)
    x0 = (w - s) // 2
    return cv2.resize(fr[:, x0:x0 + s], (size, size))


def _label(img, text, color=(255, 255, 255)):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 18) if Path("C:/Windows/Fonts/tahoma.ttf").exists() else None
    d.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
    d.text((6, 3), text, font=f, fill=color)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def render_signs(concepts, base_url="http://localhost:5181", n=6, size=260):
    import pandas as pd
    from ..motion.library import Library
    OUT.mkdir(parents=True, exist_ok=True)
    sel = pd.read_parquet(paths.WORK / "select" / "exemplars.parquet").set_index("concept")
    lib = Library()
    R = Renderer(base_url)
    made = []
    try:
        for c in concepts:
            if c not in lib.clips or c not in sel.index:
                continue
            res = R.api(f"/api/sign/{c}")
            R.load(res)
            seg = next(s for s in res["motion"]["segments"] if s["kind"] == "sign")
            info = lib.clips[c]
            r = sel.loc[c]
            a_lib = max(0, int(r.span_f0) - 4)                      # library clip frame 0 = source frame a_lib (PAD = 4)
            st0, st1 = info["stroke"]
            raw_t = np.load(r.raw_file)["t"]
            top, bot = [], []
            for k in np.linspace(0, seg["end"] - seg["start"] - 1, n).astype(int):
                fsrc = a_lib + st0 + k
                tv = float(raw_t[min(fsrc, len(raw_t) - 1)])
                v = _video_frame(paths.DATA_ROOT / r.source_file, tv, size)
                if r.mirrored:
                    v = cv2.flip(v, 1)
                top.append(v)
                bot.append(cv2.resize(R.frame((seg["start"] + k) / 25.0), (size, size)))
            sheet = np.vstack([np.hstack(top), np.hstack(bot)])
            sheet = _label(sheet, f"{c}  ·  top: {r.dataset} video{' (mirrored)' if r.mirrored else ''}  ·  bottom: avatar (same moments)")
            p = OUT / f"sign_{len(made):03d}.jpg"
            cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(p))
            made.append((c, p))
    finally:
        R.close()
    return made


def render_sentence(R, text, video=None, n=8, size=220, video_span=None, phrasebook=False):
    """our avatar for `text` (top) and, if given, a real person's video of the same sentence (bottom), n evenly spaced frames."""
    res = R.api("/api/translate", {"text": text, "phrasebook": phrasebook})
    dur = R.load(res)
    gl = " ".join(u["gloss"] for u in res["ir"]["units"])
    top = [cv2.resize(R.frame(t), (size, size)) for t in np.linspace(0.3, max(dur - 0.3, 0.4), n)]
    rows = [np.hstack(top)]
    if video is not None:
        cap = cv2.VideoCapture(str(video))
        vd = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or 25)
        cap.release()
        a, b = video_span or (0.2, max(vd - 0.2, 0.3))
        rows.append(np.hstack([_video_frame(video, t, size) for t in np.linspace(a, b, n)]))
    sheet = _label(np.vstack(rows), f"{text}  →  {gl}   (top: avatar {dur:.1f}s" + (" · bottom: real signer)" if video else ")"))
    return sheet, res


def report_renders(base_url="http://localhost:8081"):
    """the figures of result_v1.md §6: single signs (video | avatar) and generated sentences vs a real signer's recording."""
    import pandas as pd
    signs = ["สวัสดี", "ขอบคุณ", "ห้องน้ำ", "ชื่อ", "กิน", "รัก", "ไม่", "หน้ากาก", "ผีเสื้อ", "เที่ยงคืน"]
    made = render_signs(signs, base_url)
    dec = pd.read_csv(paths.PKG / "data" / "harvest_decisions.csv")
    qc = pd.read_csv(paths.HARVEST / "qc.csv")
    ph = dec[dec.decision == "phrase"].merge(qc[["id", "word", "video", "span_f0", "span_f1"]], on=["id", "word"])
    want = ["คุณกินข้าวหรือยัง", "คุณชื่ออะไร", "คุณอายุเท่าไหร่", "พรุ่งนี้คุณจะไปไหน", "เมื่อวานไปไหนมา", "วันนี้วันอะไร",
            "ห้องน้ำไปทางไหน", "ฉันไม่สบาย"]
    R = Renderer(base_url)
    out = []
    try:
        for k, w in enumerate(want):
            r = ph[ph.word == w]
            if not len(r):
                continue
            r = r.iloc[0]
            sheet, res = render_sentence(R, w, r.video, video_span=(r.span_f0 / 25.0, r.span_f1 / 25.0))
            p = OUT / f"sent_{k:02d}.jpg"
            cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(p))
            out.append(dict(text=w, file=str(p), glosses=" ".join(u["gloss"] for u in res["ir"]["units"]),
                            dur_avatar=res["ir"]["confidence"].get("duration_s"), dur_real=round((r.span_f1 - r.span_f0) / 25.0, 2)))
    finally:
        R.close()
    pd.DataFrame(out).to_csv(OUT / "sentences.csv", index=False, encoding="utf-8-sig")
    return made, out
