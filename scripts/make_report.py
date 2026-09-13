# -*- coding: utf-8 -*-
"""
make_report.py -- Turn the harvest state DB into (a) analysis-ready CSV
metadata and (b) a PDF collection report.

Outputs
-------
  metadata/metadata_master.csv     every successfully downloaded clip
  metadata/metadata_<level>.csv    one per linguistic level
  metadata/failures.csv            what did not come down, and why
  metadata/sources_manifest.csv    provenance / access-status per source
  reports/collection_report.pdf    human-readable report (Thai-capable)

Usage:  python make_report.py
"""
from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tsl_common import PROJECT_ROOT, human_bytes, human_duration  # noqa: E402

STATE_DB = PROJECT_ROOT / "state" / "harvest.db"
META_DIR = PROJECT_ROOT / "metadata"
REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORT_DIR / "_figures"

# --------------------------------------------------------------------------
# Provenance / access status for every source in scope. This is deliberately
# part of the deliverable: for a pre-training corpus, where each clip came from
# and under what terms matters as much as the bytes.
# --------------------------------------------------------------------------
SOURCE_MANIFEST: List[Dict[str, str]] = [
    {
        "source_key": "ttrs_dictionary",
        "organisation": "Thai Telecommunication Relay Service (TTRS), NECTEC/NSTDA",
        "entry_point": "https://dic.ttrs.or.th",
        "api": "https://nodedic.ttrs.or.th  +  https://apidic.ttrs.or.th",
        "status": "HARVESTED",
        "level": "word (plus a small continuous tail)",
        "labels": "Thai gloss, definition, part of speech, category, signer gender, source book",
        "robots": "no robots.txt served; no crawl restriction declared",
        "notes": "Primary word-level corpus. Public JSON API used by the site's own front-end.",
    },
    {
        "source_key": "youtube_thaipbs_bigsign",
        "organisation": "Thai PBS -- 'Big Sign' full-frame sign language",
        "entry_point": "https://www.youtube.com/channel/UCFpFQ_j7AbBjD0Dpc_ZJQqQ",
        "api": "yt-dlp",
        "status": "HARVESTED",
        "level": "continuous",
        "labels": "video title, description, subtitle tracks (th/en where present)",
        "robots": "accessed through yt-dlp with request throttling",
        "notes": "Full-screen Thai Sign Language programming; signer fills the frame.",
    },
    {
        "source_key": "youtube_keyword_sets",
        "organisation": "Mixed Thai publishers, schools, deaf-community channels",
        "entry_point": "YouTube search (9 curated Thai-language queries)",
        "api": "yt-dlp",
        "status": "HARVESTED",
        "level": "word / sentence / continuous (per query)",
        "labels": "video title, description, subtitle tracks",
        "robots": "accessed through yt-dlp with request throttling",
        "notes": "Titles are filtered for an explicit sign-language marker so that "
                 "every clip carries a usable text label.",
    },
    {
        "source_key": "nadt_th_sl",
        "organisation": "National Association of the Deaf in Thailand (สมาคมคนหูหนวกแห่งประเทศไทย)",
        "entry_point": "https://www.th-sl.com",
        "api": "n/a",
        "status": "NOT HARVESTED -- express reservation of rights",
        "level": "word (Thai Sign Language Database)",
        "labels": "Thai word + handshape/location/movement/orientation parameters",
        "robots": "Content-Signal: ai-train=no, use=reference;  User-agent: ClaudeBot -> Disallow: /",
        "notes": "The site's robots.txt both disallows this agent by name and carries an "
                 "express machine-readable reservation against AI training. Excluded. "
                 "A research licence should be requested directly from NADT.",
    },
    {
        "source_key": "royal_society",
        "organisation": "Royal Society of Thailand (สำนักงานราชบัณฑิตยสภา)",
        "entry_point": "https://royalsociety.go.th",
        "api": "n/a",
        "status": "NOT HARVESTED -- no video resource located",
        "level": "-",
        "labels": "-",
        "robots": "-",
        "notes": "The Royal Society publishes sign-language terminology in print/PDF form. "
                 "No sign-language video archive was found on its web properties.",
    },
    {
        "source_key": "parliament_direct",
        "organisation": "Parliament of Thailand / TPchannel",
        "entry_point": "https://www.parliament.go.th , TPchannel on YouTube",
        "api": "yt-dlp (keyword-scoped)",
        "status": "PARTIAL -- interpreter-inset material only",
        "level": "continuous",
        "labels": "video title; interpreter appears as a small picture-in-picture inset",
        "robots": "accessed through yt-dlp with request throttling",
        "notes": "Full session broadcasts are multi-hour with a small interpreter inset, so "
                 "they need cropping before they are useful. Only briefing-style clips "
                 "matching the sign-language filter were taken.",
    },
    {
        "source_key": "thaisignvis_tomtun",
        "organisation": "TOMTUN / ThaiSignVis",
        "entry_point": "thaisignvis.org (DNS does not resolve); tomtun.com (HTTP 520)",
        "api": "n/a",
        "status": "NOT HARVESTED -- host unreachable",
        "level": "-",
        "labels": "-",
        "robots": "-",
        "notes": "Neither host resolved to a working sign-language video service at "
                 "collection time. No public mirror or dataset release was located.",
    },
    {
        "source_key": "ku_thsl_dictionary",
        "organisation": "Kasetsart University -- ThSL digital dictionary",
        "entry_point": "https://pirun.ku.ac.th/~fhumalt/THSL/",
        "api": "n/a",
        "status": "NOT HARVESTED -- obsolete media format",
        "level": "word",
        "labels": "Thai word",
        "robots": "-",
        "notes": "The dictionary serves its signs through Adobe Shockwave/Flash, which no "
                 "longer runs and exposes no extractable video files.",
    },
]

CSV_COLUMNS = [
    "uid", "source", "level", "native_id", "label", "label_type",
    "rel_path", "filename", "bytes", "sha256",
    "duration_sec", "width", "height", "fps", "vcodec", "acodec",
    "category_th", "content_type_th", "parts_of_speech_th", "definition_th",
    "synonyms_th", "signer_gender_th", "signer_creator", "reference_book_th",
    "channel", "channel_id", "upload_date", "subtitle_files",
    "source_page", "discovered_at", "completed_at",
]


def rows_for_csv(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    q = ("SELECT * FROM items WHERE status='done' AND rel_path IS NOT NULL "
         "ORDER BY source, level, uid")
    for r in con.execute(q):
        meta = {}
        if r["meta_json"]:
            try:
                meta = json.loads(r["meta_json"])
            except json.JSONDecodeError:
                meta = {}
        yt = meta.get("youtube") or {}
        subs = meta.get("subtitle_files") or []
        out.append({
            "uid": r["uid"],
            "source": r["source"],
            "level": r["level"],
            "native_id": r["native_id"],
            "label": r["label"],
            "label_type": r["label_type"],
            "rel_path": r["rel_path"],
            "filename": Path(r["rel_path"]).name if r["rel_path"] else "",
            "bytes": r["bytes"],
            "sha256": r["sha256"],
            "duration_sec": round(r["duration"], 3) if r["duration"] else "",
            "width": r["width"], "height": r["height"],
            "fps": r["fps"], "vcodec": r["vcodec"], "acodec": r["acodec"],
            "category_th": meta.get("category_th", ""),
            "content_type_th": meta.get("content_type_th", ""),
            "parts_of_speech_th": meta.get("parts_of_speech_th", ""),
            "definition_th": (meta.get("definition_th", "") or "").replace("\n", " "),
            "synonyms_th": meta.get("synonyms_th", ""),
            "signer_gender_th": meta.get("signer_gender_th", ""),
            "signer_creator": meta.get("signer_creator", ""),
            "reference_book_th": meta.get("reference_book_th", ""),
            "channel": meta.get("channel") or yt.get("channel") or "",
            "channel_id": meta.get("channel_id") or yt.get("channel_id") or "",
            "upload_date": yt.get("upload_date", ""),
            "subtitle_files": ";".join(subs),
            "source_page": meta.get("source_page", ""),
            "discovered_at": r["discovered_at"],
            "completed_at": r["completed_at"],
        })
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel on a Thai Windows box opens it without mojibake.
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def build_figures(rows: List[Dict[str, Any]]) -> Dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs: Dict[str, Path] = {}
    ink, grid = "#1f2933", "#d9e2ec"
    palette = ["#2f6f9f", "#7aa8c7", "#c2703d", "#5c8f6b", "#9b7fb5"]

    def style(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(grid)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=ink, labelsize=8)
        ax.yaxis.grid(True, color=grid, linewidth=0.7)
        ax.set_axisbelow(True)

    # 1. clips per level
    lv = Counter(r["level"] for r in rows)
    if lv:
        fig, ax = plt.subplots(figsize=(5.4, 2.7), dpi=200)
        keys = ["word", "sentence", "continuous"]
        keys = [k for k in keys if k in lv] + [k for k in lv if k not in keys]
        ax.bar(keys, [lv[k] for k in keys], color=palette[: len(keys)], width=0.55)
        for i, k in enumerate(keys):
            ax.text(i, lv[k], "{:,}".format(lv[k]), ha="center", va="bottom",
                    fontsize=8, color=ink)
        ax.set_ylabel("clips", fontsize=8, color=ink)
        ax.set_title("Clips by linguistic level", fontsize=10, color=ink, loc="left")
        style(ax)
        ax.set_ylim(0, max(lv.values()) * 1.18)
        fig.tight_layout()
        p = FIG_DIR / "levels.png"
        fig.savefig(p); plt.close(fig)
        figs["levels"] = p

    # 2. duration histogram
    durs = [float(r["duration_sec"]) for r in rows
            if r["duration_sec"] not in ("", None)]
    if durs:
        fig, ax = plt.subplots(figsize=(5.4, 2.7), dpi=200)
        capped = [min(d, 120) for d in durs]
        ax.hist(capped, bins=40, color=palette[0], edgecolor="white", linewidth=0.4)
        ax.set_xlabel("clip duration (s, capped at 120)", fontsize=8, color=ink)
        ax.set_ylabel("clips", fontsize=8, color=ink)
        ax.set_title("Clip duration distribution", fontsize=10, color=ink, loc="left")
        style(ax)
        fig.tight_layout()
        p = FIG_DIR / "durations.png"
        fig.savefig(p); plt.close(fig)
        figs["durations"] = p

    # 3. resolution mix
    res = Counter("{}x{}".format(r["width"], r["height"])
                  for r in rows if r["width"] and r["height"])
    if res:
        top = res.most_common(8)
        fig, ax = plt.subplots(figsize=(5.4, 2.9), dpi=200)
        labels = [t[0] for t in top][::-1]
        vals = [t[1] for t in top][::-1]
        ax.barh(labels, vals, color=palette[1], height=0.6)
        for i, v in enumerate(vals):
            ax.text(v, i, " {:,}".format(v), va="center", fontsize=7.5, color=ink)
        ax.set_title("Frame size mix", fontsize=10, color=ink, loc="left")
        style(ax)
        ax.xaxis.grid(True, color=grid, linewidth=0.7)
        ax.yaxis.grid(False)
        ax.set_xlim(0, max(vals) * 1.15)
        fig.tight_layout()
        p = FIG_DIR / "resolutions.png"
        fig.savefig(p); plt.close(fig)
        figs["resolutions"] = p

    return figs


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------
def register_thai_font():
    """Find a Thai-capable TTF so Thai glosses render instead of black boxes."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        (r"C:\Windows\Fonts\LeelawUI.ttf", r"C:\Windows\Fonts\LeelaUIb.ttf"),
        (r"C:\Windows\Fonts\leelawad.ttf", r"C:\Windows\Fonts\leelawdb.ttf"),
        (r"C:\Windows\Fonts\tahoma.ttf", r"C:\Windows\Fonts\tahomabd.ttf"),
    ]
    for regular, bold in candidates:
        if os.path.exists(regular):
            try:
                pdfmetrics.registerFont(TTFont("Thai", regular))
                if os.path.exists(bold):
                    pdfmetrics.registerFont(TTFont("Thai-Bold", bold))
                else:
                    pdfmetrics.registerFont(TTFont("Thai-Bold", regular))
                from reportlab.pdfbase.pdfmetrics import registerFontFamily
                registerFontFamily("Thai", normal="Thai", bold="Thai-Bold",
                                   italic="Thai", boldItalic="Thai-Bold")
                return "Thai", "Thai-Bold"
            except Exception:  # noqa: BLE001
                continue
    return "Helvetica", "Helvetica-Bold"


def build_pdf(con: sqlite3.Connection, rows: List[Dict[str, Any]],
              figs: Dict[str, Path], out_path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    FONT, FONT_B = register_thai_font()
    INK = colors.HexColor("#1f2933")
    MUTED = colors.HexColor("#61707d")
    RULE = colors.HexColor("#d9e2ec")
    ACCENT = colors.HexColor("#2f6f9f")

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontName=FONT_B, fontSize=19,
                        leading=23, textColor=INK, alignment=TA_LEFT, spaceAfter=2)
    SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontName=FONT, fontSize=9.5,
                         leading=13, textColor=MUTED, spaceAfter=10)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName=FONT_B, fontSize=12.5,
                        leading=16, textColor=INK, spaceBefore=14, spaceAfter=5)
    H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontName=FONT_B, fontSize=10.5,
                        leading=14, textColor=INK, spaceBefore=9, spaceAfter=3)
    BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontName=FONT, fontSize=9.3,
                          leading=13.4, textColor=INK, spaceAfter=5)
    SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8.1, leading=11.4,
                           textColor=MUTED)
    CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.0, leading=10.8,
                          spaceAfter=0)
    CELLB = ParagraphStyle("CELLB", parent=CELL, fontName=FONT_B)

    def table(data, widths, align_right=(), header=True):
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0,
                  hAlign="LEFT")
        style = [
            ("FONTNAME", (0, 0), (-1, -1), FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 8.0),
            ("TEXTCOLOR", (0, 0), (-1, -1), INK),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ]
        if header:
            style += [
                ("FONTNAME", (0, 0), (-1, 0), FONT_B),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
                ("TOPPADDING", (0, 0), (-1, 0), 5),
            ]
        for c in align_right:
            style.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
        t.setStyle(TableStyle(style))
        return t

    story: List[Any] = []
    now = datetime.now(timezone.utc).astimezone()

    # ---- totals -------------------------------------------------------
    n = len(rows)
    total_bytes = sum(r["bytes"] or 0 for r in rows)
    total_secs = sum(float(r["duration_sec"]) for r in rows
                     if r["duration_sec"] not in ("", None))
    by_level = Counter(r["level"] for r in rows)
    by_source = Counter(r["source"] for r in rows)
    secs_by_level: Dict[str, float] = defaultdict(float)
    bytes_by_level: Dict[str, int] = defaultdict(int)
    for r in rows:
        if r["duration_sec"] not in ("", None):
            secs_by_level[r["level"]] += float(r["duration_sec"])
        bytes_by_level[r["level"]] += r["bytes"] or 0
    uniq_labels = len({r["label"] for r in rows if r["label"]})

    # ---- cover --------------------------------------------------------
    story.append(Paragraph("Thai Sign Language — Raw Video Collection", H1))
    story.append(Paragraph(
        "Harvest report &nbsp;·&nbsp; generated %s" % now.strftime("%Y-%m-%d %H:%M UTC%z"),
        SUB))

    kpi = [[
        Paragraph("<b>%s</b><br/><font size=7.5 color='#61707d'>video clips</font>"
                  % "{:,}".format(n), CELL),
        Paragraph("<b>%s</b><br/><font size=7.5 color='#61707d'>total footage</font>"
                  % human_duration(total_secs), CELL),
        Paragraph("<b>%s</b><br/><font size=7.5 color='#61707d'>on disk</font>"
                  % human_bytes(total_bytes), CELL),
        Paragraph("<b>%s</b><br/><font size=7.5 color='#61707d'>distinct labels</font>"
                  % "{:,}".format(uniq_labels), CELL),
    ]]
    t = Table(kpi, colWidths=[42 * mm] * 4, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 13),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f2f6f9")),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph(
        "Every clip in this collection is raw video — <b>.mp4 as downloaded</b>, no "
        "pose extraction, no feature files. Each one carries a text label (a Thai "
        "gloss, or the publisher's own title) so the corpus is usable both for "
        "self-supervised pre-training and for downstream supervised work.", BODY))

    # ---- 1. what was collected ----------------------------------------
    story.append(Paragraph("1 &nbsp; What was collected", H2))
    data = [["Level", "Clips", "Footage", "Size", "Folder"]]
    folder = {"word": "raw_data/word_level/",
              "sentence": "raw_data/sentence_level/",
              "continuous": "raw_data/continuous/"}
    for lv in ("word", "sentence", "continuous"):
        if by_level.get(lv):
            data.append([
                Paragraph(lv, CELLB),
                "{:,}".format(by_level[lv]),
                human_duration(secs_by_level[lv]),
                human_bytes(bytes_by_level[lv]),
                Paragraph(folder[lv], CELL),
            ])
    data.append([Paragraph("<b>total</b>", CELLB), "{:,}".format(n),
                 human_duration(total_secs), human_bytes(total_bytes), ""])
    story.append(table(data, [26 * mm, 20 * mm, 26 * mm, 24 * mm, 62 * mm],
                       align_right=(1, 2, 3)))

    if "levels" in figs:
        story.append(Spacer(1, 4 * mm))
        story.append(Image(str(figs["levels"]), width=135 * mm, height=67 * mm))

    # ---- 2. per source ------------------------------------------------
    story.append(Paragraph("2 &nbsp; Yield by source", H2))
    src_rows = [["Source", "Clips", "Footage", "Size"]]
    for s, c in by_source.most_common():
        ss_secs = sum(float(r["duration_sec"]) for r in rows
                      if r["source"] == s and r["duration_sec"] not in ("", None))
        ss_bytes = sum(r["bytes"] or 0 for r in rows if r["source"] == s)
        src_rows.append([Paragraph(s, CELL), "{:,}".format(c),
                         human_duration(ss_secs), human_bytes(ss_bytes)])
    story.append(table(src_rows, [62 * mm, 22 * mm, 30 * mm, 28 * mm],
                       align_right=(1, 2, 3)))

    story.append(PageBreak())

    # ---- 3. source manifest -------------------------------------------
    story.append(Paragraph("3 &nbsp; Sources attempted, and their status", H2))
    story.append(Paragraph(
        "Provenance is part of the deliverable. Two of the requested sources were "
        "deliberately left alone, and two were unreachable — the reasons are recorded "
        "here rather than buried in a log.", BODY))

    man = [["Source", "Status", "Detail"]]
    for m in SOURCE_MANIFEST:
        status = m["status"]
        colour = "#2f6f9f" if status.startswith("HARVESTED") else (
            "#c2703d" if status.startswith("PARTIAL") else "#8a4b4b")
        man.append([
            Paragraph("<b>%s</b><br/><font size=7 color='#61707d'>%s</font>"
                      % (m["organisation"], m["entry_point"]), CELL),
            Paragraph("<font color='%s'><b>%s</b></font>" % (colour, status), CELL),
            Paragraph("%s<br/><font size=7 color='#61707d'>%s</font>"
                      % (m["notes"], m["robots"]), CELL),
        ])
    story.append(table(man, [52 * mm, 32 * mm, 78 * mm]))

    # ---- 4. label coverage --------------------------------------------
    story.append(Paragraph("4 &nbsp; Label coverage", H2))
    lab_rows = [["Label type", "Clips", "What it is"]]
    lt = Counter(r["label_type"] for r in rows)
    explain = {
        "gloss": "Curated Thai gloss from the TTRS dictionary, with definition, "
                 "part of speech and category alongside.",
        "title": "Publisher's own video title, plus description and any subtitle "
                 "track that came with it.",
    }
    for k, c in lt.most_common():
        lab_rows.append([Paragraph(k or "—", CELLB), "{:,}".format(c),
                         Paragraph(explain.get(k, ""), CELL)])
    story.append(table(lab_rows, [26 * mm, 20 * mm, 116 * mm], align_right=(1,)))

    cats = Counter(r["category_th"] for r in rows if r["category_th"])
    if cats:
        story.append(Paragraph("Top vocabulary categories (TTRS)", H3))
        cat_rows = [["Category (Thai)", "Clips"]]
        for c, k in cats.most_common(14):
            cat_rows.append([Paragraph(c, CELL), "{:,}".format(k)])
        story.append(table(cat_rows, [92 * mm, 22 * mm], align_right=(1,)))
        story.append(Paragraph(
            "%d distinct categories in total." % len(cats), SMALL))

    story.append(PageBreak())

    # ---- 5. technical profile -----------------------------------------
    story.append(Paragraph("5 &nbsp; Technical profile", H2))
    story.append(Paragraph(
        "Relevant to SignDINO-style pre-training: these are the raw decode "
        "characteristics of the corpus as delivered.", BODY))

    durs = sorted(float(r["duration_sec"]) for r in rows
                  if r["duration_sec"] not in ("", None))
    if durs:
        def pct(p):
            return durs[min(len(durs) - 1, int(len(durs) * p))]
        stat_rows = [
            ["Statistic", "Value"],
            ["clips with a readable duration", "{:,}".format(len(durs))],
            ["median clip length", "%.1f s" % pct(0.50)],
            ["p10 / p90 clip length", "%.1f s  /  %.1f s" % (pct(0.10), pct(0.90))],
            ["shortest / longest", "%.1f s  /  %s" % (durs[0], human_duration(durs[-1]))],
            ["mean bitrate", "%.0f kbps" % (
                (total_bytes * 8 / 1000) / total_secs if total_secs else 0)],
        ]
        story.append(table(stat_rows, [62 * mm, 62 * mm]))

    if "durations" in figs:
        story.append(Spacer(1, 3 * mm))
        story.append(Image(str(figs["durations"]), width=135 * mm, height=67 * mm))
    if "resolutions" in figs:
        story.append(Spacer(1, 3 * mm))
        story.append(Image(str(figs["resolutions"]), width=135 * mm, height=72 * mm))

    fps_c = Counter(round(float(r["fps"])) for r in rows if r["fps"])
    codec_c = Counter(r["vcodec"] for r in rows if r["vcodec"])
    if fps_c or codec_c:
        story.append(Paragraph("Frame rate and codec", H3))
        fr = [["Frame rate", "Clips", "Video codec", "Clips"]]
        fps_list = fps_c.most_common(6)
        cod_list = codec_c.most_common(6)
        for i in range(max(len(fps_list), len(cod_list))):
            a = ("%d fps" % fps_list[i][0], "{:,}".format(fps_list[i][1])) if i < len(fps_list) else ("", "")
            b = (cod_list[i][0], "{:,}".format(cod_list[i][1])) if i < len(cod_list) else ("", "")
            fr.append([a[0], a[1], b[0], b[1]])
        story.append(table(fr, [32 * mm, 22 * mm, 32 * mm, 22 * mm],
                           align_right=(1, 3)))

    story.append(PageBreak())

    # ---- 6. reliability ------------------------------------------------
    story.append(Paragraph("6 &nbsp; Collection reliability", H2))
    counts = {r[0]: r[1] for r in con.execute(
        "SELECT status, COUNT(*) FROM items GROUP BY status")}
    disc = sum(counts.values())
    ok_rows = [["Outcome", "Items", "Share of catalogue"]]
    for k in ("done", "pending", "failed", "skipped"):
        if counts.get(k):
            ok_rows.append([k, "{:,}".format(counts[k]),
                            "%.1f%%" % (100.0 * counts[k] / disc)])
    ok_rows.append([Paragraph("<b>catalogued</b>", CELLB),
                    "{:,}".format(disc), "100.0%"])
    story.append(table(ok_rows, [40 * mm, 26 * mm, 34 * mm], align_right=(1, 2)))

    fail = con.execute(
        "SELECT source, last_error, COUNT(*) c FROM items "
        "WHERE status IN ('failed','pending') AND last_error IS NOT NULL "
        "GROUP BY source, substr(last_error,1,45) ORDER BY c DESC LIMIT 12"
    ).fetchall()
    if fail:
        story.append(Paragraph("Most common failure modes", H3))
        f_rows = [["Source", "Error", "Count"]]
        for r in fail:
            f_rows.append([Paragraph(r[0], CELL),
                           Paragraph((r[1] or "")[:150], CELL),
                           "{:,}".format(r[2])])
        story.append(table(f_rows, [34 * mm, 100 * mm, 18 * mm], align_right=(2,)))
    else:
        story.append(Paragraph("No failures were recorded.", BODY))

    story.append(Paragraph("How the collector stayed polite and stable", H3))
    for line in [
        "<b>Per-host rate limiting.</b> Every request passes a token bucket keyed on "
        "hostname — one request per configured interval plus random jitter, shared "
        "across all worker threads, so raising concurrency never raises request rate.",
        "<b>Exponential backoff with full jitter.</b> Retryable failures (408, 425, 429, "
        "5xx, timeouts, connection resets) back off over a window that doubles per "
        "attempt, sampled uniformly to avoid retry convoys. A <i>Retry-After</i> header "
        "always wins over the computed delay.",
        "<b>Adaptive braking.</b> A 429 or 503 adds a decaying per-host cooldown on top "
        "of the static spacing, so a strained server gets more room automatically.",
        "<b>Atomic writes.</b> Downloads stream to <i>.part</i> and are renamed into "
        "place only after length, sha256 and an ffprobe decode check all pass. A "
        "truncated file can never be mistaken for a good one.",
        "<b>Idempotent resume.</b> All state lives in a SQLite catalogue. Interrupting "
        "the run costs only the in-flight downloads; re-running continues from there "
        "and re-downloads nothing.",
        "<b>Content-level deduplication.</b> Clips are hashed; the same bytes reached "
        "through two different listings are recorded once.",
    ]:
        story.append(Paragraph("• " + line, BODY))

    # ---- 7. how to use -------------------------------------------------
    story.append(Paragraph("7 &nbsp; Files you now have", H2))
    files = [["Path", "What it holds"]]
    for p, d in [
        ("raw_data/word_level/", "Isolated lexical signs — one gloss per clip."),
        ("raw_data/sentence_level/", "Single utterances and short phrases."),
        ("raw_data/continuous/", "Running signing: interpreted news, songs, briefings."),
        ("raw_data/_quarantine/", "Files that downloaded but failed decode validation."),
        ("metadata/metadata_master.csv", "Every clip, one row each, all label fields."),
        ("metadata/metadata_word.csv …", "The same split by linguistic level."),
        ("metadata/failures.csv", "Anything not retrieved, with the reason."),
        ("metadata/sources_manifest.csv", "Provenance and access status per source."),
        ("state/harvest.db", "SQLite catalogue — the resume point."),
        ("*.json (beside each clip)", "Lossless per-clip record from upstream."),
        ("*.vtt (beside YouTube clips)", "Subtitle tracks, where the publisher had them."),
    ]:
        files.append([Paragraph("<font face='Courier'>%s</font>" % p, CELL),
                      Paragraph(d, CELL)])
    story.append(table(files, [62 * mm, 78 * mm]))

    story.append(Paragraph("8 &nbsp; Before you train on this", H2))
    for line in [
        "<b>Licensing is not settled by collection.</b> These clips were gathered from "
        "public endpoints for research. They are not public domain. Clearing "
        "redistribution or model release — particularly for the TTRS dictionary and "
        "any broadcaster material — is a separate step.",
        "<b>The NADT database was deliberately skipped.</b> th-sl.com carries an express "
        "machine-readable reservation against AI training and disallows this agent by "
        "name. It is the single richest Thai word-level source, so a direct research "
        "licence request to the association is the highest-value next move.",
        "<b>Continuous clips need cropping.</b> Interpreted broadcast material often puts "
        "the signer in a corner inset. Big Sign material does not — the signer fills the "
        "frame — which makes it the better continuous subset to start from.",
        "<b>Titles are weak labels.</b> YouTube-sourced rows carry the publisher's title, "
        "which names the topic rather than glossing each sign. Treat them as coarse "
        "supervision; the TTRS rows are the ones with true per-clip glosses.",
    ]:
        story.append(Paragraph("• " + line, BODY))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 12 * mm, "Thai Sign Language raw video collection")
        canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, "page %d" % doc.page)
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(18 * mm, 15 * mm, A4[0] - 18 * mm, 15 * mm)
        canvas.restoreState()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=20 * mm,
        title="Thai Sign Language Raw Video Collection",
        author="SIGN_DATA harvester",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# --------------------------------------------------------------------------
def main() -> int:
    if not STATE_DB.exists():
        print("no state DB at %s -- run harvest.py first" % STATE_DB)
        return 1
    con = sqlite3.connect(str(STATE_DB))
    con.row_factory = sqlite3.Row

    rows = rows_for_csv(con)
    META_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(META_DIR / "metadata_master.csv", rows, CSV_COLUMNS)
    for lv in ("word", "sentence", "continuous"):
        sub = [r for r in rows if r["level"] == lv]
        if sub:
            write_csv(META_DIR / ("metadata_%s.csv" % lv), sub, CSV_COLUMNS)

    fails = [{
        "uid": r["uid"], "source": r["source"], "level": r["level"],
        "label": r["label"], "url": r["url"], "status": r["status"],
        "attempts": r["attempts"], "last_error": r["last_error"],
    } for r in con.execute(
        "SELECT * FROM items WHERE status IN ('failed','skipped') "
        "OR (status='pending' AND attempts>0) ORDER BY source, uid")]
    write_csv(META_DIR / "failures.csv", fails,
              ["uid", "source", "level", "label", "url", "status",
               "attempts", "last_error"])

    write_csv(META_DIR / "sources_manifest.csv", SOURCE_MANIFEST,
              list(SOURCE_MANIFEST[0].keys()))

    figs = build_figures(rows)
    pdf = REPORT_DIR / "collection_report.pdf"
    build_pdf(con, rows, figs, pdf)
    con.close()

    total_b = sum(r["bytes"] or 0 for r in rows)
    total_s = sum(float(r["duration_sec"]) for r in rows
                  if r["duration_sec"] not in ("", None))
    print("metadata_master.csv : %d rows" % len(rows))
    print("failures.csv        : %d rows" % len(fails))
    print("collection_report.pdf -> %s" % pdf)
    print("corpus: %s of video, %s on disk" % (human_duration(total_s),
                                               human_bytes(total_b)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
