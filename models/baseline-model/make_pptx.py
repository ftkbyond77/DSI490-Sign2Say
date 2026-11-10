"""baseline.pptx — 5 slides presenting the baseline (Uni-Sign + Thai retrieval).   python models/baseline-model/make_pptx.py
Needs results/*.json and figures/fig_*.png (collect_metrics.py → run_v10_slide.py → make_figures_slides.py)."""
import json
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor as C
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches as I, Pt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
M = json.load(open(HERE / "results" / "baseline_slide_metrics.json", encoding="utf8"))
A, ISO = M["agg"], M["isolated"]
INK, MUTE, LINE, BG = C(0x1F, 0x29, 0x37), C(0x6B, 0x72, 0x80), C(0xD1, 0xD5, 0xDB), C(0xF3, 0xF4, 0xF6)
BLUE, GREEN, ORANGE, RED, PURPLE, WHITE = C(0x00, 0x72, 0xB2), C(0x00, 0x9E, 0x73), C(0xE6, 0x9F, 0x00), C(0xD5, 0x5E, 0x00), C(0x8E, 0x6B, 0xBF), C(255, 255, 255)
PALE = dict(blue=C(0xDB, 0xEA, 0xFE), yellow=C(0xFD, 0xE6, 0x8A), green=C(0xBB, 0xF7, 0xD0), purple=C(0xE9, 0xD5, 0xFF), red=C(0xFE, 0xCA, 0xCA), grey=BG, cream=C(0xFE, 0xF3, 0xC7))
FONT = "Calibri"

prs = Presentation(); prs.slide_width, prs.slide_height = I(13.333), I(7.5)


# ------------------------------------------------------------------ helpers
def text(s, x, y, w, h, paras, size=12, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, shape=None, after=3, line=None):
    tb = shape or s.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = I(.05); tf.margin_top = tf.margin_bottom = I(.03)
    paras = paras if isinstance(paras, list) else [paras]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); para.alignment = align; para.space_after = Pt(after)
        if line: para.line_spacing = line
        for sg in (p if isinstance(p, list) else [p]):
            t, b, col = (sg, bold, color) if isinstance(sg, str) else (sg[0], sg[1], sg[2] if len(sg) > 2 else color)
            r = para.add_run(); r.text = t; r.font.size = Pt(size); r.font.bold = b; r.font.color.rgb = col; r.font.name = FONT
    return tb


def box(s, x, y, w, h, fill=None, line=LINE, lw=1, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=.08):
    r = s.shapes.add_shape(shape, I(x), I(y), I(w), I(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        r.adjustments[0] = radius
    if fill is None: r.fill.background()
    else: r.fill.solid(); r.fill.fore_color.rgb = fill
    if line is None: r.line.fill.background()
    else: r.line.color.rgb = line; r.line.width = Pt(lw)
    r.shadow.inherit = False
    return r


def node(s, x, y, w, h, paras, fill, size=10, line=INK, lw=.75, color=INK, bold=False):
    b = box(s, x, y, w, h, fill, line, lw); text(s, 0, 0, 0, 0, paras, size, color, bold, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, shape=b, after=1); return b


def arrow(s, x1, y1, x2, y2, color=INK, lw=1.5, head=True):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, I(x1), I(y1), I(x2), I(y2)); c.line.color.rgb = color; c.line.width = Pt(lw)
    if head:
        ln = c.line._get_or_add_ln(); tail = etree.SubElement(ln, "{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd"); tail.set("type", "triangle")
    return c


def new_slide(num, tag, title, sub):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    box(s, 0, 0, 13.333, .09, BLUE, None, shape=MSO_SHAPE.RECTANGLE)
    text(s, .5, .22, 8, .28, [[(f"{num:02d}", True, BLUE), (f"  ·  {tag.upper()}", True, MUTE)]], 11)
    text(s, .5, .48, 12.3, .6, title, 28, INK, True)
    text(s, .5, 1.04, 12.3, .32, sub, 14, MUTE)
    text(s, 11.3, .22, 1.55, .28, "Baseline · Uni-Sign", 10, MUTE, align=PP_ALIGN.RIGHT)
    return s


def chip(s, x, y, w, h, label, fill, color=WHITE, size=10):
    node(s, x, y, w, h, label, fill, size, None, 0, color, True)


# ------------------------------------------------------------------ 1 · the model
s = new_slide(1, "The model", "Baseline = Uni-Sign, used exactly as published", "A public, pre-trained pose-based sign encoder · frozen · no Thai training")
# model card
box(s, .5, 1.6, 4.55, 5.4, BG, None, radius=.04)
text(s, .7, 1.7, 4.2, .35, "Model card", 16, BLUE, True)
rows = [("Model", "Uni-Sign — pose-only variant"), ("Paper", "Li et al., “Uni-Sign: Toward Unified Sign Language Understanding at Scale”, ICLR 2025 (arXiv 2501.15187)"),
        ("Checkpoint", "ZechengLi19/Uni-Sign · wlasl_pose_only_islr.pth"), ("Size", "90.3 M parameters — mT5-base encoder (≈ 85 M) + ST-GCN pose branch (≈ 5 M)"),
        ("Input", "2-D whole-body keypoints only (no RGB) · 25 fps"), ("Output we use", "768-d clip embedding — the model has no Thai head")]
y = 2.15
for k, v in rows:
    h = .95 if k == "Paper" else .78 if k in ("Size", "Checkpoint") else .6
    text(s, .7, y, 1.25, h, k, 11.5, MUTE, True); text(s, 1.95, y, 3.0, h, v, 12, INK); y += h + .06
# training pipeline
text(s, 5.45, 1.6, 7.4, .35, "How it was trained  →  how we use it", 16, BLUE, True)
steps = [("1", "Pre-training", "CSL-News", "1,985 h of Chinese Sign Language video–text pairs; unified sign → text generation with mT5", PALE["blue"]),
         ("2", "Fine-tuning", "WLASL-2000", "Isolated American Sign Language: 2,000 glosses, ~100 signers → the checkpoint we load", PALE["blue"]),
         ("3", "Our baseline", "Frozen, zero-shot", "No Thai training. Embed Thai clips into a bank; recognise = nearest neighbour", PALE["yellow"])]
for i, (n, a, b, c, col) in enumerate(steps):
    x = 5.45 + i * 2.52
    box(s, x, 2.05, 2.3, 2.45, col, INK, .75)
    chip(s, x + .12, 2.15, .34, .34, n, BLUE, size=12)
    text(s, x + .5, 2.12, 1.75, .4, a, 14, INK, True, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + .12, 2.6, 2.1, .4, b, 15, BLUE, True)
    text(s, x + .12, 3.05, 2.1, 1.4, c, 11.5, INK)
    if i < 2:
        arrow(s, x + 2.3, 3.27, x + 2.52, 3.27)
# why + thai side
text(s, 5.45, 4.7, 7.4, .35, "Why this baseline", 16, BLUE, True)
why = [("Public & pre-trained", "strong sign representation without any Thai data"), ("Pose input", "only keypoints → ignores background, clothes, lighting"),
       ("Fair reference", "same landmarks, bank and scorer as later versions")]
for i, (a, b) in enumerate(why):
    x = 5.45 + i * 2.52
    box(s, x, 5.1, 2.3, 1.0, None, LINE, 1); text(s, x + .08, 5.14, 2.15, .3, a, 12.5, INK, True); text(s, x + .08, 5.45, 2.15, .65, b, 11, MUTE)
box(s, 5.45, 6.3, 7.4, .7, PALE["cream"], ORANGE, 1)
text(s, 5.6, 6.32, 7.15, .66, [[("Thai side:  ", True, INK), ("bank ≈ 19k clips · 9.5k concepts   |   vocabulary 328 daily words   |   no tagger · calibration · language layer · TTS (output = text)", False, INK)]], 11.5, anchor=MSO_ANCHOR.MIDDLE)

# ------------------------------------------------------------------ 2 · architecture
s = new_slide(2, "Architecture", "From video to word: a GNN + Transformer encoder with retrieval", "Left: full model, input → output  ·  Right: how the face enters (35 %)")
PX = dict(body=BLUE, lh=GREEN, rh=RED, face=PURPLE)
text(s, .5, 1.45, 8.1, .3, "A · Pose encoder (part-wise ST-GCN)", 13, BLUE, True)
yA = 1.85; hA = 2.35
node(s, .5, yA + .55, .95, 1.2, ["Video", "25 fps", [("T×H×W×3", False, MUTE)]], PALE["blue"], 10)
arrow(s, 1.45, yA + 1.15, 1.75, yA + 1.15)
node(s, 1.75, yA + .35, 1.4, 1.6, [[("MediaPipe Holistic", True)], "pose 33", "hands 2×21", "face mesh 478", [("→ 133-slot schema", False, MUTE)]], PALE["blue"], 9.5)
arrow(s, 3.15, yA + 1.15, 3.45, yA + 1.15)
parts = [("body", "[T, 9, 3]", BLUE), ("left hand", "[T, 21, 3]", GREEN), ("right hand", "[T, 21, 3]", RED), ("face", "[T, 18, 3]", PURPLE)]
for i, (a, b, c) in enumerate(parts):
    yy = yA + .05 + i * .58
    node(s, 3.45, yy, 1.3, .5, [[(a, True), (f"  {b}", False, MUTE)]], WHITE, 8.5, c, 2)
text(s, 3.45, yA + 2.3, 1.3, .3, "x, y, confidence", 8.5, MUTE, align=PP_ALIGN.CENTER)
box(s, 5.05, yA - .05, 2.35, 2.62, None, MUTE, 1, radius=.04)
text(s, 5.05, yA - .03, 2.35, .26, "ST-GCN × 4  (one per part)", 9.5, MUTE, True, PP_ALIGN.CENTER)
inner = [("Linear 3 → 64", PALE["yellow"]), ("Spatial GCN × 3\n64 → 64 → 128 → 256", PALE["yellow"]), ("+ body joint (wrist / nose)\nhands & face only · detached", PALE["grey"]),
         ("Temporal GCN × 3\n256 ch · kernel 5 (≈ 0.5 s)", PALE["yellow"]), ("mean over joints → [T, 256]", PALE["red"])]
for i, (t, col) in enumerate(inner):
    yy = yA + .25 + i * .45
    node(s, 5.15, yy, 2.15, .4, t.split("\n") if "\n" in t else t, col, 8.5, MUTE if col == PALE["grey"] else INK, .75)
for i in range(4):
    arrow(s, 4.75, yA + .3 + i * .58, 5.05, yA + 1.15)
arrow(s, 7.4, yA + 1.15, 7.7, yA + 1.15)
node(s, 7.7, yA + .35, .9, 1.6, ["concat", "4 × 256", "= 1024", [("+ part_para", False, MUTE)], "Linear → 768"], PALE["red"], 9)
# row B
text(s, 1.2, 4.58, 7.3, .3, "B · Sequence encoder and retrieval", 13, BLUE, True)
yB = 5.0
node(s, .5, yB, 1.45, 1.35, [[("frame tokens", True)], "[T, 768]", "⊕ task prefix", "13 tokens"], PALE["blue"], 9.5)
arrow(s, 1.95, yB + .67, 2.2, yB + .67)
node(s, 2.2, yB, 1.95, 1.35, [[("mT5-base encoder", True)], "12 layers · d = 768", "12 heads · gated-GELU", [("full self-attention over frames", False, MUTE)]], PALE["green"], 9.5)
arrow(s, 4.15, yB + .67, 4.4, yB + .67)
node(s, 4.4, yB, 1.45, 1.35, [[("mean-pool over T", True)], "L2-normalise", [("z ∈ ℝ⁷⁶⁸", False, MUTE)]], PALE["purple"], 9.5)
arrow(s, 5.85, yB + .67, 6.1, yB + .67)
node(s, 6.1, yB, 1.55, 1.35, [[("cosine kNN", True)], "vs bank Z₀", "≈ 19k Thai clips", [("score = mean of top-2", False, MUTE)]], PALE["red"], 9.5)
arrow(s, 7.65, yB + .67, 7.85, yB + .67)
node(s, 7.85, yB + .22, .75, .9, [[("top-5", True)], "words"], GREEN, 10, None, 0, WHITE)
# connector A → B  (down from the concat box, left along the gap, down into row B)
arrow(s, 8.15, yA + 1.95, 8.15, 4.52, head=False); arrow(s, 8.15, 4.52, .95, 4.52, head=False); arrow(s, .95, 4.52, .95, yB)
text(s, .5, 6.5, 8.1, .6, [[("All 90.3 M parameters frozen. ", True), "Segmentation of continuous video is outside the model: hands-raised runs are cut every ≈ 49 frames (≈ 2 s), each cut → one embedding → one word."]], 10.5, MUTE)
# right: face
box(s, 8.85, 1.45, 4.1, 5.6, BG, None, radius=.03)
text(s, 9.0, 1.5, 3.8, .35, "Face branch — where it joins", 14, PURPLE, True)
text(s, 9.0, 1.82, 3.85, .8, [[("Extractor: ", True), "5 MediaPipe-Pose anchors ◇ (nose, eyes, mouth corners) → a mean face template fitted per frame → ", ("18 points", True), ": jaw 9 · inner lip 8 · nose 1"]], 10, after=0)
s.shapes.add_picture(str(FIG / "fig_face_graph.png"), I(9.0), I(2.65), height=I(1.65))
for i, (c, t) in enumerate(((BLUE, "jaw chain (9)"), (RED, "mouth ring (8)"), (GREEN, "nose hub (1)\nlinked to all"), (ORANGE, "pose anchor ◇ (5)\nused to fit"))):
    yy = 2.62 + i * .4; box(s, 11.05, yy + .08, .14, .14, c, None, shape=MSO_SHAPE.OVAL); text(s, 11.25, yy - .02, 1.7, .5, t.split("\n"), 10, INK, after=0)
fsteps = [("[T, 18, 3]  nose-relative, ÷ body scale", PALE["blue"]), ("Linear 3 → 64 · Spatial GCN × 3 → 256 (graph above)", PALE["yellow"]),
          ("⊕ nose feature of the body branch (detached)", PALE["grey"]), ("Temporal GCN × 3 · mean over 18 nodes → [T, 256]", PALE["yellow"]),
          ("JOIN: 4th block of the concat (ch. 769–1024) → Linear 1024 → 768", PALE["purple"])]
for i, (t, col) in enumerate(fsteps):
    yy = 4.5 + i * .44; node(s, 9.0, yy, 3.8, .32, t, col, 8.5, PURPLE if i == 4 else INK, 1.25 if i == 4 else .75, bold=(i == 4))
    if i < 4: arrow(s, 10.9, yy + .32, 10.9, yy + .44, lw=1.2)
text(s, 9.0, 6.62, 3.85, .4, [[("Rigid template: ", True), "head position · scale · roll only, no expression. 478-pt mesh and 52 blendshapes are not used."]], 9, MUTE)

# ------------------------------------------------------------------ 3 · performance
s = new_slide(3, "Performance", "Reads clean signs, slips on fluent ones", "Zero-shot baseline · same 328-word vocabulary · one frame per sign · missed signs shown at their position")
for x, head, sub, fig in ((.4, "TSL51 — scripted sentences", "126 test clips, slow deliberate signing (3 shown)", "fig_tsl51_baseline"),
                          (6.78, "Jack — natural signing", "5 sentences, fast and fluent (3 shown)", "fig_jack_baseline")):
    chip(s, x, 1.5, 6.15, .3, "", BLUE); text(s, x + .05, 1.5, 6.05, .3, [[(head, True, WHITE), (f"   {sub}", False, WHITE)]], 12, anchor=MSO_ANCHOR.MIDDLE)
    s.shapes.add_picture(str(FIG / f"{fig}.png"), I(x), I(1.9), width=I(6.15))
for i, (t, c) in enumerate((("top-1 = reference word", GREEN), ("reference only in top-5", ORANGE), ("wrong / extra", RED), ("= no segment for that sign", None))):
    xx = .5 + i * 2.6
    if c: box(s, xx, 6.7, .16, .16, c, None, shape=MSO_SHAPE.RECTANGLE)
    text(s, xx + (.22 if c else 0), 6.62, 2.3, .3, t if c else [[("missed", False, RED)]], 11, MUTE)
text(s, .4, 6.95, 12.5, .4, [[("Reading: ", True, INK), "orange = the right word was in the top-5 but not ranked first — the baseline often knows, but cannot decide; missed = the fixed-length cutter produced no segment for that sign."]], 11, MUTE)

# ------------------------------------------------------------------ 4 · metrics
s = new_slide(4, "Metrics", "The numbers behind it: R@1 0.58 · R@5 0.76 · WER 0.47", "Zero-shot baseline, pooled over reference words · Overall = TSL51 + Jack + data_test (136 clips)")
cols = [("Overall", "overall"), ("TSL51", "tsl51"), ("Jack", "jack")]
rows = [("Recall@1  ↑", "r1", "{:.2f}"), ("Recall@5  ↑", "r5", "{:.2f}"), ("WER  ↓", "wer", "{:.2f}"), ("   substitutions", "sub", "{:.2f}"), ("   deletions", "dele", "{:.2f}"),
        ("   insertions", "ins", "{:.2f}"), ("Segment-count error  ↓  (words / clip)", "count_err", "{:.2f}"), ("Fully correct sentences  ↑", "exact", "{:.0%}")]
tbl = s.shapes.add_table(len(rows) + 2, 4, I(.5), I(1.6), I(7.3), I(.42 * (len(rows) + 2))).table
tbl._tbl.tblPr.set("firstRow", "0"); tbl._tbl.tblPr.set("bandRow", "0")
for j, w in enumerate((3.1, 1.4, 1.4, 1.4)): tbl.columns[j].width = I(w)


def cell(i, j, v, size=13, bold=False, color=INK, fill=None, align=PP_ALIGN.CENTER):
    c = tbl.cell(i, j); c.fill.solid(); c.fill.fore_color.rgb = fill or (BG if i % 2 == 0 else WHITE); c.vertical_anchor = MSO_ANCHOR.MIDDLE
    c.margin_left = c.margin_right = I(.1); p = c.text_frame.paragraphs[0]; p.alignment = align; r = p.add_run(); r.text = v
    r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = FONT


for j, h in enumerate(["Metric"] + [c[0] for c in cols]): cell(0, j, h, 13, True, WHITE, BLUE, PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)
cell(1, 0, "clips / reference words", 11, False, MUTE, WHITE, PP_ALIGN.LEFT)
for j, (_, k) in enumerate(cols): cell(1, j + 1, f"{A[k]['n_videos']} / {A[k]['n_words']}", 11, False, MUTE, WHITE)
for i, (lab, k, f) in enumerate(rows):
    key = i < 3
    cell(i + 2, 0, lab, 13 if key else 12, key, INK if key else MUTE, None, PP_ALIGN.LEFT)
    for j, (_, g) in enumerate(cols):
        cell(i + 2, j + 1, f.format(A[g][k]), 14 if key else 12, key, (RED if k == "wer" else BLUE) if key else INK)
for i in range(len(rows) + 2): tbl.rows[i].height = I(.42)
# chart
cd = CategoryChartData(); cd.categories = [c[0] for c in cols]
for lab, k in (("R@1", "r1"), ("R@5", "r5"), ("WER", "wer")): cd.add_series(lab, [round(A[g][k], 3) for _, g in cols])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, I(8.1), I(1.55), I(4.8), I(2.9), cd); ch = gf.chart
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(11); ch.legend.font.name = FONT
for ser, col in zip(ch.plots[0].series, (BLUE, GREEN, RED)):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
pl = ch.plots[0]; pl.gap_width = 60; pl.has_data_labels = True; pl.data_labels.font.size = Pt(10); pl.data_labels.number_format = "0.00"; pl.data_labels.number_format_is_linked = False
ch.value_axis.maximum_scale = 1.0; ch.value_axis.has_major_gridlines = False; ch.value_axis.visible = False; ch.category_axis.tick_labels.font.size = Pt(11)
# isolated
text(s, 8.1, 4.55, 4.8, .3, [[("Isolated words, unseen signers", True, BLUE), ("  (open vocab, 9.5k)", False, MUTE)]], 12)
it = s.shapes.add_table(4, 4, I(8.1), I(4.88), I(4.8), I(1.3)).table; it._tbl.tblPr.set("firstRow", "0"); it._tbl.tblPr.set("bandRow", "0")
for j, w in enumerate((1.8, 1.0, 1.0, 1.0)): it.columns[j].width = I(w)
for i, row in enumerate([("Test set", "R@1", "R@5", "MRR")] + [(n, f"{ISO[k]['R1']:.2f}", f"{ISO[k]['R5']:.2f}", f"{ISO[k]['MRR']:.2f}") for k, n in (("tslone", "TSL-ONE-S"), ("ttrs", "TTRS"), ("th_sl", "th_sl"))]):
    it.rows[i].height = I(.32)
    for j, v in enumerate(row):
        c = it.cell(i, j); c.fill.solid(); c.fill.fore_color.rgb = BLUE if i == 0 else (BG if i % 2 == 0 else WHITE); c.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER; r = p.add_run(); r.text = v
        r.font.size = Pt(11.5); r.font.bold = (i == 0); r.font.color.rgb = WHITE if i == 0 else INK; r.font.name = FONT
# takeaways
box(s, .5, 5.95, 7.3, 1.1, PALE["cream"], ORANGE, 1)
text(s, .62, 5.97, 7.1, 1.06, [[("Gap R@5 − R@1 ≈ 0.18: ", True), "the right word is often in the list but not first."], [("Deletions 0.20 ≫ insertions 0.02: ", True), "the fixed cutter under-counts signs."],
                               [("Only 12 % of sentences fully correct", True), " — word accuracy is not sentence accuracy."]], 12, anchor=MSO_ANCHOR.MIDDLE, after=2)
text(s, 8.1, 6.3, 4.8, .8, "Transfer is good on clean studio clips (TSL-ONE-S 0.80) and poor on dictionary-style clips (TTRS 0.14, th_sl 0.08). Encoder: 0.15 s per clip on CPU.", 11, MUTE)

# ------------------------------------------------------------------ 5 · baseline vs v10
s = new_slide(5, "Extra · Baseline vs v10", "Same clips, what the extra stages fix", "Baseline (frozen Uni-Sign + kNN) vs current v10 · identical landmarks and vocabulary · clips as on slide 3")
for x, head, fig in ((.4, "TSL51 — scripted sentences", "fig_tsl51_vs_v10"), (6.78, "Jack — natural signing", "fig_jack_vs_v10")):
    chip(s, x, 1.5, 6.15, .3, "", BLUE); text(s, x + .05, 1.5, 6.05, .3, [[(head, True, WHITE)]], 12, anchor=MSO_ANCHOR.MIDDLE)
    s.shapes.add_picture(str(FIG / f"{fig}.png"), I(x), I(1.86), width=I(6.15))
for i, (t, c) in enumerate((("top-1 = reference word", GREEN), ("reference only in top-5", ORANGE), ("wrong / extra", RED))):
    xx = .5 + i * 2.6; box(s, xx, 7.16, .16, .16, c, None, shape=MSO_SHAPE.RECTANGLE); text(s, xx + .22, 7.08, 2.3, .3, t, 10.5, MUTE)
text(s, 8.2, 7.08, 4.7, .35, "Clips chosen for mixed baseline results; full-set numbers on slide 4.", 10.5, MUTE, align=PP_ALIGN.RIGHT)

prs.slides[0].notes_slide.notes_text_frame.text = "Baseline = Uni-Sign pose-only WLASL checkpoint (ICLR 2025), frozen, nearest-neighbour retrieval against a Thai clip bank."
out = HERE.parents[1] / "baseline.pptx"; prs.save(out); print("saved", out)
