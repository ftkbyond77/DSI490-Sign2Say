"""arc.pptx — the architecture slide of baseline.pptx, redrawn with orthogonal dark connectors (Canva-friendly).
  python models/baseline-model/make_arc.py      (needs figures/fig_face_graph.png from make_figures_slides.py)
Lines are plain 2 pt straight connectors; arrow heads are separate triangle shapes (they survive import into Canva); dashed = detached / side path."""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor as C
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches as I, Pt

HERE = Path(__file__).resolve().parent
INK, MUTE, LINE, BG = C(0x11, 0x18, 0x27), C(0x6B, 0x72, 0x80), C(0xD1, 0xD5, 0xDB), C(0xF3, 0xF4, 0xF6)
BLUE, GREEN, RED, PURPLE, WHITE = C(0x00, 0x72, 0xB2), C(0x00, 0x9E, 0x73), C(0xD5, 0x5E, 0x00), C(0x8E, 0x6B, 0xBF), C(255, 255, 255)
PALE = dict(blue=C(0xDB, 0xEA, 0xFE), yellow=C(0xFD, 0xE6, 0x8A), green=C(0xBB, 0xF7, 0xD0), purple=C(0xE9, 0xD5, 0xFF), red=C(0xFE, 0xCA, 0xCA), grey=BG)
FONT = "Calibri"

prs = Presentation(); prs.slide_width, prs.slide_height = I(13.333), I(7.5)
s = prs.slides.add_slide(prs.slide_layouts[6])


def text(x, y, w, h, paras, size=12, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, shape=None, after=2):
    tb = shape or s.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = I(.05); tf.margin_top = tf.margin_bottom = I(.02)
    for i, p in enumerate(paras if isinstance(paras, list) else [paras]):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); para.alignment = align; para.space_after = Pt(after)
        for sg in (p if isinstance(p, list) else [p]):
            t, b, col = (sg, bold, color) if isinstance(sg, str) else (sg[0], sg[1], sg[2] if len(sg) > 2 else color)
            r = para.add_run(); r.text = t; r.font.size = Pt(size); r.font.bold = b; r.font.color.rgb = col; r.font.name = FONT
    return tb


def box(x, y, w, h, fill=None, line=INK, lw=1.25, dashed=False, radius=.08):
    r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(x), I(y), I(w), I(h)); r.adjustments[0] = radius
    if fill is None: r.fill.background()
    else: r.fill.solid(); r.fill.fore_color.rgb = fill
    if line is None: r.line.fill.background()
    else:
        r.line.color.rgb = line; r.line.width = Pt(lw)
        if dashed: r.line.dash_style = MSO_LINE.DASH
    r.shadow.inherit = False
    return r


def node(x, y, w, h, paras, fill, size=10, line=INK, lw=1.25, color=INK, dashed=False):
    b = box(x, y, w, h, fill, line, lw, dashed); text(0, 0, 0, 0, paras, size, color, False, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, shape=b, after=0); return b


def line(x1, y1, x2, y2, color=INK, lw=2.0, dashed=False):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, I(x1), I(y1), I(x2), I(y2)); c.line.color.rgb = color; c.line.width = Pt(lw)
    if dashed: c.line.dash_style = MSO_LINE.DASH
    return c


def head(x, y, d, color=INK, size=.13):
    """Arrow head (filled triangle) whose tip is at (x, y), pointing d ∈ {'r','l','d','u'}."""
    L = size * 1.1; off = {"r": (-L / 2, 0), "l": (L / 2, 0), "d": (0, -L / 2), "u": (0, L / 2)}[d]; rot = {"r": 90, "l": 270, "d": 180, "u": 0}[d]
    cx, cy = x + off[0], y + off[1]
    t = s.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE, I(cx - size / 2), I(cy - L / 2), I(size), I(L)); t.rotation = rot
    t.fill.solid(); t.fill.fore_color.rgb = color; t.line.fill.background(); t.shadow.inherit = False


def arrow(x1, y1, x2, y2, color=INK, lw=2.0, dashed=False):
    """Axis-aligned arrow; the shaft stops at the base of the head."""
    d = "r" if x2 > x1 else "l" if x2 < x1 else "d" if y2 > y1 else "u"; g = .13 * 1.1
    ex, ey = {"r": (x2 - g, y2), "l": (x2 + g, y2), "d": (x2, y2 - g), "u": (x2, y2 + g)}[d]
    line(x1, y1, ex, ey, color, lw, dashed); head(x2, y2, d, color)


# ---------------------------------------------------------------- title
box(0, 0, 13.333, .09, BLUE, None, radius=0)
text(.5, .22, 8, .28, [[("02", True, BLUE), ("  ·  ARCHITECTURE", True, MUTE)]], 11)
text(.5, .48, 12.3, .6, "From video to word: a GNN + Transformer encoder with retrieval", 28, INK, True)
text(.5, 1.04, 12.3, .32, "Left: full model, input → output  ·  Right: how the face enters (35 %)", 14, MUTE)
text(11.3, .22, 1.55, .28, "Baseline · Uni-Sign", 10, MUTE, align=PP_ALIGN.RIGHT)

# ---------------------------------------------------------------- A · pose encoder
text(.5, 1.45, 8.1, .3, "A · Pose encoder (part-wise ST-GCN)", 13, BLUE, True)
yc = 3.0
node(.5, yc - .6, .95, 1.2, ["Video", "25 fps", [("T×H×W×3", False, MUTE)]], PALE["blue"], 10)
arrow(1.45, yc, 1.75, yc)
node(1.75, yc - .8, 1.4, 1.6, [[("MediaPipe Holistic", True)], "pose 33", "hands 2×21", "face mesh 478", [("→ 133-slot schema", False, MUTE)]], PALE["blue"], 9.5)
# 4 part streams  (bus in)
parts = [("body", "[T, 9, 3]", BLUE), ("left hand", "[T, 21, 3]", GREEN), ("right hand", "[T, 21, 3]", RED), ("face", "[T, 18, 3]", PURPLE)]
cys = [yc - .9, yc - .3, yc + .3, yc + .9]
line(3.15, yc, 3.45, yc); line(3.45, cys[0], 3.45, cys[3])
for (a, b, c), y in zip(parts, cys):
    arrow(3.45, y, 3.75, y); node(3.75, y - .25, 1.3, .5, [[(a, True), (f"  {b}", False, MUTE)]], WHITE, 8.5, c, 2.5)
    line(5.05, y, 5.3, y)
line(5.3, cys[0], 5.3, cys[3]); arrow(5.3, yc, 5.55, yc)
text(3.75, cys[3] + .27, 1.3, .25, "x, y, confidence", 8.5, MUTE, align=PP_ALIGN.CENTER)
# ST-GCN container
box(5.55, 1.65, 2.0, 2.7, WHITE, MUTE, 1.5, radius=.03)
text(5.55, 1.67, 2.0, .26, "ST-GCN × 4  (one per part)", 9.5, MUTE, True, PP_ALIGN.CENTER)
steps = [(["Linear 3 → 64"], PALE["yellow"], False), (["Spatial GCN × 3", "64 → 64 → 128 → 256"], PALE["yellow"], False),
         (["+ body joint (wrist / nose)", "hands & face only · detached"], WHITE, True), (["Temporal GCN × 3", "256 ch · kernel 5 (≈ 0.5 s)"], PALE["yellow"], False),
         (["mean over joints → [T, 256]"], PALE["red"], False)]
for i, (t, col, dash) in enumerate(steps):
    node(5.65, 1.97 + i * .47, 1.8, .4, t, col, 8.5, MUTE if dash else INK, 1.25, MUTE if dash else INK, dashed=dash)
arrow(7.55, yc, 7.8, yc)
node(7.8, yc - .7, .8, 1.4, ["concat", "4 × 256", "= 1024", [("+ part_para", False, MUTE)], "Linear", "→ 768"], PALE["red"], 9)

# ---------------------------------------------------------------- B · sequence encoder
text(1.2, 4.58, 7.3, .3, "B · Sequence encoder and retrieval", 13, BLUE, True)
yB = 5.0
line(8.2, yc + .7, 8.2, 4.5); line(8.2, 4.5, .975, 4.5); arrow(.975, 4.5, .975, yB)
B = [(.5, 1.45, [[("frame tokens", True)], "[T, 768]", "⊕ task prefix", "13 tokens"], PALE["blue"]),
     (2.2, 1.95, [[("mT5-base encoder", True)], "12 layers · d = 768", "12 heads · gated-GELU", [("full self-attention over frames", False, MUTE)]], PALE["green"]),
     (4.4, 1.45, [[("mean-pool over T", True)], "L2-normalise", [("z ∈ ℝ⁷⁶⁸", False, MUTE)]], PALE["purple"]),
     (6.1, 1.55, [[("cosine kNN", True)], "vs bank Z₀", "≈ 19k Thai clips", [("score = mean of top-2", False, MUTE)]], PALE["red"])]
for i, (x, w, t, col) in enumerate(B):
    node(x, yB, w, 1.35, t, col, 9.5)
    if i < 3: arrow(x + w, yB + .675, B[i + 1][0], yB + .675)
arrow(7.65, yB + .675, 7.85, yB + .675)
node(7.85, yB + .225, .75, .9, [[("top-5", True)], "words"], GREEN, 10, None, 0, WHITE)
text(.5, 6.5, 8.1, .6, [[("All 90.3 M parameters frozen. ", True), "Segmentation of continuous video is outside the model: hands-raised runs are cut every ≈ 49 frames (≈ 2 s), each cut → one embedding → one word."]], 10.5, MUTE)

# ---------------------------------------------------------------- face panel (35 %)
box(9.0, 1.45, 3.95, 5.6, BG, None, radius=.03)
text(9.15, 1.5, 3.7, .35, "Face branch — where it joins", 14, PURPLE, True)
text(9.15, 1.85, 3.7, .75, [[("Extractor: ", True), "5 MediaPipe-Pose anchors ◇ (nose, eyes, mouth corners) → a mean face template fitted per frame → ", ("18 points", True), ": jaw 9 · inner lip 8 · nose 1"]], 10)
s.shapes.add_picture(str(HERE / "figures" / "fig_face_graph.png"), I(9.1), I(2.62), height=I(1.8))
for i, (c, t) in enumerate(((BLUE, ["jaw chain (9)"]), (RED, ["mouth ring (8)"]), (GREEN, ["nose hub (1)", "linked to all"]), (C(0xE6, 0x9F, 0x00), ["pose anchor ◇ (5)", "used to fit"]))):
    yy = 2.66 + i * .4
    d = s.shapes.add_shape(MSO_SHAPE.OVAL, I(11.25), I(yy + .09), I(.15), I(.15)); d.fill.solid(); d.fill.fore_color.rgb = c; d.line.fill.background(); d.shadow.inherit = False
    text(11.45, yy, 1.45, .5, t, 10, INK, after=0)
fsteps = [("[T, 18, 3]  nose-relative, ÷ body scale", PALE["blue"]), ("Linear 3 → 64 · Spatial GCN × 3 → 256 (graph above)", PALE["yellow"]),
          ("⊕ nose feature of the body branch (detached)", WHITE), ("Temporal GCN × 3 · mean over 18 nodes → [T, 256]", PALE["yellow"]),
          ("JOIN: 4th block of the concat (ch. 769–1024) → Linear 1024 → 768", PALE["purple"])]
for i, (t, col) in enumerate(fsteps):
    yy = 4.45 + i * .46; dash = i == 2
    node(9.15, yy, 3.65, .3, t, col, 8.5, PURPLE if i == 4 else (MUTE if dash else INK), 2.25 if i == 4 else 1.25, MUTE if dash else INK, dashed=dash)
    if i < 4: arrow(10.975, yy + .3, 10.975, yy + .46)
# dashed purple path: JOIN box → concat (face is the 4th block)
line(9.15, 4.45 + 4 * .46 + .15, 8.8, 4.45 + 4 * .46 + .15, PURPLE, 2.0, True); line(8.8, 4.45 + 4 * .46 + .15, 8.8, yc, PURPLE, 2.0, True); arrow(8.8, yc, 8.6, yc, PURPLE, 2.0)
text(9.15, 6.62, 3.65, .4, [[("Rigid template: ", True), "head position · scale · roll only, no expression. 478-pt mesh and 52 blendshapes are not used."]], 9, MUTE)

out = HERE.parents[1] / "arc.pptx"; prs.save(out); print("saved", out)
