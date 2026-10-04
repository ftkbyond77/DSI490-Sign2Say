// MediaPipe Holistic in the browser — the same model (holistic_landmarker.task, tasks-vision 1.0.1) the server and the training
// data use, so a webcam frame becomes exactly the kind of landmarks the sign model was trained on.
import { FilesetResolver, HolisticLandmarker, type HolisticLandmarkerResult } from "@mediapipe/tasks-vision";

// 478-point face mesh → 68 iBUG points (must match modules/mediapipe_pose.py MESH68)
export const MESH68 = [127, 234, 93, 132, 58, 172, 136, 150, 152, 379, 365, 397, 288, 361, 323, 454, 356,
  70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
  168, 197, 5, 4, 75, 97, 2, 326, 305,
  33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
  61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
  78, 82, 13, 312, 308, 317, 14, 87];

export type LmFrame = { t: number; pose: number[][]; lh: number[][]; rh: number[][]; face: number[][]; bs: number[] };

const r4 = (v: number) => Math.round(v * 1e4) / 1e4;

export type Delegate = "GPU" | "GPU+CPU" | "CPU";
export type Landmarkers = { lm: HolisticLandmarker; delegate: Delegate; face?: HolisticLandmarker };

/** Pick the fastest set-up that works on this machine (tested on one frame, because some WebGL drivers load the face-blendshape
 *  model and then fail on the first frame with "No support of const"):
 *    GPU      — everything on the GPU (~20+ fps)
 *    GPU+CPU  — body / hands / face mesh on the GPU every frame, face expressions (blendshapes) on the CPU every few frames
 *    CPU      — everything on the CPU (~6–7 fps on a laptop) */
export async function createHolistic(prefer: "GPU" | "CPU" = "GPU"): Promise<Landmarkers> {
  const fileset = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
  const make = (delegate: "GPU" | "CPU", blendshapes = true) => HolisticLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "/models/holistic_landmarker.task", delegate },
    runningMode: "VIDEO" as const,
    outputFaceBlendshapes: blendshapes,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minHandLandmarksConfidence: 0.5,
  });
  const works = (lm: HolisticLandmarker) => {
    const c = document.createElement("canvas"); c.width = 64; c.height = 64;
    c.getContext("2d")!.fillRect(0, 0, 64, 64);
    try { lm.detectForVideo(c, 1, () => {}); return true; } catch { return false; }
  };
  if (prefer === "GPU") {
    try { const lm = await make("GPU"); if (works(lm)) return { lm, delegate: "GPU" }; lm.close(); } catch { /* next option */ }
    try {
      const lm = await make("GPU", false);
      if (works(lm)) return { lm, delegate: "GPU+CPU", face: await make("CPU") };
      lm.close();
    } catch { /* next option */ }
  }
  return { lm: await make("CPU"), delegate: "CPU" };
}

/** Which parts MediaPipe found in a frame (for the on-screen check). */
export function parts(f: LmFrame | null) {
  return { body: !!f && f.pose.length === 33, face: !!f && f.face.length === 68, brows: !!f && f.face.length === 68,
    left: !!f && f.lh.length === 21, right: !!f && f.rh.length === 21 };
}

export function pack(res: HolisticLandmarkerResult, t: number): LmFrame {
  const pose = res.poseLandmarks?.[0]?.map((p) => [r4(p.x), r4(p.y), r4(p.z), r4(p.visibility ?? 1)]) ?? [];
  const lh = res.leftHandLandmarks?.[0]?.map((p) => [r4(p.x), r4(p.y), r4(p.z)]) ?? [];
  const rh = res.rightHandLandmarks?.[0]?.map((p) => [r4(p.x), r4(p.y), r4(p.z)]) ?? [];
  const fl = res.faceLandmarks?.[0];
  const face = fl && fl.length >= 468 ? MESH68.map((i) => [r4(fl[i].x), r4(fl[i].y)]) : [];
  const cats = res.faceBlendshapes?.[0]?.categories ?? [];
  const bs = [...cats].sort((a, b) => a.index - b.index).map((c) => r4(c.score));
  return { t, pose, lh, rh, face, bs };
}

/** Run the landmarker on a frame (the camera video, or the normalised canvas); MediaPipe calls back synchronously. */
export function detect(lm: HolisticLandmarker, src: HTMLVideoElement | HTMLCanvasElement, tMs: number): LmFrame | null {
  let out: LmFrame | null = null;
  lm.detectForVideo(src, tMs, (res) => {
    out = pack(res, tMs);
  });
  return out;
}

/** A hand is "up" (signing) when it is detected above the lower chest — used for hands-down end-pointing in live mode. */
export function handsUp(f: LmFrame): boolean {
  if (f.pose.length < 17) return false;
  const ls = f.pose[11], rs = f.pose[12];
  const sw = Math.hypot(ls[0] - rs[0], ls[1] - rs[1]) || 0.2;
  const midY = (ls[1] + rs[1]) / 2;
  const up = (hand: number[][]) => hand.length === 21 && hand.reduce((s, p) => s + p[1], 0) / 21 < midY + 1.5 * sw;
  // the body's wrist points are tracked even when a hand's 21 points drop out (fast motion, 12 fps): either counts as "up"
  const wrist = (i: number) => f.pose[i] && f.pose[i][3] > 0.5 && f.pose[i][1] < midY + 1.3 * sw;
  return up(f.lh) || up(f.rh) || wrist(15) || wrist(16);
}

// ----------------------------------------------------------------------------------------------------------------------------
// Image normalisation: cameras and rooms differ (dim rooms, back light, washed-out webcams). Every ~0.3 s the frame's luminance
// is measured on a 64×36 thumbnail; a lookup table (gamma, then contrast around the image's own mean) brings it toward a standard
// exposure before MediaPipe — the same rule as modules/mediapipe_pose.py exposure_lut (server uploads). Smoothed, so it follows
// the room without flicker. On test videos it raised hand detection in dark / over-bright / low-contrast footage and left good
// footage unchanged. Landmarks are positions only, so the sign model downstream is unaffected.
export type Exposure = { mean: number; spread: number; gamma: number; contrast: number; active: boolean; label: string };
/** Display tone for the visible video: a CSS brightness / contrast filter (GPU, cheap), only when the light is clearly bad. */
export type Tone = { brightness: number; contrast: number };

export function exposureLut(mean: number, spread: number) {
  const m = Math.min(0.97, Math.max(0.03, mean));
  const g = Math.min(2.0, Math.max(0.35, Math.log(0.5) / Math.log(m)));
  const mg = m ** g;
  const sg = Math.max(spread * g * m ** (g - 1), 1e-3);
  const c = Math.min(1.8, Math.max(0.85, 0.22 / sg));
  const tgt = mg + Math.min(0.2, Math.max(-0.2, 0.5 - mg));
  return { g, c, mg, tgt };
}

export class Normalizer {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private thumb: HTMLCanvasElement;
  private tctx: CanvasRenderingContext2D;
  private last = -1e9;
  private luts: [Uint8ClampedArray, Uint8ClampedArray, Uint8ClampedArray] | null = null;
  private wb: [number, number, number] = [1, 1, 1];
  enabled = true;
  exp: Exposure = { mean: 0.5, spread: 0.22, gamma: 1, contrast: 1, active: false, label: "ปกติ" };
  tone: Tone = { brightness: 1, contrast: 1 };

  constructor() {
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
    this.thumb = document.createElement("canvas"); this.thumb.width = 160; this.thumb.height = 90;
    this.tctx = this.thumb.getContext("2d", { willReadFrequently: true })!;
  }

  /** The source MediaPipe should read: the corrected canvas, or the video itself when no correction is needed / it is off. */
  frame(v: HTMLVideoElement, t: number): HTMLVideoElement | HTMLCanvasElement {
    if (!v.videoWidth) return v;
    if (t - this.last > 300) { this.last = t; this.measure(v); }
    if (!this.enabled || !this.luts) return v;
    if (this.canvas.width !== v.videoWidth) { this.canvas.width = v.videoWidth; this.canvas.height = v.videoHeight; }
    this.ctx.drawImage(v, 0, 0);
    const img = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    const d = img.data, [R, G, B] = this.luts;
    for (let i = 0; i < d.length; i += 4) { d[i] = R[d[i]]; d[i + 1] = G[d[i + 1]]; d[i + 2] = B[d[i + 2]]; }
    this.ctx.putImageData(img, 0, 0);
    return this.canvas;
  }

  private measure(v: HTMLVideoElement) {
    const W = 160, H = 90;
    this.tctx.drawImage(v, 0, 0, W, H);
    const d = this.tctx.getImageData(0, 0, W, H).data;
    const n = W * H, Y = new Float32Array(n);
    let s = 0, s2 = 0, sr = 0, sg = 0, sb = 0;
    for (let i = 0, j = 0; i < d.length; i += 4, j++) {
      const y = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) / 255;
      Y[j] = y; s += y; s2 += y * y; sr += d[i]; sg += d[i + 1]; sb += d[i + 2];
    }
    const mean = s / n, spread = Math.sqrt(Math.max(s2 / n - mean * mean, 1e-6));
    // sharpness: variance of the Laplacian (low = blurry webcam / out of focus)
    let lap = 0, lap2 = 0, m = 0;
    for (let y = 1; y < H - 1; y += 2) for (let x = 1; x < W - 1; x += 2) {
      const k = y * W + x, L = 4 * Y[k] - Y[k - 1] - Y[k + 1] - Y[k - W] - Y[k + W];
      lap += L; lap2 += L * L; m++;
    }
    const lapVar = lap2 / m - (lap / m) ** 2;
    // gray-world white balance, half strength and clamped (keeps skin tones natural)
    const avg = (sr + sg + sb) / 3;
    const target: [number, number, number] = [sr, sg, sb].map((c) => Math.min(1.2, Math.max(0.85, 1 + 0.5 * (avg / Math.max(c, 1) - 1)))) as [number, number, number];
    const { g, c, mg, tgt } = exposureLut(mean, spread);
    const k = this.exp.active || this.luts ? 0.35 : 1;                              // smoothing toward the target
    const gamma = Math.exp(Math.log(this.exp.gamma) + k * (Math.log(g) - Math.log(this.exp.gamma)));
    const contrast = this.exp.contrast + k * (c - this.exp.contrast);
    this.wb = this.wb.map((w, i) => w + k * (target[i] - w)) as [number, number, number];
    const wbActive = this.wb.some((w) => Math.abs(w - 1) >= 0.04);
    // correct only clearly bad light (dark room, glare, washed-out webcam): good frames go to MediaPipe untouched, as in v7
    const bad = mean < 0.3 || mean > 0.72 || spread < 0.11;
    const active = bad && (Math.abs(gamma - 1) >= 0.08 || Math.abs(contrast - 1) >= 0.06);
    if (active) {
      this.luts = [0, 1, 2].map((ch) => {
        const L = new Uint8ClampedArray(256);
        for (let x = 0; x < 256; x++) L[x] = Math.round(((x / 255) ** gamma - mg) * contrast * 255 + tgt * 255);
        return L;
      }) as [Uint8ClampedArray, Uint8ClampedArray, Uint8ClampedArray];
    } else this.luts = null;
    void lapVar; void wbActive;
    this.tone = active ? { brightness: Math.min(1.8, Math.max(0.75, 0.5 / Math.max(mean, 0.05))), contrast: Math.min(1.6, Math.max(0.9, contrast)) }
      : { brightness: 1, contrast: 1 };
    const label = mean < 0.3 ? "มืด → เพิ่มแสง" : mean > 0.7 ? "สว่างจ้า → ลดแสง" : spread < 0.12 ? "ภาพจาง → เพิ่มคอนทราสต์" : wbActive ? "ปรับสมดุลสี" : "ปรับเล็กน้อย";
    this.exp = { mean, spread, gamma, contrast, active, label };
  }
}

// ----------------------------------------------------------------------------------------------------------------------------
// Live emotion: the same blendshape formula and neutral-face thresholds as modules/face.py, averaged over the last ~0.6 s.
const BS = ["_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
  "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
  "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
  "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
  "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
  "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
  "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"];
const BI: Record<string, number> = Object.fromEntries(BS.map((n, i) => [n, i]));
const EMO_REF: Record<string, number> = { happy: 0.545, sad: 0.05, angry: 0.365, surprise: 0.15, fear: 0.05, disgust: 0.15 };
export const EMO_TH: Record<string, string> = { neutral: "เป็นกลาง", happy: "ดีใจ", sad: "เศร้า", angry: "โกรธ/ขมวดคิ้ว", surprise: "ประหลาดใจ", fear: "กลัว", disgust: "รังเกียจ" };
export const EMO_ICON: Record<string, string> = { neutral: "😐", happy: "😊", sad: "😢", angry: "😠", surprise: "😮", fear: "😨", disgust: "😖" };
/** What the face shows → what it may mean (the live label above the face box). */
export const EMO_FACE: Record<string, [string, string]> = { neutral: ["หน้าปกติ", "เป็นกลาง"], happy: ["ยิ้ม", "มีความสุข"], sad: ["มุมปากตก", "เศร้า"],
  angry: ["ขมวดคิ้ว", "ไม่พอใจ / หรือกำลังถาม"], surprise: ["ตาโต ปากอ้า", "ประหลาดใจ"], fear: ["ตาเบิก", "กลัว / กังวล"], disgust: ["ย่นจมูก", "รังเกียจ"] };
export type Emotion = { emotion: string; level: number };

export class EmotionMeter {
  private win: number[][] = [];
  update(bs: number[]): Emotion {
    if (bs.length >= 52) { this.win.push(bs); if (this.win.length > 15) this.win.shift(); }
    if (this.win.length < 3) return { emotion: "neutral", level: 0 };
    const avg = (n: string) => this.win.reduce((a, r) => a + r[BI[n]], 0) / this.win.length;
    const m = (...names: string[]) => names.reduce((s, n) => s + avg(n), 0) / names.length;
    const smile = m("mouthSmileLeft", "mouthSmileRight"), squint = m("eyeSquintLeft", "eyeSquintRight", "cheekSquintLeft", "cheekSquintRight");
    const frown = m("mouthFrownLeft", "mouthFrownRight"), up = m("browInnerUp", "browOuterUpLeft", "browOuterUpRight"), down = m("browDownLeft", "browDownRight");
    const press = m("mouthPressLeft", "mouthPressRight"), jaw = m("jawOpen"), wide = m("eyeWideLeft", "eyeWideRight"), sneer = m("noseSneerLeft", "noseSneerRight");
    const sc: Record<string, number> = {
      happy: smile * 1.2 + 0.3 * squint, sad: frown * 0.9 + 0.5 * up * frown * 4, angry: down * 0.8 + 0.6 * press,
      surprise: 0.6 * up + 0.6 * jaw + 0.8 * wide - 0.2, fear: 0.5 * wide + 0.4 * up * wide * 3, disgust: 1.2 * sneer,
    };
    let best = "neutral", r = 1;
    for (const k of Object.keys(sc)) { const x = Math.max(0, sc[k]) / EMO_REF[k]; if (x >= r) { r = x; best = k; } }
    return { emotion: best, level: best === "neutral" ? 0 : Math.min(1, (r - 1) / 1.5) };
  }
  reset() { this.win = []; }
}

// ----------------------------------------------------------------------------------------------------------------------------
// Drawing: body + hands (labelled boxes) and one box around the face with the live emotion — the face itself stays uncovered
// (its 68 points and blendshapes are still extracted and checked in the "ตรวจพบ" bar). The canvas is mirrored by CSS (selfie view), so text is drawn flipped back to stay readable.
const POSE_EDGES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24]];
const HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14],
  [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
function box(pts: number[][], w: number, h: number, pad: number) {
  let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
  for (const p of pts) { x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]); }
  const px = (x1 - x0) * pad, py = (y1 - y0) * pad;
  return { x: (x0 - px) * w, y: (y0 - py) * h, w: (x1 - x0 + 2 * px) * w, h: (y1 - y0 + 2 * py) * h };
}

/** A tag above a box. (x, y) = the box corner that appears top-LEFT on the mirrored screen. */
function tag(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, col: string, fontPx: number) {
  ctx.save();
  ctx.font = `600 ${fontPx}px "Noto Sans Thai", "Leelawadee UI", Tahoma, sans-serif`;
  const tw = ctx.measureText(text).width, th = fontPx + 6;
  ctx.translate(x, Math.max(y, th)); ctx.scale(-1, 1);     // undo the CSS mirror for the text only
  ctx.fillStyle = col; ctx.fillRect(0, -th, tw + 10, th);
  ctx.fillStyle = "#0b1020"; ctx.fillText(text, 5, -5);
  ctx.restore();
}

export function draw(ctx: CanvasRenderingContext2D, f: LmFrame | null, w: number, h: number, active: boolean, emo?: Emotion) {
  ctx.clearRect(0, 0, w, h);
  if (!f) return;
  const fontPx = Math.max(13, Math.round(h / 34));
  const line = (a: number[], b: number[]) => { ctx.beginPath(); ctx.moveTo(a[0] * w, a[1] * h); ctx.lineTo(b[0] * w, b[1] * h); ctx.stroke(); };
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.lineWidth = 2.5; ctx.strokeStyle = "rgba(255,255,255,0.55)";
  if (f.pose.length === 33) for (const [a, b] of POSE_EDGES) if (f.pose[a][3] > 0.5 && f.pose[b][3] > 0.5) line(f.pose[a], f.pose[b]);
  for (const [hand, col, name] of [[f.lh, "#22d3ee", "มือซ้าย"], [f.rh, "#f472b6", "มือขวา"]] as const) {
    if (hand.length !== 21) continue;
    // no box: the 21 tracked points themselves show the hand is caught (they are the model's input), tips drawn larger
    const r = Math.max(3, h / 200);
    ctx.strokeStyle = col; ctx.lineWidth = Math.max(2, r * 0.8);
    for (const [a, b] of HAND_EDGES) line(hand[a], hand[b]);
    for (const [k, p] of hand.entries()) {
      ctx.beginPath(); ctx.arc(p[0] * w, p[1] * h, [4, 8, 12, 16, 20].includes(k) ? r * 1.6 : r, 0, 2 * Math.PI);
      ctx.fillStyle = "#0b1020"; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = col; ctx.stroke(); ctx.fillStyle = col; ctx.fill();
    }
    ctx.strokeStyle = col;
    const small = Math.round(fontPx * 0.8);
    tag(ctx, name, hand[0][0] * w + small * 2, hand[0][1] * h + small * 2.4, col, small);   // a small label under the wrist
  }
  if (f.face.length === 68) {
    const b = box(f.face, w, h, 0.15);
    b.y -= b.h * 0.12; b.h *= 1.12;                          // the 68 points stop at the brows: leave room for the forehead
    const col = active ? "#facc15" : "#f59e0b";
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash([8, 5]); ctx.strokeRect(b.x, b.y, b.w, b.h); ctx.setLineDash([]);
    if (emo) { const [look, mean] = EMO_FACE[emo.emotion] ?? ["", EMO_TH[emo.emotion]];
      tag(ctx, `${EMO_ICON[emo.emotion]} ${look} → ${mean}${emo.level > 0 ? ` ${Math.round(emo.level * 100)}%` : ""}`, b.x + b.w, b.y, col, fontPx); }
  }
}
