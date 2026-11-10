/* MediaPipe Holistic in a Web Worker (classic script: MediaPipe loads its WASM runtime with importScripts).
 * Why: creating the landmarker (WASM compile, GPU shader build, a test frame) and every inference ran on the page's main thread, so the
 * screen could not be clicked while it loaded and stuttered while the camera was on. Here the page only snapshots a frame
 * (createImageBitmap) and receives the landmarks back; the same model, the same options, the same output as lib/holistic.ts.
 * The landmark packing below (MESH68, pack) must stay identical to lib/holistic.ts.
 *
 * Hand recovery (port of modules/hand_recovery.py, lab4-handExtractor "method C"): when Holistic loses a hand while the wrist is raised, a Hand
 * Landmarker (CPU, IMAGE mode, public/models/hand_landmarker.task) is run on a zoomed, forearm-aligned crop — the hand's previous box first, then the
 * pose wrist — and the result is accepted only if it sits on the pose wrist. OFF unless the page sends {recover:true} (localStorage "thaislm.hand.recover" = "1"):
 * real-Chrome A/B on the iPhone clips: ≤ 1 point fewer missing hands, p95 frame time 64 → 216 ms.
 *
 * Messages in : {type:"init", prefer:"GPU"|"CPU", last, recover, smooth}  ·  {type:"detect", id, t, bmp, face}  ·  {type:"close"}
 * Messages out: {type:"ready", delegate, face, remember} | {type:"fail", message} | {type:"frame", id, frame, bs} | {type:"error", id, message}
 */
importScripts("/mediapipe/vision_bundle.js");   // defines the global `Vision` (copied from @mediapipe/tasks-vision by scripts/copy-mediapipe.mjs)

const MESH68 = [127, 234, 93, 132, 58, 172, 136, 150, 152, 379, 365, 397, 288, 361, 323, 454, 356,
  70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
  168, 197, 5, 4, 75, 97, 2, 326, 305,
  33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
  61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
  78, 82, 13, 312, 308, 317, 14, 87];
const r4 = (v) => Math.round(v * 1e4) / 1e4;

function pack(res, t) {
  const pose = (res.poseLandmarks && res.poseLandmarks[0] || []).map((p) => [r4(p.x), r4(p.y), r4(p.z), r4(p.visibility == null ? 1 : p.visibility)]);
  const lh = (res.leftHandLandmarks && res.leftHandLandmarks[0] || []).map((p) => [r4(p.x), r4(p.y), r4(p.z)]);
  const rh = (res.rightHandLandmarks && res.rightHandLandmarks[0] || []).map((p) => [r4(p.x), r4(p.y), r4(p.z)]);
  const fl = res.faceLandmarks && res.faceLandmarks[0];
  const face = fl && fl.length >= 468 ? MESH68.map((i) => [r4(fl[i].x), r4(fl[i].y)]) : [];
  const cats = (res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories) || [];
  const bs = [...cats].sort((a, b) => a.index - b.index).map((c) => r4(c.score));
  return { t, pose, lh, rh, face, bs };
}

// ---------------------------------------------------------------------------------------------------------------------------- hand recovery
const SIDE = { L: { wrist: 15, elbow: 13, pinky: 17, index: 19, thumb: 21, hip: 23 }, R: { wrist: 16, elbow: 14, pinky: 18, index: 20, thumb: 22, hip: 24 } };
const OUT = 256;
let handLm = null, recoverOn = true, smoothOn = false;
let rs = null;                                              // per-stream recovery state
const newState = () => ({ i: 0, prev: { L: null, R: null }, vel: { L: [0, 0], R: [0, 0] }, cool: { L: 0, R: 0 }, fails: { L: 0, R: 0 }, filt: { L: null, R: null }, canvas: null, last: [] });

class OneEuro {
  constructor(mc = 1.0, beta = 8.0, dc = 1.0) { this.mc = mc; this.beta = beta; this.dc = dc; this.x = null; this.dx = null; this.t = null; }
  static a(cut, dt) { const r = 2 * Math.PI * cut * dt; return r / (r + 1); }
  push(x, t) {                                                 // x: Float64Array
    if (!this.x || t - this.t > 0.5) { this.x = Float64Array.from(x); this.dx = new Float64Array(x.length); this.t = t; return this.x; }
    const dt = Math.max(t - this.t, 1e-3), ad = OneEuro.a(this.dc, dt);
    for (let k = 0; k < x.length; k++) {
      this.dx[k] += ad * ((x[k] - this.x[k]) / dt - this.dx[k]);
      this.x[k] += OneEuro.a(this.mc + this.beta * Math.abs(this.dx[k]), dt) * (x[k] - this.x[k]);
    }
    this.t = t; return this.x;
  }
}

const bbox = (p) => { let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9; for (const q of p) { x0 = Math.min(x0, q[0]); y0 = Math.min(y0, q[1]); x1 = Math.max(x1, q[0]); y1 = Math.max(y1, q[1]); } return [x0, y0, x1, y1]; };
const sizeOf = (p) => { const b = bbox(p); return Math.max(b[2] - b[0], b[3] - b[1]); };
const toPx = (lm, W, H) => lm.map((q) => [q[0] * W, q[1] * H]);

function poseRoi(pose, side, W, H) {
  const ix = SIDE[side];
  if (pose.length < 25 || pose[ix.wrist][3] < 0.3) return null;
  const P = (i) => [pose[i][0] * W, pose[i][1] * H];
  const w = P(ix.wrist), e = P(ix.elbow), t = [P(ix.pinky), P(ix.index), P(ix.thumb)];
  const tips = [(t[0][0] + t[1][0] + t[2][0]) / 3, (t[0][1] + t[1][1] + t[2][1]) / 3];
  const tipVis = (pose[ix.pinky][3] + pose[ix.index][3] + pose[ix.thumb][3]) / 3;
  const c = tipVis > 0.3 ? [(w[0] + tips[0]) / 2, (w[1] + tips[1]) / 2] : w;
  const fore = pose[ix.elbow][3] > 0.3 ? Math.hypot(w[0] - e[0], w[1] - e[1]) : 0;
  const sh = pose[11][3] > 0.3 && pose[12][3] > 0.3 ? Math.hypot(P(11)[0] - P(12)[0], P(11)[1] - P(12)[1]) : 0;
  const s = Math.max(1.15 * fore, 0.55 * sh, 3.2 * Math.hypot(tips[0] - w[0], tips[1] - w[1]), 48);
  return { c, s: Math.min(s, 0.9 * Math.max(W, H)) };
}
function forearmAngle(pose, side, W, H) {
  const ix = SIDE[side];
  if (pose[ix.wrist][3] < 0.3 || pose[ix.elbow][3] < 0.3) return 0;
  const dx = (pose[ix.wrist][0] - pose[ix.elbow][0]) * W, dy = (pose[ix.wrist][1] - pose[ix.elbow][1]) * H;
  return Math.atan2(dx, -dy) * 180 / Math.PI;
}
const raised = (pose, side) => { const ix = SIDE[side]; const hip = pose[ix.hip]; return hip[3] < 0.3 || pose[ix.wrist][1] < hip[1] + 0.05; };
function plausible(lm, pose, side, W, H, tol) {
  const ix = SIDE[side];
  if (pose.length < 25 || pose[ix.wrist][3] < 0.3) return false;                // cannot be checked against the body → the caller decides
  const fore = pose[ix.elbow][3] > 0.3 ? Math.hypot((pose[ix.wrist][0] - pose[ix.elbow][0]) * W, (pose[ix.wrist][1] - pose[ix.elbow][1]) * H) : 0.15 * Math.max(W, H);
  return Math.hypot(lm[0][0] * W - pose[ix.wrist][0] * W, lm[0][1] * H - pose[ix.wrist][1] * H) <= tol * Math.max(fore, 30);
}
// a recovered hand must have a believable size: 0.5–2× the last hand of this side, 0.33–1.5× the forearm (Holistic's own hands, 2,959 frames: 1st–96th percentile)
function sizeOk(lm, pose, side, W, H) {
  const size = sizeOf(toPx(lm, W, H)), pv = rs.prev[side], ix = SIDE[side];
  if (pv && rs.i - pv.i <= 12) { const e = sizeOf(toPx(pv.lm, W, H)); if (e > 0 && !(0.5 * e <= size && size <= 2.0 * e)) return false; }
  if (pose.length >= 25 && pose[ix.wrist][3] >= 0.3 && pose[ix.elbow][3] >= 0.3) {
    const fore = Math.hypot((pose[ix.wrist][0] - pose[ix.elbow][0]) * W, (pose[ix.wrist][1] - pose[ix.elbow][1]) * H);
    if (fore > 0 && !(0.33 * fore <= size && size <= 1.5 * fore)) return false;
  }
  return true;
}

// crop = a rotated square of side s centred on c (px), drawn into a 256×256 canvas:  img = k·R(a)·(crop − OUT/2) + c
function cropTo(bmp, c, s, ang) {
  if (!rs.canvas) rs.canvas = new OffscreenCanvas(OUT, OUT);
  const ctx = rs.canvas.getContext("2d");
  const k = s / OUT, a = ang * Math.PI / 180, cs = Math.cos(a), sn = Math.sin(a);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "rgb(114,114,114)"; ctx.fillRect(0, 0, OUT, OUT);
  ctx.setTransform(cs / k, -sn / k, sn / k, cs / k, OUT / 2 - (cs * c[0] + sn * c[1]) / k, OUT / 2 - (-sn * c[0] + cs * c[1]) / k);
  ctx.drawImage(bmp, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return { canvas: rs.canvas, k, cs, sn };
}
function backProject(lms, c, k, cs, sn, W, H) {
  return lms.map((p) => {
    const x = (p.x * OUT - OUT / 2) * k, y = (p.y * OUT - OUT / 2) * k;
    return [(cs * x - sn * y + c[0]) / W, (sn * x + cs * y + c[1]) / H, p.z * OUT * k / W];
  });
}

function recoverSide(bmp, pose, side, other, W, H) {
  if (rs.cool[side] > 0) { rs.cool[side]--; return null; }
  const rois = [], pv = rs.prev[side];
  if (pv && rs.i - pv.i <= 12) {                                                  // 1) where the hand just was (+ velocity)
    const p = toPx(pv.lm, W, H), b = bbox(p), n = rs.i - pv.i;
    rois.push({ kind: "track", c: [(b[0] + b[2]) / 2 + rs.vel[side][0] * n, (b[1] + b[3]) / 2 + rs.vel[side][1] * n], s: Math.max(sizeOf(p) * 2.2, 40), ang: 0 });
  }
  const pr = pose.length >= 25 ? poseRoi(pose, side, W, H) : null;
  if (pr && raised(pose, side)) {                                                 // 2) the pose wrist
    rois.push({ kind: "pose", c: pr.c, s: pr.s, ang: forearmAngle(pose, side, W, H) });
    if (rs.fails[side] % 3 === 2) rois.push({ kind: "wide", c: pr.c, s: pr.s * 1.5, ang: 0 });
  }
  const cont = !!pv && rs.i - pv.i <= 2;                                          // the hand was there a moment ago → a continuation
  for (const roi of rois) {
    const { canvas, k, cs, sn } = cropTo(bmp, roi.c, roi.s, roi.ang);
    const r = handLm.detect(canvas);
    for (let h = 0; h < (r.landmarks || []).length; h++) {
      const sc = r.handedness && r.handedness[h] && r.handedness[h][0] ? r.handedness[h][0].score : 0;
      if (sc < (cont ? 0.6 : 0.75)) continue;                                      // a hand that re-appears from nowhere must be convincing:
      const lm = backProject(r.landmarks[h], roi.c, k, cs, sn, W, H);              // on the wrist and sure of itself (lone frames flip borderline words)
      if (!plausible(lm, pose, side, W, H, cont ? 1.4 : 0.7) && !(cont && (pose.length < 25 || pose[SIDE[side].wrist][3] < 0.3))) continue;
      if (!sizeOk(lm, pose, side, W, H)) continue;
      if (other.length === 21) {
        const op = toPx(other, W, H);
        if (Math.hypot(lm[0][0] * W - op[0][0], lm[0][1] * H - op[0][1]) < 0.25 * sizeOf(op)) continue;     // that is the other hand
      }
      rs.fails[side] = 0; rs.last.push(side + ":" + (roi.kind || "?"));
      return lm;
    }
  }
  rs.fails[side]++; rs.cool[side] = !rois.length ? 0 : rs.fails[side] < 3 ? 2 : Math.min(2 * rs.fails[side], 10);   // a hand that is really absent is tried less and less often
  return null;
}

/** frame: pack() output (mutated). Returns the number of hands recovered. */
function recoverHands(bmp, frame) {
  if (!recoverOn || !handLm || !frame || !rs) return 0;
  const W = bmp.width, H = bmp.height, pose = frame.pose;
  const res = { L: frame.lh, R: frame.rh };
  let got = 0; rs.last = [];
  for (const s of ["L", "R"]) {
    if (res[s].length === 21) continue;
    const lm = recoverSide(bmp, pose, s, res[s === "L" ? "R" : "L"], W, H);
    if (lm) { res[s] = lm.map((q) => [r4(q[0]), r4(q[1]), r4(q[2])]); got++; }
  }
  for (const s of ["L", "R"]) {
    if (res[s].length !== 21) { rs.filt[s] = null; continue; }
    const pv = rs.prev[s];
    if (pv) {
      const a = res[s][0], b = pv.lm[0], n = Math.max(rs.i - pv.i, 1);
      rs.vel[s] = [0.6 * rs.vel[s][0] + 0.4 * (a[0] - b[0]) * W / n, 0.6 * rs.vel[s][1] + 0.4 * (a[1] - b[1]) * H / n];
    }
    rs.prev[s] = { lm: res[s].map((q) => q.slice()), i: rs.i };                    // the RAW hand is what the next ROI follows
    if (smoothOn) {
      const f = rs.filt[s] || (rs.filt[s] = { eu: new OneEuro(), size: null });
      const sz = Math.max(sizeOf(toPx(res[s], W, H)), 1);
      f.size = f.size == null ? sz : 0.9 * f.size + 0.1 * sz;
      const x = new Float64Array(63);
      res[s].forEach((q, j) => { x[3 * j] = q[0] * W / f.size; x[3 * j + 1] = q[1] * H / f.size; x[3 * j + 2] = q[2] * W / f.size; });
      const y = f.eu.push(x, frame.t / 1000);
      res[s] = res[s].map((q, j) => [r4(y[3 * j] * f.size / W), r4(y[3 * j + 1] * f.size / H), r4(y[3 * j + 2] * f.size / W)]);
    }
  }
  frame.lh = res.L; frame.rh = res.R;
  rs.i++;
  return got;
}

async function initHand() {
  if (handLm) return;
  try {
    const fileset = await filesetP;
    const buf = await fetch("/models/hand_landmarker.task").then((r) => { if (!r.ok) throw new Error("hand model " + r.status); return r.arrayBuffer(); });
    handLm = await Vision.HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetBuffer: new Uint8Array(buf), delegate: "CPU" },                   // CPU: the GPU belongs to Holistic (rare, small crops)
      runningMode: "IMAGE", numHands: 2, minHandDetectionConfidence: 0.4, minHandPresenceConfidence: 0.4, minTrackingConfidence: 0.4,
    });
  } catch (e) { handLm = null; console.warn("[hand-recovery] unavailable:", e); }
}

let filesetP = null, modelP = null;
let lm = null, faceLm = null, faceN = 0;

async function make(delegate, blendshapes = true) {
  const [fileset, model] = await Promise.all([filesetP, modelP]);
  return Vision.HolisticLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetBuffer: model.slice(), delegate },
    runningMode: "VIDEO",
    outputFaceBlendshapes: blendshapes,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minHandLandmarksConfidence: 0.5,
  });
}
// some WebGL drivers load the model and then fail on the first frame ("No support of const"): test one frame before trusting a set-up
async function works(l) {
  let bmp;
  try { bmp = await createImageBitmap(new ImageData(64, 64)); l.detectForVideo(bmp, 1, () => {}); return true; } catch (e) { return false; }
  finally { if (bmp) bmp.close(); }
}
function dispose() { try { handLm && handLm.close(); } catch (e) { /* closed */ } handLm = null; try { lm && lm.close(); } catch (e) { /* closed */ } try { faceLm && faceLm.close(); } catch (e) { /* closed */ } lm = faceLm = null; faceN = 0; }

async function init(prefer, last, opts) {
  dispose();
  recoverOn = !!(opts && opts.recover === true); smoothOn = !!(opts && opts.smooth); rs = newState();
  filesetP = filesetP || Vision.FilesetResolver.forVisionTasks("/mediapipe/wasm");
  modelP = modelP || fetch("/models/holistic_landmarker.task")
    .then((r) => { if (!r.ok) throw new Error("model " + r.status); return r.arrayBuffer(); })
    .then((b) => new Uint8Array(b)).catch((e) => { modelP = null; throw e; });
  let remember = null;
  if (prefer === "GPU") {
    if (last !== "GPU+CPU" && last !== "CPU") {           // full GPU: skipped when it failed on this machine before
      try { const l = await make("GPU"); if (await works(l)) { lm = l; return { delegate: "GPU", face: false, remember: "GPU" }; } l.close(); } catch (e) { /* next option */ }
    }
    if (last !== "CPU") {
      try {
        const l = await make("GPU", false);
        if (await works(l)) { lm = l; faceLm = await make("CPU"); return { delegate: "GPU+CPU", face: true, remember: "GPU+CPU" }; }   // expressions on the CPU every few frames
        l.close();
      } catch (e) { /* next option */ }
    }
    remember = "CPU";
  }
  lm = await make("CPU");
  return { delegate: "CPU", face: false, remember };
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === "init") {
    try { const r = await init(m.prefer, m.last, m); if (recoverOn) initHand(); self.postMessage({ type: "ready", ...r }); }
    catch (err) { self.postMessage({ type: "fail", message: String((err && err.message) || err) }); }
  } else if (m.type === "detect") {
    try {
      if (!lm) throw new Error("landmarker not ready");
      let frame = null;
      lm.detectForVideo(m.bmp, m.t, (res) => { frame = pack(res, m.t); });
      let rec = 0;
      if (frame) { try { rec = recoverHands(m.bmp, frame); } catch (e3) { rec = 0; } }           // never let the recovery layer break a frame
      let bs = null;
      if (m.face && faceLm && frame && frame.face.length && faceN++ % 8 === 0) {      // blendshapes every 8th frame (same cadence as the main-thread path)
        let g = null;
        faceLm.detectForVideo(m.bmp, m.t, (res) => { g = pack(res, m.t); });
        if (g) bs = g.bs;
      }
      self.postMessage({ type: "frame", id: m.id, frame, bs, rec });
    } catch (err) { self.postMessage({ type: "error", id: m.id, message: String((err && err.message) || err) }); }
    finally { try { m.bmp.close(); } catch (e2) { /* closed */ } }
  } else if (m.type === "close") { dispose(); self.close(); }
};
