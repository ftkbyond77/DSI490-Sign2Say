/* MediaPipe Holistic in a Web Worker (classic script: MediaPipe loads its WASM runtime with importScripts).
 * Why: creating the landmarker (WASM compile, GPU shader build, a test frame) and every inference ran on the page's main thread, so the
 * screen could not be clicked while it loaded and stuttered while the camera was on. Here the page only snapshots a frame
 * (createImageBitmap) and receives the landmarks back; the same model, the same options, the same output as lib/holistic.ts.
 * The landmark packing below (MESH68, pack) must stay identical to lib/holistic.ts.
 *
 * Messages in : {type:"init", prefer:"GPU"|"CPU", last}  ·  {type:"detect", id, t, bmp, face}  ·  {type:"close"}
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
function dispose() { try { lm && lm.close(); } catch (e) { /* closed */ } try { faceLm && faceLm.close(); } catch (e) { /* closed */ } lm = faceLm = null; faceN = 0; }

async function init(prefer, last) {
  dispose();
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
    try { const r = await init(m.prefer, m.last); self.postMessage({ type: "ready", ...r }); }
    catch (err) { self.postMessage({ type: "fail", message: String((err && err.message) || err) }); }
  } else if (m.type === "detect") {
    try {
      if (!lm) throw new Error("landmarker not ready");
      let frame = null;
      lm.detectForVideo(m.bmp, m.t, (res) => { frame = pack(res, m.t); });
      let bs = null;
      if (m.face && faceLm && frame && frame.face.length && faceN++ % 8 === 0) {      // blendshapes every 8th frame (same cadence as the main-thread path)
        let g = null;
        faceLm.detectForVideo(m.bmp, m.t, (res) => { g = pack(res, m.t); });
        if (g) bs = g.bs;
      }
      self.postMessage({ type: "frame", id: m.id, frame, bs });
    } catch (err) { self.postMessage({ type: "error", id: m.id, message: String((err && err.message) || err) }); }
    finally { try { m.bmp.close(); } catch (e2) { /* closed */ } }
  } else if (m.type === "close") { dispose(); self.close(); }
};
