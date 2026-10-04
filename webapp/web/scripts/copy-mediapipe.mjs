// Serve MediaPipe's WASM runtime from our own origin (no CDN at run time): node_modules → public/mediapipe/wasm
import { cpSync, existsSync, mkdirSync } from "node:fs";
const src = "node_modules/@mediapipe/tasks-vision/wasm";
if (existsSync(src)) {
  mkdirSync("public/mediapipe/wasm", { recursive: true });
  cpSync(src, "public/mediapipe/wasm", { recursive: true });
  console.log("copied MediaPipe wasm → public/mediapipe/wasm");
}
// The facial-expression model (face-api FaceExpressionNet) the same way: node_modules → public/models/face-api
const fa = "node_modules/@vladmandic/face-api/model";
if (existsSync(fa)) {
  mkdirSync("public/models/face-api", { recursive: true });
  for (const f of ["face_expression_model-weights_manifest.json", "face_expression_model.bin"]) cpSync(`${fa}/${f}`, `public/models/face-api/${f}`);
  console.log("copied face expression model → public/models/face-api");
}
