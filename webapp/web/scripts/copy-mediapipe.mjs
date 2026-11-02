// Serve MediaPipe's WASM runtime from our own origin (no CDN at run time): node_modules → public/mediapipe/wasm
import { cpSync, existsSync, mkdirSync } from "node:fs";
const src = "node_modules/@mediapipe/tasks-vision/wasm";
if (existsSync(src)) {
  mkdirSync("public/mediapipe/wasm", { recursive: true });
  cpSync(src, "public/mediapipe/wasm", { recursive: true });
  console.log("copied MediaPipe wasm → public/mediapipe/wasm");
}
