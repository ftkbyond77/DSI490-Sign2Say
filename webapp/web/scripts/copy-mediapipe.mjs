// Serve MediaPipe's WASM runtime from our own origin (no CDN at run time): node_modules → public/mediapipe
//   wasm/            the runtime files (loaded by FilesetResolver)
//   vision_bundle.js the classic (non-module) build of tasks-vision — the Web Worker (public/workers/holistic.worker.js) importScripts() it
import { copyFileSync, cpSync, existsSync, mkdirSync } from "node:fs";
const root = "node_modules/@mediapipe/tasks-vision";
if (existsSync(`${root}/wasm`)) {
  mkdirSync("public/mediapipe/wasm", { recursive: true });
  cpSync(`${root}/wasm`, "public/mediapipe/wasm", { recursive: true });
  copyFileSync(`${root}/vision_bundle.js`, "public/mediapipe/vision_bundle.js");
  console.log("copied MediaPipe wasm + vision_bundle.js → public/mediapipe");
}
