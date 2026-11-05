// The landmark detector the camera screens use. MediaPipe runs in a Web Worker (public/workers/holistic.worker.js) so loading it and
// every frame's inference no longer freeze the page; if a browser cannot (no OffscreenCanvas / worker failed to start) it falls back
// to the previous main-thread landmarker (lib/holistic.ts), so recognition never depends on the worker.
import { createHolistic, detect as detectMain, type Delegate, type LmFrame, type Landmarkers } from "./holistic";

export interface Detector {
  delegate: Delegate;
  /** One frame → its landmarks (null: nothing found). `face`: also refresh the blendshape expressions (every 8th such frame). */
  detect(src: CanvasImageSource, t: number, face: boolean): Promise<LmFrame | null>;
  close(): void;
}

const SETUP_KEY = "thaislm.mp.setup";                    // the same remembered set-up as lib/holistic.ts
const remembered = (): string | null => { try { return localStorage.getItem(SETUP_KEY); } catch { return null; } };
const remember = (d: string | null) => { try { if (d) localStorage.setItem(SETUP_KEY, d); else localStorage.removeItem(SETUP_KEY); } catch { /* storage blocked */ } };
const WORKER_URL = "/workers/holistic.worker.js?v=1";

type Ready = { delegate: Delegate; face: boolean; remember: string | null };
type Frame = { frame: LmFrame | null; bs: number[] | null };

class WorkerDetector implements Detector {
  delegate: Delegate = "CPU";
  private w: Worker;
  private seq = 0;
  private waiting = new Map<number, { ok: (f: Frame) => void; bad: (e: Error) => void }>();
  private onReady: { ok: (r: Ready) => void; bad: (e: Error) => void } | null = null;
  constructor() {
    this.w = new Worker(WORKER_URL);
    this.w.onmessage = (e: MessageEvent) => {
      const m = e.data;
      if (m.type === "ready") { this.onReady?.ok(m); this.onReady = null; }
      else if (m.type === "fail") { this.onReady?.bad(new Error(m.message)); this.onReady = null; }
      else if (m.type === "frame") { this.waiting.get(m.id)?.ok({ frame: m.frame, bs: m.bs }); this.waiting.delete(m.id); }
      else if (m.type === "error") { this.waiting.get(m.id)?.bad(new Error(m.message)); this.waiting.delete(m.id); }
    };
    this.w.onerror = (e) => this.fail(new Error(e.message || "worker error"));
  }
  private fail(err: Error) {
    this.onReady?.bad(err); this.onReady = null;
    this.waiting.forEach((p) => p.bad(err)); this.waiting.clear();
  }
  init(prefer: "GPU" | "CPU"): Promise<void> {
    return new Promise<Ready>((ok, bad) => { this.onReady = { ok, bad }; this.w.postMessage({ type: "init", prefer, last: remembered() }); })
      .then((r) => { this.delegate = r.delegate; if (prefer === "GPU") remember(r.remember); });
  }
  async detect(src: CanvasImageSource, t: number, face: boolean): Promise<LmFrame | null> {
    const bmp = await createImageBitmap(src as ImageBitmapSource);             // a snapshot of this frame; ownership moves to the worker
    const id = ++this.seq;
    const r = await new Promise<Frame>((ok, bad) => { this.waiting.set(id, { ok, bad }); this.w.postMessage({ type: "detect", id, t, bmp, face }, [bmp]); });
    if (r.frame && r.bs) r.frame.bs = r.bs;
    return r.frame;
  }
  close() { try { this.w.postMessage({ type: "close" }); } catch { /* gone */ } this.w.terminate(); this.fail(new Error("closed")); }
}

/** Fallback: the previous behaviour, everything on the main thread. */
class MainDetector implements Detector {
  private n = 0;
  constructor(private l: Landmarkers) {}
  get delegate() { return this.l.delegate; }
  async detect(src: CanvasImageSource, t: number, face: boolean): Promise<LmFrame | null> {
    const f = detectMain(this.l.lm, src as HTMLVideoElement | HTMLCanvasElement, t);
    if (face && f && this.l.face && f.face.length && this.n++ % 8 === 0) { const g = detectMain(this.l.face, src as HTMLVideoElement | HTMLCanvasElement, t); if (g) f.bs = g.bs; }
    return f;
  }
  close() { this.l.lm.close(); this.l.face?.close(); }
}

const canWorker = () => typeof Worker !== "undefined" && typeof OffscreenCanvas !== "undefined" && typeof createImageBitmap === "function";

/** A new detector (not shared): the worker if it can start, otherwise the main-thread landmarker. */
export async function createDetector(prefer: "GPU" | "CPU" = "GPU"): Promise<Detector> {
  if (canWorker()) {
    const d = new WorkerDetector();
    try { await d.init(prefer); return d; }
    catch (e) { console.warn("[mediapipe] worker unavailable, using the main thread:", e); d.close(); }
  }
  return new MainDetector(await createHolistic(prefer));
}

// One detector for the whole app, started in the background as soon as a camera screen opens.
let shared: Promise<Detector> | null = null;
export function getDetector(): Promise<Detector> {
  shared ??= createDetector("GPU").then((d) => { performance.mark("thaislm:mediapipe-ready"); return d; }).catch((e) => { shared = null; throw e; });
  return shared;
}
export function replaceDetector(d: Detector) { shared = Promise.resolve(d); }
export function preloadDetector() {
  const go = () => { getDetector().catch(() => { /* reported when the camera is opened */ }); };
  if ("requestIdleCallback" in window) (window as Window & { requestIdleCallback: (f: () => void, o?: { timeout: number }) => number }).requestIdleCallback(go, { timeout: 1200 });
  else setTimeout(go, 300);
}
