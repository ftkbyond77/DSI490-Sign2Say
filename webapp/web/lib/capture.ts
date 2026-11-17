// v12 prompted live capture (prompt-opt-v12 §1I): records the CONTINUOUS landmark stream of a whole live session (every frame, real
// timestamps, the engine's live state) + prompt marks, and uploads it in chunks to POST /v1/capture/chunk → cache/live_labeled/<session>/.
// Recognition is untouched: the engine calls frameTap.fn only when a capture session is open (null otherwise = v10 behaviour).
import type { LmFrame } from "./holistic";

export type CaptureMark = { i: number; id: string; kind: string; prompt: string; gloss?: string; t_show: number; t_next?: number; typed?: string };
export type CapturePrompt = { id: string; kind: string; prompt: string; gloss?: string; seconds?: number; note?: string };
export type CaptureScript = { name: string; title?: string; prompts: CapturePrompt[] };

export const frameTap: { fn: ((f: LmFrame, liveState: string) => void) | null } = { fn: null };
export const resultTap: { fn: ((kind: string, payload: unknown) => void) | null } = { fn: null };

const API = "/api";              // same path the rest of the client uses (gateway in production, next.config rewrite in dev)
const r4 = (x: number) => Math.round(x * 1e4) / 1e4;
const pts = (a: number[][]) => a.map((p) => p.map(r4));

export class CaptureSession {
  session: string;
  seq = 0;
  frames: unknown[] = [];
  events: unknown[] = [];
  marks: CaptureMark[] = [];
  nFrames = 0;
  closed = false;
  constructor(public script: string, public width: number, public height: number) {
    const d = new Date();
    const p = (n: number) => String(n).padStart(2, "0");
    this.session = `${script}_${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  }
  frame(f: LmFrame, liveState: string) {
    if (this.closed) return;
    this.frames.push({ t: Math.round(f.t * 10) / 10, s: liveState, pose: pts(f.pose), lh: pts(f.lh), rh: pts(f.rh), face: pts(f.face), bs: f.bs.map(r4) });
    this.nFrames++;
    if (this.frames.length >= 300) void this.flush();
  }
  event(kind: string, payload: unknown) { if (!this.closed) this.events.push({ t: performance.now(), kind, payload }); }
  async flush(final = false) {
    const body = { session: this.session, script: this.script, seq: this.seq++, width: this.width, height: this.height, final,
                   frames: this.frames, events: this.events, marks: final ? this.marks : [], ua: navigator.userAgent };
    this.frames = []; this.events = [];
    try {
      await fetch(`${API}/v1/capture/chunk`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    } catch (e) { console.warn("capture upload failed", e); }
  }
}

export async function loadScript(name: string): Promise<CaptureScript> {
  const r = await fetch(`${API}/v1/capture/script/${encodeURIComponent(name)}`);
  if (!r.ok) throw new Error(`script ${name}: ${r.status}`);
  return r.json();
}
