// Face branch, client side (v11): cuts a 48×48 grayscale picture of the face from the camera frame and asks the server's face model
// (sign-api /v1/face, models/face-model-*-v1) for the expression. The sign path never sees any of it: the face box is placed from the
// MediaPipe face points that are already extracted (read only), the pictures go to their own endpoint / their own request field, and
// every failure just means "no label" (the blendshape label takes over) — never a change to the landmarks or the sign request.
// ≤ ~5 crops/s; the label on screen is the server's smoothed (EMA + hysteresis) answer, carried between calls as a small state.
import type { Emotion } from "./holistic";

export type FaceCrop = { t: number; g: string; side: number; occ: number };
type Resp = { ok: boolean; smoothed?: { expression: string; confidence: number; valid: boolean }; state?: unknown };

const B64 = (u: Uint8Array) => { let s = ""; for (let i = 0; i < u.length; i += 0x2000) s += String.fromCharCode(...u.subarray(i, i + 0x2000)); return btoa(s); };

export class ExpressionModel {
  failed = false;
  private cfg = { scale: 1.22, shift_up: 0.16 };
  private every = 180;                                  // ms between crops
  private busy = false;
  private last = 0;
  private fresh = 0;                                    // time of the last answer
  private state: unknown = null;
  private shown: Emotion | null = null;
  private pending: FaceCrop[] = [];
  private buf: FaceCrop[] = [];                         // the last ~70 s of crops: an utterance's crops are taken from here
  private c96 = typeof document !== "undefined" ? document.createElement("canvas") : null;
  private c48 = typeof document !== "undefined" ? document.createElement("canvas") : null;
  private loading: Promise<void> | null = null;

  load() {
    this.loading ??= (async () => {
      try {
        const r = await fetch("/api/v1/face/info");
        const j = await r.json();
        if (!r.ok || !j.ok) throw new Error("face model not loaded");
        if (j.crop) this.cfg = { scale: j.crop.scale ?? this.cfg.scale, shift_up: j.crop.shift_up ?? this.cfg.shift_up };
        if (this.c96 && this.c48) { this.c96.width = this.c96.height = 96; this.c48.width = this.c48.height = 48; }
        this.failed = false;
      } catch (e) { this.failed = true; this.loading = null; console.warn("face expression model unavailable, using blendshapes", e); }   // loading = null: asked again when the camera opens
    })();
    return this.loading;
  }

  private crop(src: CanvasImageSource, vw: number, vh: number, face: number[][], hands: number[][][], t: number): FaceCrop | null {
    if (!this.c96 || !this.c48) return null;
    let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
    for (const p of face) { x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]); }
    const cx = (x0 + x1) / 2 * vw, cy = ((y0 + y1) / 2 - (y1 - y0) * this.cfg.shift_up) * vh;     // same rule as modules/face_fer.py crop_box
    const side = Math.max((x1 - x0) * vw, (y1 - y0) * vh) * this.cfg.scale;
    if (!(side > 24)) return null;
    const a = this.c96.getContext("2d", { willReadFrequently: true })!, b = this.c48.getContext("2d", { willReadFrequently: true })!;
    a.imageSmoothingEnabled = b.imageSmoothingEnabled = true; a.imageSmoothingQuality = b.imageSmoothingQuality = "high";
    a.drawImage(src, cx - side / 2, cy - side / 2, side, side, 0, 0, 96, 96);
    b.drawImage(this.c96, 0, 0, 48, 48);
    const d = b.getImageData(0, 0, 48, 48).data, g = new Uint8Array(2304);
    for (let i = 0; i < 2304; i++) g[i] = Math.round(0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]);
    // hand over the face (share of a hand's points inside the face box): the server turns it into the quality gate
    let occ = 0;
    for (const h of hands) if (h.length) {
      let n = 0; for (const p of h) if (p[0] > x0 - 0.05 && p[0] < x1 + 0.05 && p[1] > y0 - 0.1 && p[1] < y1 + 0.05) n++;
      occ = Math.max(occ, n / h.length);
    }
    return { t: Math.round(t), g: B64(g), side: Math.round(side), occ: Math.round(occ * 100) / 100 };
  }

  private flush() {
    if (this.busy || !this.pending.length) return;
    this.busy = true;
    const crops = this.pending.splice(0, 12);
    fetch("/api/v1/face", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ crops, state: this.state }) })
      .then((r) => r.json() as Promise<Resp>)
      .then((j) => {
        if (!j.ok || !j.smoothed) return;
        this.state = j.state ?? null; this.fresh = performance.now();
        this.shown = { emotion: j.smoothed.expression, level: j.smoothed.expression === "unknown" ? 0 : j.smoothed.confidence };
      })
      .catch(() => {})
      .finally(() => { this.busy = false; });
  }

  /** Feed the current video frame + the 68 face points (0..1) + the hands' points; returns the smoothed expression, or null when there is none (yet) —
   *  the caller then falls back to the blendshape label. */
  update(src: CanvasImageSource, vw: number, vh: number, face: number[][], t: number, hands: number[][][] = []): Emotion | null {
    if (this.failed || face.length !== 68) return this.current(t);
    if (t - this.last >= this.every) {
      const c = this.crop(src, vw, vh, face, hands, t);
      this.last = t;
      if (c) { this.pending.push(c); this.buf.push(c); while (this.buf.length && t - this.buf[0].t > 70000) this.buf.shift(); }
    }
    this.flush();
    return this.current(t);
  }

  private current(t: number): Emotion | null { return this.shown && performance.now() - this.fresh < 2500 ? this.shown : (void t, null); }

  /** The crops of one utterance (frames[0].t … last frame t, same clock) with times relative to its first frame, for the translate request. */
  takeCrops(t0: number, t1: number): FaceCrop[] {
    return this.buf.filter((c) => c.t >= t0 - 100 && c.t <= t1 + 100).map((c) => ({ ...c, t: c.t - t0 }));
  }

  reset() { this.shown = null; this.state = null; this.pending = []; }
}
