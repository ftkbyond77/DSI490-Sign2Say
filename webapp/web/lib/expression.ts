// Live facial expression from a trained model: face-api's FaceExpressionNet (7 classes, 112×112 face crop, ~330 kB, served from our own
// origin). The face is cropped with MediaPipe's own face landmarks, so no second face detector runs. At most ~7 predictions/s, off the
// drawing path; the label on screen is the smoothed (EMA) last result, so it shows on every frame.
import type { Emotion } from "./holistic";

type Net = { predictExpressions: (c: HTMLCanvasElement) => Promise<Record<string, number>> };
const NAME: Record<string, string> = { neutral: "neutral", happy: "happy", sad: "sad", angry: "angry", fearful: "fear", disgusted: "disgust", surprised: "surprise" };

export class ExpressionModel {
  private net: Net | null = null;
  private loading: Promise<void> | null = null;
  private busy = false;
  private last = 0;
  private ema: Record<string, number> = {};
  private crop = typeof document !== "undefined" ? document.createElement("canvas") : null;
  failed = false;

  load() {
    this.loading ??= (async () => {
      try {
        const fa = await import("@vladmandic/face-api");
        const tf = fa.tf as unknown as { setBackend: (b: string) => Promise<boolean>; ready: () => Promise<void> };
        await tf.setBackend("webgl"); await tf.ready();
        await fa.nets.faceExpressionNet.loadFromUri("/models/face-api");
        const net = fa.nets.faceExpressionNet as unknown as Net;
        if (this.crop) { this.crop.width = this.crop.height = 112; await net.predictExpressions(this.crop); }  // compile the shaders now (~0.2 s), not mid-sign
        this.net = net;
      } catch (e) { this.failed = true; console.warn("expression model unavailable, using blendshapes", e); }
    })();
    return this.loading;
  }

  /** Feed the current video frame + the 68 face points (0..1); returns the smoothed expression, or null before the first prediction. */
  update(src: CanvasImageSource, vw: number, vh: number, face: number[][], t: number): Emotion | null {
    if (this.net && this.crop && face.length === 68 && !this.busy && t - this.last > 140) {
      let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
      for (const p of face) { x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]); }
      const cx = (x0 + x1) / 2 * vw, cy = ((y0 + y1) / 2 - (y1 - y0) * 0.08) * vh;   // the 68 points stop at the brows: shift up a little
      const side = Math.max((x1 - x0) * vw, (y1 - y0) * vh) * 1.25;
      if (side > 24) {
        this.busy = true; this.last = t;
        const c = this.crop; c.width = c.height = 112;
        c.getContext("2d")!.drawImage(src, cx - side / 2, cy - side / 2, side, side, 0, 0, 112, 112);
        this.net.predictExpressions(c).then((p) => {
          for (const k of Object.keys(NAME)) this.ema[k] = (this.ema[k] ?? p[k]) * 0.6 + p[k] * 0.4;
        }).catch(() => {}).finally(() => { this.busy = false; });
      }
    }
    const ks = Object.keys(this.ema);
    if (!ks.length) return null;
    const best = ks.reduce((a, b) => (this.ema[b] > this.ema[a] ? b : a));
    return { emotion: NAME[best], level: this.ema[best] };
  }
  reset() { this.ema = {}; }
}
