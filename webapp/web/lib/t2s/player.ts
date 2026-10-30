// Motion timeline playback: decodes the API payload (int16 quaternions, uint8 morphs at 25 fps) and samples it at display rate
// with per-bone slerp, so playback is smooth at 60 / 120 Hz and speed can change without re-requesting.
import * as THREE from "three";
import type { Pose } from "./avatar";

export type Segment = { unit: number; kind: string; start: number; end: number; label: string };
export type MotionPayload = {
  fps: number; frames: number; bones: string[]; morphs: string[]; q: string; m: string; segments: Segment[];
};

function b64(s: string): Uint8Array {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export class Motion {
  readonly fps: number; readonly frames: number; readonly B: number; readonly M: number;
  readonly q: Float32Array; readonly m: Float32Array; readonly segments: Segment[];
  readonly bones: string[]; readonly morphs: string[];
  constructor(p: MotionPayload) {
    this.fps = p.fps; this.frames = p.frames; this.bones = p.bones; this.morphs = p.morphs;
    this.B = p.bones.length; this.M = p.morphs.length; this.segments = p.segments;
    const qi = new Int16Array(b64(p.q).buffer);
    this.q = new Float32Array(qi.length);
    for (let i = 0; i < qi.length; i++) this.q[i] = qi[i] / 32767;
    const mi = b64(p.m);
    this.m = new Float32Array(mi.length);
    for (let i = 0; i < mi.length; i++) this.m[i] = mi[i] / 255;
  }
  get duration() { return (this.frames - 1) / this.fps; }
}

const qa = new THREE.Quaternion(), qb = new THREE.Quaternion();

export function sample(mo: Motion, t: number, out: Pose) {
  const f = Math.min(Math.max(t * mo.fps, 0), mo.frames - 1);
  const i0 = Math.floor(f), i1 = Math.min(i0 + 1, mo.frames - 1), u = f - i0;
  const B = mo.B;
  for (let b = 0; b < B; b++) {
    const o0 = (i0 * B + b) * 4, o1 = (i1 * B + b) * 4;
    qa.set(mo.q[o0], mo.q[o0 + 1], mo.q[o0 + 2], mo.q[o0 + 3]);
    qb.set(mo.q[o1], mo.q[o1 + 1], mo.q[o1 + 2], mo.q[o1 + 3]);
    qa.slerp(qb, u).normalize();
    out.q[b * 4] = qa.x; out.q[b * 4 + 1] = qa.y; out.q[b * 4 + 2] = qa.z; out.q[b * 4 + 3] = qa.w;
  }
  const M = mo.M;
  for (let k = 0; k < M; k++) out.m[k] = mo.m[i0 * M + k] * (1 - u) + mo.m[i1 * M + k] * u;
  return f;
}

export class Player {
  motion: Motion | null = null;
  t = 0; speed = 1; playing = false; loop = false;
  pose: Pose = { q: new Float32Array(0), m: new Float32Array(0) };
  onEnd?: () => void;

  load(mo: Motion) {
    this.motion = mo; this.t = 0;
    this.pose = { q: new Float32Array(mo.B * 4), m: new Float32Array(mo.M) };
  }
  play() { if (this.motion) { if (this.t >= this.motion.duration) this.t = 0; this.playing = true; } }
  pause() { this.playing = false; }
  seek(t: number) { if (this.motion) this.t = Math.min(Math.max(t, 0), this.motion.duration); }

  /** advance and sample; returns the current fractional frame (or -1 if nothing loaded) */
  tick(dt: number): number {
    const mo = this.motion;
    if (!mo) return -1;
    if (this.playing) {
      this.t += dt * this.speed;
      if (this.t >= mo.duration) {
        if (this.loop) this.t = 0;
        else { this.t = mo.duration; this.playing = false; this.onEnd?.(); }
      }
    }
    return sample(mo, this.t, this.pose);
  }

  segmentAt(frame: number): Segment | undefined {
    return this.motion?.segments.find((s) => frame >= s.start && frame < s.end);
  }
}
