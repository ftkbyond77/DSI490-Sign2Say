import type { LmFrame } from "./holistic";

/** Experimental causal median, NOT a wait-until-still gate. Missing hands stay missing.
 * Reset at missing observations/time gaps; timestamps/body/face are never smoothed. */
export class HandConsensus {
  private left: number[][][] = [];
  private right: number[][][] = [];
  private last = -Infinity;
  constructor(readonly window: number = 0) {}
  reset() { this.left = []; this.right = []; this.last = -Infinity; }
  apply(f: LmFrame): LmFrame {
    if (!this.window) return f;
    if (f.t <= this.last || f.t - this.last > 250) this.reset();
    this.last = f.t;
    const filter = (hand: number[][], history: number[][][]) => {
      if (hand.length !== 21) { history.length = 0; return hand; }
      history.push(hand); if (history.length > this.window) history.shift();
      // No partial/even-window median: retain current observation during warmup.
      if (history.length < this.window) return hand;
      return hand.map((p, i) => p.map((_, j) => {
        const values = history.map(h => h[i][j]).sort((a, b) => a - b);
        return values[values.length >> 1];
      }));
    };
    return { ...f, lh: filter(f.lh, this.left), rh: filter(f.rh, this.right) };
  }
}
export function experimentOptions(search: string) {
  const q = new URLSearchParams(search);
  return { window: [3, 5].includes(Number(q.get("consensus"))) ? Number(q.get("consensus")) : 0,
    faceEvery: [2, 3].includes(Number(q.get("faceEvery"))) ? Number(q.get("faceEvery")) : 1,
    enabled: q.get("liveExp") === "1" };
}
