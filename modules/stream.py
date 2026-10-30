"""Streaming runtime (v9): live camera → words that appear while the person signs, instead of one result per finished clip.

    st = StreamingTranslator(pipe)                # pipe = SignPipeline
    for frame in camera:                          # frame = one MediaPipe raw frame (pose / lh / rh / face / bs) + timestamp (s)
        ev = st.push(frame, t)                    # → None, or {"partial": [...], "confirmed": [new words], "latency": …}
    st.flush()

How (stateful, bounded work):
  * frames go into a buffer; every `step_s` the pipeline runs on the buffer window [anchor − context, now] (≤ max_window_s), never on
    the whole history — work per step is bounded no matter how long the person signs;
  * a sign segment is CONFIRMED once it ended ≥ `lookahead_s` before now (the decoder has seen what follows it, so a merge / split
    can no longer change it) and the same word was read on the previous step at an overlapping place (stability); confirmed words
    are emitted once, never revised; the anchor moves to the end of the confirmed sign;
  * everything else is PARTIAL (shown greyed, may change) — the flicker is measured: how often a partial word changes before it
    is confirmed;
  * a hands-down pause ≥ `rest_s` confirms whatever is left (end of utterance).
Latency of a word = confirmation time − the sign's end time (the wait a viewer has after the hands finished that sign), with the
compute counted: a step starts only when the previous one has finished, and its words exist only after its processing time.
"""
from __future__ import annotations

import time

import numpy as np

KEYS = ("pose", "lh", "rh", "face", "bs")


class StreamingTranslator:
    def __init__(self, pipe, step_s=0.5, lookahead_s=0.8, context_s=1.0, max_window_s=8.0, rest_s=0.8, alphabet=None, hw=(720, 1280)):
        self.pipe, self.step_s, self.look, self.ctx, self.maxw, self.rest_s = pipe, step_s, lookahead_s, context_s, max_window_s, rest_s
        self.alphabet, self.hw = alphabet, hw
        self.frames, self.ts = [], []
        self.anchor = 0.0                     # stream time before which everything is final
        self.last_run, self.prev = -1e9, []
        self.confirmed, self.events, self.flicker = [], [], 0
        self.proc_ms = []
        self.busy_until = -1e9                # stream time when the previous step's result is ready (one step at a time)

    def _raw(self, a, b):
        fr = self.frames[a:b]
        return dict(t=np.array(self.ts[a:b], np.float64), hw=np.array(self.hw, np.int32), **{k: np.stack([f[k] for f in fr]) for k in KEYS})

    def _hands_seen(self, f):
        return float(np.abs(f["lh"]).sum() + np.abs(f["rh"]).sum()) > 0

    def push(self, frame, t):
        self.frames.append(frame); self.ts.append(float(t))
        if t - self.last_run < self.step_s or t < self.busy_until:
            return None
        return self._step(t)

    def _step(self, now, final=False):
        self.last_run = now
        t0 = max(self.anchor - self.ctx, now - self.maxw)
        a = int(np.searchsorted(self.ts, t0)); b = len(self.ts)
        if b - a < 8:
            return None
        w0 = time.perf_counter()
        r = self.pipe.run_landmarks(self._raw(a, b), alphabet=self.alphabet)
        self.proc_ms.append((time.perf_counter() - w0) * 1000)
        done = now + self.proc_ms[-1] / 1000.0           # the result exists only after the compute (real-time accounting)
        self.busy_until = done
        base = self.ts[a]
        segs = [dict(word=s["nearest"], status=s["status"], p=s.get("p_correct", 0.0), t0=base + s["t0"], t1=base + s["t1"])
                for s in r["segments"] if s["status"] != "transition"]
        recent = [self._hands_seen(f) for f in self.frames[-int(self.rest_s * 25):]]
        resting = final or (len(recent) >= int(self.rest_s * 25) * 0.9 and not any(recent))
        new = []
        for s in segs:
            if s["t1"] <= self.anchor + 0.05:
                continue                                      # already confirmed region
            if s["t0"] < self.anchor and self.anchor - s["t0"] > 0.5 * (s["t1"] - s["t0"]):
                continue                                      # mostly inside a confirmed sign: its re-cut tail, not a new sign
            last = self.confirmed[-1] if self.confirmed else (new[-1] if new else None)
            if last is not None and last["word"] == s["word"] and s["t0"] - last["t1"] < 0.4:
                last["t1"] = max(last["t1"], s["t1"]); self.anchor = max(self.anchor, s["t1"])
                continue                                      # the same sign continued across a step boundary → extend, not repeat
            stable = any(p["word"] == s["word"] and min(p["t1"], s["t1"]) - max(p["t0"], s["t0"]) > 0 for p in self.prev)
            if (now - s["t1"] >= self.look and stable) or resting:
                if s["status"] == "unknown":
                    pass                                      # "[?]" segments are not spoken, but they still close the region
                else:
                    new.append(dict(s, confirmed_at=done, latency=done - s["t1"]))
                self.anchor = max(self.anchor, s["t1"])
        for p in self.prev:                                   # flicker: a partial word that changed while still partial
            if p["t1"] > self.anchor and not any(min(p["t1"], s["t1"]) - max(p["t0"], s["t0"]) > 0 and p["word"] == s["word"] for s in segs):
                self.flicker += 1
        self.prev = [s for s in segs if s["t1"] > self.anchor]
        self.confirmed += new
        if len(self.ts) > 4000:                               # keep memory bounded: drop frames long before the window
            cut = int(np.searchsorted(self.ts, self.anchor - self.ctx - 1.0))
            self.frames, self.ts = self.frames[cut:], self.ts[cut:]
        ev = dict(t=now, partial=[s["word"] for s in self.prev], confirmed=[s["word"] for s in new], proc_ms=self.proc_ms[-1])
        self.events.append(ev)
        return ev

    def flush(self):
        if self.ts:
            self._step(self.ts[-1] + self.look + 0.01, final=True)
        return self.confirmed


def replay(pipe, raw, fps=25.0, **kw):
    """Feed a recorded MediaPipe raw dict through the streaming translator in stream time → (confirmed words, stats)."""
    st = StreamingTranslator(pipe, hw=tuple(int(x) for x in raw["hw"]), **kw)
    T = len(raw["t"])
    for i in range(T):
        st.push({k: raw[k][i] for k in KEYS}, float(raw["t"][i] - raw["t"][0]))
    st.flush()
    lat = [c["latency"] for c in st.confirmed]
    words = [c["word"] for c in st.confirmed]
    dup = sum(1 for i in range(1, len(st.confirmed)) if st.confirmed[i]["word"] == st.confirmed[i - 1]["word"]
              and st.confirmed[i]["t0"] - st.confirmed[i - 1]["t1"] < 0.3)
    return words, dict(latency=lat, proc_ms=st.proc_ms, flicker=st.flicker, duplicates=dup, steps=len(st.proc_ms),
                       duration_s=float(raw["t"][-1] - raw["t"][0]) if T else 0.0)
