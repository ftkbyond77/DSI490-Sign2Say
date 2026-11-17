"""Labeled live-mode simulation (EA-for-v12): conditions on real MediaPipe clips with references.
usage: live_sim.py <scratch_dir> <real|w_t51_tune|...> <cond,cond,...>"""
import sys, json, time, numpy as np
from pathlib import Path
ROOT = Path(r"D:/SLM-labs/SLM-Labs-SignDINO"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
import logging; logging.disable(logging.WARNING)
S = Path(sys.argv[1]); which = sys.argv[2]; conds = sys.argv[3].split(",")
import decision as D
from modules.pipeline import SignPipeline
from modules import stream as ST


def ed(ref, hyp):
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        p = d.copy(); d[0] = i
        for j, h in enumerate(hyp, 1):
            d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (r != h))
    return int(d[-1])


def flags(raw):
    """handsUp (1.5 sw / wrist 1.3) and handsRaised (0.8 sw, real hand) per frame, as webapp/web/lib/holistic.ts"""
    p = raw["pose"]; ls, rs = p[:, 11], p[:, 12]
    sw = np.hypot(ls[:, 0] - rs[:, 0], ls[:, 1] - rs[:, 1]); sw[sw == 0] = 0.2
    mid = (ls[:, 1] + rs[:, 1]) / 2
    has = lambda h: np.abs(h).sum((1, 2)) > 0
    my = lambda h: h[:, :, 1].mean(1)
    upk = lambda k: (has(raw["lh"]) & (my(raw["lh"]) < mid + k * sw)) | (has(raw["rh"]) & (my(raw["rh"]) < mid + k * sw))
    wrist = lambda i: (p[:, i, 3] > 0.5) & (p[:, i, 1] < mid + 1.3 * sw)
    return upk(1.5) | wrist(15) | wrist(16), upk(0.8), has(raw["lh"]) | has(raw["rh"])


def sub(raw, idx, t=None):
    T = len(raw["t"])
    out = {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == T and k != "hw" else v) for k, v in raw.items()}
    if t is not None:
        out["t"] = np.asarray(t, np.float64)
    return out


def live_utts(raw, end_ms, pre=500, tail=400):
    """the browser's live state machine -> list of index arrays (utterances), number dropped as too short"""
    up, raised, hands = flags(raw); t = raw["t"] * 1000.0; T = len(t)
    utts, dropped, state, upSince, start, lastUp, u0 = [], 0, "w", None, 0, 0, 0
    for i in range(T):
        if state == "w":
            if raised[i]:
                if upSince is None:
                    upSince = t[i]
            else:
                upSince = None
            if upSince is not None and t[i] - upSince >= 250:
                state = "s"; u0 = int(np.searchsorted(t, t[i] - pre)); start = t[i]; lastUp = t[i]
        else:
            if up[i]:
                lastUp = t[i]
            if t[i] - lastUp >= end_ms or t[i] - start >= 20000 or i == T - 1:
                idx = np.arange(u0, i + 1); idx = idx[t[idx] <= lastUp + tail]
                dur = t[idx[-1]] - t[idx[0]] if len(idx) > 1 else 0
                upms = float(np.sum(np.diff(t[idx])[up[idx][1:]])) if len(idx) > 1 else 0
                if dur < 1200 or upms < 600 or hands[idx].sum() < 3:
                    dropped += 1
                else:
                    utts.append(idx)
                state, upSince = "w", None
    return utts, dropped


DTS = json.load(open(S / "live_dts.json"))


def retime(raw, rng, stall=True):
    """re-sample a regular clip at the frame times of a real live capture (stall=False: the every-8th-frame 200 ms stall removed)"""
    t = raw["t"] * 1000.0; d = np.array(DTS[rng.randint(len(DTS))])
    if not stall:
        d = np.where(d > 120, np.median(d), d)
    tt = [0.0]; k = rng.randint(len(d))
    while tt[-1] < t[-1]:
        tt.append(tt[-1] + d[k % len(d)]); k += 1
    tt = np.array(tt[:-1]); idx = np.clip(np.searchsorted(t, tt), 0, len(t) - 1); keep = np.r_[True, np.diff(idx) > 0]
    return idx[keep], t[idx[keep]] / 1000.0


RUNS = [x for x in json.load(open(S / "live_handloss.json"))["runs"]["live"] if x > 0.05]


def drop(raw, rng, extra):
    """replay real live hand-loss runs inside the signing part until `extra` of its frames lost a hand (70 % one hand, 30 % both)"""
    r = dict(raw); r["lh"] = raw["lh"].copy(); r["rh"] = raw["rh"].copy(); t = raw["t"]
    a = (np.abs(r["lh"]).sum((1, 2)) > 0) | (np.abs(r["rh"]).sum((1, 2)) > 0); idx = np.where(a)[0]
    if len(idx) < 10:
        return r
    lo, hi = idx[0], idx[-1]; lost = np.zeros(len(t), bool); guard = 0
    while lost[lo:hi + 1].mean() < extra and guard < 200:
        guard += 1; i = rng.randint(lo, hi + 1); d = RUNS[rng.randint(len(RUNS))]; j = int(np.searchsorted(t, t[i] + d)); j = min(max(j, i + 1), hi + 1)
        u = rng.rand()
        if u < 0.7:
            (r["lh"] if rng.rand() < 0.5 else r["rh"])[i:j] = 0
        else:
            r["lh"][i:j] = 0; r["rh"][i:j] = 0
        lost[i:j] = True
    return r


if which == "real":
    items = D.build_items("data_test") + D.build_items("team") + D.build_items("iphone_word") + D.build_items("iphone_sentence")
    pipe = SignPipeline(model_dir=ROOT / "models/model-v10", backend="onnx", device="cpu", vocab="conversation", use_llm=False, use_face=False)
else:
    items = D.build_items(which)
    pipe = SignPipeline(model_dir=ROOT / "models/model-v10", backend="onnx", device="cpu", vocab="conversation", use_llm=False, use_face=False,
                        bank_file=ROOT / "artifacts/eval/bank_eval.npz")
if len(sys.argv) > 4:
    items = items[:int(sys.argv[4])]
print("items", len(items), "fps", [round(float(1 / np.median(np.diff(it["raw"]["t"]))), 1) for it in items[:5]], flush=True)


def run(raw):
    segs = [s for s in pipe.run_landmarks(raw)["segments"] if s["status"] != "transition"]
    return [(s["nearest"], s["status"], float(s.get("p_correct", 0) or 0)) for s in segs]


res = {c: [] for c in conds}; t00 = time.time()
out = S / f"live_sim_{which}_{'-'.join(conds)[:60]}.json"
for n, it in enumerate(items):
    raw, ref = dict(it["raw"]), it["ref"]; raw["t"] = np.asarray(raw["t"], np.float64) - float(raw["t"][0])
    T = len(raw["t"])
    for c in conds:
        rng = np.random.RandomState(1000 + n)          # the same re-timing for every condition of a clip
        try:
            extra = {}
            if c == "full":
                segs = run(raw)
            elif c == "u12":
                segs = run(sub(raw, np.arange(0, T, 2)))
            elif c in ("jit", "nostall"):
                i, t = retime(raw, rng, stall=(c == "jit")); segs = run(sub(raw, i, t)); extra["fps"] = (len(i) - 1) / max(t[-1] - t[0], 1e-6)
            elif c.startswith("drop"):
                segs = run(drop(raw, rng, int(c[4:]) / 100.0))
            elif c.startswith("zoom"):
                z = int(c[4:]) / 100.0; r2 = dict(raw)
                for k in ("pose", "lh", "rh", "face"):
                    v = raw[k].copy(); m = np.abs(v[..., :2]).sum(-1, keepdims=True) > 0
                    v[..., :2] = np.where(m, (v[..., :2] - [0.5, 0.45]) * z + [0.5, 0.45], v[..., :2]); r2[k] = v
                segs = run(r2)
            elif c.startswith("trim") or c.startswith("live"):
                pp = c[4:].split("_"); end = int(pp[0]); pre_, tail_ = (int(pp[1]), int(pp[2])) if len(pp) == 3 else (500, 400); base = raw
                if c.startswith("live"):
                    i, t = retime(raw, rng); base = sub(raw, i, t)
                utts, dropped = live_utts(base, end, pre_, tail_); segs = []
                for idx in utts:
                    r2 = sub(base, idx); r2["t"] = r2["t"] - r2["t"][0]; segs += run(r2)
                extra.update(n_utt=len(utts), dropped=dropped, kept_frac=sum(len(u) for u in utts) / len(base["t"]))
            elif c.startswith("stream"):
                base = raw
                if c == "stream_jit":
                    i, t = retime(raw, rng); base = sub(raw, i, t)
                words, st = ST.replay(pipe, base); segs = [(w, "accepted", 1.0) for w in words]
                extra.update(lat=float(np.mean(st["latency"])) if st["latency"] else None, flicker=st["flicker"], steps=st["steps"],
                             proc=float(np.mean(st["proc_ms"])) if st["proc_ms"] else None)
            top = [w for w, _, _ in segs]; shown = [w for w, s, _ in segs if s == "accepted"]
            res[c].append(dict(id=it["id"], n_ref=len(ref), e_top=ed(ref, top), e_shown=ed(ref, shown), n_hyp=len(top), n_shown=len(shown),
                               shown_ok=sum(1 for w in shown if w in ref), p=[p for _, _, p in segs], ref=ref, top=top, shown=shown, **extra))
        except Exception as e:
            res[c].append(dict(id=it["id"], n_ref=len(ref), e_top=len(ref), e_shown=len(ref), n_hyp=0, n_shown=0, shown_ok=0, p=[], ref=ref, top=[],
                               shown=[], err=repr(e)[:300]))
    if n % 5 == 0:
        print(n, len(items), round(time.time() - t00), flush=True)
    json.dump(res, open(out, "w"), ensure_ascii=False, default=float)
for c in conds:
    R = res[c]; N = sum(r["n_ref"] for r in R)
    print(f"{c:10s} n={len(R)} topWER={sum(r['e_top'] for r in R)/N:.3f} shownWER={sum(r['e_shown'] for r in R)/N:.3f} hyp/ref={sum(r['n_hyp'] for r in R)/N:.2f} "
          f"shown/ref={sum(r['n_shown'] for r in R)/N:.2f} shown_prec={sum(r['shown_ok'] for r in R)/max(1,sum(r['n_shown'] for r in R)):.2f} "
          f"medP={np.median([p for r in R for p in r['p']] or [0]):.3f} err={sum('err' in r for r in R)} {[r['err'] for r in R if 'err' in r][:1]}")
