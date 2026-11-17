"""v12 prompted live capture (prompt-opt-v12 §1I, Appendix C).

  python main.py live-capture script [--name livel_a] [--seed 12]   # → artifacts/eval/live_scripts/<name>.json (the page loads it)
  python main.py live-replay <session> [--policy manual|live|both] [--model v10] [--end 1500 --pre 500 --tail 400 ...]
  python main.py live-capture list                                  # recorded sessions + their LIVE-L split

Recording: open http://localhost:3000/sign/live?script=livel_a (API with the capture endpoints running), Space = start / next prompt.
The session (continuous landmark stream + prompt marks) lands in cache/live_labeled/<session>/ — landmarks only, never video, never
uploaded anywhere, never a training sample (L-tune = calibration / live-rule tuning, L-test = evaluation only).

Scoring of one session: the SAME stream is cut two ways and both go through the same pipeline —
  manual = by the prompt marks (what start / stop would send), live = by the browser's hands-up / hands-down rule (modules/live_rule.py,
  checked frame-by-frame against the TypeScript in tests/test_live_rule.py). Live utterances are attributed to the prompt whose window
  holds their midpoint. Per prompt: top-1 words, accepted (displayed) words → WER top-1 / displayed WER per kind; idle prompts →
  accepted words per 10 s; OOV prompts → share with an accepted word.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from modules.utils import ART, CACHE, PREP, read_json, write_json

SCRIPTS = ART / "eval" / "live_scripts"
SESS = CACHE / "live_labeled"
FOCUS = ["กิน", "คุณ", "ยาย", "ไป", "อะไร", "เหงา", "น้อง", "น้า", "เที่ยว", "ข้าว", "อา", "ย่า", "กลับ", "ญาติ", "ดู", "มา", "ส่ง"]
PAIRS = [("ไป", "กลับ"), ("ยาย", "ย่า"), ("น้า", "อา"), ("กิน", "ดู"), ("คุณ", "ฉัน")]
FREE = [["ฉัน", "กิน", "ข้าว"], ["คุณ", "ชื่อ", "อะไร"], ["พรุ่งนี้", "ฉัน", "ไป", "โรงเรียน"], ["ฉัน", "ชอบ", "กิน", "ปลา"],
        ["วันนี้", "ฉัน", "เหนื่อย", "มาก"], ["คุณ", "อยู่", "ที่ไหน"], ["น้อง", "ชอบ", "ดู"], ["ยาย", "ไป", "โรงพยาบาล"],
        ["ฉัน", "ไม่", "เข้าใจ"], ["พ่อ", "ทำงาน", "กรุงเทพ"], ["เพื่อน", "มา", "บ้าน"], ["ลูก", "ไป", "โรงเรียน"],
        ["ขอบคุณ", "มาก"], ["เรา", "กลับ", "บ้าน", "ด้วยกัน"]]
IDLE = ["วางมือบนโต๊ะ / ตัก", "เกาหัว / จัดแว่น / จัดผม", "ยกมือค้างไว้นิ่ง ๆ (ไม่ทำท่า)", "ดื่มน้ำ / หยิบของ"]


def cmd_script(a):
    rng = np.random.RandomState(a.seed)
    T = read_json(PREP / "manifest" / "tiers.json")
    W = T["words"]
    sent = read_json(PREP / "params" / "tsl51_thai_sentences.json")
    prompts = []
    words = [w for w in FOCUS if w in W] * 2
    rest = [w for w in sorted(W) if w not in FOCUS and W[w]["tier"] in ("core", "extended") and not any(ch.isdigit() for ch in w)]
    by = {}
    for w in rest:                                                   # stratified: tier × signers in the bank (≤ 2 / 3–5 / ≥ 6)
        s = W[w].get("signers", 0)
        by.setdefault((W[w]["tier"], 0 if s <= 2 else 1 if s <= 5 else 2), []).append(w)
    keys = sorted(by)
    k = 0
    while len(words) < len([w for w in FOCUS if w in W]) * 2 + 26:
        g = by[keys[k % len(keys)]]
        if g:
            words.append(g.pop(rng.randint(len(g))))
        k += 1
    rng.shuffle(words)
    prompts += [dict(id=f"w{i:02d}", kind="word", prompt=w, gloss=w) for i, w in enumerate(words)]
    for i, (x, y) in enumerate(PAIRS):
        for j, (p, q) in enumerate(((x, y), (y, x))):
            prompts.append(dict(id=f"p{i}{j}", kind="pair", prompt=f"{p}  →  {q}", gloss=f"{p} {q}", note="สองท่าต่อกัน"))
    t51 = list(sent.items())
    rng.shuffle(t51)
    free = [s for s in FREE if all(w in W for w in s)]
    for i, (g, th) in enumerate(t51[:10]):
        prompts.append(dict(id=f"s{i:02d}", kind="sentence", prompt=th, gloss=g, note="TSL51"))
    for i, g in enumerate(free[:10]):
        prompts.append(dict(id=f"f{i:02d}", kind="sentence", prompt=" ".join(g), gloss=" ".join(g),
                            note="เว้นช่วงสั้น ๆ กลางประโยค" if i < 5 else "ความเร็วปกติ"))
    for i, x in enumerate(IDLE):
        prompts.append(dict(id=f"i{i}", kind="idle", prompt=x, seconds=20))
    for i in range(8):
        prompts.append(dict(id=f"o{i}", kind="oov", prompt="ทำท่าที่รู้จัก แต่ไม่อยู่ในคลังคำ แล้วพิมพ์ชื่อท่า"))
    SCRIPTS.mkdir(parents=True, exist_ok=True)
    out = dict(name=a.name, title="ชุดทดสอบโหมดสด LIVE-L", seed=a.seed, created=time.strftime("%Y-%m-%d %H:%M"), prompts=prompts,
               counts={k_: sum(p["kind"] == k_ for p in prompts) for k_ in ("word", "pair", "sentence", "idle", "oov")},
               free_sentences_dropped=[s for s in FREE if s not in free])
    write_json(SCRIPTS / f"{a.name}.json", out)
    print("script", SCRIPTS / f"{a.name}.json", out["counts"], "| free sentences not in vocabulary:", out["free_sentences_dropped"])


def load_session(name):
    d = SESS / name
    parts = sorted(d.glob("part_*.json.gz"))
    if not parts:
        raise FileNotFoundError(d)
    frames, events, marks, meta = [], [], [], {}
    for p in parts:
        b = json.load(gzip.open(p, "rt", encoding="utf-8"))
        frames += b.get("frames") or []; events += b.get("events") or []
        if b.get("marks"):
            marks = b["marks"]
        meta = dict(width=b.get("width"), height=b.get("height"), script=b.get("script"), final=bool(b.get("final")) or meta.get("final", False))
    frames.sort(key=lambda f: f["t"])
    return frames, marks, events, meta


def _words(pipe, frames, hw):
    if len(frames) < 8:
        return [], []
    raw = pipe.landmarks_from_browser(frames, hw)
    r = pipe.run_landmarks(raw, title="replay")
    live = [s for s in r["segments"] if s["status"] != "transition"]
    return [s["nearest"] for s in live], [s["nearest"] for s in live if s["status"] == "accepted"]


def cmd_replay(a):
    from modules.decode import wer
    from modules.live_rule import frames_to_arrays, live_utts
    from modules.pipeline import SignPipeline
    frames, marks, events, meta = load_session(a.session)
    if not marks:
        sys.exit("session has no prompt marks (not finished?)")
    hw = (int(meta["height"]), int(meta["width"]))
    pipe = SignPipeline(model_dir=ROOT / "models" / f"model-{a.model}", backend="onnx", device="cpu", vocab="conversation", use_llm=False, use_face=False)
    t = np.array([f["t"] for f in frames])
    out = dict(session=a.session, model=a.model, n_frames=len(frames), seconds=float((t[-1] - t[0]) / 1000), policies={})
    pol = ["manual", "live"] if a.policy == "both" else [a.policy]
    rule = dict(end_ms=a.end, pre=a.pre, tail=a.tail, wrist_grace_ms=a.wrist_grace, continue_ms=a.cont, cap_cut=a.cap_cut)
    for policy in pol:
        per = {m["id"]: dict(m, top=[], shown=[]) for m in marks}
        if policy == "manual":
            for m in marks:
                t1 = m.get("t_next", t[-1])
                fr = [f for f in frames if m["t_show"] <= f["t"] <= t1]
                per[m["id"]]["top"], per[m["id"]]["shown"] = _words(pipe, fr, hw)
        else:
            raw = frames_to_arrays(frames)
            utts, dropped = live_utts(raw, **rule)
            out.setdefault("live_rule", dict(rule, utterances=len(utts), dropped=dropped,
                                             capped=int(sum((t[u[-1]] - t[u[0]]) >= 19500 for u in utts))))
            for u in utts:
                mid = (t[u[0]] + t[u[-1]]) / 2
                m = next((m for m in marks if m["t_show"] <= mid <= m.get("t_next", t[-1])), None)
                if m is None:
                    continue
                top, shown = _words(pipe, [frames[i] for i in u], hw)
                per[m["id"]]["top"] += top; per[m["id"]]["shown"] += shown
                per[m["id"]].setdefault("n_utts", 0); per[m["id"]]["n_utts"] += 1
        agg = {}
        for kind in ("word", "pair", "sentence"):
            rows = [p for p in per.values() if p["kind"] == kind]
            if rows:
                refs = [p["gloss"].split() for p in rows]
                agg[kind] = dict(n=len(rows), WER=float(np.mean([wer(r, p["top"]) for r, p in zip(refs, rows)])),
                                 displayed_WER=float(np.mean([wer(r, p["shown"]) for r, p in zip(refs, rows)])),
                                 split_or_merged=float(np.mean([p.get("n_utts", 1) != 1 for p in rows])) if policy == "live" else None)
        idle = [p for p in per.values() if p["kind"] == "idle"]
        if idle:
            sec = sum((p.get("t_next", t[-1]) - p["t_show"]) / 1000 for p in idle)
            agg["idle"] = dict(n=len(idle), accepted_words_per_10s=10 * sum(len(p["shown"]) for p in idle) / max(sec, 1e-6))
        oov = [p for p in per.values() if p["kind"] == "oov"]
        if oov:
            agg["oov"] = dict(n=len(oov), shown_as_word=float(np.mean([len(p["shown"]) > 0 for p in oov])), typed=[p.get("typed") for p in oov])
        out["policies"][policy] = dict(summary=agg, prompts=list(per.values()))
        print(policy, json.dumps(agg, ensure_ascii=False))
    if "manual" in out["policies"] and "live" in out["policies"]:
        g = {k: out["policies"]["live"]["summary"][k]["WER"] - out["policies"]["manual"]["summary"][k]["WER"]
             for k in ("word", "pair", "sentence") if k in out["policies"]["live"]["summary"]}
        out["live_minus_manual_WER"] = g
        print("live − manual WER", g)
    write_json(ART / "reports" / f"live_replay_{a.session}_{a.model}.json", out)


def cmd_list(a):
    rows = sorted(p.name for p in SESS.glob("*") if p.is_dir()) if SESS.exists() else []
    for i, r in enumerate(rows):
        print(r, "L-tune" if i < 2 else "L-test (sealed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("script"); p.add_argument("--name", default="livel_a"); p.add_argument("--seed", type=int, default=12)
    p = sub.add_parser("replay"); p.add_argument("session"); p.add_argument("--policy", default="both"); p.add_argument("--model", default="v10")
    p.add_argument("--end", type=float, default=1500); p.add_argument("--pre", type=float, default=500); p.add_argument("--tail", type=float, default=400)
    p.add_argument("--wrist_grace", type=float, default=None); p.add_argument("--cont", type=float, default=None); p.add_argument("--cap_cut", action="store_true")
    sub.add_parser("list")
    a = ap.parse_args()
    dict(script=cmd_script, replay=cmd_replay, list=cmd_list)[a.cmd](a)
