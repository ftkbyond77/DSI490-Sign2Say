"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { openWidestCamera } from "@/lib/camera";
import { ExpressionModel } from "@/lib/expression";
import type { HolisticLandmarker } from "@mediapipe/tasks-vision";
import { createHolistic, detect, draw, EmotionMeter, handsUp, Normalizer, parts, type Delegate, type Emotion, type LmFrame, type Tone } from "@/lib/holistic";
import { info, speak, translateFramesStream, vocabList, translateVideoOnServerStream, type Alphabet, type Options, type Progress, type Speak, type TranslateResult } from "@/lib/api";
import { b64ToBytes, playWav, unlockAudio } from "@/lib/audio";
import { clearSigns, signWords } from "@/lib/mysigns";
import ResultCard from "./ResultCard";

type Status = "idle" | "loading" | "camera" | "recording" | "recorded" | "processing";
type Mode = "manual" | "live";

const LIVE_PREROLL_MS = 500;   // frames kept from before the hands went up
const LIVE_UP_MS = 250;        // hands up this long → an utterance starts
const LIVE_END_OPTIONS = [900, 1500, 2500];   // hands down this long → the utterance is complete (word-by-word signers need longer)
const LIVE_MAX_MS = 20000;
const MIN_SIGN_MS = 600;       // the model's shortest sign is 14 frames at 25 fps (0.56 s): shorter input cannot hold a sign
const LIVE_MIN_MS = 1200;      // live mode: a scratch / adjusting the body is shorter than a signed word with its hold
const LIVE_MIN_UP_MS = 600;    // … and the hands must have been up (signing height) this long in total
const BASELINE_FRAMES = 75;    // resting face (hands down) kept for the experimental question cue

const STAGE_TH: Record<string, string> = {
  browser: "MediaPipe ในเบราว์เซอร์ (แยกจุด landmark)", upload: "ส่งข้อมูลให้เซิร์ฟเวอร์", received: "เซิร์ฟเวอร์รับข้อมูลแล้ว",
  mediapipe: "เซิร์ฟเวอร์แยกจุด landmark จากวิดีโอ", schema: "จัดรูปท่าทาง", tagger: "หาช่วงที่ทำท่ามือ", spotting: "จำคำภาษามือ",
  face: "อ่านสีหน้า", language: "เรียบเรียงประโยค", tts: "สร้างเสียงพูด", done: "เสร็จ",
};
const ALPHA_TH: Partial<Record<Alphabet, string>> = { en: "อังกฤษ A–Z", th: "ไทย ก–ฮ + สระ", off: "ปิด" };
const SPEAK_TH: Record<Speak, string> = { off: "ไม่ออกเสียง", confident: "ออกเสียงเฉพาะที่มั่นใจ", all: "ออกเสียงทุกผล (รวมที่ไม่แน่ใจ)" };

const durMs = (f: LmFrame[]) => (f.length > 1 ? f[f.length - 1].t - f[0].t : 0);
const median = (rows: number[][]) => rows[0].map((_, j) => { const v = rows.map((r) => r[j]).sort((a, b) => a - b); return v[v.length >> 1]; });
const load = <T,>(k: string, d: T): T => { try { const v = localStorage.getItem(k); return v ? (JSON.parse(v) as T) : d; } catch { return d; } };
const save = (k: string, v: unknown) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } };


export default function SignStudio() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lmRef = useRef<HolisticLandmarker | null>(null);
  const delegateRef = useRef<Delegate>("GPU");
  const faceLmRef = useRef<HolisticLandmarker | null>(null);      // GPU+CPU mode: expressions on the CPU every few frames
  const nRef = useRef(0);
  const fileLmRef = useRef<{ lm: HolisticLandmarker; t: number } | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number>(0);
  const framesRef = useRef<LmFrame[]>([]);
  const statusRef = useRef<Status>("idle");
  const modeRef = useRef<Mode>("manual");
  const live = useRef({ state: "waiting" as "waiting" | "signing", pre: [] as LmFrame[], utt: [] as LmFrame[], upSince: 0, lastUp: 0, start: 0 });
  const fpsRef = useRef({ n: 0, t: 0 });
  const errRef = useRef({ n: 0, switching: false });
  const restRef = useRef<number[][]>([]);
  const partsRef = useRef({ t: 0 });
  const normRef = useRef<Normalizer | null>(null);
  const emoRef = useRef(new EmotionMeter());
  const exprRef = useRef(new ExpressionModel());
  const lastEmoRef = useRef<Emotion | undefined>(undefined);

  const [status, setStatus] = useState<Status>("idle");
  const [mode, setMode] = useState<Mode>("manual");
  const [opts, setOpts] = useState<Options>({ vocab: "conversation", llm: true, speak: "confident", faceQuestion: false, alphabet: "en" });
  const [nFrames, setNFrames] = useState(0);
  const [fps, setFps] = useState(0);
  const [delegate, setDelegate] = useState<Delegate | "">("");
  const [seen, setSeen] = useState(parts(null));
  const [exposure, setExposure] = useState("");
  const [tone, setTone] = useState<Tone>({ brightness: 1, contrast: 1 });
  const [autoLight, setAutoLight] = useState(true);
  const [endMs, setEndMs] = useState(1500);
  const endRef = useRef(1500);
  endRef.current = endMs;
  const autoLightRef = useRef(true);
  autoLightRef.current = autoLight;
  const [liveState, setLiveState] = useState("waiting");
  const [liveNote, setLiveNote] = useState("");
  const [speaking, setSpeaking] = useState("");
  const [mySigns, setMySigns] = useState(0);
  const [vocab, setVocab] = useState<string[]>([]);
  const [camSize, setCamSize] = useState<[number, number]>([16, 9]);
  const refreshSigns = () => setMySigns(Object.keys(signWords()).length);
  const [results, setResults] = useState<TranslateResult[]>([]);
  const resultSeq = useRef(0);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState<Progress | null>(null);
  const [model, setModel] = useState<{ version?: string; conversation_concepts?: number; concepts_full?: number; llm?: boolean } | null>(null);
  const optsRef = useRef(opts);
  optsRef.current = opts;

  const set = (s: Status) => { statusRef.current = s; setStatus(s); };

  useEffect(() => {
    info().then(setModel).catch(() => setModel(null));
    vocabList().then(setVocab).catch(() => setVocab([]));
    setOpts((o) => ({ ...o, ...load<Partial<Options>>("thaislm.opts.v3", {}) }));     // v3: earlier saved settings (full vocabulary …) reset
    setEndMs(load("thaislm.endMs", 1500)); setAutoLight(load("thaislm.autoLight", true)); refreshSigns();
  }, []);
  useEffect(() => { save("thaislm.opts.v3", opts); }, [opts]);
  useEffect(() => { save("thaislm.endMs", endMs); }, [endMs]);
  useEffect(() => { save("thaislm.autoLight", autoLight); if (normRef.current) normRef.current.enabled = autoLight; }, [autoLight]);
  useEffect(() => { modeRef.current = mode; }, [mode]);

  /** Speak a result according to the speech setting (the server already attached the audio when it was allowed). */
  const say = useCallback(async (r: TranslateResult) => {
    const o = optsRef.current;
    if (o.speak === "off") return;
    const tentative = !r.sentence;
    if (tentative && o.speak !== "all") return;
    const text = (r.sentence || r.language?.tentative_sentence_th || "").replaceAll("[?]", " ").trim();
    if (!text) return;
    setSpeaking(tentative ? `🔊 (ไม่แน่ใจ) ${text}` : `🔊 ${text}`);
    try {
      const bytes = r.audio_wav_b64 ? b64ToBytes(r.audio_wav_b64) : await speak(text, r.face?.emotion, r.face?.intensity);
      await playWav(bytes);
    } catch (e) { setError(`เล่นเสียงไม่ได้ (${(e as Error).message}) — กดปุ่ม 🔊 บนการ์ดผลลัพธ์`); }
    finally { setSpeaking(""); }
  }, []);

  const handle = useCallback(async (r: TranslateResult) => {
    r = { ...r, uid: ++resultSeq.current };        // a stable key: each card keeps its own taught words when a new result is prepended
    setResults((prev) => [r, ...prev].slice(0, 12));
    await say(r);
  }, [say]);

  /** Input checks shared by manual / live; returns an error text, or "" when the frames can hold a sign. */
  const check = (frames: LmFrame[], source: "manual" | "live") => {
    const ms = durMs(frames);
    const withHands = frames.filter((f) => f.lh.length || f.rh.length).length;
    if (source === "live") {
      let upMs = 0;
      for (let i = 1; i < frames.length; i++) if (handsUp(frames[i])) upMs += frames[i].t - frames[i - 1].t;
      return ms < LIVE_MIN_MS || upMs < LIVE_MIN_UP_MS || withHands < 3 ? `ข้ามการขยับสั้น ๆ (${(ms / 1000).toFixed(1)} วิ) — ไม่ใช่ท่าภาษามือ` : "";
    }
    if (!frames.length) return "ไม่ได้รับจุด landmark เลย — MediaPipe ยังไม่ทำงาน (ดูแถบ “ตรวจพบ” บนภาพกล้อง)";
    if (ms < MIN_SIGN_MS) return `บันทึกได้แค่ ${(ms / 1000).toFixed(1)} วินาที (${frames.length} เฟรม) — ทำท่าอย่างน้อย ${MIN_SIGN_MS / 1000} วินาที`;
    if (!withHands) return "ไม่เห็นมือในช่วงที่บันทึก — ยกมือให้อยู่ในกรอบกล้อง";
    return "";
  };

  const request = useCallback((frames: LmFrame[], w: number, h: number, onProgress: (x: Progress) => void, signal?: AbortSignal) => {
    const rest = restRef.current;
    return translateFramesStream(frames, w, h, optsRef.current, onProgress, rest.length >= 10 ? median(rest) : null, signal);
  }, []);

  /** frames → server (progress streamed back) → card + speech. */
  const send = useCallback(async (frames: LmFrame[], w: number, h: number, source: "manual" | "live", ready?: Promise<TranslateResult | null>) => {
    const why = check(frames, source);
    if (why) { if (source === "live") setLiveNote(why); else setError(why); return false; }
    setError(""); setLiveNote("");
    try {
      let r = ready ? await ready : null;                          // live mode: usually already computed during the pause
      if (!r) r = await request(frames, w, h, setProgress);
      setProgress({ stage: "done", p: 1 });
      await handle(r);                         // unsure signs are shown as [?] (no guessed word) so the user can teach them
    } catch (e) { setError(`ประมวลผลไม่สำเร็จ: ${(e as Error).message}`); }
    finally { setTimeout(() => setProgress(null), 600); }
    return true;
  }, [handle, request]);

  const switchToCpu = useCallback(async () => {
    if (errRef.current.switching) return;
    errRef.current.switching = true;
    try {
      lmRef.current?.close(); lmRef.current = null;
      faceLmRef.current?.close(); faceLmRef.current = null;
      const { lm, delegate: d } = await createHolistic("CPU");
      lmRef.current = lm; delegateRef.current = d; setDelegate(d);
    } catch (e) { setError(`MediaPipe ใช้งานไม่ได้: ${(e as Error).message}`); }
    finally { errRef.current = { n: 0, switching: false }; }
  }, []);

  const loop = useCallback(() => {
    try {
      const v = videoRef.current, c = canvasRef.current, lm = lmRef.current;
      if (!(v && c && lm && v.readyState >= 2)) return;
      const t = performance.now();
      let f: LmFrame | null = null;
      if (!normRef.current) { normRef.current = new Normalizer(); normRef.current.enabled = autoLightRef.current; }
      const N = normRef.current;
      let src: HTMLVideoElement | HTMLCanvasElement = v;
      try {
        src = N.frame(v, t);
        f = detect(lm, src, t); errRef.current.n = 0;
        const fl = faceLmRef.current;
        if (f && fl && f.face.length && nRef.current++ % 8 === 0) {        // expressions (blendshapes) every 8th frame on the CPU
          const g = detect(fl, src, t);
          if (g) f.bs = g.bs;
        }
      } catch {                                    // a frame failed (e.g. the GPU path) — never stop the loop; move to the CPU after a few
        if (++errRef.current.n >= 3 && delegateRef.current !== "CPU") switchToCpu();
        return;
      }
      const ctx = c.getContext("2d");
      // live expression: the trained expression model on the face crop; blendshape rules only if it could not load. Held between updates.
      if (f && f.face.length) {
        const m = exprRef.current.failed ? null : exprRef.current.update(src, c.width, c.height, f.face, t);
        if (m) lastEmoRef.current = m;
        else if (f.bs.length >= 52) lastEmoRef.current = emoRef.current.update(f.bs);
      } else if (f) { lastEmoRef.current = undefined; exprRef.current.reset(); }
      const emo = lastEmoRef.current;
      if (ctx) draw(ctx, f, c.width, c.height, statusRef.current === "recording" || live.current.state === "signing", emo);
      if (t - partsRef.current.t > 250) {
        partsRef.current.t = t; setSeen(parts(f));
        const e = N.exp;
        setExposure(N.enabled && e.active ? `${e.label} (ความสว่างเดิม ${Math.round(e.mean * 100)}%)` : "");
        setTone(N.enabled ? N.tone : { brightness: 1, contrast: 1 });
      }
      if (!f) return;
      const fr = fpsRef.current; fr.n++;
      if (t - fr.t > 1000) { setFps(Math.round((fr.n * 1000) / (t - fr.t))); fr.n = 0; fr.t = t; }
      const up = handsUp(f);
      if (!up && f.bs.length >= 52 && statusRef.current !== "recording") {          // resting face, for the question cue
        restRef.current.push(f.bs); if (restRef.current.length > BASELINE_FRAMES) restRef.current.shift();
      }
      if (statusRef.current === "recording") { framesRef.current.push(f); if (framesRef.current.length % 5 === 0) setNFrames(framesRef.current.length); }
      if (modeRef.current === "live" && (statusRef.current === "camera" || statusRef.current === "processing")) {
        const L = live.current;
        if (L.state === "waiting") {
          L.pre.push(f); while (L.pre.length && t - L.pre[0].t > LIVE_PREROLL_MS) L.pre.shift();
          if (up) { if (!L.upSince) L.upSince = t; } else L.upSince = 0;
          if (L.upSince && t - L.upSince >= LIVE_UP_MS) { L.state = "signing"; L.utt = [...L.pre]; L.start = t; L.lastUp = t; setLiveState("signing"); setLiveNote(""); }
        } else {
          L.utt.push(f);
          if (up) L.lastUp = t;
          if (t - L.lastUp >= endRef.current || t - L.start >= LIVE_MAX_MS) {
            const frames = L.utt.filter((x) => x.t <= L.lastUp + 400);
            L.state = "waiting"; L.pre = []; L.utt = []; L.upSince = 0;
            setLiveState("processing");
            send(frames, v.videoWidth, v.videoHeight, "live").finally(() => setLiveState(live.current.state));
          }
        }
      }
    } finally {
      rafRef.current = requestAnimationFrame(loop);
    }
  }, [send, switchToCpu, request]);  // eslint-disable-line react-hooks/exhaustive-deps

  const openCamera = async () => {
    unlockAudio();
    setError(""); set("loading");
    try {
      if (!lmRef.current) {
        const { lm, delegate: d, face } = await createHolistic("GPU");
        lmRef.current = lm; faceLmRef.current = face ?? null; delegateRef.current = d; setDelegate(d);
        exprRef.current.load();                               // in the background: the label falls back to blendshapes until it is ready
      }
      const s = await openWidestCamera();
      streamRef.current = s;
      const v = videoRef.current!;
      v.srcObject = s; await v.play();
      const c = canvasRef.current!; c.width = v.videoWidth; c.height = v.videoHeight;
      setCamSize([v.videoWidth, v.videoHeight]);          // the stage takes the camera's own shape (4:3 sensors show their full height)
      set("camera");
      cancelAnimationFrame(rafRef.current); rafRef.current = requestAnimationFrame(loop);
    } catch (e) { setError(`เปิดกล้องไม่ได้: ${(e as Error).message}`); set("idle"); }
  };

  const closeCamera = () => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop()); streamRef.current = null;
    setSeen(parts(null)); set("idle");
  };
  useEffect(() => () => closeCamera(), []);  // eslint-disable-line react-hooks/exhaustive-deps

  const start = () => { unlockAudio(); framesRef.current = []; setNFrames(0); setError(""); set("recording"); };
  const stop = () => { setNFrames(framesRef.current.length); set("recorded"); };
  const process = async () => {
    unlockAudio();
    const v = videoRef.current!;
    set("processing");
    await send(framesRef.current, v.videoWidth, v.videoHeight, "manual");
    set(streamRef.current ? "camera" : "idle");
  };

  const onFile = async (file: File, onServer: boolean) => {
    unlockAudio();
    setError(""); const prev = statusRef.current; set("processing");
    try {
      if (onServer) {
        const r = await translateVideoOnServerStream(file, optsRef.current, setProgress);
        setProgress({ stage: "done", p: 1 }); await handle({ ...r, title: file.name }); return;
      }
      if (!fileLmRef.current) { const { lm } = await createHolistic("CPU"); fileLmRef.current = { lm, t: 1000 }; }
      const F = fileLmRef.current;
      const FN = new Normalizer(); FN.enabled = autoLight;
      const v = document.createElement("video");
      v.muted = true; v.playsInline = true; v.src = URL.createObjectURL(file);
      await new Promise((ok, bad) => { v.onloadeddata = ok; v.onerror = () => bad(new Error("เบราว์เซอร์เปิดไฟล์วิดีโอนี้ไม่ได้ (codec ไม่รองรับ) — ลองปุ่ม “ให้เซิร์ฟเวอร์แยกจุด”")); });
      const frames: LmFrame[] = [];
      const base = F.t;                                    // VIDEO-mode timestamps must keep increasing across files
      for (let t = 0; t < v.duration; t += 1 / 25) {
        v.currentTime = t;
        await new Promise((ok) => { v.onseeked = ok; });
        const f = detect(F.lm, FN.frame(v, t * 1000), base + t * 1000);
        if (f) frames.push({ ...f, t: t * 1000 });
        if (frames.length % 5 === 0) setProgress({ stage: "browser", p: 0.6 * (t / v.duration) });
      }
      F.t = base + v.duration * 1000 + 1000;
      const r = await request(frames, v.videoWidth, v.videoHeight, (x) => setProgress({ stage: x.stage, p: 0.6 + 0.4 * x.p }));
      setProgress({ stage: "done", p: 1 });
      await handle({ ...r, title: file.name });
      URL.revokeObjectURL(v.src);
    } catch (e) { setError(`ประมวลผลวิดีโอไม่สำเร็จ: ${(e as Error).message}`); }
    finally { setTimeout(() => setProgress(null), 600); set(prev === "processing" ? "idle" : prev === "recording" ? "camera" : prev); }
  };

  const camOn = status === "camera" || status === "recording" || status === "recorded" || (status === "processing" && !!streamRef.current);
  const chip = (ok: boolean, label: string) => <span className={`part ${ok ? "on" : ""}`}>{ok ? "✓" : "✗"} {label}</span>;

  return (
    <div className="studio">
      <section className="stage">
        <div className={`viewport ${status === "recording" || liveState === "signing" ? "rec" : ""}`} style={{ aspectRatio: `${camSize[0]} / ${camSize[1]}` }}>
          <video ref={videoRef} playsInline muted className="mirror"
            style={autoLight && (tone.brightness !== 1 || tone.contrast !== 1) ? { filter: `brightness(${tone.brightness.toFixed(2)}) contrast(${tone.contrast.toFixed(2)})` } : undefined} />
          <canvas ref={canvasRef} className="mirror overlay" />
          {!camOn && <div className="placeholder">กด “เปิดกล้อง” เพื่อเริ่ม<br /><small>MediaPipe Holistic ทำงานในเบราว์เซอร์ของคุณ — ส่งเฉพาะจุด landmark ไปที่เซิร์ฟเวอร์</small></div>}
          {camOn && <div className="hud">{delegate && `${delegate} · `}{fps} fps · {mode === "live" ? (liveState === "signing" ? "● กำลังทำท่า" : liveState === "processing" ? "… กำลังแปล" : "รอยกมือ") : status === "recording" ? `● REC ${nFrames} เฟรม` : status === "recorded" ? `${nFrames} เฟรม พร้อมประมวลผล` : "พร้อม"}</div>}
          {camOn && exposure && <div className="exposure">ปรับภาพอัตโนมัติ: {exposure}</div>}
          {camOn && <div className="parts">ตรวจพบ: {chip(seen.body, "ลำตัว")}{chip(seen.face, "หน้า")}{chip(seen.brows, "คิ้ว/สีหน้า")}{chip(seen.left, "มือซ้าย")}{chip(seen.right, "มือขวา")}</div>}
        </div>
        {camOn && fps > 0 && fps < 8 && <p className="note">เครื่องประมวลผลได้ {fps} fps (ช้า) — ทำท่าช้าลงเล็กน้อย และให้มืออยู่ในกรอบชัด ๆ</p>}
        {liveNote && mode === "live" && <p className="note">{liveNote}</p>}
        <div className="controls">
          {!camOn ? <button className="primary" onClick={openCamera} disabled={status === "loading"}>{status === "loading" ? "กำลังโหลดโมเดล…" : "เปิดกล้อง"}</button>
            : <button onClick={closeCamera}>ปิดกล้อง</button>}
          <div className="seg">
            <button className={mode === "manual" ? "on" : ""} onClick={() => { unlockAudio(); setMode("manual"); }}>บันทึกเอง</button>
            <button className={mode === "live" ? "on" : ""} onClick={() => { unlockAudio(); setMode("live"); if (statusRef.current === "recording") set("camera"); }}>โหมดสด (ลดมือ = จบประโยค)</button>
          </div>
          {mode === "live" && <label className="small">จบประโยคเมื่อลดมือนาน <select value={endMs} onChange={(e) => setEndMs(Number(e.target.value))}>
            {LIVE_END_OPTIONS.map((ms) => <option key={ms} value={ms}>{(ms / 1000).toFixed(1)} วินาที</option>)}</select></label>}
          {mode === "manual" && camOn && <>
            <button className="rec" onClick={start} disabled={status === "recording" || status === "processing"}>เริ่ม</button>
            <button onClick={stop} disabled={status !== "recording"}>หยุด</button>
            <button className="primary" onClick={process} disabled={status !== "recorded"}>ประมวลผล</button>
          </>}
        </div>
        <div className="controls small">
          <label>เสียงพูด <select value={opts.speak} onChange={(e) => { unlockAudio(); setOpts({ ...opts, speak: e.target.value as Speak }); }}>
            {(Object.keys(SPEAK_TH) as Speak[]).map((s) => <option key={s} value={s}>{SPEAK_TH[s]}</option>)}</select></label>
          <label title="ตัวอักษรที่สะกดด้วยนิ้ว (เช่น ชื่อ J A C K) — ทำท่าแต่ละตัวค้างไว้ครู่หนึ่ง">สะกดนิ้ว <select value={opts.alphabet} onChange={(e) => setOpts({ ...opts, alphabet: e.target.value as Alphabet })}>
            {(Object.keys(ALPHA_TH) as Alphabet[]).map((a) => <option key={a} value={a}>{ALPHA_TH[a]}</option>)}</select></label>
          <label>คลังคำ <select value={opts.vocab} onChange={(e) => setOpts({ ...opts, vocab: e.target.value as Options["vocab"] })}>
            <option value="conversation">บทสนทนาประจำวัน{model?.conversation_concepts ? ` (${model.conversation_concepts} คำ)` : ""}</option>
            <option value="full">ทั้งหมด{model?.concepts_full ? ` (${model.concepts_full} คำ)` : ""} — ช้ากว่าและคลาดเคลื่อนง่าย</option></select></label>
          <label><input type="checkbox" checked={opts.llm} onChange={(e) => setOpts({ ...opts, llm: e.target.checked })} /> เรียบเรียงด้วย LLM{model && !model.llm ? " (ไม่มีคีย์ → กฎ)" : ""}</label>
          <label title="วัดแสง/คอนทราสต์/สมดุลสี/ความคมของภาพทุก 0.3 วินาที แล้วปรับทั้งภาพที่เห็นและภาพที่ส่งให้ MediaPipe">
            <input type="checkbox" checked={autoLight} onChange={(e) => setAutoLight(e.target.checked)} /> ปรับภาพอัตโนมัติ</label>
          <label title="วัดเทียบกับหน้าปกติของคุณตอนลดมือ — ยังไม่ผ่านการทดสอบกับคนอื่น จึงปิดไว้เป็นค่าเริ่มต้น">
            <input type="checkbox" checked={opts.faceQuestion} onChange={(e) => setOpts({ ...opts, faceQuestion: e.target.checked })} /> ทดลอง: ขมวดคิ้วท้ายประโยค = คำถาม (ไหม)</label>
          <span className="mysigns" title="กดที่คำบนการ์ดผลลัพธ์เพื่อยืนยัน/แก้คำ — ระบบจะจำท่าของคุณ (เก็บในเบราว์เซอร์นี้เท่านั้น)">
            ท่าของฉัน: {mySigns} คำ {mySigns > 0 && <button className="ghost" onClick={() => { clearSigns(); refreshSigns(); }}>ล้าง</button>}</span>
          <label className="file">อัปโหลดวิดีโอ
            <input type="file" accept="video/*" onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f, false); e.target.value = ""; }} /></label>
          <label className="file">ให้เซิร์ฟเวอร์แยกจุด
            <input type="file" accept="video/*" onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f, true); e.target.value = ""; }} /></label>
        </div>
        {progress && (
          <div className="pbar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress.p * 100)}>
            <div className="pbar-track"><div className="pbar-fill" style={{ width: `${Math.max(3, Math.round(progress.p * 100))}%` }} /></div>
            <span className="pbar-label">{STAGE_TH[progress.stage] ?? progress.stage} · {Math.round(progress.p * 100)}%</span>
          </div>
        )}
        {speaking && <p className="speaking">{speaking}</p>}
        {error && <p className="error">{error}</p>}
      </section>
      <section className="results">
        {results.length === 0 ? <div className="empty">ผลการแปลจะแสดงที่นี่<br /><small>ทำภาษามือเป็นประโยค เช่น “ฉัน รัก เพื่อน” แล้วกดประมวลผล หรือใช้โหมดสด</small></div>
          : results.map((r, i) => <ResultCard key={r.uid ?? i} r={r} latest={i === 0} onSaved={refreshSigns} vocab={vocab} />)}
        {model && <p className="meta">model {model.version} · MediaPipe Holistic 1.0.1</p>}
        <datalist id="vocab-list">{vocab.map((w) => <option key={w} value={w} />)}</datalist>
      </section>
    </div>
  );
}
