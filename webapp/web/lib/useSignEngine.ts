"use client";
// The sign → text engine shared by every screen that has a camera (/sign, /sign/live, /chat): webcam + MediaPipe Holistic in the
// browser, manual recording or live mode (hands up = an utterance starts, hands down = it ends), video upload, the server request
// with streamed progress, and speech. The logic is the one of the original SignStudio (components/SignStudio.tsx, kept at /classic);
// only the UI moved out, so each screen draws it its own way.
import { useCallback, useEffect, useRef, useState } from "react";
import type { HolisticLandmarker } from "@mediapipe/tasks-vision";
import { openWidestCamera } from "@/lib/camera";
import { ExpressionModel } from "@/lib/expression";
import { createHolistic, detect, drawV2 as draw, EmotionMeter, getHolistic, preloadHolistic, replaceHolistic, handsUp, Normalizer, parts, type Delegate, type Emotion, type LmFrame, type Tone } from "@/lib/holistic";
import { info, speak, translateFramesStream, translateVideoOnServerStream, vocabList, type Options, type Progress, type TranslateResult } from "@/lib/api";
import { b64ToBytes, playWav, unlockAudio } from "@/lib/audio";
import { signWords } from "@/lib/mysigns";

export type Status = "idle" | "loading" | "camera" | "recording" | "recorded" | "processing";
export type Mode = "manual" | "live";
export type LiveState = "waiting" | "signing" | "processing";
export type ModelInfo = { version?: string; conversation_concepts?: number; concepts_full?: number; llm?: boolean };

const LIVE_PREROLL_MS = 500;   // frames kept from before the hands went up
const LIVE_UP_MS = 250;        // hands up this long → an utterance starts
export const LIVE_END_OPTIONS = [900, 1500, 2500];   // hands down this long → the utterance is complete (word-by-word signers need longer)
const LIVE_MAX_MS = 20000;
const MIN_SIGN_MS = 600;       // the model's shortest sign is 14 frames at 25 fps (0.56 s): shorter input cannot hold a sign
const LIVE_MIN_MS = 1200;      // live mode: a scratch / adjusting the body is shorter than a signed word with its hold
const LIVE_MIN_UP_MS = 600;    // … and the hands must have been up (signing height) this long in total
const BASELINE_FRAMES = 75;    // resting face (hands down) kept for the experimental question cue

export const STAGE_TH: Record<string, string> = {
  browser: "MediaPipe ในเบราว์เซอร์ (แยกจุด landmark)", upload: "ส่งข้อมูลให้เซิร์ฟเวอร์", received: "เซิร์ฟเวอร์รับข้อมูลแล้ว",
  mediapipe: "เซิร์ฟเวอร์แยกจุด landmark จากวิดีโอ", schema: "จัดรูปท่าทาง", tagger: "หาช่วงที่ทำท่ามือ", spotting: "จำคำภาษามือ",
  face: "อ่านสีหน้า", language: "เรียบเรียงประโยค", tts: "สร้างเสียงพูด", done: "เสร็จ",
};

const durMs = (f: LmFrame[]) => (f.length > 1 ? f[f.length - 1].t - f[0].t : 0);
const median = (rows: number[][]) => rows[0].map((_, j) => { const v = rows.map((r) => r[j]).sort((a, b) => a - b); return v[v.length >> 1]; });
export const load = <T,>(k: string, d: T): T => { try { const v = localStorage.getItem(k); return v ? (JSON.parse(v) as T) : d; } catch { return d; } };
export const save = (k: string, v: unknown) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } };

export const DEFAULT_OPTS: Options = { vocab: "conversation", llm: true, speak: "confident", faceQuestion: false, alphabet: "en" };

/** Settings shared by every screen (and remembered in this browser) — the same keys the original studio used. */
export function useSignSettings() {
  const [opts, setOpts] = useState<Options>(DEFAULT_OPTS);
  const [endMs, setEndMs] = useState(1500);
  const [autoLight, setAutoLight] = useState(false);   // pitty: off by default — whole-frame correction washed out skin, hands missed 2.2× more often (pitty/README.md)
  const loaded = useRef(false);
  useEffect(() => {
    setOpts((o) => ({ ...o, ...load<Partial<Options>>("thaislm.opts.v3", {}) }));     // v3: earlier saved settings (full vocabulary …) reset
    setEndMs(load("thaislm.endMs", 1500)); setAutoLight(load("thaislm.autoLight", false));
    loaded.current = true;
  }, []);
  useEffect(() => { if (loaded.current) save("thaislm.opts.v3", opts); }, [opts]);
  useEffect(() => { if (loaded.current) save("thaislm.endMs", endMs); }, [endMs]);
  useEffect(() => { if (loaded.current) save("thaislm.autoLight", autoLight); }, [autoLight]);
  return { opts, setOpts, endMs, setEndMs, autoLight, setAutoLight };
}
export type SignSettings = ReturnType<typeof useSignSettings>;

/** Number of words the user taught (personal sign memory) — refreshed when a word is saved. */
export function useMySignCount() {
  const [n, setN] = useState(0);
  const refresh = useCallback(() => setN(Object.keys(signWords()).length), []);
  useEffect(() => {
    refresh();
    window.addEventListener("thaislm:mysigns", refresh);
    window.addEventListener("storage", refresh);
    return () => { window.removeEventListener("thaislm:mysigns", refresh); window.removeEventListener("storage", refresh); };
  }, [refresh]);
  return n;
}
/** Tell every screen that the personal sign memory changed. */
export const signsChanged = () => window.dispatchEvent(new Event("thaislm:mysigns"));

export function useSignEngine(settings: SignSettings, onResult?: (r: TranslateResult) => void) {
  const { opts, endMs, autoLight } = settings;
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
  const onResultRef = useRef(onResult);
  onResultRef.current = onResult;

  const [status, setStatus] = useState<Status>("idle");
  const [mode, setModeState] = useState<Mode>("manual");
  const [nFrames, setNFrames] = useState(0);
  const [fps, setFps] = useState(0);
  const [delegate, setDelegate] = useState<Delegate | "">("");
  const [seen, setSeen] = useState(parts(null));
  const [exposure, setExposure] = useState("");
  const [tone, setTone] = useState<Tone>({ brightness: 1, contrast: 1 });
  const [liveState, setLiveState] = useState<LiveState>("waiting");
  const [liveNote, setLiveNote] = useState("");
  const [speaking, setSpeaking] = useState("");
  const [vocab, setVocab] = useState<string[]>([]);
  const [camSize, setCamSize] = useState<[number, number]>([4, 3]);
  const [results, setResults] = useState<TranslateResult[]>([]);
  const resultSeq = useRef(0);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState<Progress | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [apiDown, setApiDown] = useState(false);
  const optsRef = useRef(opts);
  optsRef.current = opts;
  const endRef = useRef(endMs);
  endRef.current = endMs;
  const autoLightRef = useRef(autoLight);
  autoLightRef.current = autoLight;

  const set = (s: Status) => { statusRef.current = s; setStatus(s); };

  useEffect(() => {
    preloadHolistic();                                     // MediaPipe starts loading now, not when the camera button is pressed
    const ex = exprRef.current;
    const t = window.setTimeout(() => ex.load(), 2500);     // the expression model after it (small; not competing with MediaPipe)
    info().then((m) => { setModel(m); setApiDown(false); }).catch(() => { setModel(null); setApiDown(true); });
    vocabList().then(setVocab).catch(() => setVocab([]));
    return () => window.clearTimeout(t);
  }, []);
  useEffect(() => { if (normRef.current) normRef.current.enabled = autoLight; }, [autoLight]);

  /** Speak a result according to the speech setting (the server already attached the audio when it was allowed). */
  const say = useCallback(async (r: TranslateResult, force = false) => {
    const o = optsRef.current;
    const tentative = !r.sentence;
    if (!force && (o.speak === "off" || (tentative && o.speak !== "all"))) return;
    const text = (r.sentence || r.language?.tentative_sentence_th || "").replaceAll("[?]", " ").trim();
    if (!text) return;
    if (force) unlockAudio();
    setSpeaking(text);
    try {
      const bytes = !force && r.audio_wav_b64 ? b64ToBytes(r.audio_wav_b64) : await speak(text, r.face?.emotion, r.face?.intensity);
      await playWav(bytes);
    } catch (e) { setError(`เล่นเสียงไม่ได้ (${(e as Error).message}) — กดปุ่ม “ฟังเสียง” อีกครั้ง`); }
    finally { setSpeaking(""); }
  }, []);

  const handle = useCallback(async (r: TranslateResult) => {
    r = { ...r, uid: ++resultSeq.current };        // a stable key: each card keeps its own taught words when a new result is prepended
    setResults((prev) => [r, ...prev].slice(0, 12));
    onResultRef.current?.(r);
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
    if (!frames.length) return "ไม่ได้รับจุด landmark เลย — MediaPipe ยังไม่ทำงาน";
    if (ms < MIN_SIGN_MS) return `บันทึกได้แค่ ${(ms / 1000).toFixed(1)} วินาที (${frames.length} เฟรม) — ทำท่าอย่างน้อย ${MIN_SIGN_MS / 1000} วินาที`;
    if (!withHands) return "ไม่เห็นมือในช่วงที่บันทึก — ยกมือให้อยู่ในกรอบกล้อง";
    return "";
  };

  const request = useCallback((frames: LmFrame[], w: number, h: number, onProgress: (x: Progress) => void, signal?: AbortSignal) => {
    const rest = restRef.current;
    return translateFramesStream(frames, w, h, optsRef.current, onProgress, rest.length >= 10 ? median(rest) : null, signal);
  }, []);

  /** frames → server (progress streamed back) → result + speech. */
  const send = useCallback(async (frames: LmFrame[], w: number, h: number, source: "manual" | "live") => {
    const why = check(frames, source);
    if (why) { if (source === "live") setLiveNote(why); else setError(why); return false; }
    setError(""); setLiveNote("");
    try {
      const r = await request(frames, w, h, setProgress);
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
      const l = await createHolistic("CPU");
      replaceHolistic(l);                                  // the shared one failed on this machine: every screen uses the CPU one now
      lmRef.current = l.lm; delegateRef.current = l.delegate; setDelegate(l.delegate);
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
      if (ctx) draw(ctx, f, c.width, c.height, statusRef.current === "recording" || live.current.state === "signing", lastEmoRef.current);
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
  }, [send, switchToCpu]);

  const opening = useRef<Promise<boolean> | null>(null);
  const openCamera = useCallback(async () => {
    unlockAudio();
    if (streamRef.current) return true;
    if (opening.current) return opening.current;            // already opening (e.g. two quick clicks): one camera stream only
    opening.current = openNow().finally(() => { opening.current = null; });
    return opening.current;

  async function openNow() {
    setError(""); set("loading");
    try {
      if (!lmRef.current) {
        const { lm, delegate: d, face } = await getHolistic();      // usually ready already: preloaded when the screen opened
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
      return true;
    } catch (e) { setError(`เปิดกล้องไม่ได้: ${(e as Error).message}`); set("idle"); return false; }
  }
  }, [loop]);

  const closeCamera = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop()); streamRef.current = null;
    const L = live.current; L.state = "waiting"; L.pre = []; L.utt = []; L.upSince = 0;
    setLiveState("waiting"); setSeen(parts(null)); set("idle");
  }, []);
  useEffect(() => () => closeCamera(), [closeCamera]);

  const setMode = useCallback((m: Mode) => {
    unlockAudio();
    modeRef.current = m; setModeState(m);
    if (m === "live" && statusRef.current === "recording") set("camera");
    if (m === "manual") { const L = live.current; L.state = "waiting"; L.pre = []; L.utt = []; L.upSince = 0; setLiveState("waiting"); }
  }, []);

  /** Live mode on: the camera opens (if needed) and every hands-up … hands-down becomes a sentence. */
  const startLive = useCallback(async () => { setMode("live"); return openCamera(); }, [openCamera, setMode]);
  const stopLive = useCallback(() => setMode("manual"), [setMode]);

  const startRec = () => { unlockAudio(); framesRef.current = []; setNFrames(0); setError(""); set("recording"); };
  const stopRec = () => { setNFrames(framesRef.current.length); set("recorded"); };
  const process = async () => {
    unlockAudio();
    const v = videoRef.current!;
    set("processing");
    await send(framesRef.current, v.videoWidth, v.videoHeight, "manual");
    set(streamRef.current ? "camera" : "idle");
  };

  const onFile = async (file: File, onServer = false) => {
    unlockAudio();
    setError(""); const prev = statusRef.current; set("processing");
    try {
      if (onServer) {
        const r = await translateVideoOnServerStream(file, optsRef.current, setProgress);
        setProgress({ stage: "done", p: 1 }); await handle({ ...r, title: file.name }); return;
      }
      if (!fileLmRef.current) { const { lm } = await createHolistic("CPU"); fileLmRef.current = { lm, t: 1000 }; }
      const F = fileLmRef.current;
      const FN = new Normalizer(); FN.enabled = autoLightRef.current;
      const v = document.createElement("video");
      v.muted = true; v.playsInline = true; v.src = URL.createObjectURL(file);
      await new Promise((ok, bad) => { v.onloadeddata = ok; v.onerror = () => bad(new Error("เบราว์เซอร์เปิดไฟล์วิดีโอนี้ไม่ได้ (codec ไม่รองรับ) — ลอง “ให้เซิร์ฟเวอร์แยกจุด” ในการตั้งค่า")); });
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

  return {
    videoRef, canvasRef, status, mode, setMode, liveState, liveNote, nFrames, fps, delegate, seen, exposure, tone, camSize,
    speaking, vocab, results, error, setError, progress, model, apiDown, camOn, autoLight,
    openCamera, closeCamera, startLive, stopLive, startRec, stopRec, process, onFile, say,
  };
}
export type SignEngine = ReturnType<typeof useSignEngine>;
