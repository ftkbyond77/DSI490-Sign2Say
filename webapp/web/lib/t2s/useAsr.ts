"use client";
// Microphone → Thai / English text (reverse-speech, Whisper), from reverse_text2sign/service/web/src/main.ts.
// While the person speaks, the audio so far is sent every second: chunks closed by a pause are recognised once and stay (committed),
// the open tail is recognised quickly (tentative). On stop the final text comes back with the recogniser's own text kept apart,
// correction SUGGESTIONS (accepted with a click, never silent) and a reliability check: an impossible transcript is not signed.
import { useCallback, useEffect, useRef, useState } from "react";
import { api, type AsrResult } from "./api";

export type AsrState = "idle" | "recording" | "final";

export function useAsr(onFinal: (r: AsrResult) => void, speechDevice = "cpu") {
  const [state, setState] = useState<AsrState>("idle");
  const [seconds, setSeconds] = useState(0);
  const [committed, setCommitted] = useState("");
  const [tentative, setTentative] = useState("");
  const [progress, setProgress] = useState<{ p: number; elapsed: number; expect: number } | null>(null);
  const [error, setError] = useState("");
  const rec = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const session = useRef("");
  const inflight = useRef(false);
  const mime = useRef("audio/webm");
  const t0 = useRef(0);
  const timer = useRef(0);
  const progTimer = useRef(0);
  const langRef = useRef<"th" | "en">("th");
  const onFinalRef = useRef(onFinal);
  onFinalRef.current = onFinal;

  // Real-time progress of the final recognition. The recogniser cannot report its own progress, so the bar runs on the time this
  // recording is EXPECTED to take (CPU / Whisper-th small ≈ 1.8 s + 0.16 s per second of audio, GPU ≈ 0.45 s + 0.17 s/s): it eases
  // towards 92 % at the expected time, then creeps, and jumps to 100 % when the text arrives.
  const runProgress = (on: boolean, audioSec = 0) => {
    clearInterval(progTimer.current);
    if (!on) { setProgress((p) => (p ? { ...p, p: 100 } : null)); window.setTimeout(() => setProgress(null), 350); return; }
    const expect = speechDevice === "cuda" ? 0.45 + 0.17 * audioSec : 1.8 + 0.16 * audioSec;
    const start = performance.now();
    setProgress({ p: 0, elapsed: 0, expect });
    progTimer.current = window.setInterval(() => {
      const t = (performance.now() - start) / 1000, u = t / expect;
      setProgress({ p: u <= 1 ? 92 * (1 - Math.pow(1 - u, 2)) : 92 + 7 * (1 - Math.exp(-(u - 1) * 1.5)), elapsed: t, expect });
    }, 50);
  };

  const tick = async () => {
    const r0 = rec.current;
    if (inflight.current || !r0 || r0.state !== "recording" || !chunks.current.length) return;
    inflight.current = true;
    try {
      const r = await api.asrStream(new Blob(chunks.current, { type: mime.current }), langRef.current, session.current, false);
      session.current = r.session;
      if (rec.current?.state === "recording") { setCommitted(r.committed ?? ""); setTentative(r.tentative ?? ""); }
    } catch { /* a missed partial is harmless; the final pass decides */ } finally { inflight.current = false; }
  };

  const stop = useCallback(() => { if (rec.current?.state === "recording") rec.current.stop(); }, []);

  const start = useCallback(async (lang: "th" | "en" = "th") => {
    if (rec.current?.state === "recording") { rec.current.stop(); return; }
    setError("");
    langRef.current = lang;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, noiseSuppression: true, echoCancellation: true } });
      const r = new MediaRecorder(stream, MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? { mimeType: "audio/webm;codecs=opus" } : undefined);
      rec.current = r;
      mime.current = r.mimeType || "audio/webm";
      chunks.current = []; session.current = ""; setCommitted(""); setTentative("");
      r.ondataavailable = (e) => { if (e.data.size) { chunks.current.push(e.data); tick(); } };
      r.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        clearInterval(timer.current);
        setState("final"); setCommitted(""); setTentative("");
        runProgress(true, (performance.now() - t0.current) / 1000);
        try {
          const res = await api.asrStream(new Blob(chunks.current, { type: mime.current }), langRef.current, session.current, true);
          runProgress(false);
          setState("idle");
          onFinalRef.current(res);
        } catch (e) { runProgress(false); setState("idle"); setError("ถอดเสียงไม่สำเร็จ: " + (e as Error).message); }
      };
      r.start(1000);                                              // a chunk every second → partial recognition
      t0.current = performance.now();
      setState("recording"); setSeconds(0);
      timer.current = window.setInterval(() => {
        const sec = (performance.now() - t0.current) / 1000;
        setSeconds(sec);
        if (sec > 30) r.stop();
      }, 250);
    } catch (e) { setError("เปิดไมโครโฟนไม่ได้: " + (e as Error).message); }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => () => { clearInterval(timer.current); clearInterval(progTimer.current); if (rec.current?.state === "recording") rec.current.stop(); }, []);

  return { state, seconds, committed, tentative, progress, error, setError, start, stop };
}
