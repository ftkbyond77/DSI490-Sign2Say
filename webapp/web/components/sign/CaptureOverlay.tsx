"use client";
// v12 prompted live capture (prompt-opt-v12 §1I / Appendix C): shown on /sign/live?script=<name>. One prompt at a time, a 3-2-1 cue,
// Space / → = next. The whole session's landmark stream (every frame, real timestamps, the live state), the prompt marks, the results and
// the streamed words are uploaded in chunks (lib/capture.ts). Recognition runs exactly as in normal live mode.
import { useEffect, useRef, useState } from "react";
import { CaptureSession, frameTap, loadScript, resultTap, type CaptureMark, type CaptureScript } from "@/lib/capture";
import type { SignEngine } from "@/lib/useSignEngine";

const KIND_TH: Record<string, string> = { word: "คำ", pair: "คู่คำใกล้กัน", sentence: "ประโยค", idle: "ไม่ทำภาษามือ", oov: "ท่าที่ไม่อยู่ในคลังคำ" };

export default function CaptureOverlay({ name, e }: { name: string; e: SignEngine }) {
  const [script, setScript] = useState<CaptureScript | null>(null);
  const [err, setErr] = useState("");
  const [i, setI] = useState(-1);                  // -1 = not started
  const [cue, setCue] = useState(0);
  const [left, setLeft] = useState(0);
  const [typed, setTyped] = useState("");
  const sess = useRef<CaptureSession | null>(null);
  const done = i >= (script?.prompts.length ?? 0);

  useEffect(() => { loadScript(name).then(setScript).catch((x) => setErr(String(x))); }, [name]);
  useEffect(() => () => { frameTap.fn = null; resultTap.fn = null; }, []);
  useEffect(() => { if (sess.current && e.liveWords) sess.current.event("stream", e.liveWords); }, [e.liveWords]);

  const mark = (k: number) => {
    const s = sess.current; if (!s || !script) return;
    const now = performance.now();
    const prev = s.marks[s.marks.length - 1];
    if (prev && prev.t_next === undefined) { prev.t_next = now; if (typed) prev.typed = typed; }
    if (k < script.prompts.length) {
      const p = script.prompts[k];
      s.marks.push({ i: k, id: p.id, kind: p.kind, prompt: p.prompt, gloss: p.gloss, t_show: now } as CaptureMark);
      setCue(3);
      if (p.kind === "idle" && p.seconds) setLeft(p.seconds);
    }
    setTyped("");
  };

  const start = () => {
    const v = e.videoRef.current;
    if (!script || !v) return;
    const s = new CaptureSession(script.name, v.videoWidth, v.videoHeight);
    sess.current = s;
    frameTap.fn = (f, st) => s.frame(f, st);
    resultTap.fn = (kind, payload) => {
      const r = payload as { segments?: { nearest?: string; status?: string; f0?: number; f1?: number; p_correct?: number }[]; sentence_th?: string };
      s.event(kind, { sentence: r.sentence_th, segments: (r.segments || []).map((x) => ({ w: x.nearest, s: x.status, p: x.p_correct, f0: x.f0, f1: x.f1 })) });
    };
    setI(0); mark(0);
  };

  const next = async () => {
    if (!script || i < 0 || done) return;
    const k = i + 1;
    mark(k); setI(k);
    if (k >= script.prompts.length && sess.current) {
      await sess.current.flush(true); sess.current.closed = true; frameTap.fn = null; resultTap.fn = null;
    }
  };

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if ((ev.target as HTMLElement)?.tagName === "INPUT") { if (ev.key === "Enter") { ev.preventDefault(); void next(); } return; }
      if (ev.key === " " || ev.key === "ArrowRight") { ev.preventDefault(); if (i < 0) start(); else void next(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });
  useEffect(() => { if (cue > 0) { const id = setTimeout(() => setCue(cue - 1), 700); return () => clearTimeout(id); } }, [cue]);
  useEffect(() => {
    if (left > 0 && cue === 0) { const id = setTimeout(() => { if (left <= 1) void next(); setLeft(left - 1); }, 1000); return () => clearTimeout(id); }
  }, [left, cue]);   // eslint-disable-line react-hooks/exhaustive-deps

  const p = script && i >= 0 && !done ? script.prompts[i] : null;
  return (
    <div className="capture glass" style={{ position: "absolute", top: 72, left: "50%", transform: "translateX(-50%)", zIndex: 30, padding: "12px 18px",
      borderRadius: 14, minWidth: 320, maxWidth: "min(92vw, 640px)", textAlign: "center" }} role="dialog" aria-label="บันทึกชุดทดสอบโหมดสด">
      {err && <p className="err">{err}</p>}
      {!script && !err && <small>กำลังโหลดสคริปต์ {name}…</small>}
      {script && i < 0 && (
        <>
          <b>{script.title || script.name}</b> · {script.prompts.length} ข้อ
          <p style={{ margin: "6px 0" }}>นั่งแบบใช้งานปกติ · ทำท่าตามข้อความครั้งเดียว แล้ววางมือลง จากนั้นกด Space</p>
          <button className="btn coral sm" onClick={start} disabled={!e.camOn}>{e.camOn ? "เริ่มบันทึก (Space)" : "เปิดกล้องก่อน"}</button>
        </>
      )}
      {p && (
        <>
          <small>{i + 1} / {script!.prompts.length} · {KIND_TH[p.kind] || p.kind}{p.note ? ` · ${p.note}` : ""}</small>
          <div style={{ fontSize: 34, fontWeight: 700, margin: "6px 0", opacity: cue > 0 ? 0.35 : 1 }}>{p.prompt}</div>
          {cue > 0 ? <div style={{ fontSize: 22 }}>{cue}</div>
            : p.kind === "idle" ? <small>อยู่ในท่านี้อีก {left} วินาที</small>
            : p.kind === "oov" ? <input value={typed} onChange={(ev) => setTyped(ev.target.value)} placeholder="พิมพ์ชื่อท่าที่ทำ แล้วกด Enter" aria-label="ชื่อท่า" />
            : <small>ทำท่า แล้ววางมือลง · Space = ข้อต่อไป</small>}
        </>
      )}
      {script && done && <p>บันทึกเสร็จแล้ว ({sess.current?.nFrames ?? 0} เฟรม) · session <code>{sess.current?.session}</code></p>}
    </div>
  );
}
