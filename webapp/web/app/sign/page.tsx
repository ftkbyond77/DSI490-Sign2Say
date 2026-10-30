"use client";
// v2-C — แปลภาษามือ: camera on the left (live mode / manual recording / upload), the latest result in teal, the previous one, and the
// “teach your sign” prompt on the right.
import Link from "next/link";
import { useEffect, useState } from "react";
import AppShell, { greeting } from "@/components/AppShell";
import SettingsSheet, { QuickOptions } from "@/components/SettingsSheet";
import { CamToggle, CameraFeed, Dot, Seen, statusOf } from "@/components/sign/CameraView";
import Words, { cues, shownText, useSigns } from "@/components/sign/Words";
import { Ic, ProgressBar, Sentence } from "@/components/ui";
import type { TranslateResult } from "@/lib/api";
import { useSignEngine, useSignSettings, type SignEngine } from "@/lib/useSignEngine";

export default function SignPage() {
  const settings = useSignSettings();
  const e = useSignEngine(settings);
  const [hello, setHello] = useState("สวัสดี");
  const [showSettings, setShowSettings] = useState(false);
  useEffect(() => setHello(greeting()), []);
  useEffect(() => { e.setMode("live"); }, []);   // eslint-disable-line react-hooks/exhaustive-deps -- live mode is the default here
  const st = statusOf(e);
  const [latest, prev, ...older] = e.results;
  const unsure = latest ? countUnsure(latest) : 0;

  return (
    <AppShell>
      <div className="title-row rise">
        <div><div className="hello">{hello}</div><h1>แปลภาษามือ</h1></div>
        <div className="options">
          <QuickOptions s={settings} model={e.model} />
          <button className="icon-btn" onClick={() => setShowSettings(true)} aria-label="ตั้งค่า" title="ตั้งค่า"><Ic n="sliders-teal" /></button>
        </div>
      </div>

      <div className="workspace">
        <section className="cam-col rise d1">
          <div className={`camera ${e.mode === "live" && e.liveState === "signing" ? "signing" : ""} ${e.status === "recording" ? "rec" : ""}`}>
            <CameraFeed engine={e} onOpen={() => e.openCamera()} label={e.mode === "live" ? "เปิดกล้อง · เริ่มโหมดสด" : "เปิดกล้อง · บันทึกเอง"} />
            <div className="tl"><span className="pill"><Dot tone={st.tone} />{st.text}{st.sub && ` · ${st.sub}`}</span></div>
            <div className="tr">
              <CamToggle e={e} onOpen={() => e.openCamera()} />
              <Link href="/sign/live" className="glass-btn" aria-label="กล้องเต็มจอ" title="กล้องเต็มจอ (โหมดสด)">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ccfbf1" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" /></svg>
              </Link>
            </div>
            {e.camOn && <div className="bl"><Seen e={e} /></div>}
          </div>
          <Actions e={e} />
          <ProgressBar progress={e.progress} />
          {e.liveNote && e.mode === "live" && <p className="note">{e.liveNote}</p>}
          {e.camOn && e.fps > 0 && e.fps < 8 && <p className="note">เครื่องประมวลผลได้ {e.fps} fps (ช้า) — ทำท่าช้าลงเล็กน้อย และให้มืออยู่ในกรอบชัด ๆ</p>}
          {e.exposure && <p className="note">ปรับภาพอัตโนมัติ: {e.exposure}</p>}
          {e.error && <p className="err">{e.error}</p>}
          {e.apiDown && <p className="err">เชื่อมต่อเซิร์ฟเวอร์แปลภาษามือไม่ได้ — ตรวจว่า sign-api ทำงานอยู่</p>}
          <div className="cam-meta">
            <span>{e.camOn ? `${e.delegate} · ${e.fps} fps` : "ภาพประมวลผลในเบราว์เซอร์ ส่งเฉพาะจุด landmark"}</span>
          </div>
        </section>

        <section className="results" aria-live="polite">
          {latest ? <Latest key={latest.uid} r={latest} e={e} /> : (
            <div className="latest empty rise d2">
              <span className="cap">ผลล่าสุด</span>
              <p>ผลการแปลจะแสดงที่นี่ — ทำภาษามือเป็นประโยค เช่น “ฉัน รัก เพื่อน” แล้วลดมือ</p>
            </div>
          )}
          {prev && <Previous key={prev.uid} r={prev} e={e} label="ก่อนหน้า" />}
          {older.length > 0 && (
            <details className="more-results">
              <summary>ผลก่อนหน้าอีก {older.length} ประโยค</summary>
              {older.map((r) => <Previous key={r.uid} r={r} e={e} />)}
            </details>
          )}
          <div className="teach rise d3">
            <span className="iconbox"><Ic n="star-white" s={22} /></span>
            <div>
              <b>สอนท่าของคุณ</b>
              <p>{unsure > 0 ? `มี ${unsure} คำที่ระบบไม่แน่ใจ — แตะคำสีเหลือง เพื่อให้จำท่าภาษามือของคุณ` : "แตะคำที่ระบบไม่แน่ใจ เพื่อให้จำท่าภาษามือของคุณ"}</p>
            </div>
          </div>
        </section>
      </div>
      <datalist id="vocab-list">{e.vocab.map((w) => <option key={w} value={w} />)}</datalist>
      {showSettings && <SettingsSheet s={settings} model={e.model} onClose={() => setShowSettings(false)} onServerFile={(f) => e.onFile(f, true)} />}
    </AppShell>
  );
}

function countUnsure(r: TranslateResult) {
  return r.segments.filter((s) => s.status !== "transition" && s.status !== "accepted").length;
}

/** The three buttons under the camera (v2-C “เริ่มโหมดสด · บันทึกเอง · อัปโหลดวิดีโอ”), which follow what the camera is doing. */
function Actions({ e }: { e: SignEngine }) {
  const busy = e.status === "processing" || e.status === "loading";
  const recording = e.status === "recording" || e.status === "recorded";
  const upload = (
    <label className={`btn secondary file ${busy ? "disabled" : ""}`}>อัปโหลดวิดีโอ
      <input type="file" accept="video/*" disabled={busy} onChange={(ev) => { const f = ev.target.files?.[0]; if (f) e.onFile(f); ev.target.value = ""; }} />
    </label>
  );
  return (
    <div className="mode-box">
      {/* which mode the camera is in is always visible; switching never closes the camera */}
      <div className="mode-tabs" role="radiogroup" aria-label="โหมดกล้อง">
        <button role="radio" aria-checked={e.mode === "live"} className={e.mode === "live" ? "on" : ""} disabled={recording}
          onClick={() => e.setMode("live")}><span className={`dot ${e.mode === "live" && e.camOn ? "live" : "off"}`} />โหมดสด</button>
        <button role="radio" aria-checked={e.mode === "manual"} className={e.mode === "manual" ? "on" : ""}
          onClick={() => e.setMode("manual")}>บันทึกเอง</button>
      </div>
      <p className="mode-help">{e.mode === "live"
        ? "ไม่ต้องกดปุ่มระหว่างทำ — ยกมือเพื่อเริ่ม ทำภาษามือเป็นประโยค แล้วลดมือ ระบบแปลและอ่านออกเสียงเอง ทำต่อได้เรื่อย ๆ"
        : "กดเริ่มบันทึก → ทำภาษามือ → กดหยุด → กดแปล เหมาะกับการทดสอบทีละประโยค"}</p>
      <div className="actions">
        {!e.camOn ? <button className="btn" onClick={() => e.openCamera()} disabled={busy}>{e.status === "loading" ? "กำลังเปิดกล้อง…" : e.mode === "live" ? "เปิดกล้อง · เริ่มโหมดสด" : "เปิดกล้อง"}</button>
          : e.mode === "live" ? <Link href="/sign/live" className="btn">เต็มจอ</Link>
          : e.status === "recording" ? <button className="btn rec" onClick={e.stopRec}>■ หยุดบันทึก</button>
          : e.status === "recorded" ? <><button className="btn" onClick={e.process}>แปลท่าที่บันทึก</button><button className="btn secondary" onClick={e.startRec}>บันทึกใหม่</button></>
          : <button className="btn coral" onClick={e.startRec} disabled={busy}>● เริ่มบันทึก</button>}
        {!recording && upload}
        {e.camOn && !recording && <button className="btn secondary" onClick={e.closeCamera}>ปิดกล้อง</button>}
      </div>
    </div>
  );
}

function Latest({ r, e }: { r: TranslateResult; e: SignEngine }) {
  const { text, tentative } = shownText(r);
  const signs = useSigns(r);
  const c = cues(r);
  const speaking = !!e.speaking;
  return (
    <article className="latest">
      <span className="cap">ผลล่าสุด{r.title && r.title !== "camera" ? ` · ${r.title}` : ""}{tentative && text ? " · ยังไม่มั่นใจ" : ""}</span>
      <p className={`sentence ${tentative || !text ? "tentative" : ""}`}>
        {text ? <Sentence text={text} /> : signs.length ? `เห็น ${signs.length} ท่า แต่ยังไม่มั่นใจว่าเป็นคำอะไร — แตะ [?] เพื่อบอกคำที่ทำจริง` : "ไม่พบท่ามือ — ยกมือให้อยู่ในกรอบแล้วลองอีกครั้ง"}
      </p>
      <Words r={r} onTeal vocab={e.vocab} />
      {c.length > 0 && <div className="cues">{c.map((x) => <span key={x}>{x}</span>)}</div>}
      {text && <button className={`listen ${speaking ? "on" : ""}`} onClick={() => e.say(r, true)} disabled={speaking}>
        <Ic n="speaker-teal" />{speaking ? "กำลังพูด…" : "ฟังเสียง"}</button>}
    </article>
  );
}

function Previous({ r, e, label }: { r: TranslateResult; e: SignEngine; label?: string }) {
  const { text } = shownText(r);
  const n = useSigns(r).length;
  return (
    <article className="prev">
      <span className="cap">{label ?? (r.title && r.title !== "camera" ? r.title : "ก่อนหน้า")}
        {text && <button onClick={() => e.say(r, true)}>ฟังอีกครั้ง</button>}</span>
      <p className="sentence">{text ? <Sentence text={text} /> : n ? `ยังไม่มั่นใจ ${n} ท่า` : "ไม่พบท่ามือ"}</p>
      <Words r={r} vocab={e.vocab} />
    </article>
  );
}
