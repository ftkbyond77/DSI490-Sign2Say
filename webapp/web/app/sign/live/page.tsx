"use client";
// v2-B — กล้องเต็มจอ: live mode on the whole screen. The latest sentence is the big caption; earlier sentences and the avatar's
// replies are listed on the right; the reply button lets the other person type, and the avatar answers in the picture-in-picture.
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import AvatarStage from "@/components/avatar/AvatarStage";
import SettingsSheet from "@/components/SettingsSheet";
import { CamToggle, CameraFeed, Dot, statusOf } from "@/components/sign/CameraView";
import Words, { shownText } from "@/components/sign/Words";
import { Ic, Logo, ProgressBar, Sentence } from "@/components/ui";
import { api } from "@/lib/t2s/api";
import { useAvatar } from "@/lib/t2s/useAvatar";
import { useSignEngine, useSignSettings } from "@/lib/useSignEngine";

type Item = { id: number; kind: "sign" | "speech"; text: string };

export default function LivePage() {
  const router = useRouter();
  const settings = useSignSettings();
  const [items, setItems] = useState<Item[]>([]);
  const seq = useRef(0);
  const e = useSignEngine(settings, (r) => {
    const { text } = shownText(r);
    if (text) setItems((x) => [{ id: ++seq.current, kind: "sign" as const, text }, ...x].slice(0, 8));
  });
  const hostRef = useRef<HTMLDivElement>(null);
  const a = useAvatar(hostRef);
  const [replyOpen, setReplyOpen] = useState(false);
  const [reply, setReply] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [showPip, setShowPip] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => { e.startLive(); }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const sendReply = async () => {
    const t = reply.trim();
    if (!t || busy) return;
    setBusy(true); setErr("");
    try {
      const r = await api.translate(t, "auto", 1.0, "text", a.avatarId, "tsl51", null);
      setShowPip(true); a.show(r, true);
      setItems((x) => [{ id: ++seq.current, kind: "speech" as const, text: t }, ...x].slice(0, 8));
      setReply(""); setReplyOpen(false);
    } catch (x) { setErr(`แปลเป็นภาษามือไม่สำเร็จ: ${(x as Error).message}`); }
    finally { setBusy(false); }
  };

  const st = statusOf(e);
  const live = e.mode === "live" && e.camOn;
  const latest = e.results[0];
  const shown = latest ? shownText(latest) : null;
  const handsN = (e.seen.left ? 1 : 0) + (e.seen.right ? 1 : 0);
  const units = a.current?.ir.units ?? [];
  const cur = a.unit >= 0 ? units[a.unit] : null;

  return (
    <div className="immersive">
      <CameraFeed engine={e} onOpen={e.startLive} label="เปิดกล้อง" />
      {e.camOn && <div className={`guide ${e.liveState === "signing" ? "signing" : ""}`} aria-hidden />}

      <div className="im-top">
        <Logo variant="glass" />
        <div className="live-pill glass" role="status">
          <Dot tone={st.tone} /><b>{st.text}</b>{st.sub && <span>· {st.sub}</span>}
        </div>
        <div className="right">
          {e.camOn && <div className="det glass">มือ {handsN}/2 · หน้า {e.seen.face ? "✓" : "✗"}</div>}
          <CamToggle e={e} onOpen={e.startLive} className="im-btn" />
          <button className="im-btn" onClick={() => setShowSettings(true)} aria-label="ตั้งค่า"><Ic n="sliders-mint" /></button>
        </div>
      </div>

      {items.length > 0 && (
        <aside className="history glass" aria-label="ก่อนหน้านี้">
          <h3>ก่อนหน้านี้</h3>
          {items.slice(latest ? 1 : 0).length === 0 && <small style={{ color: "var(--teal-200)" }}>ประโยคก่อนหน้าจะแสดงที่นี่</small>}
          {items.filter((it, i) => !(i === 0 && it.kind === "sign" && latest)).map((it) => (
            <div key={it.id} className={`item ${it.kind}`}><small>{it.kind === "sign" ? "ภาษามือ" : "ผู้พูด → อวาตาร์"}</small><b>{it.text}</b></div>
          ))}
        </aside>
      )}

      <div className="pip" hidden={!showPip}>
        <AvatarStage a={a} hostRef={hostRef} small>
          <span className="label">อวาตาร์ตอบ</span>
          {cur && <span className="cur" key={a.unit}>{cur.gloss}</span>}
        </AvatarStage>
      </div>

      {e.error && <div className="im-err"><p className="err">{e.error}</p></div>}
      {err && <div className="im-err"><p className="err">{err}</p></div>}

      {replyOpen ? (
        <form className="reply glass" onSubmit={(ev) => { ev.preventDefault(); sendReply(); }}>
          <input autoFocus value={reply} onChange={(ev) => setReply(ev.target.value)} placeholder="พิมพ์ตอบ แล้วอวาตาร์จะทำภาษามือให้…" maxLength={400} aria-label="ข้อความตอบ" />
          <button type="submit" className="btn coral sm" disabled={!reply.trim() || busy}>{busy ? "กำลังแปล…" : "ส่ง"}</button>
          <button type="button" className="btn sm" style={{ background: "transparent", border: "1px solid var(--glass-line)" }} onClick={() => setReplyOpen(false)}>ยกเลิก</button>
        </form>
      ) : e.progress && !shown ? (
        <div className="im-progress"><ProgressBar progress={e.progress} /></div>
      ) : shown && (
        <div className="caption glass new" key={latest!.uid} aria-live="polite">
          <p className={shown.tentative ? "tentative" : ""}>{shown.text ? <Sentence text={shown.text} /> : "ไม่พบท่ามือ"}</p>
          <Words r={latest!} onTeal hint vocab={e.vocab} />
        </div>
      )}
      {!shown && !replyOpen && !e.progress && live && <div className="caption glass"><span className="hint">ยกมือแล้วทำภาษามือเป็นประโยค · ลดมือเพื่อให้ระบบแปลและอ่านออกเสียง</span></div>}

      <div className="im-controls">
        <label className="side" aria-label="อัปโหลดวิดีโอ" title="อัปโหลดวิดีโอ"><Ic n="upload-mint" s={24} />
          <input type="file" accept="video/*" onChange={(ev) => { const f = ev.target.files?.[0]; if (f) e.onFile(f); ev.target.value = ""; }} />
        </label>
        {live ? <button className="stop" onClick={() => { e.stopLive(); e.closeCamera(); router.push("/sign"); }} aria-label="หยุดโหมดสด" title="หยุดโหมดสด"><i /></button>
          : <button className="stop start" onClick={() => e.startLive()} aria-label="เริ่มโหมดสด" disabled={e.status === "loading"}><i /></button>}
        <button className="side" onClick={() => setReplyOpen(!replyOpen)} aria-label="ตอบกลับด้วยอวาตาร์" title="ตอบกลับ (อวาตาร์ทำภาษามือ)"><Ic n="chat-mint" s={24} /></button>
      </div>
      <Link href="/chat" className="sr">ไปห้องสนทนา</Link>
      <datalist id="vocab-list">{e.vocab.map((w) => <option key={w} value={w} />)}</datalist>
      {showSettings && <SettingsSheet s={settings} model={e.model} onClose={() => setShowSettings(false)} />}
    </div>
  );
}
