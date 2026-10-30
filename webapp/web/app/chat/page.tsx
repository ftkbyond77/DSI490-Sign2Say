"use client";
// v2-A — ห้องสนทนา: two people on one screen. The signer signs to the camera (live mode) → teal bubbles, spoken aloud; the other
// person types or speaks → coral bubbles, signed by the avatar. Both engines of the app run side by side.
import { useEffect, useRef, useState } from "react";
import AvatarStage from "@/components/avatar/AvatarStage";
import SettingsSheet from "@/components/SettingsSheet";
import { CamToggle, CameraFeed, Dot, statusOf } from "@/components/sign/CameraView";
import Words, { shownText } from "@/components/sign/Words";
import AppShell from "@/components/AppShell";
import { Ic, Sentence, Typing } from "@/components/ui";
import type { TranslateResult } from "@/lib/api";
import { api, type Result } from "@/lib/t2s/api";
import { useAsr } from "@/lib/t2s/useAsr";
import { useAvatar } from "@/lib/t2s/useAvatar";
import { useSignEngine, useSignSettings } from "@/lib/useSignEngine";

type Msg = { id: number; at: string; kind: "sign"; r: TranslateResult } | { id: number; at: string; kind: "speech"; text: string; via: "voice" | "text"; t2s: Result };
const now = () => new Date().toLocaleTimeString("th-TH", { hour: "2-digit", minute: "2-digit" });

export default function ChatPage() {
  const settings = useSignSettings();
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const seq = useRef(0);
  const e = useSignEngine(settings, (r) => setMsgs((m) => [...m, { id: ++seq.current, at: now(), kind: "sign", r }]));
  const hostRef = useRef<HTMLDivElement>(null);
  const a = useAvatar(hostRef);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [playingId, setPlayingId] = useState(-1);
  const listRef = useRef<HTMLDivElement>(null);

  const send = async (text: string, via: "voice" | "text" = "text") => {
    const t = text.trim();
    if (!t || busy) return;
    setBusy(true); setErr("");
    try {
      const r = await api.translate(t, "auto", 1.0, via === "voice" ? "speech" : "text", a.avatarId, "tsl51", null);
      const id = ++seq.current;
      setMsgs((m) => [...m, { id, at: now(), kind: "speech", text: t, via, t2s: r }]);
      setDraft("");
      a.show(r, true); setPlayingId(id);
    } catch (x) { setErr(`แปลเป็นภาษามือไม่สำเร็จ: ${(x as Error).message}`); }
    finally { setBusy(false); }
  };
  const mic = useAsr((r) => { if (r.text.trim() && r.reliable) send(r.text, "voice"); else { setDraft(r.text); setErr(r.text ? "ระบบไม่แน่ใจข้อความที่ได้ยิน — ตรวจแล้วกดส่ง" : "ไม่ได้ยินเสียงพูด"); } });

  useEffect(() => { listRef.current?.scrollTo({ top: listRef.current.scrollHeight }); }, [msgs.length, busy, e.liveState, e.status]);
  useEffect(() => { if (!a.playing) setPlayingId(-1); }, [a.playing]);

  const st = statusOf(e);
  const live = e.mode === "live" && e.camOn;
  const translatingSign = e.liveState === "processing" || e.status === "processing";
  const units = a.current?.ir.units ?? [];
  const u = a.unit >= 0 ? units[a.unit] : null;

  return (
    <AppShell fill>
      <div className="title-row rise">
        <div><div className="hello">ผู้ใช้ภาษามือ ⇄ คนที่พิมพ์หรือพูด</div><h1>บทสนทนา</h1></div>
        <div className="options">
          <button className="pill" style={{ border: "none", boxShadow: "var(--shadow-teal)", fontWeight: 400, color: "var(--teal-800)" }}
            onClick={() => { if (msgs.length && confirm("เริ่มบทสนทนาใหม่? ข้อความทั้งหมดจะหายไป")) setMsgs([]); }} title="เริ่มบทสนทนาใหม่">
            <span className={`dot ${live ? "live" : msgs.length ? "" : "off"}`} style={{ width: 8, height: 8 }} />บทสนทนาใหม่ · {msgs.length} ข้อความ
          </button>
          <button className="icon-btn" onClick={() => setShowSettings(true)} aria-label="ตั้งค่า"><Ic n="sliders-teal" /></button>
        </div>
      </div>

      <div className="chat-main">
        <div className="tiles">
          <div className="tile">
            <CameraFeed engine={e} onOpen={e.startLive} label="เปิดกล้องผู้ใช้ภาษามือ" compact />
            <div className="tl"><span className="pill teal"><Dot tone={st.tone} />{live ? (e.liveState === "signing" ? "กำลังทำท่า…" : e.liveState === "processing" ? "กำลังอ่านท่า…" : "ยกมือเพื่อเริ่ม") : st.text}</span></div>
            <span className="name">ผู้ใช้ภาษามือ</span>
            <div className="tr"><CamToggle e={e} onOpen={e.startLive} /></div>
          </div>
          <div className="tile avatar">
            <AvatarStage a={a} hostRef={hostRef} small>
              {u && <div className="tl" style={{ position: "absolute", left: 14, top: 14, zIndex: 2 }}><span className="pill rose" key={a.unit}>{u.gloss} · {a.unit + 1}/{units.length}</span></div>}
              <span className="name">อวาตาร์ (แปลจากผู้พูด)</span>
            </AvatarStage>
          </div>
        </div>

        <section className="convo">
          <div className="msgs" ref={listRef} aria-live="polite">
            {msgs.length === 0 && !translatingSign && (
              <div className="chat-empty">
                <h2>คุยกันสองคน ในหน้าจอเดียว</h2>
                <p>นั่งหน้าจอเดียวกัน ผลัดกันสื่อสาร — ระบบแปลให้ทั้งสองทาง</p>
                <div className="roles">
                  <div className="role sign">
                    <b>ผู้ใช้ภาษามือ</b>
                    <ol><li>กด <em>✋ ทำภาษามือต่อ</em> (เปิดกล้อง)</li><li>ยกมือ ทำภาษามือเป็นประโยค</li><li>ลดมือ — ข้อความขึ้นฝั่งซ้ายและอ่านออกเสียง</li></ol>
                  </div>
                  <div className="role speech">
                    <b>คนที่พิมพ์หรือพูด</b>
                    <ol><li>พิมพ์ในช่องด้านล่าง หรือกดไมค์แล้วพูด</li><li>กดส่ง</li><li>อวาตาร์ทางซ้ายทำภาษามือให้ดู (ดูซ้ำได้)</li></ol>
                  </div>
                </div>
              </div>
            )}
            {msgs.map((m) => m.kind === "sign" ? (
              <div key={m.id} className="msg sign">
                <span className="who">ภาษามือ · {m.at}</span>
                <div className="bubble">
                  {(() => { const { text, tentative } = shownText(m.r, e.minP); return <p className={tentative ? "tentative" : ""}>{text ? <Sentence text={text} /> : "ไม่พบท่ามือ"}</p>; })()}
                  <Words r={m.r} onTeal vocab={e.vocab} minP={e.minP} />
                  {shownText(m.r, e.minP).text && <button className="replay-chip" style={{ color: "var(--teal-700)" }} onClick={() => e.say(m.r, true)}><Ic n="speaker-teal" s={14} />ฟังเสียง</button>}
                </div>
              </div>
            ) : (
              <div key={m.id} className="msg speech">
                <span className="who">{m.via === "voice" ? "เสียงพูด" : "ข้อความ"} · {m.at}</span>
                <div className="bubble">
                  <p>{m.text}</p>
                  <button className={`replay-chip ${playingId === m.id ? "on" : ""}`} onClick={() => { a.show(m.t2s, true); setPlayingId(m.id); }}>
                    <Ic n="play-coral" s={14} />{playingId === m.id ? "อวาตาร์กำลังทำภาษามือ…" : "ดูอวาตาร์ทำภาษามืออีกครั้ง"}
                  </button>
                </div>
              </div>
            ))}
            {translatingSign && <Typing text="กำลังแปลท่ามือ…" />}
            {busy && <div style={{ alignSelf: "flex-end" }}><Typing rose text="กำลังเตรียมท่าภาษามือ…" /></div>}
            {mic.state === "recording" && <div style={{ alignSelf: "flex-end" }}><Typing rose text={mic.committed || mic.tentative ? `${mic.committed} ${mic.tentative}` : "กำลังฟัง…"} /></div>}
            {mic.state === "final" && <div style={{ alignSelf: "flex-end" }}><Typing rose text="กำลังถอดเสียง…" /></div>}
          </div>
          {(err || e.error || mic.error || e.liveNote) && <div className="composer-note">{err || e.error || mic.error || e.liveNote}</div>}
          <div className="composer">
            <button className={`btn ${live ? "secondary" : ""}`} onClick={() => (live ? (e.stopLive(), e.closeCamera()) : e.startLive())} disabled={e.status === "loading"}>
              {e.status === "loading" ? "กำลังเปิดกล้อง…" : live ? "■ หยุดกล้อง" : "✋ ทำภาษามือต่อ"}
            </button>
            <form className="input" onSubmit={(ev) => { ev.preventDefault(); send(draft); }}>
              <input value={draft} onChange={(ev) => setDraft(ev.target.value)} placeholder="พิมพ์หรือพูด แล้วอวาตาร์จะทำภาษามือให้…" aria-label="ข้อความถึงผู้ใช้ภาษามือ" maxLength={400} />
              <button type="button" className={`sq ${mic.state === "recording" ? "on" : ""}`} onClick={() => mic.start("th")} disabled={mic.state === "final"}
                aria-label={mic.state === "recording" ? "หยุดพูด" : "กดเพื่อพูด"}><Ic n="mic-coral-19" s={19} /></button>
              <button type="submit" className="sq send" disabled={!draft.trim() || busy} aria-label="ส่ง"><Ic n="send-white" s={19} /></button>
            </form>
          </div>
        </section>
      </div>
      <datalist id="vocab-list">{e.vocab.map((w) => <option key={w} value={w} />)}</datalist>
      {showSettings && <SettingsSheet s={settings} model={e.model} onClose={() => setShowSettings(false)} />}
    </AppShell>
  );
}
