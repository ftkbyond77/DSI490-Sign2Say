"use client";
// The camera (video + MediaPipe overlay, mirrored) with its status pill — used as the card on /sign, a tile on /chat and full screen on /sign/live.
import type { SignEngine } from "@/lib/useSignEngine";

export type Tone = "ready" | "live" | "rec" | "busy" | "off";

/** What the camera is doing, in the words of the design (“พร้อม · ยกมือเพื่อเริ่ม”, “กำลังทำท่า”, “กำลังอ่านท่า…”). */
export function statusOf(e: SignEngine): { text: string; sub?: string; tone: Tone } {
  if (e.status === "loading") return { text: "กำลังเปิดกล้อง…", tone: "busy" };
  if (!e.camOn) return e.status === "processing" ? { text: "กำลังแปลวิดีโอ…", tone: "busy" } : { text: "กล้องปิดอยู่", tone: "off" };
  if (e.mode === "live") {
    if (e.liveState === "signing") return { text: "กำลังทำท่า", sub: "ลดมือเพื่อจบประโยค", tone: "live" };
    if (e.liveState === "processing") return { text: "กำลังอ่านท่า…", tone: "busy" };
    return { text: "โหมดสด", sub: "ยกมือเพื่อเริ่ม", tone: "live" };
  }
  if (e.status === "recording") return { text: `กำลังบันทึก · ${e.nFrames} เฟรม`, tone: "rec" };
  if (e.status === "recorded") return { text: "บันทึกแล้ว · กดแปล", tone: "ready" };
  if (e.status === "processing") return { text: "กำลังอ่านท่า…", tone: "busy" };
  return { text: "บันทึกเอง", sub: "กดเริ่มบันทึก", tone: "ready" };
}

export function Dot({ tone }: { tone: Tone }) {
  return <span className={`dot ${tone === "live" ? "live" : tone === "rec" ? "rec" : tone === "busy" ? "busy" : tone === "off" ? "off" : ""}`} />;
}

/** <video> + <canvas> of the engine; while the camera is off: a framed empty state with a button that opens it (`onOpen`). */
export function CameraFeed({ engine: e, onOpen, label = "เปิดกล้อง", hint = "ยืนให้เห็นตัวถึงเอว มือทั้งสองข้างอยู่ในกรอบ", compact = false }: {
  engine: SignEngine; onOpen?: () => void; label?: string; hint?: string; compact?: boolean;
}) {
  const filter = e.autoLight && (e.tone.brightness !== 1 || e.tone.contrast !== 1)
    ? { filter: `brightness(${e.tone.brightness.toFixed(2)}) contrast(${e.tone.contrast.toFixed(2)})` } : undefined;
  return (
    <>
      <video ref={e.videoRef} playsInline muted className="mirror" style={{ ...filter, visibility: e.camOn ? "visible" : "hidden" }} />
      <canvas ref={e.canvasRef} className="mirror" style={{ visibility: e.camOn ? "visible" : "hidden" }} />
      {!e.camOn && (
        <div className={`cam-empty ${compact ? "compact" : ""}`}>
          <span className="corners" aria-hidden><i /><i /><i /><i /></span>
          {e.status === "loading" ? <div className="cam-open busy"><span className="spinner teal" /></div> : onOpen && (
            <button className="cam-open" onClick={onOpen} aria-label={label}>
              <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M15 10l4.55-2.28A1 1 0 0 1 21 8.62v6.76a1 1 0 0 1-1.45.9L15 14" /><rect x="3" y="6" width="12" height="12" rx="3" /></svg>
            </button>
          )}
          <b>{e.status === "loading" ? "กำลังโหลดโมเดล MediaPipe…" : label}</b>
          {!compact && <small>{hint}</small>}
        </div>
      )}
    </>
  );
}

/** “ตรวจพบ” — which parts MediaPipe finds right now. */
export function Seen({ e }: { e: SignEngine }) {
  const s = e.seen;
  const c = (ok: boolean, l: string) => <span className={ok ? "on" : ""}>{ok ? `✓ ${l}` : l}</span>;
  return <div className="seen" aria-label="สิ่งที่กล้องเห็น">{c(s.body, "ตัว")}{c(s.face, "หน้า")}{c(s.left, "มือซ้าย")}{c(s.right, "มือขวา")}</div>;
}

/** One icon button that opens / closes the camera (camera ↔ camera with a slash), for the corner of every camera view. */
export function CamToggle({ e, onOpen, className = "glass-btn" }: { e: SignEngine; onOpen: () => void; className?: string }) {
  const on = e.camOn, busy = e.status === "loading";
  return (
    <button className={`${className} cam-toggle ${on ? "on" : ""}`} onClick={on ? e.closeCamera : onOpen} disabled={busy}
      aria-label={on ? "ปิดกล้อง" : "เปิดกล้อง"} title={on ? "ปิดกล้อง" : "เปิดกล้อง"} aria-pressed={on}>
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        <path d="M15 10l4.55-2.28A1 1 0 0 1 21 8.62v6.76a1 1 0 0 1-1.45.9L15 14" /><rect x="3" y="6" width="12" height="12" rx="3" />
        {on && <path d="M2 2l20 20" />}
      </svg>
    </button>
  );
}
