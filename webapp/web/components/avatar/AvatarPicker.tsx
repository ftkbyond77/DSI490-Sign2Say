"use client";
// เลือกอวาตาร์: a button on the stage showing the current character → a gallery of every character (large pictures). The choice is
// remembered in this browser (useAvatar keeps it under "t2s-avatar"), so /text, /chat and the full-screen camera use the same one.
import { useEffect, useState } from "react";
import Portal from "@/components/Portal";
import type { AvatarEntry } from "@/lib/t2s/api";
import type { AvatarCtl } from "@/lib/t2s/useAvatar";

const num = (c: AvatarEntry) => c.label.replace(/\D+/g, "");
// the character's own name from its file (VRM meta title, e.g. "Satoru Gojo") — stable when files are added; "แบบที่ N" only when it has none
const hasName = (t: string) => /[A-Za-z฀-๿]{2}/.test(t) && !/^AvatarSample/i.test(t);
export const avatarName = (c: AvatarEntry, i: number) =>
  c.label === "Default" ? "ค่าเริ่มต้น" : hasName(c.title ?? "") ? c.title : `แบบที่ ${num(c) || i + 1}`;

function Thumb({ c, i }: { c: AvatarEntry; i: number }) {
  return c.thumb ? <img src={c.thumb} alt="" loading="lazy" /> : <span className="initial">{c.label === "Default" ? "D" : num(c) || i + 1}</span>;
}

export default function AvatarPicker({ a, small = false }: { a: AvatarCtl; small?: boolean }) {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open) return;
    const k = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [open]);
  if (!a.ready || a.cast.length < 2) return null;
  const i = Math.max(0, a.cast.findIndex((c) => c.id === a.avatarId));
  const cur = a.cast[i];
  return (
    <>
      <button className={`pick-btn ${small ? "small" : ""}`} onClick={() => setOpen(true)} aria-haspopup="dialog" title="เลือกอวาตาร์">
        <span className="thumb"><Thumb c={cur} i={i} /></span>
        {!small && <span className="txt"><small>อวาตาร์</small>{avatarName(cur, i)}</span>}
        {!small && <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M6 9l6 6 6-6" /></svg>}
      </button>
      {open && (
        <Portal>
          <div className="scrim center" onClick={() => setOpen(false)}>
            <div className="sheet picker" role="dialog" aria-modal="true" aria-label="เลือกอวาตาร์" onClick={(e) => e.stopPropagation()}>
              <div className="sheet-head"><h2>เลือกอวาตาร์</h2><button className="close" onClick={() => setOpen(false)} aria-label="ปิด">×</button></div>
              <p className="muted">ท่าภาษามือจะปรับให้เข้ากับสัดส่วนร่างกายของแต่ละตัว — ระบบจำตัวที่เลือกไว้ทุกหน้า</p>
              <div className="pick-grid" role="radiogroup" aria-label="อวาตาร์">
                {a.cast.map((c, k) => {
                  const on = c.id === a.avatarId, busy = a.switchingId !== undefined && a.switchingId === c.id;
                  return (
                    <button key={c.id ?? "default"} role="radio" aria-checked={on} className={`pick ${on ? "on" : ""}`} style={{ animationDelay: `${k * 0.04}s` }}
                      disabled={a.switchingId !== undefined} title={[c.title, c.author && `โดย ${c.author}`].filter(Boolean).join(" · ")}
                      onClick={async () => { await a.switchAvatar(c); setOpen(false); }}>
                      <span className="thumb"><Thumb c={c} i={k} />{busy && <span className="busy" />}</span>
                      <b>{avatarName(c, k)}</b>
                      {on && <span className="tick" aria-hidden>✓</span>}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </Portal>
      )}
    </>
  );
}
