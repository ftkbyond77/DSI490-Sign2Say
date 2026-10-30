"use client";
// v2-D header button “วิธีใช้”: how both directions work, in three steps each.
import { useEffect, useState } from "react";
import Portal from "./Portal";

export default function HelpButton() {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open) return;
    const k = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [open]);
  return (
    <>
      <button className="btn secondary" onClick={() => setOpen(true)}>วิธีใช้</button>
      {open && (<Portal>
        <div className="scrim center" onClick={() => setOpen(false)}>
          <div className="sheet" role="dialog" aria-modal="true" aria-label="วิธีใช้" onClick={(e) => e.stopPropagation()}>
            <div className="sheet-head"><h2>วิธีใช้ ThaiSLM</h2><button className="close" onClick={() => setOpen(false)} aria-label="ปิด">×</button></div>
            <div className="help-grid">
              <section>
                <h3>ฉันใช้ภาษามือ</h3>
                <ol>
                  <li>เปิดกล้อง ยืนให้เห็นตัวถึงเอว มือทั้งสองข้างอยู่ในกรอบ</li>
                  <li>ยกมือแล้วทำภาษามือเป็นประโยค เว้นจังหวะสั้น ๆ ระหว่างคำ</li>
                  <li>ลดมือ — ระบบแปลเป็นข้อความและอ่านออกเสียงให้</li>
                </ol>
                <p className="muted">คำที่ระบบไม่แน่ใจจะแสดงเป็น [?] แตะเพื่อบอกคำที่ถูก ระบบจะจำท่าของคุณไว้ในเบราว์เซอร์นี้</p>
              </section>
              <section className="rose">
                <h3>ฉันพิมพ์หรือพูด</h3>
                <ol>
                  <li>พิมพ์ประโยค หรือกดไมค์แล้วพูด (ไทยหรืออังกฤษ)</li>
                  <li>ตรวจคำที่ระบบได้ยิน แก้ได้ก่อนแปล</li>
                  <li>ดูอวาตาร์ทำภาษามือ ปรับความเร็ว หรือแตะคำเพื่อดูซ้ำ</li>
                </ol>
                <p className="muted">ลำดับท่าเรียงตามไวยากรณ์ภาษามือไทย (TSL) — สลับเป็นลำดับคำภาษาไทยได้</p>
              </section>
            </div>
            <p className="muted">ภาพจากกล้องประมวลผลในเบราว์เซอร์ ส่งไปเซิร์ฟเวอร์เฉพาะจุด landmark ของร่างกาย ไม่ส่งวิดีโอ</p>
          </div>
        </div>
      </Portal>)}
    </>
  );
}
