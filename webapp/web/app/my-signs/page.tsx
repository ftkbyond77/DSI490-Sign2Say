"use client";
// ท่าของฉัน — the words this user taught (personal sign memory, this browser only): see them, forget one, or clear all.
import Link from "next/link";
import { useEffect, useState } from "react";
import AppShell from "@/components/AppShell";
import { clearSigns, removeWord, signWords } from "@/lib/mysigns";
import { signsChanged } from "@/lib/useSignEngine";

export default function MySigns() {
  const [words, setWords] = useState<Record<string, number>>({});
  const refresh = () => setWords(signWords());
  useEffect(() => {
    refresh();
    window.addEventListener("thaislm:mysigns", refresh);
    return () => window.removeEventListener("thaislm:mysigns", refresh);
  }, []);
  const list = Object.entries(words).sort((a, b) => a[0].localeCompare(b[0], "th"));
  return (
    <AppShell>
      <div className="title-row rise"><div><div className="hello">จำไว้ในเบราว์เซอร์นี้เท่านั้น</div><h1>ท่าของฉัน</h1></div></div>
      <section className="panel">
        <h2>คำที่สอนแล้ว {list.length > 0 && `· ${list.length} คำ`}</h2>
        <p className="muted">เมื่อแตะคำบนผลการแปลแล้วเลือกคำที่ถูก ระบบจะเก็บท่าของคุณไว้ (สูงสุด 3 ตัวอย่างต่อคำ) และใช้เทียบทุกครั้งที่คุณทำภาษามือ
          โมเดลกลางไม่เปลี่ยน — ข้อมูลนี้ไม่ออกจากเบราว์เซอร์ ยกเว้นตอนส่งไปแปล</p>
        {list.length === 0 ? (
          <div className="empty-state">ยังไม่มีคำที่สอน — ไปที่ <Link href="/sign" style={{ color: "var(--teal-700)", textDecoration: "underline" }}>แปลภาษามือ</Link> แล้วแตะคำสีเหลือง [?]</div>
        ) : (
          <>
            <div className="word-grid">
              {list.map(([w, n]) => (
                <span key={w} className="w">{w} <small>×{n}</small>
                  <button onClick={() => { removeWord(w); signsChanged(); refresh(); }} aria-label={`ลืมคำ ${w}`} title="ลืมคำนี้">×</button></span>
              ))}
            </div>
            <div><button className="btn coral-outline sm" onClick={() => { if (confirm("ลบท่าที่สอนทั้งหมด?")) { clearSigns(); signsChanged(); refresh(); } }}>ลบทั้งหมด</button></div>
          </>
        )}
      </section>
    </AppShell>
  );
}
