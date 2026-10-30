"use client";
// คลังคำ — the words the sign model can read (conversation vocabulary). Tap a word to see the avatar sign it (/text?q=…).
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import AppShell from "@/components/AppShell";
import { vocabList } from "@/lib/api";

export default function Vocab() {
  const [words, setWords] = useState<string[] | null>(null);
  const [q, setQ] = useState("");
  useEffect(() => { vocabList().then((v) => setWords(v.filter((w) => !/^[A-Z]$/.test(w)))).catch(() => setWords([])); }, []);
  const shown = useMemo(() => (words ?? []).filter((w) => !q.trim() || w.includes(q.trim())).sort((a, b) => a.localeCompare(b, "th")), [words, q]);
  return (
    <AppShell>
      <div className="title-row rise"><div><div className="hello">คำที่ระบบอ่านจากภาษามือได้</div><h1>คลังคำ</h1></div></div>
      <section className="panel">
        <input className="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="ค้นหาคำ…" aria-label="ค้นหาคำ" />
        <p className="muted">{words === null ? "กำลังโหลด…" : `${shown.length} จาก ${words.length} คำ · แตะคำเพื่อดูอวาตาร์ทำท่า`} · สะกดนิ้วได้ทั้ง A–Z และ ก–ฮ</p>
        {words && words.length === 0 && <div className="empty-state">โหลดคลังคำไม่ได้ — ตรวจว่า sign-api ทำงานอยู่</div>}
        <div className="word-grid">
          {shown.slice(0, 600).map((w) => <Link key={w} href={`/text?q=${encodeURIComponent(w)}`}>{w}</Link>)}
        </div>
      </section>
    </AppShell>
  );
}
