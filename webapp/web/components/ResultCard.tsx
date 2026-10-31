"use client";

import { useState } from "react";
import { speak, type TranslateResult } from "@/lib/api";
import { saveSign } from "@/lib/mysigns";
import { playWav, unlockAudio } from "@/lib/audio";

const LABEL: Record<string, string> = { accepted: "มั่นใจ", uncertain: "ไม่แน่ใจ", unknown: "ไม่รู้จัก", transition: "ช่วงเปลี่ยนท่า" };

export default function ResultCard({ r, latest, onSaved, vocab = [] }: { r: TranslateResult; latest: boolean; onSaved?: () => void; vocab?: string[] }) {
  const [typed, setTyped] = useState("");
  const lang = r.language ?? {};
  const [open, setOpen] = useState(-1);                 // which word's "is this right?" menu is open
  const [saved, setSaved] = useState<Record<number, string>>({});
  const confirm = (i: number, word: string) => {        // personal sign memory: this is how *I* sign `word`
    const z = signs[i].emb;
    if (z) { saveSign(word, z); setSaved({ ...saved, [i]: word }); onSaved?.(); }
    setOpen(-1); setTyped("");
  };
  const shown = r.sentence || "";
  const tentative = !shown && lang.tentative_sentence_th;
  const signs = r.segments.filter((s) => s.status !== "transition");
  const dur = Math.max(r.duration_s || 0, 0.1);
  const sayText = (shown || lang.tentative_sentence_th || "").replaceAll("[?]", " ").trim();   // [?] marks are never read out
  const replay = async () => {                     // a click: the user asks for it, so an uncertain reading is spoken too
    if (!sayText) return;
    unlockAudio();
    await playWav(await speak(sayText, r.face?.emotion, r.face?.intensity));
  };
  const cues = [r.face?.question_yesno && (r.face?.question_furrow ? "คำถาม ไหม (ขมวดคิ้ว · ทดลอง)" : "คำถาม (คิ้วยก)"), r.face?.negation && "ปฏิเสธ (ส่ายหัว)",
    r.face?.affirmation && "ยืนยัน (พยักหน้า)", r.face?.emotion && r.face.emotion !== "neutral" && `อารมณ์: ${r.face.emotion_th}`].filter(Boolean) as string[];
  return (
    <article className={`card ${latest ? "latest" : ""}`}>
      <header>
        <span className="title">{r.title && r.title !== "camera" ? r.title : "กล้อง"} · {r.duration_s?.toFixed(1)} วิ</span>
        <button className="ghost" onClick={replay} disabled={!sayText} title={shown ? "ฟังอีกครั้ง" : "ฟังคำอ่านที่ไม่แน่ใจ"}>🔊{!shown && sayText ? "?" : ""}</button>
      </header>
      <p className={`sentence ${shown ? "" : "muted"}`}>
        {shown || (signs.length ? "ยังไม่มั่นใจ (ต่ำกว่า 80%) — กด [?] เพื่อสอนคำที่ทำจริง" : "ไม่พบท่ามือ")}
      </p>
      <div className="chips">
        {signs.map((s, i) => (
          <span key={i} className="chipwrap">
            <button className={`chip ${saved[i] ? "accepted saved" : s.status}`} onClick={() => { setOpen(open === i ? -1 : i); setTyped(""); }}
              title={`${LABEL[s.status]} p=${s.p} · ${s.t0}–${s.t1}s — กดเพื่อยืนยัน/แก้คำ แล้วบันทึกเป็นท่าของฉัน`}>
              {saved[i] ? `${saved[i]} ✓` : s.status === "accepted" ? s.word : "[?]"}
            </button>
            {open === i && (
              <span className="menu">
                <small>คำที่ถูกคือ… (บันทึกเป็นท่าของฉัน)</small>
                {s.candidates.map((c) => <button key={c} onClick={() => confirm(i, c)} disabled={!s.emb}>{c}</button>)}
                <span className="teach">
                  <input list="vocab-list" value={typed} placeholder="พิมพ์คำที่ทำจริง…" onChange={(e) => setTyped(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && vocab.includes(typed.trim())) { confirm(i, typed.trim()); setTyped(""); } }} />
                  <button disabled={!s.emb || !vocab.includes(typed.trim())} onClick={() => { confirm(i, typed.trim()); setTyped(""); }}
                    title={!s.emb ? "ช่วงนี้ไม่มีข้อมูลท่า" : !vocab.includes(typed.trim()) ? "พิมพ์หรือเลือกคำจากรายการ" : "บันทึกเป็นท่าของฉัน"}>บันทึก</button>
                </span>
                {typed.trim() && !vocab.includes(typed.trim()) && <small>ไม่มีคำนี้ในคลังคำ — เลือกจากรายการ</small>}
                <button className="ghost" onClick={() => { setOpen(-1); setTyped(""); }}>ปิด</button>
              </span>
            )}
          </span>
        ))}
      </div>
      <div className="timeline">
        {r.segments.map((s, i) => (
          <span key={i} className={`bar ${s.status}`} style={{ left: `${(100 * s.t0) / dur}%`, width: `${Math.max(0.8, (100 * (s.t1 - s.t0)) / dur)}%` }} />
        ))}
      </div>
      {cues.length > 0 && <div className="cues">{cues.map((c) => <span key={c}>{c}</span>)}</div>}
      <footer>
        {lang.provider && <span>ภาษา: {lang.provider}</span>}
        <span>คลังคำ: {r.vocab}</span>
        {r.mirrored_to_right_hand && <span>ถนัดซ้าย → สะท้อนเป็นขวา</span>}
        <span>{Math.round(r.timings_ms?.total_ms ?? 0)} ms</span>
      </footer>
    </article>
  );
}
