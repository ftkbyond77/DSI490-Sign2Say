"use client";
// The words of one result as chips (Figma “WordChip”: Confident ✓ / Uncertain “?” / On teal). Tapping a chip opens “the right word
// is …”: the segment is remembered as the user's own way of signing that word (personal sign memory, this browser only).
import { useState } from "react";
import type { TranslateResult } from "@/lib/api";
import { saveSign } from "@/lib/mysigns";
import { signsChanged } from "@/lib/useSignEngine";

export function useSigns(r: TranslateResult) {
  return r.segments.filter((s) => s.status !== "transition");
}

export default function Words({ r, onTeal = false, hint = false, vocab = [] }: { r: TranslateResult; onTeal?: boolean; hint?: boolean; vocab?: string[] }) {
  const signs = useSigns(r);
  const [open, setOpen] = useState(-1);
  const [typed, setTyped] = useState("");
  const [saved, setSaved] = useState<Record<number, string>>({});
  const confirm = (i: number, word: string) => {
    const z = signs[i].emb;
    if (z) { saveSign(word, z); setSaved((s) => ({ ...s, [i]: word })); signsChanged(); }
    setOpen(-1); setTyped("");
  };
  if (!signs.length) return null;
  const t = typed.trim(), known = vocab.includes(t);
  return (
    <div className="chips">
      {signs.map((s, i) => {
        const ok = s.status === "accepted";
        const label = saved[i] ? `✓ ${saved[i]}` : ok ? (s.kind === "letter" ? s.word : `✓ ${s.word}`) : `[?]${hint ? " แตะเพื่อแก้" : ""}`;   // unsure: no guessed word (v7.3) — the candidates are in the menu
        const cls = saved[i] ? `chip ${onTeal ? "teal" : ""} saved` : ok ? `chip ${onTeal ? "teal" : ""}` : "chip unsure";
        return (
          <span key={i} className="chipwrap">
            <button className={cls} onClick={() => { setOpen(open === i ? -1 : i); setTyped(""); }} aria-expanded={open === i}
              title={`${ok ? "มั่นใจ" : "ไม่แน่ใจ"} ${Math.round(s.p * 100)}% · ${s.t0}–${s.t1} วิ — แตะเพื่อยืนยันหรือแก้คำ`}>{label}</button>
            {open === i && (
              <span className="menu" role="dialog" aria-label="แก้คำ">
                <small>คำที่ถูกคือ… (ระบบจะจำท่าของคุณ)</small>
                {s.candidates.slice(0, 4).map((c) => <button key={c} className="opt" onClick={() => confirm(i, c)} disabled={!s.emb}>{c}</button>)}
                <span className="row">
                  <input list="vocab-list" value={typed} placeholder="พิมพ์คำที่ทำจริง…" onChange={(e) => setTyped(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && known) confirm(i, t); }} />
                  <button className="save" disabled={!s.emb || !known} onClick={() => confirm(i, t)}>บันทึก</button>
                </span>
                {t && !known && <small>ไม่มีคำนี้ในคลังคำ — เลือกจากรายการ</small>}
                {!s.emb && <small>ช่วงนี้ไม่มีข้อมูลท่า บันทึกไม่ได้</small>}
                <button className="x" onClick={() => setOpen(-1)}>ปิด</button>
              </span>
            )}
          </span>
        );
      })}
    </div>
  );
}

/** Non-manual cues the server read from the face (shown under a result). */
export function cues(r: TranslateResult) {
  return [r.face?.question_yesno && (r.face?.question_furrow ? "คำถาม ไหม (ขมวดคิ้ว)" : "คำถาม (คิ้วยก)"), r.face?.negation && "ปฏิเสธ (ส่ายหัว)",
    r.face?.affirmation && "ยืนยัน (พยักหน้า)", r.face?.emotion && r.face.emotion !== "neutral" && `อารมณ์: ${r.face.emotion_th}`].filter(Boolean) as string[];
}

/** The sentence to show for a result: the confident sentence, else the tentative reading (marked), else nothing. */
export function shownText(r: TranslateResult) {
  const s = r.sentence || "";
  return { text: s || r.language?.tentative_sentence_th || "", tentative: !s };
}
