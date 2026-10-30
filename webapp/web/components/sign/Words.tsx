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

/** The word filter (settings → filter): a sign's word is shown when the server accepted it, or when its P(correct) reaches the chosen
 *  threshold (0 = no threshold: every reading shown); otherwise it is [?]. The same rule for live and video results. */
export const shows = (s: { status: string; p?: number }, minP: number) => s.status === "accepted" || (s.p ?? 0) >= minP;

export default function Words({ r, onTeal = false, hint = false, vocab = [], minP = 0.8 }: { r: TranslateResult; onTeal?: boolean; hint?: boolean; vocab?: string[]; minP?: number }) {
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
        const ok = s.status === "accepted", shown = shows(s, minP);       // shown but not accepted: let through by the filter (dashed)
        const label = saved[i] ? `✓ ${saved[i]}` : ok ? (s.kind === "letter" ? s.word : `✓ ${s.word}`) : shown ? s.word : `[?]${hint ? " แตะเพื่อแก้" : ""}`;   // below the filter: no guessed word — the candidates are in the menu
        const cls = saved[i] ? `chip ${onTeal ? "teal" : ""} saved` : ok ? `chip ${onTeal ? "teal" : ""}` : shown ? `chip ${onTeal ? "teal" : ""} low` : "chip unsure";
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

/** The sentence to show for a result: the confident sentence, else the tentative reading (marked), else nothing. When the filter
 *  lets through words the server did not accept, the server's sentence would not hold them: the words themselves are shown
 *  (marked tentative), [?] for the ones still below the threshold. */
export function shownText(r: TranslateResult, minP = 0.8) {
  const signs = r.segments.filter((s) => s.status !== "transition");
  if (signs.some((s) => s.status !== "accepted" && shows(s, minP)))
    return { text: signs.map((s) => (shows(s, minP) ? s.word : "[?]")).join(" "), tentative: true };
  const s = r.sentence || "";
  return { text: s || r.language?.tentative_sentence_th || "", tentative: !s };
}
