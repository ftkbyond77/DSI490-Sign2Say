"use client";
// ตั้งค่า (the sliders button): every option of the original studio's control row, remembered in this browser.
import Link from "next/link";
import { useEffect } from "react";
import Portal from "./Portal";
import type { Alphabet, MinP, Options, Speak } from "@/lib/api";
import { LIVE_END_OPTIONS, type ModelInfo, type SignSettings } from "@/lib/useSignEngine";

export const SPEAK_TH: Record<Speak, string> = { off: "ไม่ออกเสียง", confident: "ออกเสียงเฉพาะที่มั่นใจ", all: "ออกเสียงทุกผล (รวมที่ไม่แน่ใจ)" };
export const ALPHA_TH: Partial<Record<Alphabet, string>> = { en: "A–Z", th: "ก–ฮ", off: "ปิด" };
export const MINP_TH: [MinP, string][] = [[0, "ไม่มี threshold"], [0.5, "threshold 0.50"], [0.8, "threshold 0.80"]];

/** The word filter: which unsure words are shown (else [?]) — live mode and video results alike. */
function MinPSelect({ s }: { s: SignSettings }) {
  const { opts, setOpts } = s;
  return (
    <select value={opts.minP ?? 0.8} onChange={(e) => setOpts({ ...opts, minP: Number(e.target.value) as MinP })}>
      {MINP_TH.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
    </select>
  );
}

/** The two quick selects of the title row (v2-C “คลังคำ ประจำวัน ▾”, “สะกดนิ้ว A–Z ▾”). */
export function QuickOptions({ s, model }: { s: SignSettings; model?: ModelInfo | null }) {
  const { opts, setOpts } = s;
  return (
    <>
      <label className="select">คลังคำ
        <select value={opts.vocab} onChange={(e) => setOpts({ ...opts, vocab: e.target.value as Options["vocab"] })}>
          <option value="conversation">ประจำวัน{model?.conversation_concepts ? ` (${model.conversation_concepts})` : ""}</option>
          <option value="full">ทั้งหมด{model?.concepts_full ? ` (${model.concepts_full})` : ""}</option>
        </select>
      </label>
      <label className="select">สะกดนิ้ว
        <select value={opts.alphabet} onChange={(e) => setOpts({ ...opts, alphabet: e.target.value as Alphabet })}>
          {(Object.keys(ALPHA_TH) as Alphabet[]).map((a) => <option key={a} value={a}>{ALPHA_TH[a]}</option>)}
        </select>
      </label>
      <label className="select">filter
        <MinPSelect s={s} />
      </label>
    </>
  );
}

export default function SettingsSheet({ s, model, onClose, onServerFile }: {
  s: SignSettings; model?: ModelInfo | null; onClose: () => void; onServerFile?: (f: File) => void;
}) {
  const { opts, setOpts, endMs, setEndMs, autoLight, setAutoLight } = s;
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  return (
    <Portal><div className="scrim" onClick={onClose}>
      <aside className="sheet" role="dialog" aria-modal="true" aria-label="ตั้งค่า" onClick={(e) => e.stopPropagation()}>
        <div className="sheet-head"><h2>ตั้งค่า</h2><button className="close" onClick={onClose} aria-label="ปิด">×</button></div>
        <label className="field"><span>เสียงพูด</span>
          <select value={opts.speak} onChange={(e) => setOpts({ ...opts, speak: e.target.value as Speak })}>
            {(Object.keys(SPEAK_TH) as Speak[]).map((x) => <option key={x} value={x}>{SPEAK_TH[x]}</option>)}
          </select>
        </label>
        <label className="field"><span>filter คำที่ไม่แน่ใจ</span>
          <MinPSelect s={s} />
          <small>คำที่ความมั่นใจต่ำกว่า threshold แสดงเป็น [?] · ไม่มี threshold = แสดงทุกคำที่อ่านได้ · คำที่ระบบยืนยันแล้วแสดงเสมอ (ทั้งโหมดสดและวิดีโอ)</small>
        </label>
        <label className="field"><span>คลังคำ</span>
          <select value={opts.vocab} onChange={(e) => setOpts({ ...opts, vocab: e.target.value as Options["vocab"] })}>
            <option value="conversation">บทสนทนาประจำวัน{model?.conversation_concepts ? ` (${model.conversation_concepts} คำ)` : ""}</option>
            <option value="full">ทั้งหมด{model?.concepts_full ? ` (${model.concepts_full} คำ)` : ""}</option>
          </select>
          <small>คลังคำทั้งหมดมีคำมากกว่า แต่ช้ากว่าและคลาดเคลื่อนง่าย</small>
        </label>
        <label className="field"><span>สะกดนิ้ว</span>
          <select value={opts.alphabet} onChange={(e) => setOpts({ ...opts, alphabet: e.target.value as Alphabet })}>
            <option value="en">อังกฤษ A–Z</option><option value="th">ไทย ก–ฮ + สระ</option><option value="off">ปิด</option>
          </select>
          <small>ทำท่าตัวอักษรแต่ละตัวค้างไว้ครู่หนึ่ง เช่น ชื่อ J A C K</small>
        </label>
        <label className="field"><span>โหมดสด: จบประโยคเมื่อลดมือนาน</span>
          <select value={endMs} onChange={(e) => setEndMs(Number(e.target.value))}>
            {LIVE_END_OPTIONS.map((ms) => <option key={ms} value={ms}>{(ms / 1000).toFixed(1)} วินาที</option>)}
          </select>
          <small>ถ้าทำภาษามือทีละคำ ให้เลือกนานขึ้น</small>
        </label>
        <hr />
        <label className="check"><input type="checkbox" checked={opts.llm} onChange={(e) => setOpts({ ...opts, llm: e.target.checked })} />
          <span>เรียบเรียงประโยคด้วย LLM<small>{model && !model.llm ? "เซิร์ฟเวอร์ไม่มีคีย์ — ใช้กฎแทน" : "ใช้เมื่อมีคำที่ไม่แน่ใจ"}</small></span></label>
        <label className="check"><input type="checkbox" checked={autoLight} onChange={(e) => setAutoLight(e.target.checked)} />
          <span>ปรับภาพอัตโนมัติ<small>ปรับแสงและคอนทราสต์ของภาพที่ส่งให้ MediaPipe (ห้องมืด / กล้องสีซีด)</small></span></label>
        <label className="check"><input type="checkbox" checked={opts.faceQuestion} onChange={(e) => setOpts({ ...opts, faceQuestion: e.target.checked })} />
          <span>ทดลอง: ขมวดคิ้วท้ายประโยค = คำถาม (ไหม)<small>ยังไม่ผ่านการทดสอบกับคนอื่น จึงปิดไว้</small></span></label>
        {onServerFile && <>
          <hr />
          <label className="field"><span>อัปโหลดวิดีโอให้เซิร์ฟเวอร์แยกจุด</span>
            <input type="file" accept="video/*" onChange={(e) => { const f = e.target.files?.[0]; if (f) { onServerFile(f); onClose(); } e.target.value = ""; }} />
            <small>ใช้เมื่อเบราว์เซอร์เปิดไฟล์วิดีโอไม่ได้ (codec ไม่รองรับ)</small>
          </label>
        </>}
        <hr />
        <p className="muted">{model?.version ? `โมเดล ${model.version} · ` : ""}MediaPipe Holistic 1.0.1 · <Link className="link" href="/classic">โหมดเดิม (สำหรับทดสอบ)</Link></p>
      </aside>
    </div></Portal>
  );
}
