"use client";
// v10: the words of the utterance being signed, while it is signed (WebSocket /api/v1/live). Confirmed words stay; partial words
// (faded) may still change. A word below the filter (settings → filter) is shown as [?] — never the guessed word — and a partial
// one as "…"; "no threshold" shows every reading.
import type { LiveWords as Words } from "@/lib/api";
import { shows } from "./Words";

export default function LiveWords({ w, busy = false, minP = 0.8 }: { w: Words; busy?: boolean; minP?: number }) {
  const empty = !w.confirmed.length && !w.partial.length;
  return (
    <div className="live-words" aria-live="polite" aria-label="คำที่อ่านได้ระหว่างทำท่า">
      {empty && <span className="lw wait">{busy ? "กำลังอ่านท่า…" : "กำลังดูท่า…"}</span>}
      {w.confirmed.map((x, i) => (
        <span key={`c${i}`} className={shows(x, minP) ? "lw ok" : "lw unsure"}>{shows(x, minP) ? x.word : "[?]"}</span>
      ))}
      {w.partial.map((x, i) => (
        <span key={`p${i}`} className="lw partial">{shows(x, minP) ? x.word : "…"}</span>
      ))}
    </div>
  );
}
