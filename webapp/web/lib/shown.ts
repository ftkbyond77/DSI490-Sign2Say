import type { TranslateResult } from "./api";

/** The word filter (settings → filter): a sign's word is shown (and spoken) when the server accepted it, or when its P(correct) reaches the
 *  chosen threshold (0 = no threshold: every reading); otherwise it is [?] on screen and a pause "…" in the voice. Live and video alike. */
export const shows = (s: { status: string; p?: number }, minP: number) => s.status === "accepted" || (s.p ?? 0) >= minP;

/** The sentence to show for a result — and to speak, word for word: the confident sentence, else the tentative reading (marked), else nothing.
 *  When the filter lets through words the server did not accept, the server's sentence would not hold them: the words themselves are shown
 *  (marked tentative), [?] for the ones still below the threshold. (webapp/api/app.py `_spoken_text` is the same rule.) */
export function shownText(r: TranslateResult, minP = 0.8) {
  const signs = r.segments.filter((s) => s.status !== "transition");
  if (signs.some((s) => s.status !== "accepted" && shows(s, minP)))
    return { text: signs.map((s) => (shows(s, minP) ? s.word : "[?]")).join(" "), tentative: true };
  const s = r.sentence || "";
  return { text: s || r.language?.tentative_sentence_th || "", tentative: !s };
}

/** Is there anything to say? (a text of only [?] has no word.) */
export const sayable = (text: string) => text.replaceAll("[?]", "").replaceAll("…", "").trim().length > 0;
