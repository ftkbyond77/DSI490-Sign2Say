import type { LmFrame } from "./holistic";
import { userBank } from "./mysigns";

export type Segment = { t0: number; t1: number; status: "accepted" | "uncertain" | "unknown" | "transition"; word: string; p: number; candidates: string[];
  emb?: string; kind?: "sign" | "letter" };
export type TranslateResult = {
  uid?: number;                 // client-side identity of one result card (never from the server)
  title?: string;
  duration_s: number;
  sentence: string;
  segments: Segment[];
  language: { sentence_th?: string; tentative_sentence_th?: string; meaning_th?: string; evidence_gloss?: string; evidence_guard?: string; provider?: string };
  face: { question_yesno?: boolean; question_furrow?: boolean; question_wh?: boolean; negation?: boolean; affirmation?: boolean; emotion?: string;
    emotion_th?: string; intensity?: number; brow_furrow_rel?: number | null };
  timings_ms: Record<string, number>;
  vocab: string;
  model: string;
  mirrored_to_right_hand?: boolean;
  audio_wav_b64?: string | null;
  spoken?: "sentence" | "tentative";
  speculative?: boolean;
};
/** speak: "off" | "confident" (only a sentence the evidence supports) | "all" (also an uncertain reading, marked as such). */
export type Speak = "off" | "confident" | "all";
export type Alphabet = "en" | "th" | "both" | "off";
export type Options = { vocab: "conversation" | "full"; llm: boolean; speak: Speak; faceQuestion: boolean; alphabet: Alphabet };
export type Progress = { stage: string; p: number };
type OnProgress = (x: Progress) => void;

async function check(r: Response) {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export async function translateFrames(frames: LmFrame[], width: number, height: number, o: Options): Promise<TranslateResult> {
  const t0 = frames.length ? frames[0].t : 0;
  const body = { frames: frames.map((f) => ({ ...f, t: f.t - t0 })), width, height, vocab: o.vocab, llm: o.llm, speak: o.speak !== "off",
    speak_tentative: o.speak === "all" };
  return check(await fetch("/api/v1/translate", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }));
}

/** Read an NDJSON progress stream: {"stage","p"} lines, then {"result"} (or {"error"}). */
async function readStream(r: Response, onProgress: OnProgress): Promise<TranslateResult> {
  if (!r.ok || !r.body) throw new Error(`${r.status} ${await r.text()}`);
  const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (value) buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
      if (!line) continue;
      const m = JSON.parse(line);
      if (m.result) return m.result as TranslateResult;
      if (m.error) throw new Error(m.error);
      if (m.stage) onProgress({ stage: m.stage, p: m.p });
    }
    if (done) throw new Error("การเชื่อมต่อหลุดก่อนได้ผลลัพธ์");
  }
}

export async function translateFramesStream(frames: LmFrame[], width: number, height: number, o: Options, onProgress: OnProgress,
  faceBaseline?: number[] | null, signal?: AbortSignal): Promise<TranslateResult> {
  const t0 = frames.length ? frames[0].t : 0;
  const body = { frames: frames.map((f) => ({ ...f, t: f.t - t0 })), width, height, vocab: o.vocab, llm: o.llm, speak: o.speak !== "off",
    speak_tentative: o.speak === "all", face_question: o.faceQuestion, face_baseline: faceBaseline ?? null, user_bank: userBank(), alphabet: o.alphabet };
  onProgress({ stage: "upload", p: 0.01 });
  return readStream(await fetch("/api/v1/translate/stream", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal }), onProgress);
}

export async function translateVideoOnServerStream(file: File, o: Options, onProgress: OnProgress): Promise<TranslateResult> {
  const fd = new FormData();
  fd.append("file", file);
  const q = new URLSearchParams({ vocab: o.vocab, llm: String(o.llm), speak: String(o.speak !== "off"), speak_tentative: String(o.speak === "all"),
    face_question: String(o.faceQuestion), alphabet: o.alphabet });
  onProgress({ stage: "upload", p: 0.01 });
  return readStream(await fetch(`/api/v1/translate/video/stream?${q}`, { method: "POST", body: fd }), onProgress);
}

export async function translateVideoOnServer(file: File, o: Options): Promise<TranslateResult> {
  const fd = new FormData();
  fd.append("file", file);
  const q = new URLSearchParams({ vocab: o.vocab, llm: String(o.llm), speak: String(o.speak !== "off"), speak_tentative: String(o.speak === "all") });
  return check(await fetch(`/api/v1/translate/video?${q}`, { method: "POST", body: fd }));
}

export async function info() {
  return check(await fetch("/api/v1/info"));
}

/** Words a user can teach (personal sign memory): the conversation vocabulary + the English letters. */
export async function vocabList(): Promise<string[]> {
  const v = await check(await fetch("/api/v1/vocab"));
  return [...(v.concepts as string[]), ...Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i))];
}

export async function speak(text: string, emotion = "neutral", intensity = 0): Promise<ArrayBuffer> {
  const r = await fetch("/tts/v1/speak", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text, emotion, intensity }) });
  if (!r.ok) throw new Error(`tts ${r.status}`);
  return r.arrayBuffer();
}
