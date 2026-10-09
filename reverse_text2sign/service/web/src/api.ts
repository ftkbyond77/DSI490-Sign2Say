// Thin typed client for reverse-api (same origin: nginx proxies /api).
import type { MotionPayload } from "./player";

export type Unit = {
  gloss: string; kind: "sign" | "fingerspell" | "text" | "phrase"; concept: string | null; source_words: string[]; match: string;
  role: string | null; letters: string[];
  clip: null | { clip_id: string; dataset: string; signer: string; tier: string; mirrored: boolean; consensus: number | null;
    n_signers: number; fidelity?: { wrist: number; elbow: number; fingertip: number | null } };
  manual: null | { dominant: string; two_handed: boolean; location: string; movement?: number; movement_kind: string };
  nmm: Record<string, unknown>; confidence: { translation?: number; motion?: number };
  timing: null | { start_s: number; end_s: number }; note: string | null;
};
export type Clause = { text: string; type: string; negated: boolean; moves: string[]; units: number[]; nmm?: Record<string, string> };
export type IR = {
  input: { text: string; lang: string; source: string; mt?: unknown }; thai: string; tokens: string[]; clauses: Clause[];
  units: Unit[]; warnings: string[];
  confidence: { translation: number; motion: number; coverage: number; signs: number; units: number; duration_s?: number;
    t_total_ms?: number; t_language_ms?: number; t_motion_ms?: number };
};
export type Result = { ir: IR; motion: MotionPayload };
export type AvatarEntry = { id: string | null; label: string; file: string; title: string; author: string; vrm_version: string;
  url: string; thumb: string | null; bytes: number; scale: number; optimised: boolean };
const av = (a: string | null) => (a ? `?avatar=${encodeURIComponent(a)}` : "");

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).detail ?? msg; } catch { /* not json */ }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

export const api = {
  health: () => fetch("/api/health").then((r) => j<{ ok: boolean; signs: number; english: boolean; semantic: boolean }>(r)),
  rest: (avatar: string | null = null) => fetch("/api/rest" + av(avatar)).then((r) => j<MotionPayload>(r)),
  avatars: () => fetch("/api/avatars").then((r) => j<{ avatars: AvatarEntry[] }>(r)),
  examples: () => fetch("/api/examples").then((r) => j<{ examples: string[] }>(r)),
  translate: (text: string, lang: string, speed: number, source = "text", avatar: string | null = null) =>
    fetch("/api/translate", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, lang, speed, source, avatar }) }).then((r) => j<Result>(r)),
  render: (ir: IR, speed: number, avatar: string | null = null) =>
    fetch("/api/render", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ir, speed, avatar }) })
      .then((r) => j<Result>(r)),
  lookup: (q: string) => fetch(`/api/lookup?q=${encodeURIComponent(q)}`).then((r) => j<{ items: { text: string; concept: string }[] }>(r)),
  asr: (blob: Blob, lang: string) => {
    const fd = new FormData();
    fd.append("audio", blob, "speech.webm");
    fd.append("lang", lang);
    return fetch("/api/asr", { method: "POST", body: fd }).then((r) => j<{ text: string; lang: string; ms: number }>(r));
  },
  vrma: (ir: IR, speed: number, avatar: string | null = null) =>
    fetch("/api/export/vrma", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ir, speed, avatar }) })
      .then(async (r) => { if (!r.ok) throw new Error(String(r.status)); return r.blob(); }),
};
