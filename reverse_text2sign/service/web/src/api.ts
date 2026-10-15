// Thin typed client for reverse-api (same origin: nginx proxies /api).
import type { MotionPayload } from "./player";

export type Provenance = {
  origin?: string; dataset?: string; clip_id?: string; signer?: string; tier?: string; mirrored?: boolean; n_signers?: number;
  licence?: string; source?: string | null; qc?: Record<string, number>; reconstruction?: string[];
  fidelity?: { wrist?: number; elbow?: number; fingertip?: number | null }; assembled_from?: string; letters?: string[]; note?: string;
};
export type Lexeme = {
  concept_id: string; gloss: string; sense: string | null; variant_id: string | null; n_variants: number;
  variants: { key: string; variant_id: string | null; sense: string | null; dataset: string }[];
};
export type Unit = {
  gloss: string; kind: "sign" | "fingerspell" | "text" | "phrase"; concept: string | null; source_words: string[]; match: string;
  role: string | null; letters: string[];
  clip: null | { clip_id: string; dataset: string; signer: string; tier: string; mirrored: boolean; consensus: number | null;
    n_signers: number; fidelity?: { wrist: number; elbow: number; fingertip: number | null } };
  manual: null | { dominant: string; two_handed: boolean; location: string; movement?: number; movement_kind: string };
  nmm: Record<string, unknown>; confidence: { translation?: number; motion?: number };
  timing: null | { start_s: number; end_s: number }; note: string | null;
  lexeme?: Lexeme | null; provenance?: Provenance | null;
  sense?: { word: string; sense: string; label: string; concept: string | null; evidence: string[]; ambiguous: boolean;
    alternatives: string[] } | null;
};
export type NmmPlan = { type: string; source?: string; ref?: string; brows?: string; head?: string; applied?: boolean; reason?: string };
export type NmmMark = { type: string; frames: number[]; units: number[]; ref: string; source: string; note?: string };
export type Clause = { text: string; type: string; negated: boolean; moves: string[]; units: number[]; nmm?: NmmPlan[]; recorded?: boolean };
export type IR = {
  input: { text: string; lang: string; source: string; mt?: unknown; asr?: Record<string, unknown> }; thai: string; tokens: string[];
  clauses: Clause[]; units: Unit[]; warnings: string[]; version?: string;
  grammar?: { mode: string; rules?: string; phrasebook?: boolean }; versions?: Record<string, string | number | null>; nmm?: NmmMark[];
  confidence: { translation: number; motion: number; coverage: number; signs: number; units: number; duration_s?: number;
    t_total_ms?: number; t_language_ms?: number; t_motion_ms?: number };
};
export type Result = { ir: IR; motion: MotionPayload };
export type AvatarCheck = { ok: boolean; vrm: string; folded: string[]; face_direct: number; face_expression: number;
  face_missing: string[]; warnings: string[]; licence: Record<string, string | null> };
export type AvatarEntry = { id: string | null; label: string; file: string; title: string; author: string; vrm_version: string;
  url: string; thumb: string | null; bytes: number; scale: number; optimised: boolean; check?: AvatarCheck };
export type Suggestion = { original: string; corrected: string; gain_nats: number; sound: string[]; probability: number | null };
export type AsrResult = { text: string; raw: string; lang: string; ms: number; reliable: boolean; flags: string[];
  suggestions: Suggestion[]; suggested_text: string | null; model?: string; device?: string; session: string };
export type AsrPartial = { session: string; committed: string; tentative: string; final: boolean; ms: number };
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
  health: () => fetch("/api/health").then((r) => j<{ ok: boolean; signs: number; english: boolean;
    speech: { device?: string; th_model?: string } | null; versions?: Record<string, string> }>(r)),
  rest: (avatar: string | null = null) => fetch("/api/rest" + av(avatar)).then((r) => j<MotionPayload>(r)),
  avatars: () => fetch("/api/avatars").then((r) => j<{ avatars: AvatarEntry[] }>(r)),
  examples: () => fetch("/api/examples").then((r) => j<{ examples: string[] }>(r)),
  translate: (text: string, lang: string, speed: number, source = "text", avatar: string | null = null, grammar = "tsl51",
    asr: Record<string, unknown> | null = null) =>
    fetch("/api/translate", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, lang, speed, source, avatar, grammar, asr }) }).then((r) => j<Result>(r)),
  render: (ir: IR, speed: number, avatar: string | null = null) =>
    fetch("/api/render", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ir, speed, avatar }) })
      .then((r) => j<Result>(r)),
  lookup: (q: string) => fetch(`/api/lookup?q=${encodeURIComponent(q)}`).then((r) => j<{ items: { text: string; concept: string }[] }>(r)),
  asr: (blob: Blob, lang: string) => {
    const fd = new FormData();
    fd.append("audio", blob, "speech.webm");
    fd.append("lang", lang);
    return fetch("/api/asr", { method: "POST", body: fd }).then((r) => j<AsrResult>(r));
  },
  asrStream: (blob: Blob, lang: string, session: string, final: boolean) => {
    const fd = new FormData();
    fd.append("audio", blob, "speech.webm");
    fd.append("lang", lang);
    fd.append("session", session);
    fd.append("final", final ? "1" : "0");
    return fetch("/api/asr/stream", { method: "POST", body: fd }).then((r) => j<AsrResult & AsrPartial>(r));
  },
  avatarThumbnail: (id: string, blob: Blob) => {
    const fd = new FormData();
    fd.append("image", blob, "thumb.png");
    return fetch(`/api/avatars/${encodeURIComponent(id)}/thumbnail`, { method: "POST", body: fd }).then((r) => j<{ thumb: string }>(r));
  },
  vrma: (ir: IR, speed: number, avatar: string | null = null) =>
    fetch("/api/export/vrma", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ir, speed, avatar }) })
      .then(async (r) => { if (!r.ok) throw new Error(String(r.status)); return r.blob(); }),
};
