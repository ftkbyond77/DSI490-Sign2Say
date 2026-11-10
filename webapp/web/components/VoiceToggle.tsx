"use client";
// Male / Female voice of the speech (speech v3 profiles) — top right next to the filter; remembered in this browser. Default: male.
import type { Voice } from "@/lib/api";
import type { SignSettings } from "@/lib/useSignEngine";

const VOICES: [Voice, string][] = [["male", "เสียงชาย"], ["female", "เสียงหญิง"]];

export default function VoiceToggle({ s, glass = false }: { s: SignSettings; glass?: boolean }) {
  const { opts, setOpts } = s;
  const cur = opts.voice ?? "male";
  return (
    <div className={`voice-toggle ${glass ? "glass" : ""}`} role="group" aria-label="Male / Female Voice" title="Male / Female Voice">
      {VOICES.map(([v, t]) => (
        <button key={v} type="button" className={cur === v ? "on" : ""} aria-pressed={cur === v} onClick={() => setOpts({ ...opts, voice: v })}>{t}</button>
      ))}
    </div>
  );
}
