// Small building blocks shared by every screen (Figma components: Logo, Button, WordChip, status pill, progress).
import Link from "next/link";
import type { Progress } from "@/lib/api";
import { STAGE_TH } from "@/lib/useSignEngine";

/** An icon exported from the Figma file (public/ui/*.svg), at its design size. */
export function Ic({ n, s = 20, className }: { n: string; s?: number; className?: string }) {
  return <img src={`/ui/${n}.svg`} width={s} height={s} alt="" aria-hidden className={className} />;
}

export function Logo({ variant = "default", href = "/" }: { variant?: "default" | "on-teal" | "glass"; href?: string }) {
  const icon = variant === "on-teal" ? "hand-teal" : variant === "glass" ? "hand-white-22" : "logo-hand";
  return (
    <Link href={href} className={`logo ${variant}`} aria-label="ThaiSLM — หน้าแรก">
      <span className="mark"><Ic n={icon} s={variant === "default" ? 24 : 22} /></span>
      <span className="name">ThaiSLM</span>
    </Link>
  );
}

export function Typing({ text, rose = false }: { text: string; rose?: boolean }) {
  return <div className={`typing ${rose ? "rose" : ""}`} role="status"><span><i /><i /><i /></span>{text}</div>;
}

export function ProgressBar({ progress, rose = false, label }: { progress: Progress | { p: number; stage?: string } | null; rose?: boolean; label?: string }) {
  if (!progress) return null;
  const pct = Math.round(progress.p * 100);
  return (
    <div className={`pbar ${rose ? "rose" : ""}`} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
      <div className="track"><div className="fill" style={{ width: `${Math.max(3, pct)}%` }} /></div>
      <small>{label ?? `${STAGE_TH[progress.stage ?? ""] ?? progress.stage ?? ""} · ${pct}%`}</small>
    </div>
  );
}

/** Thai sentence with the uncertain part (words written “คำ?” by the server) in amber, as in the design (“ปู่ เหนื่อย?”). */
export function Sentence({ text }: { text: string }) {
  // unsure parts: “[?]”, “[?] ≈ คำ”, or a word ending in “?”
  const parts = text.split(/(\[\?\](?:\s*≈\s*\S+)?|[^\s[\]]+\?(?=\s|$))/g);
  return <>{parts.map((p, i) => (i % 2 ? <span key={i} className="q">{p}</span> : p))}</>;
}

export function SpeakerIcon() {
  return <Ic n="speaker-teal" s={20} />;
}
