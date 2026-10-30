"use client";
// The avatar stage (v2-E “Avatar stage”): the three-vrm canvas, the sign being shown (“ข้าว · ท่าที่ 3 จาก 6”), loading state and the
// character picker. The canvas is created by useAvatar inside `hostRef`; the stage only frames it, whatever the character looks like.
import type { AvatarCtl } from "@/lib/t2s/useAvatar";
import AvatarPicker from "./AvatarPicker";

export const KIND_TH: Record<string, string> = { sign: "ท่ามือ", fingerspell: "สะกดนิ้ว", text: "ไม่มีท่า (แสดงข้อความ)", phrase: "ประโยคจากผู้ใช้ภาษามือจริง" };

export default function AvatarStage({ a, hostRef, cast = true, small = false, children }: {
  a: AvatarCtl; hostRef: React.RefObject<HTMLDivElement | null>; cast?: boolean; small?: boolean; children?: React.ReactNode;
}) {
  const units = a.current?.ir.units ?? [];
  const u = a.unit >= 0 ? units[a.unit] : null;
  const signIdx = u ? units.filter((x, i) => i <= a.unit).length : 0;
  return (
    <div className="stage" ref={hostRef}>
      {/* neutral silhouette while the signer loads — the signer can be any character (a realistic person later), so no cartoon here */}
      {!a.ready && <div className="sil" aria-hidden><i /><b /></div>}
      {!a.ready && !small && <div className="loading">{a.failed ? <span>{a.failed}</span> : <><div className="spinner" /><span>{a.loadText}</span></>}</div>}
      {u && !small && (
        <div className="cur-sign" aria-live="polite">
          <b key={a.unit} className={`k-${u.kind}`}>{u.gloss}</b>
          <small>ท่าที่ {signIdx} จาก {units.length}{u.kind !== "sign" ? ` · ${KIND_TH[u.kind]}` : ""}</small>
        </div>
      )}
      {cast && <div className={`pick-at ${small ? "small" : ""}`}><AvatarPicker a={a} small={small} /></div>}
      {children}
    </div>
  );
}
