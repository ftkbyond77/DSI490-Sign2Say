import type { LmFrame } from "./holistic";
export type Framing = { ready: boolean; message: string; shoulderWidth: number | null; centerX: number | null; handsNearEdge: boolean };
/** UX heuristics, not training-distribution thresholds or recognition confidence. */
export function framing(f: LmFrame | null): Framing {
  const base: Framing = { ready: false, message: "จัดกล้องให้เห็นศีรษะและไหล่ทั้งสองข้าง", shoulderWidth: null, centerX: null, handsNearEdge: false };
  const valid = (p?: number[]) => !!p && p.length >= 4 && p.every(Number.isFinite) && p[3] > .5;
  if (!f || ![0,11,12].every(i => valid(f.pose[i]))) return base;
  const [nose, left, right] = [f.pose[0],f.pose[11],f.pose[12]];
  const centerX = (left[0]+right[0])/2, shoulderWidth = Math.abs(left[0]-right[0]);
  const handsNearEdge = [...f.lh,...f.rh].some(p => p[0]<.04 || p[0]>.96 || p[1]<.04 || p[1]>.96);
  const out = {...base, centerX, shoulderWidth, handsNearEdge};
  if (shoulderWidth > .46 || nose[1] < .12) return {...out, message: "ถอยจากกล้องเล็กน้อย ให้เหลือพื้นที่เหนือหัวและข้างลำตัว"};
  if (shoulderWidth < .18) return {...out, message: "ขยับใกล้กล้องเล็กน้อย ให้เห็นมือชัดขึ้น"};
  if (Math.abs(centerX-.5) > .12) return {...out, message: "ขยับลำตัวเข้ากลางกรอบ"};
  const sy = (left[1]+right[1])/2;
  if (sy < .28 || sy > .65 || nose[1] > .45) return {...out, message: "ปรับมุมกล้อง ให้ศีรษะอยู่ในวงและเห็นช่วงเอว"};
  if (handsNearEdge) return {...out, message: "มือใกล้ขอบภาพ ลองถอยเล็กน้อยเพื่อเพิ่มพื้นที่"};
  return {...out, ready: true, message: "ตำแหน่งลำตัวอยู่ในช่วงแนะนำ · ลองยกมือและกางแขนเช็กพื้นที่"};
}
export function drawFraming(ctx: CanvasRenderingContext2D, w: number, h: number, state: Framing) {
  ctx.save(); ctx.strokeStyle = state.ready ? "#4ade80" : "#fbbf24";
  ctx.lineWidth = Math.max(2,w/400); ctx.setLineDash([10,8]);
  // Coordinates in the same source-image space as the mirrored video and landmarks.
  ctx.strokeRect(.06*w,.06*h,.88*w,.88*h);
  ctx.beginPath(); ctx.ellipse(.5*w,.24*h,.10*w,.14*h,0,0,2*Math.PI); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(.30*w,.43*h); ctx.lineTo(.70*w,.43*h);
  ctx.moveTo(.30*w,.43*h);ctx.lineTo(.24*w,.85*h);
  ctx.moveTo(.70*w,.43*h);ctx.lineTo(.76*w,.85*h);ctx.stroke();ctx.restore();
}
