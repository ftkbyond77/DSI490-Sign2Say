"use client";
// Section 4 — closing statement. A draggable liquid-glass lens (liquid-glass-js) refracts the big type; it lives only while the section is on screen.
import Link from "next/link";
import { useEffect, useRef } from "react";
import HelpButton from "../HelpButton";
import { useInView } from "./useLanding";

export default function Closing() {
  const [wrap, near] = useInView<HTMLElement>("-20% 0px -20% 0px");
  const bg = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = bg.current, title = el?.querySelector(".lg-title");
    if (!near || !el || !title) return;
    type Lens = { destroy(): void; moveTo(x: number, y: number): unknown; lensX: number; lensY: number; glassEl: HTMLElement; params: { width: number; height: number } };
    let glass: Lens | null = null, dead = false, raf = 0, grabbed = false, idleAt = 0, cx = 0, cy = 0;
    const release = () => { if (grabbed) { grabbed = false; idleAt = performance.now(); } };
    const textBox = document.createRange(); textBox.selectNodeContents(title);
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const t0 = performance.now();
    // The lens drifts around the headline on its own, and the visitor can grab it — but it never leaves the headline's frame.
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      if (!glass) return;
      const f = textBox.getBoundingClientRect(), { width: w, height: h } = glass.params;   // the glyphs' box, not the (full-width) block, so the lens stays off the buttons
      const minX = f.left, maxX = f.right, minY = f.top, maxY = f.bottom;          // allowed area for the lens centre = the headline's box
      if (grabbed) {                                                              // the library moves it under the pointer; we only clamp
        cx = glass.lensX + w / 2; cy = glass.lensY + h / 2;
      } else {
        const t = (now - t0) / 1000, wander = now - idleAt > 900 && !reduced;
        const tx = minX + (maxX - minX) * (0.5 + 0.46 * Math.sin(t * 0.42));
        const ty = minY + (maxY - minY) * (0.5 + 0.42 * Math.sin(t * 0.29 + 1.3));
        const rx = minX + (maxX - minX) * 0.5, ry = minY + (maxY - minY) * 0.5;      // reduced motion: sit in the middle
        const [gx, gy] = wander ? [tx, ty] : reduced ? [rx, ry] : [cx || tx, cy || ty];
        if (!cx && !cy) { cx = gx; cy = gy; }
        const k = wander ? 0.07 : 0.2;                                              // ease back onto the path after a release
        cx += (gx - cx) * k; cy += (gy - cy) * k;
      }
      cx = Math.min(maxX, Math.max(minX, cx)); cy = Math.min(maxY, Math.max(minY, cy));
      const x = cx - w / 2, y = cy - h / 2;
      if (Math.abs(glass.lensX - x) > 0.4 || Math.abs(glass.lensY - y) > 0.4) glass.moveTo(x, y);
    };
    import("liquid-glass-js").then(({ default: LiquidGlass }) => {
      if (dead) return;
      const r = el.getBoundingClientRect(), w = Math.min(340, r.width * 0.62), h = Math.min(190, w * 0.58);
      glass = new LiquidGlass({
        background: el, width: w, height: h, radius: h / 2, scale: 42, depth: 38, curvature: 3, convexity: 1, chroma: 0.12,
        blur: 0, glow: 0.35, edge: 0.7, tint: 0.06, tintColor: "#ffffff", zIndex: 40,
      }) as unknown as Lens;
      const g = glass.glassEl;
      g.addEventListener("pointerdown", () => { grabbed = true; });
      g.addEventListener("pointerup", release); g.addEventListener("pointercancel", release);
      window.addEventListener("pointerup", release);   // safety: a release that lands outside the lens must still hand it back to the drift
      raf = requestAnimationFrame(tick);
    }).catch((e) => console.warn("[landing] liquid glass disabled:", e));
    return () => { dead = true; cancelAnimationFrame(raf); window.removeEventListener("pointerup", release); glass?.destroy(); };
  }, [near]);

  return (
    <section ref={wrap} className="closing" aria-labelledby="closing-title">
      <div className="lg-bg" ref={bg}>
        <p className="lg-eyebrow">Sign2Say</p>
        <h2 id="closing-title" className="lg-title">ทุกท่ามือ<br />ควรถูก<br />ได้ยิน</h2>
        <p className="lg-sub">ลากแก้วเพื่อส่องดูตัวอักษร</p>
        {/* static markup on purpose: the lens clones this block, so nothing here may animate in */}
        <div className="closing-ui">
          <p className="closing-lede">เริ่มได้ทันทีบนเบราว์เซอร์ — เปิดกล้อง หรือพิมพ์ข้อความ</p>
          <div className="lp-actions">
            <Link href="/sign" className="lp-btn light">เริ่มแปลภาษามือ<span aria-hidden>→</span></Link>
            <Link href="/text" className="lp-btn glass">พิมพ์ / พูดให้อวาตาร์</Link>
          </div>
        </div>
      </div>
      <footer className="lp-foot">
        <span>© Sign2Say · แปลภาษามือไทย</span>
        <nav aria-label="ลิงก์เพิ่มเติม">
          <Link href="/vocab">คลังคำ</Link><Link href="/my-signs">ท่าของฉัน</Link><Link href="/classic">โหมดเดิม (สำหรับทดสอบ)</Link>
          <HelpButton className="linkbtn" label="วิธีใช้" />
        </nav>
      </footer>
    </section>
  );
}
