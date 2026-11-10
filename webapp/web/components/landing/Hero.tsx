"use client";
// Section 1 — name, promise and the two services in one breath, over a living teal gradient.
import dynamic from "next/dynamic";
import Link from "next/link";
import { motion, useScroll, useTransform } from "motion/react";
import { useEffect, useRef, useState } from "react";
import GlBoundary, { webglOk } from "./GlBoundary";
import { useInView, useReducedMotion } from "./useLanding";

const HeroBackdrop = dynamic(() => import("./HeroBackdrop"), { ssr: false });
const LiquidMark = dynamic(() => import("./LiquidMark"), { ssr: false, loading: () => null });

export default function Hero() {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end start"] });
  const y = useTransform(scrollYProgress, [0, 1], [0, 140]);
  const fade = useTransform(scrollYProgress, [0, 0.7], [1, 0]);
  const [vis, seen] = useInView<HTMLDivElement>("0px");
  const reduced = useReducedMotion();
  const [gl, setGl] = useState(false);
  useEffect(() => setGl(webglOk()), []);

  return (
    <section ref={ref} className="hero-lp" aria-labelledby="hero-title">
      <div className="hero-bg" ref={vis} aria-hidden>
        {gl && seen && <GlBoundary><HeroBackdrop animate={!reduced} /></GlBoundary>}
        <div className="hero-veil" />
      </div>

      <motion.div className="hero-copy" style={{ y, opacity: fade }}>
        <div className="mark3d" aria-hidden>{gl && <GlBoundary><LiquidMark speed={reduced ? 0 : 0.6} /></GlBoundary>}</div>
        <motion.span className="hero-pill" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15, duration: 0.7 }}>
          <i />Thai Sign Language
        </motion.span>
        <motion.h1 id="hero-title" initial={{ opacity: 0, y: 30, filter: "blur(10px)" }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }} transition={{ delay: 0.25, duration: 1, ease: [0.22, 1, 0.36, 1] }}>
          Sign<span>2Say</span>
        </motion.h1>
        <motion.p className="lead" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.45, duration: 0.8 }}>
          สะพานเชื่อมภาษามือไทยกับทุกคน
        </motion.p>
        <motion.p className="sub" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.55, duration: 0.8 }}>
          แปลภาษามือเป็นข้อความและเสียงผ่านกล้องในเบราว์เซอร์ หรือพิมพ์–พูดแล้วให้อวาตาร์ 3 มิติทำภาษามือให้ — ครบสองทางในหน้าเดียว ไม่ต้องติดตั้งอะไร
        </motion.p>
        <motion.ul className="svc-chips" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.65, duration: 0.8 }}>
          <li><b>ภาษามือ</b><span aria-hidden>→</span>ข้อความ + เสียงพูด</li>
          <li><b>ข้อความ / เสียง</b><span aria-hidden>→</span>อวาตาร์ภาษามือ</li>
        </motion.ul>
        <motion.div className="lp-actions" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.75, duration: 0.8 }}>
          <Link href="/sign" className="lp-btn primary">เริ่มแปลภาษามือ<span aria-hidden>→</span></Link>
          <a href="#demo" className="lp-btn ghost">ดูวิธีทำงาน</a>
        </motion.div>
      </motion.div>

      <a href="#demo" className="scroll-cue" aria-label="เลื่อนลง"><span /></a>
    </section>
  );
}
