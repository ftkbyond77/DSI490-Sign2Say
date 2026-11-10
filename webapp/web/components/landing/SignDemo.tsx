"use client";
// Section 2 — the product in motion: one looping clip that walks through Human → Extracting → Predicting → Speech.
// The clip is an illustration of the flow (made from a recorded signer; the sentence is scripted, not live model output).
import { useEffect, useRef } from "react";
import Reveal from "./Reveal";
import { useInView, useReducedMotion } from "./useLanding";

export default function SignDemo() {
  const [box, inView] = useInView<HTMLDivElement>("0px");
  const vid = useRef<HTMLVideoElement>(null);
  const reduced = useReducedMotion();

  // play while it is on screen, rest when it is not (saves battery; the clip loops forever)
  useEffect(() => {
    const v = vid.current;
    if (!v) return;
    if (reduced || !inView) v.pause();
    else v.play().catch(() => { /* autoplay blocked: the poster stays visible */ });
  }, [inView, reduced]);

  return (
    <section id="demo" className="demo" aria-labelledby="demo-title">
      <div className="demo-head">
        <Reveal><span className="eyebrow">ภาษามือ → ข้อความ</span></Reveal>
        <Reveal delay={0.06}><h2 id="demo-title">รู้จำภาษามือแบบ <em>Real-time</em></h2></Reveal>
        <Reveal delay={0.1}><p className="demo-sub">ทำท่าหน้ากล้อง ระบบตรวจจับมือและติดตามจุดข้อต่อทุกเฟรม แล้วเรียบเรียงเป็นประโยคไทยพร้อมเสียงพูด</p></Reveal>
      </div>
      <Reveal delay={0.12} className="demo-stage">
        <div className="demo-clip" ref={box}>
          <video ref={vid} muted loop playsInline autoPlay preload="auto" poster="/landing/hero-poster.jpg"
            aria-label="ตัวอย่างการทำงาน: คนทำภาษามือหน้ากล้อง ระบบแปลเป็นข้อความไทยทีละคำ แล้วอ่านออกเสียง">
            <source src="/landing/hero.mp4" type="video/mp4" />
          </video>
        </div>
        <p className="demo-note">ภาพจำลองเพื่ออธิบายการทำงาน</p>
      </Reveal>
    </section>
  );
}
