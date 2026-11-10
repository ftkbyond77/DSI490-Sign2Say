"use client";
// Floating glass top bar: brand, section anchors, one call to action. Gets a stronger surface once the page has moved.
import Link from "next/link";
import { useEffect, useState } from "react";
import HelpButton from "../HelpButton";
import { Ic } from "../ui";

export default function Nav() {
  const [solid, setSolid] = useState(false);
  useEffect(() => {
    const on = () => setSolid(window.scrollY > 40);
    on();
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, []);
  return (
    <header className={`lp-nav ${solid ? "solid" : ""}`}>
      <nav aria-label="เมนูหลัก">
        <Link href="/" className="brand" aria-label="Sign2Say — หน้าแรก"><span className="mk"><Ic n="hand-white-22" s={20} /></span>Sign2Say</Link>
        <div className="links">
          <a href="#demo">วิธีทำงาน</a>
          <a href="#services">บริการ</a>
          <HelpButton className="linkbtn" label="วิธีใช้" />
        </div>
        <Link href="/sign" className="cta">เริ่มใช้งาน<span aria-hidden>→</span></Link>
      </nav>
    </header>
  );
}
