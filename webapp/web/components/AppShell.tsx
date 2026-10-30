"use client";
// v2-C: the teal side menu (desktop) / top bar (phone) around the sign-language screens.
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Ic, Logo } from "./ui";
import { useMySignCount } from "@/lib/useSignEngine";

const NAV = [
  { href: "/sign", label: "แปลภาษามือ", icon: "hand-white-22", on: "nav-hand-on" },
  { href: "/text", label: "แปลข้อความเป็นท่า", icon: "nav-text" },
  { href: "/chat", label: "บทสนทนา", icon: "nav-chat" },
  { href: "/my-signs", label: "ท่าของฉัน", icon: "nav-star", badge: true },
  { href: "/vocab", label: "คลังคำ", icon: "nav-book" },
];

function Nav({ compact = false }: { compact?: boolean }) {
  const path = usePathname();
  const n = useMySignCount();
  return (
    <>
      {NAV.map((x) => {
        const on = path === x.href;
        return (
          <Link key={x.href} href={x.href} className={`nav ${on ? "on" : ""}`} aria-current={on ? "page" : undefined} title={compact ? x.label : undefined}>
            {x.on ? <><Ic n={x.icon} className="ic-off" /><Ic n={x.on} className="ic-on" /></> : <Ic n={x.icon} />}
            <span className="label">{x.label}</span>
            {x.badge && n > 0 && <span className="badge">{n}</span>}
          </Link>
        );
      })}
    </>
  );
}

const TIPS: Record<string, string> = {
  "/text": "พิมพ์ทีละประโยคสั้น ๆ แตะคำในลำดับภาษามือเพื่อดูท่านั้นซ้ำ และลดความเร็วได้ถ้าดูไม่ทัน",
  "/chat": "ผลัดกันพูดทีละคน ผู้ใช้ภาษามือลดมือเมื่อจบประโยค อีกคนรอให้อวาตาร์ทำท่าจบก่อนตอบ",
  "/my-signs": "สอนคำละ 2–3 ครั้ง ระบบจะจำท่าของคุณได้แม่นขึ้น",
};
function Tip() {
  const path = usePathname();
  return <div className="tip"><b>เคล็ดลับ</b>{TIPS[path] ?? "ให้เห็นตัวถึงเอว มือทั้งสองข้างอยู่ในกรอบ และเว้นจังหวะสั้น ๆ ระหว่างคำ"}</div>;
}

/** tone: the content background (teal for the sign side, rose for text → sign); fill: the content is exactly one screen high (chat). */
export default function AppShell({ children, tone = "teal", fill = false }: { children: React.ReactNode; tone?: "teal" | "rose"; fill?: boolean }) {
  // the side menu folds to icons only (remembered in this browser, the same on every page)
  // the state is <html data-sidebar="folded"> (set before paint by app/layout.tsx); React only mirrors it for the button's label
  const [folded, setFolded] = useState(false);
  useEffect(() => setFolded(document.documentElement.dataset.sidebar === "folded"), []);
  const toggle = () => {
    const next = !folded;
    if (next) document.documentElement.dataset.sidebar = "folded"; else delete document.documentElement.dataset.sidebar;
    try { localStorage.setItem("thaislm.sidebar", next ? "folded" : "open"); } catch { /* storage blocked */ }
    setFolded(next);
  };
  return (
    <div className={`shell ${tone}`}>
      <aside className="sidebar">
        <div className="side-head">
          <Logo variant="on-teal" />
          <button className="fold" onClick={toggle} aria-expanded={!folded} aria-label={folded ? "ขยายเมนู" : "ย่อเมนู"} title={folded ? "ขยายเมนู" : "ย่อเมนู"}>
            {/* side-panel icon: a window with its left panel; the small arrow says which way the menu will go */}
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <rect x="3" y="4" width="18" height="16" rx="3" /><path d="M9 4v16" />
              <path className="arr" d={folded ? "M13.5 10l2 2-2 2" : "M15.5 10l-2 2 2 2"} />
            </svg>
          </button>
        </div>
        <Nav compact={folded} />
        <div className="spacer" />
        <Tip />
      </aside>
      <header className="mobile-bar">
        <Logo variant="on-teal" />
        <nav><Nav compact /></nav>
      </header>
      <main className={`content ${fill ? "fill" : ""}`}>{children}</main>
    </div>
  );
}

/** Greeting for the title row (the design's “สวัสดี [ชื่อผู้ใช้]” — there are no accounts, so the time of day). */
export function greeting() {
  const h = new Date().getHours();
  return h < 12 ? "สวัสดีตอนเช้า" : h < 17 ? "สวัสดีตอนบ่าย" : "สวัสดีตอนเย็น";
}
