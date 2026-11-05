"use client";
// Section 3 — one row per service (text on the left, its picture on the right), each explained on its own.
import Link from "next/link";
import Reveal from "./Reveal";

const DOTS: [number, number][] = [[100, 98], [100, 72], [84, 58], [78, 40], [96, 44], [96, 24], [110, 46], [112, 28], [122, 54], [130, 38], [136, 66], [150, 58]];

function SignViz() {
  return (
    <svg viewBox="0 0 200 110" aria-hidden>
      <rect x="64" y="14" width="96" height="94" rx="6" fill="none" stroke="#fff" strokeWidth="1.5" strokeDasharray="10 6" opacity=".7" />
      <path d="M100 98V72M100 72L84 58M84 58L78 40M100 72L96 44M96 44L96 24M100 72L110 46M110 46L112 28M100 72L122 54M122 54L130 38M100 72L136 66M136 66L150 58" stroke="#fff" strokeWidth="1.4" strokeLinecap="round" fill="none" opacity=".9" />
      {DOTS.map(([x, y]) => <circle key={`${x}-${y}`} cx={x} cy={y} r="3" fill="#fff" />)}
    </svg>
  );
}
function VoiceViz() {
  return (
    <>
      <div className="wave">{Array.from({ length: 30 }, (_, i) => <i key={i} style={{ animationDelay: `${(i % 9) * 0.11}s` }} />)}</div>
      <div className="svc-bubble"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="3" width="6" height="12" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" /></svg>สวัสดีครับ</div>
    </>
  );
}
function ChatViz() {
  return (
    <div className="chat-viz">
      <span className="b l"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M8 13V6a1.5 1.5 0 0 1 3 0v5M11 11V4.5a1.5 1.5 0 0 1 3 0V11M14 11V6a1.5 1.5 0 0 1 3 0v7c0 4-2.5 7-6 7-3 0-4.5-2-6-5l-1-2a1.5 1.5 0 0 1 2.5-1.5L8 13" /></svg>ท่าภาษามือ → ข้อความ</span>
      <span className="b r">พิมพ์ / พูด → อวาตาร์</span>
    </div>
  );
}

const ITEMS = [
  { no: "01", href: "/sign", cls: "c1", viz: <SignViz />, title: "ฉันใช้ภาษามือ", cta: "เริ่มทำภาษามือ",
    lede: "เปิดกล้องแล้วทำภาษามือเป็นประโยค ระบบแปลเป็นข้อความไทยและอ่านออกเสียงให้ทันที",
    points: ["เปิดกล้อง ยืนให้เห็นถึงเอว แล้วทำภาษามือเป็นประโยค", "ลดมือ — ระบบแปลเป็นข้อความและอ่านออกเสียงให้ทันที", "คำที่ไม่แน่ใจแสดงเป็น [?] แตะเพื่อบอกคำที่ถูก ระบบจะจำท่าของคุณ"] },
  { no: "02", href: "/text", cls: "c2", viz: <VoiceViz />, title: "ฉันพิมพ์หรือพูด", cta: "เริ่มพิมพ์ / พูด",
    lede: "พิมพ์ข้อความหรือพูดใส่ไมค์ แล้วดูอวาตาร์ 3 มิติทำภาษามือให้ — เหมาะกับการสื่อสารกับคนที่ใช้ภาษามือ",
    points: ["พิมพ์หรือพูดภาษาไทยก็ได้", "ปรับความเร็วของอวาตาร์ได้"] },
  { no: "03", href: "/chat", cls: "c3", viz: <ChatViz />, title: "คุยกันสองคน", cta: "เปิดห้องสนทนา",
    lede: "คนหนึ่งใช้ภาษามือ อีกคนพิมพ์หรือพูด ในหน้าจอเดียว คุยกันได้โดยไม่ต้องมีล่ามข้างตัว",
    points: ["ฝั่งภาษามือ: ท่าถูกแปลเป็นข้อความ", "ฝั่งพิมพ์/พูด: ข้อความถูกแสดงเป็นท่าโดยอวาตาร์"] },
];

export default function Services() {
  return (
    <section id="services" className="svc" aria-labelledby="svc-title">
      <div className="svc-wrap">
        <div className="svc-top">
          <Reveal><span className="eyebrow">Service</span></Reveal>
          <Reveal delay={0.05}><h2 id="svc-title"><em>Bidirectional</em><br />Translation</h2></Reveal>
        </div>
        {ITEMS.map((it) => (
          <article key={it.no} className="svc-row" aria-labelledby={`svc-${it.no}`}>
            <div className="svc-text">
              <Reveal><span className="svc-no">{it.no}</span></Reveal>
              <Reveal delay={0.05}><h3 id={`svc-${it.no}`}>{it.title}</h3></Reveal>
              <Reveal delay={0.1}><p className="lede">{it.lede}</p></Reveal>
              <ul className="ticks">
                {it.points.map((t, i) => <Reveal key={t} as="li" delay={0.14 + i * 0.06} y={16}><span className="tick" aria-hidden>✓</span>{t}</Reveal>)}
              </ul>
              <Reveal delay={0.2}><Link href={it.href} className="svc-cta">{it.cta}<span aria-hidden>→</span></Link></Reveal>
            </div>
            <Reveal delay={0.1}>
              <Link href={it.href} className={`svc-card ${it.cls}`} aria-label={`${it.title} — ${it.cta}`} tabIndex={-1}>
                <div className="viz" aria-hidden>{it.viz}</div>
                <span className="go" aria-hidden>→</span>
              </Link>
            </Reveal>
          </article>
        ))}
      </div>
    </section>
  );
}
