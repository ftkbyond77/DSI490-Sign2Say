import Link from "next/link";
import HelpButton from "@/components/HelpButton";
import { Ic, Logo } from "@/components/ui";

// v2-D Kiosk — หน้าแรก: pick the direction, then follow the three steps on screen.
export default function Home() {
  return (
    <main className="home">
      <header className="home-head rise"><Logo /><HelpButton /></header>
      <section className="hero rise d1">
        <h1>วันนี้อยากสื่อสาร<em>แบบไหน</em>?</h1>
        <p>เลือกทิศทางการแปล แล้วทำตาม 3 ขั้นตอนบนจอ</p>
      </section>
      <section className="choices">
        <Link href="/sign" className="choice sign rise d2">
          <div className="icons">
            <span className="iconbox" style={{ background: "rgba(255,255,255,.18)" }}><Ic n="hand-white" s={36} /></span>
            <Ic n="arrow-mint" s={28} />
            <span className="iconbox" style={{ background: "#fff" }}><Ic n="chat-teal" s={36} /></span>
          </div>
          <div><h2>ฉันใช้ภาษามือ</h2><p className="sub">ทำภาษามือหน้ากล้อง → ได้ข้อความและเสียงพูด</p></div>
          <ol className="steps">
            <li><span className="num">1</span>เปิดกล้อง ยืนให้เห็นถึงเอว</li>
            <li><span className="num">2</span>ยกมือแล้วทำภาษามือเป็นประโยค</li>
            <li><span className="num">3</span>ลดมือ ระบบอ่านออกเสียงให้</li>
          </ol>
          <span className="btn secondary">เริ่มทำภาษามือ <span className="arrow">→</span></span>
        </Link>
        <Link href="/text" className="choice text rise d3">
          <div className="icons">
            <span className="iconbox" style={{ background: "var(--rose-50)" }}><Ic n="mic-coral" s={36} /></span>
            <Ic n="arrow-coral" s={28} />
            <span className="iconbox" style={{ background: "var(--rose-400)" }}><Ic n="person-white" s={36} /></span>
          </div>
          <div><h2>ฉันพิมพ์หรือพูด</h2><p className="sub">ข้อความหรือเสียง → อวาตาร์ทำภาษามือให้</p></div>
          <ol className="steps">
            <li><span className="num">1</span>พิมพ์ หรือกดไมค์แล้วพูด</li>
            <li><span className="num">2</span>ตรวจคำที่ระบบได้ยิน</li>
            <li><span className="num">3</span>ดูอวาตาร์ทำภาษามือ ปรับความเร็วได้</li>
          </ol>
          <span className="btn coral">เริ่มพิมพ์ / พูด <span className="arrow">→</span></span>
        </Link>
      </section>
      <Link href="/chat" className="banner rise d4">
        <span className="iconbox"><Ic n="chats-teal" s={28} /></span>
        <div className="text"><h3>คุยกันสองคน</h3><p>คนหนึ่งใช้ภาษามือ อีกคนพิมพ์หรือพูด ในหน้าจอเดียว</p></div>
        <span className="btn">เปิดห้องสนทนา</span>
      </Link>
      <footer className="home-foot rise d5">
        <Link href="/vocab">คลังคำ</Link><Link href="/my-signs">ท่าของฉัน</Link><Link href="/classic">โหมดเดิม (สำหรับทดสอบ)</Link>
      </footer>
    </main>
  );
}
