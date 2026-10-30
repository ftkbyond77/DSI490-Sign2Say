import type { Metadata } from "next";
import Link from "next/link";
import SignStudio from "@/components/SignStudio";
import "./classic.css";

export const metadata: Metadata = { title: "ThaiSLM — โหมดเดิม (สำหรับทดสอบ)" };

// The original v7 studio with every control on one page — kept for debugging and comparisons (same engine as the new screens).
export default function Classic() {
  return (
    <main className="classic">
      <header className="top">
        <h1>ThaiSLM <span>โหมดเดิม (สำหรับทดสอบ) · <Link href="/">กลับหน้าหลัก</Link></span></h1>
        <p>ทุกตัวเลือกอยู่ในหน้าเดียว — ใช้เทียบผลและหาข้อผิดพลาด</p>
      </header>
      <SignStudio />
    </main>
  );
}
