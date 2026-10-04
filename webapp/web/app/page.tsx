import SignStudio from "@/components/SignStudio";

export default function Page() {
  return (
    <main>
      <header className="top">
        <h1>ThaiSLM <span>ภาษามือไทย → ข้อความ → เสียงพูด</span></h1>
        <p>เปิดกล้อง ทำภาษามือเป็นประโยค แล้วระบบจะแปลเป็นภาษาไทยและพูดออกมา · คำที่ระบบไม่มั่นใจจะแสดงเป็น “คำ?” หรือ “[?] ≈ คำใกล้เคียง” — ไม่เดาเกินหลักฐาน</p>
      </header>
      <SignStudio />
    </main>
  );
}
