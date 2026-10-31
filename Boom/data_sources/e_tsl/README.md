# e-TSL catalogue

Source: http://e-tsl.eng.chula.ac.th/browsex.php

เปิด e_tsl_catalog.ipynb เรียก scrape.py จริง โดยอ่าน cache เป็นค่าเริ่มต้น words.csv/words.json คือรายการคำ sign_variants.csv คือท่าทางเลือกและลิงก์วิดีโอ root IDs และ URL รูปสัญลักษณ์

{
  "entries": 2049,
  "unique_word_strings": 2049,
  "sign_variants": 2196,
  "variants_with_video": 2192,
  "unique_video_urls": 2192,
  "exact_project_concept_matches": 1394,
  "exact_target_matches": [
    "ไป",
    "ภาษามือ",
    "วิ่ง",
    "กิน",
    "ดู",
    "ข้าว",
    "ไม่มี",
    "คุณ",
    "ยาย",
    "ญาติ",
    "ป้า"
  ],
  "notes": [
    "word overlap is not proof of duplicate videos",
    "No video downloaded or bank modified",
    "Root IDs are source symbols, not validated 53-dimensional attributes",
    "Alphabet/number entries and lexical variants require review",
    "No license inferred from public access"
  ]
}

robots.txt ตอบ 404 ไม่ใช่ใบอนุญาตใช้สื่อ ยังไม่ได้ยืนยัน licence สำหรับเผยแพร่ชุดข้อมูลใหม่ ไม่ได้เพิ่ม bank ไม่ได้ดาวน์โหลดวิดีโอทั้งหมด source root symbols ยังไม่ใช่ attributes ที่ผ่านการตรวจ label

## Label policy
ใช้ label=source_label=wrd_word_th ตรงเว็บทุกแถว target คือคำค้นไม่ใช่ label มี clip_manifest.csv/json สำหรับ 12 คลิปที่ดาวน์โหลดไว้ mapped_concept_id แยกและยังไม่อนุมัติ รัน source_labels.py เพื่อ validate และสร้างหน้ารีวิวใหม่
