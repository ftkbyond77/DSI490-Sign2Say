# โครงสร้างงานทีม

สำเนาโค้ดจาก GitHub แยกตาม branch ไม่มีการ merge ทับ production และไม่มีการ push

| เจ้าของ | branch | commit | จำนวนไฟล์ต่างจาก main |
|---|---|---|---|
| Jack | main | f08a56dbfbf6 | 0 |
| pitty | pitty | abafdedccb11 | 14 |
| Boom | boom | 5b31439e4f21 | 2298 |

- Jack/git_main: โค้ดฐานของแจ็คตามที่ผู้ใช้ระบุ
- pitty/git_pitty: โค้ด branch ของพิตตี้ แยกจากงาน pitty เดิม
- Boom/git_boom: โค้ด branch ของบุ๋มที่ขึ้น Git แล้ว ไม่รวมงาน local ที่ยังไม่ commit
- แต่ละโฟลเดอร์มี DIFF_FROM_MAIN.tsv และ BRANCH_SOURCE.json
- Boom/phase2.1combined: งานรวมและผลทดลองล่าสุดในเครื่อง
- Boom/phase2.2: symlink รองรับงาน test ที่กำลังรันและพาธเก่า ไม่ใช่โฟลเดอร์งานซ้ำ

main มีไฟล์ LFS ที่ดาวน์โหลดแล้ว ส่วน pitty/boom เก็บ LFS pointer ตาม Git; สำเนาเหล่านี้ใช้ดู source/diff ไม่ควรใช้รันโมเดลจนเตรียม artifacts จริง การเทียบกับ main ไม่ใช่หลักฐานว่าแต่ละคนเป็นผู้เขียนทุกบรรทัด
