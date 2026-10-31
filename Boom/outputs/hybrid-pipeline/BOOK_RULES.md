# กฎที่ถอดจากภาพในหนังสือ (ชุดทดลอง)

เลขหน้าเล่มและเลขหน้า PDF ระบุแยกกัน ตรวจภาพจริงของคำหลัก 6 คำในเล่ม 1 และ 3 ไม่ได้อ้างว่าตรวจครบทุกคำในเล่ม 1–6 ค่าตัวเลขเป็นสมมติฐานทางวิศวกรรม ไม่ใช่ค่ามาตรฐานที่หนังสือระบุ การจัดหมวดเป็นการตีความภาพที่ยังไม่ได้ตรวจโดยผู้เชี่ยวชาญ

| concept / variant | ref_book_page (PDF) | num_hands | bimanual_type | primary_location | movement_trajectory | repetition | landmark_heuristic_rule |
|---|---|---|---|---|---|---|---|
| สวัสดี / wai | เล่ม 1 หน้า 1 (PDF 11) | 2 | symmetric | chest_torso | static | false | Y ข้อมือ [-0.5, 0.65] / shoulder width; dist(L,R) ≤ 0.5 |
| สวัสดี / salute | เล่ม 1 หน้า 1 (PDF 11) | 1 | single | head_face | linear | false | Y ข้อมือ [-0.9, 0.3] / shoulder width |
| ขอบคุณ / wai | เล่ม 1 หน้า 2 (PDF 12) | 2 | symmetric | chest_torso | static | false | Y ข้อมือ [-0.5, 0.65] / shoulder width; dist(L,R) ≤ 0.5 |
| ขอบคุณ / outward | เล่ม 1 หน้า 2 (PDF 12) | 1 | single | chest_torso | linear | false | Y ข้อมือ [-0.3, 0.9] / shoulder width |
| ขอโทษ / wai | เล่ม 1 หน้า 3 (PDF 13) | 2 | symmetric | chest_torso | static | false | Y ข้อมือ [-0.5, 0.65] / shoulder width; dist(L,R) ≤ 0.5 |
| ขอโทษ / rub_circle | เล่ม 1 หน้า 3 (PDF 13) | 2 | asymmetric | chest_torso | circling | true | Y ข้อมือ [-0.2, 1.1] / shoulder width; dist(L,R) ≤ 0.9; {"circling_min": 0.35, "speed_balance_max": 0.65} |
| ภาษามือ / two_circles | เล่ม 1 หน้า 7 (PDF 17) | 2 | symmetric | chest_torso | circling | true | Y ข้อมือ [-0.5, 1.1] / shoulder width; {"circling_min": 0.35} |
| เพื่อน / swap_contact | เล่ม 1 หน้า 84 (PDF 95) | 2 | asymmetric | chest_torso | alternating | null | Y ข้อมือ [0, 1.2] / shoulder width; dist(L,R) ≤ 0.95; {"alternating_min": 0.4} |
| ข้าว / finger_motion | เล่ม 3 หน้า 46 (PDF 59) | 1 | single | chest_torso | linear | true | Y ข้อมือ [-0.1, 1.1] / shoulder width |

`null` หมายถึงภาพไม่ยืนยันเรื่องทำซ้ำ จึงไม่เดาเป็น false

## หมายเหตุราย variant

- สวัสดี / wai: ภาพประนมมือ; ไม่มีลูกศรการเคลื่อนในภาพนี้ ไม่แปลว่าข้อมือต้องนิ่งตลอดคลิป
- สวัสดี / salute: มือเดียวเคลื่อนจากใกล้หน้าผากออกไป; ไม่บังคับข้างตามรูป
- ขอบคุณ / wai: ภาพประนมมือ; ไม่มีลูกศรการเคลื่อนในภาพนี้ ไม่แปลว่าข้อมือต้องนิ่งตลอดคลิป
- ขอบคุณ / outward: ภาพมือสองตำแหน่งแสดงลำดับการเคลื่อนของมือเดียว ไม่ใช่สองมือพร้อมกัน
- ขอโทษ / wai: ภาพประนมมือ; ไม่มีลูกศรการเคลื่อนในภาพนี้ ไม่แปลว่าข้อมือต้องนิ่งตลอดคลิป
- ขอโทษ / rub_circle: มือหนึ่งวนเหนือ/บนอีกมือ; ค่าพิกัดเป็น proxy ของภาพ ไม่ตรวจสัมผัสจริง
- ภาษามือ / two_circles: สองมือวนในพื้นที่หน้าลำตัวและมีลำดับ 1–3; ไม่บังคับว่ามือต้องชิดกันหรือมีความเร็วเท่ากันทุกเฟรม
- เพื่อน / swap_contact: ภาพ 1–2 สลับบทบาทนิ้วและมือ; ไม่ยืนยันว่าต้องทำซ้ำหลายรอบ การสลับ Y ข้อมือเป็นเพียง proxy ซึ่งอาจไม่เกิดชัด
- ข้าว / finger_motion: ภาพเน้นนิ้วขยับพร้อมลูกศรซ้ำ ไม่ใช่ท่ากินที่ปาก; ไม่ใช้วิถีข้อมือตัดสินการเคลื่อนนิ้ว
