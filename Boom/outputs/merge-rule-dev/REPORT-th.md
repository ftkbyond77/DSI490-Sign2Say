# กฎ Merge บน Dev clips — ยังไม่พร้อม deploy

## ขอบเขต

ใช้เพียง clip1_browser.json.gz และ clip2_browser.json.gz (landmarks ที่มีอยู่) ไม่อ่านวิดีโอ clip3 ไม่สกัด ไม่ทำ inference และไม่ปรับค่าจาก clip3 ไม่แก้ production model/decoder/threshold และไม่ rerank สวัสดีคลิป 1 ด้วยกฎคำเฉพาะ

## กฎทดลอง v0

- ช่วงเรียงเวลา ไม่ซ้อนกัน และ gap < 0.2 วินาที (0.2 พอดีไม่ผ่าน)
- Top-5 มีคำซ้ำกันอย่างน้อย 2 คำ ไม่เช็กชื่อ “ภาษามือ” เป็นพิเศษ
- ตรวจ motion รอบกึ่งกลางรอยต่อ ±0.16 วินาที ใช้ raw observations ก่อนเติมจุดและก่อน reflection ปรับแกน x ตาม aspect แล้ว normalize ตามไหล่
- มือที่ผ่านต้องมี adjacent-frame observations ≥60%, median speed ≥0.15 shoulder-width/s, max adjacent step ≤0.35 shoulder-width, boundary/context speed ratio อยู่ระหว่าง 0.25–4.0
- v0 ต้องมีมือผ่านอย่างน้อย 1 ข้าง และช่วงรวมยาวไม่เกิน 3.2 วินาที
- เป็นกฎเสนอให้ merge คู่ที่ติดกัน ยังไม่ใช้ greedy chaining ทั้งประโยค และไม่สร้าง p ใหม่โดยอาศัย calibration เก่าที่ไม่ได้ประเมินกับกฎนี้

## ผลจริง 3 คู่จาก 2 คลิป

|คู่|Gap|คำที่ซ้ำ|v0|หมายเหตุ|
|---|---:|---|---|---|
|clip1 ช่วง 1–2|0.00s|ข้าว, ภาษามือ|รวม|น่าสงสัย: ขวากระโดดสูงสุด 0.812 shoulder-width และ boundary speed ratio 6.50 แต่ซ้ายผ่าน จึงไม่ควรยอมเพียงเพราะคำหลังรวมดูดี|
|clip2 ช่วง 1–2|0.16s|ไม่มี|ไม่รวม|ปฏิเสธเพราะ candidate overlap ไม่ผ่าน แม้ motion ผ่าน เป็น negative candidate; ยังไม่มี time-aligned gloss ยืนยัน|
|clip2 ช่วง 2–3|0.00s|ขอโทษ, ภาษามือ|รวม|มือทั้งสองผ่าน และ temporal ensemble หลังรวมให้ภาษามือเป็นอันดับ 1 support 0.3383|

ทดสอบ sensitivity 9 combinations ของ min speed {0.1,0.15,0.2} และ min boundary ratio {0.15,0.25,0.4}: ผลตัดสินของแต่ละคู่เหมือนเดิมทุกค่า หมายถึงไม่ไวต่อสองค่าช่วงแคบนี้เท่านั้น ไม่ได้แปลว่าทนต่อคนใหม่หรือ boundary ใหม่

## Candidate v1 หลังเห็น dev: ต้องผ่านทั้งสองมือ

เพิ่ม min_continuous_hands=2: clip1 1–2 ไม่รวม, clip2 1–2 ไม่รวม, clip2 2–3 รวม เหตุผลคือ abstain เมื่อมีข้อมูลขัดแย้ง/ไม่พอจากมืออีกข้าง

นี่เป็นการปรับจาก dev ที่เห็นแล้ว ไม่ใช่ผลยืนยันอิสระ และอาจพลาดการรวมท่ามือเดียวที่ถูกต้อง จึงเก็บเป็น candidate ไม่เปิดใช้งานจริง ไม่ใช้ min_continuous_hands=2 เป็นข้อสรุปว่าทุกคำต้องใช้สองมือ

## Negative controls

Unit tests 8 ข้อผ่าน: continuous positive, gap เท่ากับ 0.2, candidate disjoint, pause, coordinate jump, hand missing, time overlap, combined duration เกินเกณฑ์

7 กรณีปฏิเสธนี้เป็นข้อมูลจำลองสำหรับตรวจกลไก ไม่ใช่ 7 คลิปภาษามือจริง กรณีคำคนละคำแต่ motion ต่อเนื่องและ candidate ซ้ำกันยังสามารถผ่านกฎได้ เพราะ motion continuity ไม่เท่ากับ semantic identity จึงยังรับประกันว่าไม่รวมคนละท่าไม่ได้

## เงื่อนไขก่อนปลดล็อก held-out

ต้องยืนยันหน่วยท่าจริงและขอบเขตบน dev โดยผู้รู้ภาษามือ โดยเฉพาะ 1.44–4.48 s ว่าเป็นคำเดียว หรือ คุยกัน + ภาษามือ สองท่า ถ้าเป็นสองท่า การที่ merge แล้วภาษามือขึ้นอันดับ 1 จะเป็นการกลืนอีกคำ ไม่ใช่ผลดี

เพิ่ม negative dev ที่เป็นคนละคำจริงต่อกัน ทั้งแบบมี/ไม่มี pause และแบบ top-5 ซ้ำกัน พร้อมกรณีท่ามือเดียว แล้ววัด false merge กับ missed merge ล็อกกฎก่อนเปิด clip3 ซึ่งขณะนี้ยังไม่มีผล inference
