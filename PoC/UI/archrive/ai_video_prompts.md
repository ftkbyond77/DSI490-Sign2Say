# Prompts สำหรับสร้างคลิปผู้ทำภาษามือ (สังเคราะห์)

ใช้กับ Veo / Kling / Runway / Sora หรือ Adobe Firefly Video
สร้าง **แยก 3 คลิป (ประโยคละคลิป)** คุมท่ามือได้ง่ายกว่าคลิปยาว
บันทึกเป็น `assets/clip_1.mp4`, `clip_2.mp4`, `clip_3.mp4` แล้วบอกผมเพื่อเปลี่ยน CLIPS ใน hero.html

## Prompt หลัก (ใช้ร่วมกันทั้ง 3 คลิป)

> Static front-facing medium shot, chest-up, of a young East Asian woman aged 24–26 (Chinese ancestry),
> shoulder-length dark hair with a small hair clip on one side, soft natural makeup, plain light-teal top,
> seated in front of a clean bright minimal background, soft even daylight. She is communicating in sign
> language, both hands clearly visible inside the frame, hands move slowly and deliberately, face relaxed
> and expressive. No on-screen text, captions, logos or UI overlays. Camera locked off, no zoom, no cut, 1080p, 25 fps, about 4 seconds, neutral resting hands
> at the start and end.

## เพิ่มท้าย prompt ตามประโยค (ตัวอย่าง — เลือกท่าที่ผู้รู้ภาษามือไทยยืนยันแล้ว)

1. `She signs a greeting: right hand raised to the forehead then outward, then both hands to the chest.`
2. `She signs a request: one fist taps an open flat palm twice, then points to her chest.`
3. `She signs thanks: flat hand from the chin moving outward twice, then a thumbs-up with both hands.`

## ข้อควรระวัง
- ท่าภาษามือจาก AI มักผิดหลักภาษามือไทย ให้ผู้รู้ตรวจก่อนใช้จริง
- ตรวจนิ้วมือ (จำนวนนิ้ว/การบิดเบี้ยว) ทุกเฟรมสำคัญ ถ้าเพี้ยนให้สร้างใหม่
- ปิดเสียง ผมจะใส่เสียงพูดไทยจาก TTS ของโปรเจกต์ให้ทีหลังได้
