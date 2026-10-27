# สังเกตการณ์ replay live landmarks: v8.1 เทียบ cache v7.2

การ replay นี้เป็นการสังเกตผลของโมเดล ไม่ใช่การประเมินความถูกต้อง: ไม่มีเฉลยครบ จึงไม่คำนวณ WER หรือกล่าวอ้าง accuracy. ตัวอย่างทั้งหมดเป็นผู้ลงนามคนเดียวตามข้อมูลที่ผู้ใช้ให้มา.

## วิธีและขอบเขต

- ใช้ cache จริง 53 ไฟล์ (3169 frames), source modules จาก commit `99e99eee8dd6eda84c21d4e289721153dd80aa3a` และโมเดล `Jack-model` ใน `_inputs/` เท่านั้น (config v8.1: `hand_w=1.2`, `accept_min=0.8`); manifest และ SHA-256 ของไฟล์ staged 101 รายการตรวจผ่าน.
- แปลงเฟรมด้วย `SignPipeline.landmarks_from_browser(frames, (height, width))` ของ v8.1 ซึ่ง delegate ไป `modules.mediapipe_pose.from_browser` แบบเดียวกับ `_frames_job` ของ API; ไม่อ่าน/ถอดรหัสวิดีโอหรือภาพใบหน้า.
- ใช้ ONNX Runtime CPU (`onnxruntime 1.23.2`), conversation vocab, face analysis เปิด, alphabet จาก cache, LLM ปิดเพื่อไม่เรียก network; warm-up แบบสังเคราะห์ก่อนจับเวลา. ORT threads=4.
- Python ที่ใช้จริง: 3.11.8 จาก host environment. พาธ `Boom/venv-sign2say/bin/python` ที่ระบุไม่มีอยู่ใน worktree นี้; จึงใช้ Python 3.11.8 ที่มีอยู่แทน. รันด้วย `python3 -B Boom/v81_replay/replay_v81.py --threads 4` และ session-local `onnxruntime==1.23.2` ใน `PYTHONPATH` (ใน venv ที่มีแพ็กเกจนี้แล้ว ไม่ต้องเติม target path).
- Cache ไม่มี request option `llm`, `face_question`, `face_baseline`, `spell_only` หรือ user-bank embeddings (มีเพียง `n_user_signs`; ทั้ง 53 ไฟล์เป็น 0). สมมติ `face_question=false`, `face_baseline=null`, `spell_only=false`, ไม่มี personal signs; ปิด LLM แบบตั้งใจเพื่อไม่ใช้บริการภายนอก. ไม่ประมาณข้อมูล landmark; ใช้ frame fields ที่ cache มี และ adapter จะเติมศูนย์ให้ fields ที่ว่างตาม implementation.
- latency v8.1 วัด wall-clock ต่อรายการตั้งแต่แปลง browser frames จน `run_landmarks` คืนผล ไม่รวมสร้าง pipeline/warm-up/เขียน CSV; ค่า v7.2 นำจาก `result.timings_ms.total_ms` ใน cache.

## กลุ่มเวลา 20261027 07:49–07:53

มี 14 ไฟล์ในช่วงเวลาที่กำหนด; 13 ไฟล์ผ่านเงื่อนไขเพิ่มว่า segment หนึ่งใน 2 segment ที่ไม่ใช่ transition ตอนท้ายของผล v7.2 มีคำ `ขอบคุณ` หรือ `กรุงเทพ`. ค่า `result.sentence` เก่าที่ตรงข้อความ `สวัสดี ขอบคุณ` มี 0/14 ไฟล์—ดังนั้นชื่อวลีนี้ไม่ถือเป็น label จริง.

| กลุ่ม | คำ | v7.2 Top-1 อย่างน้อย 1 segment | v7.2 Accepted | v8.1 Top-1 อย่างน้อย 1 segment | v8.1 Accepted |
|---|---|---:|---:|---:|---:|
| ทั้งหมดในช่วงเวลา (n=14) | สวัสดี | 14/14 | 8/14 | 14/14 | 12/14 |
| ทั้งหมดในช่วงเวลา (n=14) | ขอบคุณ | 4/14 | 0/14 | 6/14 | 0/14 |
| ช่วงเวลา + clue ท้าย (n=13) | สวัสดี | 13/13 | 7/13 | 13/13 | 11/13 |
| ช่วงเวลา + clue ท้าย (n=13) | ขอบคุณ | 4/13 | 0/13 | 6/13 | 0/13 |

Top-1 นับจาก candidate อันดับแรกของแต่ละ segment; Accepted นับเฉพาะ segment ที่ status เป็น `accepted` และคำอันดับแรกตรงคำเป้าหมาย. นับเป็นจำนวนไฟล์ที่มีอย่างน้อยหนึ่ง segment เช่นนั้น ไม่ใช่จำนวน segment และไม่เท่ากับความถูกต้องเมื่อไม่มีเฉลย.
ใน Top-5: `score` คือ candidate score จาก encoder; `support` คือ score หลัง fusion ที่เรียง candidate ใน v8.1 (v7.2 cache ไม่มี `support`).

### รายชื่อและผลช่วงท้าย

| ไฟล์ | v7.2 sentence | v7.2 สอง segment ท้าย (word/status/p; Top-5) | v8.1 sentence | v8.1 สอง segment ท้าย (word/status/p; Top-5) |
|---|---|---|---|---|
| * `20261027_074947_285.json.gz` | สวัสดี | ขอบคุณ/uncertain/p=0.557 [Top5: ขอบคุณ(score=0.690), กรุงเทพ(score=0.543), วัด(score=0.463), ขอโทษ(score=0.430), ถ้า(score=0.342)]<br>เกิด/uncertain/p=0.469 [Top5: เกิด(score=0.547), ไป(score=0.424), เที่ยว(score=0.423), กิน(score=0.420), กรุงเทพ(score=0.395)] | สวัสดี | ฉัน/uncertain/p=0.780 [Top5: ฉัน(score=0.745, support=1.358), กิน(score=0.703, support=1.243), ข้าว(score=0.749, support=1.221), ขนมปัง(score=0.622, support=1.124), เกิด(score=0.772, support=1.089)]<br>ขอบคุณ/uncertain/p=0.747 [Top5: ขอบคุณ(score=0.543, support=0.996), กรุงเทพ(score=0.335, support=0.641), เกิด(score=0.221, support=0.624), อย่า(score=0.351, support=0.617), ประตู(score=0.356, support=0.540)] |
| * `20261027_075012_046.json.gz` | — | ข้าว/uncertain/p=0.270 [Top5: ข้าว(score=0.549), ไป(score=0.512), กิน(score=0.486), น้อง(score=0.480), เกิด(score=0.474)]<br>ขอบคุณ/uncertain/p=0.130 [Top5: ขอบคุณ(score=0.551), กรุงเทพ(score=0.517), เกิด(score=0.398), ถ้า(score=0.408), วัด(score=0.379)] | สวัสดี | สวัสดี/accepted/p=0.931 [Top5: สวัสดี(score=0.685, support=1.091), พรุ่งนี้(score=0.305, support=0.593), โกรธ(score=0.332, support=0.564), ก่อน(score=0.328, support=0.546), ลม(score=0.310, support=0.538)]<br>ขอบคุณ/uncertain/p=0.649 [Top5: ขอบคุณ(score=0.520, support=0.907), กรุงเทพ(score=0.329, support=0.641), เกิด(score=0.328, support=0.583), ครอบครัว(score=0.145, support=0.536), สวัสดี(score=0.221, support=0.495)] |
| * `20261027_075021_560.json.gz` | สวัสดี | ขอบคุณ/unknown/p=0.088 [Top5: ขอบคุณ(score=0.546), กรุงเทพ(score=0.513), วิ่ง(score=0.366), ภาษามือ(score=0.434), ประตู(score=0.401)]<br>เกิด/uncertain/p=0.150 [Top5: เกิด(score=0.425), กรุงเทพ(score=0.375), แต่งงาน(score=0.325), วัด(score=0.295), สบายดี(score=0.284)] | สวัสดี | สวัสดี/accepted/p=0.924 [Top5: สวัสดี(score=0.676, support=1.054), พรุ่งนี้(score=0.281, support=0.562), โกรธ(score=0.338, support=0.554), ก่อน(score=0.348, support=0.551), ฉัน(score=0.329, support=0.538)]<br>ขอบคุณ/uncertain/p=0.593 [Top5: ขอบคุณ(score=0.450, support=0.831), กรุงเทพ(score=0.379, support=0.656), สวัสดี(score=0.204, support=0.545), อย่า(score=0.269, support=0.528), ครอบครัว(score=0.109, support=0.504)] |
| * `20261027_075035_536.json.gz` | สวัสดี | เกิด/uncertain/p=0.240 [Top5: เกิด(score=0.578), กิน(score=0.563), ข้าว(score=0.572), ไป(score=0.492), เที่ยว(score=0.514)]<br>กรุงเทพ/uncertain/p=0.214 [Top5: กรุงเทพ(score=0.540), กิน(score=0.432), วัด(score=0.416), สวัสดี(score=0.416), ครอบครัว(score=0.391)] | สวัสดี | สวัสดี/accepted/p=0.932 [Top5: สวัสดี(score=0.703, support=1.068), ก่อน(score=0.392, support=0.641), พรุ่งนี้(score=0.322, support=0.573), ทำไม(score=0.282, support=0.569), ไป(score=0.330, support=0.517)]<br>ขอบคุณ/uncertain/p=0.215 [Top5: ขอบคุณ(score=0.381, support=0.811), กรุงเทพ(score=0.436, support=0.807), สวัสดี(score=0.275, support=0.631), ครอบครัว(score=0.218, support=0.608), ปรบมือ(score=0.276, support=0.494)] |
| * `20261027_075047_839.json.gz` | — | สวัสดี/uncertain/p=0.741 [Top5: สวัสดี(score=0.726), กิน(score=0.628), ไป(score=0.571), ข้าว(score=0.669), น้อง(score=0.563)]<br>กรุงเทพ/uncertain/p=0.338 [Top5: กรุงเทพ(score=0.595), วัด(score=0.454), ขอโทษ(score=0.468), ขอบคุณ(score=0.442), ครอบครัว(score=0.480)] | สวัสดี | สวัสดี/accepted/p=0.932 [Top5: สวัสดี(score=0.710, support=1.110), พรุ่งนี้(score=0.381, support=0.696), ไป(score=0.439, support=0.660), กิน(score=0.374, support=0.618), น้อง(score=0.428, support=0.595)]<br>กรุงเทพ/uncertain/p=0.177 [Top5: กรุงเทพ(score=0.508, support=0.847), ครอบครัว(score=0.367, support=0.733), ขอบคุณ(score=0.248, support=0.722), สวัสดี(score=0.319, support=0.636), วัด(score=0.380, support=0.568)] |
| * `20261027_075116_961.json.gz` | — | เกิด/uncertain/p=0.408 [Top5: เกิด(score=0.765), ข้าว(score=0.729), กิน(score=0.712), ไป(score=0.559), เที่ยว(score=0.686)]<br>กรุงเทพ/uncertain/p=0.303 [Top5: กรุงเทพ(score=0.552), ขอบคุณ(score=0.502), วัด(score=0.427), ประตู(score=0.412), ขอโทษ(score=0.422)] | — | สวัสดี/uncertain/p=0.836 [Top5: สวัสดี(score=0.604, support=1.037), กิน(score=0.537, support=0.830), เศร้า(score=0.340, support=0.706), พรุ่งนี้(score=0.370, support=0.668), ฉัน(score=0.388, support=0.631)]<br>กรุงเทพ/uncertain/p=0.273 [Top5: กรุงเทพ(score=0.527, support=0.872), ขอบคุณ(score=0.387, support=0.807), ครอบครัว(score=0.264, support=0.622), วัด(score=0.390, support=0.594), สวัสดี(score=0.256, support=0.529)] |
| * `20261027_075125_714.json.gz` | สวัสดี | สวัสดี/accepted/p=0.852 [Top5: สวัสดี(score=0.775), ไป(score=0.610), กิน(score=0.614), น้อง(score=0.510), ข้าว(score=0.601)]<br>กรุงเทพ/uncertain/p=0.298 [Top5: กรุงเทพ(score=0.577), วัด(score=0.459), กิน(score=0.442), สวัสดี(score=0.359), ปิด(score=0.388)] | สวัสดี | สวัสดี/accepted/p=0.949 [Top5: สวัสดี(score=0.759, support=1.147), พรุ่งนี้(score=0.393, support=0.713), กิน(score=0.378, support=0.641), ก่อน(score=0.404, support=0.621), ไป(score=0.409, support=0.611)]<br>กรุงเทพ/uncertain/p=0.383 [Top5: กรุงเทพ(score=0.535, support=0.920), ครอบครัว(score=0.318, support=0.722), วัด(score=0.435, support=0.658), ขอบคุณ(score=0.261, support=0.655), สวัสดี(score=0.317, support=0.636)] |
| * `20261027_075140_168.json.gz` | — | สวัสดี/uncertain/p=0.662 [Top5: สวัสดี(score=0.694), กิน(score=0.626), ข้าว(score=0.558), ไป(score=0.516), เกิด(score=0.577)]<br>กรุงเทพ/uncertain/p=0.365 [Top5: กรุงเทพ(score=0.579), สวัสดี(score=0.432), ขอโทษ(score=0.496), วัด(score=0.395), วิ่ง(score=0.364)] | — | สวัสดี/uncertain/p=0.903 [Top5: สวัสดี(score=0.658, support=1.052), กิน(score=0.409, support=0.693), พรุ่งนี้(score=0.329, support=0.651), เศร้า(score=0.276, support=0.603), ไป(score=0.350, support=0.560)]<br>กรุงเทพ/uncertain/p=0.400 [Top5: กรุงเทพ(score=0.556, support=0.989), สวัสดี(score=0.507, support=0.985), วัด(score=0.459, support=0.777), ครอบครัว(score=0.283, support=0.746), ขอบคุณ(score=0.316, support=0.684)] |
| * `20261027_075149_137.json.gz` | สวัสดี | สวัสดี/accepted/p=0.874 [Top5: สวัสดี(score=0.812), ไป(score=0.575), กิน(score=0.579), ข้าว(score=0.539), น้อง(score=0.497)]<br>ขอบคุณ/uncertain/p=0.486 [Top5: ขอบคุณ(score=0.723), กรุงเทพ(score=0.512), ประตู(score=0.353), วัด(score=0.487), สบายดี(score=0.485)] | สวัสดี | สวัสดี/accepted/p=0.962 [Top5: สวัสดี(score=0.806, support=1.203), พรุ่งนี้(score=0.428, support=0.761), พี่(score=0.415, support=0.731), ไป(score=0.456, support=0.717), กิน(score=0.386, support=0.696)]<br>ขอบคุณ/uncertain/p=0.869 [Top5: ขอบคุณ(score=0.638, support=1.107), กรุงเทพ(score=0.440, support=0.768), ครอบครัว(score=0.245, support=0.646), สวัสดี(score=0.244, support=0.571), วัด(score=0.329, support=0.541)] |
| * `20261027_075216_456.json.gz` | สวัสดี | สวัสดี/accepted/p=0.823 [Top5: สวัสดี(score=0.763), กิน(score=0.635), ไป(score=0.595), ข้าว(score=0.584), น้อง(score=0.538)]<br>กรุงเทพ/uncertain/p=0.220 [Top5: กรุงเทพ(score=0.525), วัด(score=0.496), ครอบครัว(score=0.458), กิน(score=0.447), สวัสดี(score=0.338)] | สวัสดี | สวัสดี/accepted/p=0.951 [Top5: สวัสดี(score=0.740, support=1.157), พรุ่งนี้(score=0.371, support=0.700), พี่(score=0.335, support=0.582), ไป(score=0.387, support=0.582), ก่อน(score=0.347, support=0.582)]<br>กรุงเทพ/uncertain/p=0.310 [Top5: กรุงเทพ(score=0.522, support=0.908), ครอบครัว(score=0.367, support=0.770), สวัสดี(score=0.314, support=0.665), วัด(score=0.473, support=0.647), ขอบคุณ(score=0.239, support=0.631)] |
| * `20261027_075223_813.json.gz` | สวัสดี | สวัสดี/accepted/p=0.850 [Top5: สวัสดี(score=0.751), ไป(score=0.587), กิน(score=0.625), ข้าว(score=0.597), เที่ยว(score=0.552)]<br>กรุงเทพ/uncertain/p=0.280 [Top5: กรุงเทพ(score=0.529), ขอบคุณ(score=0.507), ประตู(score=0.437), วัด(score=0.407), ครอบครัว(score=0.457)] | สวัสดี | สวัสดี/accepted/p=0.946 [Top5: สวัสดี(score=0.746, support=1.157), พรุ่งนี้(score=0.391, support=0.745), ก่อน(score=0.415, support=0.661), พี่(score=0.348, support=0.639), ไป(score=0.377, support=0.627)]<br>ขอบคุณ/uncertain/p=0.381 [Top5: ขอบคุณ(score=0.433, support=0.947), กรุงเทพ(score=0.527, support=0.903), ครอบครัว(score=0.310, support=0.738), สวัสดี(score=0.305, support=0.678), ประตู(score=0.386, support=0.576)] |
| * `20261027_075238_515.json.gz` | — | สวัสดี/uncertain/p=0.776 [Top5: สวัสดี(score=0.730), กิน(score=0.561), ไป(score=0.522), ข้าว(score=0.513), พรุ่งนี้(score=0.389)]<br>กรุงเทพ/uncertain/p=0.247 [Top5: กรุงเทพ(score=0.566), วัด(score=0.446), ขอโทษ(score=0.538), รัก(score=0.435), สวัสดี(score=0.393)] | สวัสดี | ครอบครัว/uncertain/p=0.368 [Top5: ครอบครัว(score=0.494, support=0.981), กรุงเทพ(score=0.458, support=0.894), ปิด(score=0.476, support=0.838), วัด(score=0.463, support=0.799), สวัสดี(score=0.280, support=0.678)]<br>กรุงเทพ/uncertain/p=0.243 [Top5: กรุงเทพ(score=0.475, support=0.846), สวัสดี(score=0.370, support=0.700), ครอบครัว(score=0.279, support=0.658), ขอบคุณ(score=0.229, support=0.615), วัด(score=0.372, support=0.597)] |
| * `20261027_075305_836.json.gz` | — | สวัสดี/uncertain/p=0.721 [Top5: สวัสดี(score=0.703), กิน(score=0.630), ข้าว(score=0.602), ไป(score=0.563), น้อง(score=0.540)]<br>กรุงเทพ/uncertain/p=0.453 [Top5: กรุงเทพ(score=0.599), วัด(score=0.431), ที่ไหน(score=0.412), ประตู(score=0.379), สวัสดี(score=0.406)] | สวัสดี | สวัสดี/accepted/p=0.931 [Top5: สวัสดี(score=0.703, support=1.114), กิน(score=0.426, support=0.749), เศร้า(score=0.344, support=0.673), พรุ่งนี้(score=0.367, support=0.659), ไป(score=0.403, support=0.602)]<br>กรุงเทพ/uncertain/p=0.399 [Top5: กรุงเทพ(score=0.585, support=0.959), สวัสดี(score=0.367, support=0.814), ครอบครัว(score=0.267, support=0.695), วัด(score=0.400, support=0.663), ประตู(score=0.379, support=0.640)] |
| † `20261027_075318_107.json.gz` | สวัสดี | สวัสดี/accepted/p=0.854 [Top5: สวัสดี(score=0.773), ไป(score=0.510), เที่ยว(score=0.476), ข้าว(score=0.467), กิน(score=0.465)]<br>เกิด/uncertain/p=0.347 [Top5: เกิด(score=0.568), กรุงเทพ(score=0.554), ขอบคุณ(score=0.478), ไป(score=0.535), ขอโทษ(score=0.483)] | สวัสดี | สวัสดี/accepted/p=0.967 [Top5: สวัสดี(score=0.769, support=1.205), พรุ่งนี้(score=0.359, support=0.692), ไป(score=0.408, support=0.666), กิน(score=0.313, support=0.598), พี่(score=0.310, support=0.594)]<br>กรุงเทพ/unknown/p=0.114 [Top5: กรุงเทพ(score=0.452, support=0.814), ขอบคุณ(score=0.354, support=0.747), ครอบครัว(score=0.291, support=0.740), สวัสดี(score=0.161, support=0.563), วัด(score=0.281, support=0.546)] |

`*` = time window + clue; `†` = อยู่ในช่วงเวลาแต่ไม่ผ่าน clue และไม่รวมในกลุ่ม strict: `20261027_075318_107.json.gz` (ท้าย v7.2: สวัสดี/accepted/p=0.854 [Top5: สวัสดี(score=0.773), ไป(score=0.510), เที่ยว(score=0.476), ข้าว(score=0.467), กิน(score=0.465)]<br>เกิด/uncertain/p=0.347 [Top5: เกิด(score=0.568), กรุงเทพ(score=0.554), ขอบคุณ(score=0.478), ไป(score=0.535), ขอโทษ(score=0.483)]).

## สรุป 53 ไฟล์

| รุ่น | Segments: accepted | uncertain | unknown | transition | latency เฉลี่ย |
|---|---:|---:|---:|---:|---:|
| v7.2 (cache) | 23 | 113 | 21 | 0 | 20974.7 ms (n=53) |
| v8.1 (replay) | 28 | 115 | 21 | 3 | 987.8 ms (n=53) |

ผลต่อไฟล์และ Top-5 แบบเต็มอยู่ใน [`outputs/per_file.csv`](outputs/per_file.csv). จำนวน segments ใน CSV เป็นผลที่แต่ละ pipeline ส่งคืน; transition แสดงแยกและไม่นำไปนับเป็นคำ Top-1 ในรายงานช่วงท้าย.

## ข้อจำกัด

- เป็นการสังเกต cache จากผู้ลงนามคนเดียวตามที่แจ้ง ไม่ใช่ตัวอย่างอิสระหลายคน และไม่ได้ยืนยันว่า gesture ที่ทำจริงตรงกับ segment ใด.
- cache เก็บผลเดิมและ landmarks แต่ไม่เก็บทุก option ที่ส่งใน request; เงื่อนไข offline ข้างต้นทำให้ sentence layer ของ v8.1 ไม่ใช่การ replay LLM ใน production.
- ไม่มีการอ่านชุดข้อมูล TSL51 หรือ Eval-Test, ไม่มีการเทรน, Docker หรือการเรียกบริการ inference/LLM ภายนอก; ติดตั้ง ONNX Runtime ตามเวอร์ชันใน requirements ลง session-local package target นอก repo. staged modules/model/cache ไม่ถูกแก้ไข.

### Traceback จากการลอง Torch backend ก่อนหน้า (แก้โดยใช้ ONNX backend ของ pipeline)

Torch backend ต้องการ vendored `third_party/unisign/stgcn_layers.py` ซึ่งไม่ได้อยู่ในชุด staging ที่ SHA-verified; ไม่ได้นำโค้ดจาก repo หลักหรือ external checkout มาปน. เปลี่ยนมาใช้ ONNX backend ที่ pipeline รองรับและโมเดล Jack-model มีอยู่ครบ:

```text
Traceback (most recent call last):
  File "/Users/thawaphorn/copilot-worktrees/DSI490-Sign2Say/boomthawa-vigilant-robot/Boom/v81_replay/replay_v81.py", line 482, in main
    rows, _ = run_replay(args.threads)
              ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/thawaphorn/copilot-worktrees/DSI490-Sign2Say/boomthawa-vigilant-robot/Boom/v81_replay/replay_v81.py", line 438, in run_replay
    pipeline = SignPipeline(
               ^^^^^^^^^^^^^
  File "/Users/thawaphorn/copilot-worktrees/DSI490-Sign2Say/boomthawa-vigilant-robot/Boom/v81_replay/_inputs/v81/modules/pipeline.py", line 98, in __init__
    self.be = make_backend(md, backend, device, **({} if backend == "onnx" else {"tagger_arch": self.cfg.get("tagger_arch", "gru")}))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/thawaphorn/copilot-worktrees/DSI490-Sign2Say/boomthawa-vigilant-robot/Boom/v81_replay/_inputs/v81/modules/runtime.py", line 115, in make_backend
    return OnnxBackend(model_dir, device, **kw) if backend == "onnx" else TorchBackend(model_dir, device, **kw)
                                                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/thawaphorn/copilot-worktrees/DSI490-Sign2Say/boomthawa-vigilant-robot/Boom/v81_replay/_inputs/v81/modules/runtime.py", line 90, in __init__
    from .encoder import load_encoder
  File "/Users/thawaphorn/copilot-worktrees/DSI490-Sign2Say/boomthawa-vigilant-robot/Boom/v81_replay/_inputs/v81/modules/encoder.py", line 29, in <module>
    from .unisign import UniSignPose
  File "/Users/thawaphorn/copilot-worktrees/DSI490-Sign2Say/boomthawa-vigilant-robot/Boom/v81_replay/_inputs/v81/modules/unisign.py", line 20, in <module>
    from stgcn_layers import Graph, get_stgcn_chain  # noqa: E402
ModuleNotFoundError: No module named 'stgcn_layers'
```
