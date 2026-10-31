# v8.1 port — experimental branch, no live cutover

วันที่ตรวจ 2026-10-28 (Asia/Bangkok). Branch `boom-v81-port` เริ่มจาก `origin/boom` commit `61ac19f` ไม่ใช่ main; code source `origin/jack_master` commit `99e99eee8dd6eda84c21d4e289721153dd80aa3a`. Jack model card อ้าง `c6a1cfe71f07bd5f845f199a89cb6010029cb283` จึงบันทึกทั้งสอง provenance ไม่อ้างว่าเป็น commit เดียวกัน

## ผลสำคัญ

พอร์ตแล้วโหลดและ replay ได้จริงทั้ง ONNX/Torch บน 53 Live requests เดิม โดยใช้ model assets ชุดเดียวกันกับ PR #6 และรักษา hand canonicalization metadata ของ Phase 2.2 ไม่มีการเทรนหรือจูนค่า ไม่มี TSL51 test/Eval-Test ไม่มีคำสั่ง Docker ที่เรียกใช้งานในงานนี้ ไม่มีการเปลี่ยน container/live checkout

**ผลนี้คือ backend/port consistency ไม่ใช่ accuracy** ไม่มีเฉลยครบจึงไม่คำนวณ WER และไม่อ้างว่าการตรงกันระหว่าง backend พิสูจน์ว่าทายถูก

## ไฟล์ที่นำเข้าและการรวมโค้ด

|ไฟล์|การเปลี่ยน/เหตุผล|
|---|---|
|modules/handshape.py|นำจาก Jack: descriptor + HandBank + fusion ใช้ hand_w จาก model config|
|modules/decode.py|นำจาก Jack: decision features, calibrated decisions, merge/split, candidate reranking|
|modules/pipeline.py|นำ v8.1 front/back/decide/hand cue แล้วคง standardise() 5 ค่า และ standardise_with_metadata() 6 ค่า; result มี hand_canonicalization, mirrored_to_right_hand และ model_version|
|modules/runtime.py|นำเฉพาะ variant_lut จาก Jack; คง Torch device resolution ของ boom|
|modules/grammar.py|นำ support-key guard/rerank API ของ Jack|
|modules/schema.py|ไม่แก้: canonical_hands_with_metadata และ canonical_hands() คืน 3 ค่ายังคงอยู่|
|modules/utils.py|ไม่แก้: รองรับ THAISLM_MODELS อยู่แล้ว และต้องคง default models/ รวม device/inference_precision ของเรา|
|modules/encoder.py, unisign.py, spanhead.py, third_party/unisign|คงของ boom: Torch ใช้ dependency นี้ได้จริง ไม่เอา staged modules ที่ขาด third_party มาใช้|
|modules/lexicon.py|ไม่แก้: diff ของ Jack เพิ่มเพียง iPhone test references ไม่ใช่ dependency ของ inference ที่ต้อง port|
|scripts/decision.py|ไม่ได้นำเข้า: เป็นชุด tune/train/report ไม่ใช่ script ที่ pipeline เรียกขณะ inference; decision ที่ใช้จริงอยู่ใน decode/pipeline|
|webapp/api/app.py|คง perf_mode/client_fps, r.get(hand_canonicalization); เพิ่ม model_version ใน response/load log และส่ง canonicalization metadata แบบ optional; schema sentence/segments/face/timings_ms เดิม|
|webapp/api/request_cache.py|เก็บ result.model_version เพิ่ม โดยคง client capture metadata และ safe None handling|
|docker-compose.yml|ตัวแปร host THAISLM_MODELS เลือก directory mount; default ./models และ THAISLM_MODELS_DIR fallback ยังใช้ได้|
|Boom/v81_replay/replay_v81.py|จาก PR #6 commit 9c2a85b; เพิ่ม --ported/--backend และ input/output paths โดยคง parser/row protocol เดิม; ไม่ต้อง copy model/cache เข้ากิต|
|Boom/v81_port/*|คู่มือ, isolated compose, parity comparison, tests, legacy response check, รายงานนี้|

เลือกนำเฉพาะไฟล์/ฟังก์ชัน ไม่ merge ทั้ง branch จึงไม่มี text conflict จาก git merge แต่แก้ semantic conflict: metadata ถูก Jack standardise เดิมตัดทิ้ง, Jack default model/device ต่างจาก boom, และ staged Torch เดิมหา stgcn_layers ไม่เจอ การรวมครั้งนี้คง schema/device/third_party ของ boom และเชื่อม v8.1 decision ผ่าน API เดิม PR #5 และการแก้ที่ยังไม่ commit ใน live checkout ไม่ถูกดึงมาปน

ข้อจำกัดการรวม: pipeline ของ Jack แทนเส้นทาง symbolic dictionary experiment เดิม; draft นี้มุ่ง hand-shape cue/decision ของ v8.1 ไม่รับรองผล THAISLM_SYMBOLIC_WEIGHT จาก Phase 2.1 และไม่เปิด experiment นั้นอัตโนมัติ

## การตรวจจริง

- `Boom/phase2.2/tests`: **23/23 ผ่าน** บน Python 3.11 ของ Boom/venv-sign2say
- API/cache contract tests ใหม่: **2/2 ผ่าน** (model_version, perf_mode/client_fps, hand metadata และ variant mapping)
- Import pipeline / สร้าง SignPipeline(Jack-model): **ผ่าน ONNX และ Torch CPU**; Torch ใช้ third_party/unisign ของ branch นี้จริง
- Jack effective config: version v8.1, hand_w=1.2; accept_min=0.8 แต่ effective tau_accept=**0.92** เพราะใช้ max(calibration threshold, accept_min)
- Replay ทั้งสอง backend: **53/53 ครบ** ไม่มี ground truth เติมเอง; input SHA256 ทั้ง 53 และ model SHA256 ทั้ง 26 ตรง staging ของ PR #6
- SHA256 encoder.pt/tagger.pt/bank.npz ระหว่าง Jack-model และ original model เท่ากันทั้งสามไฟล์ (อ่านเทียบเท่านั้น)
- Run เสร็จแล้วตรวจ model hashes ไม่เปลี่ยน
- ใช้ clip จริง 20261026_130052_394 กับโมเดลเดิม v7.2 ตรวจ rollback response: **ผ่าน** standardise คืน 5, with_metadata คืน 6 และ response keys ครบ; เป็น smoke check ไม่ใช่ equivalence test ทั้ง v7.2
- `git diff --check` และ AST parse ผ่าน
- ไม่ได้ build/start/stop Docker จึงยังไม่รับรอง container build/browser smoke test; API test deps ติดตั้งแยกไว้ใน outputs/api_deps ไม่แก้ venv/live environment

## Torch เทียบ ONNX

|คู่เทียบ|Top-1 sequence ตรง/53|accepted words ตรง/53|accepted count ตรง/53|status sequence ตรง/53|accepted รวมก่อน→หลัง|median latency ก่อน→หลัง ms|
|---|---:|---:|---:|---:|---:|---:|
|PR6 ONNX → port Torch|53|53|53|53|28 → 28|750.1 → 952.1|
|port ONNX → port Torch|53|53|53|53|28 → 28|1460.3 → 952.1|
|PR6 ONNX → port ONNX|53|53|53|53|28 → 28|750.1 → 1460.3|

Top-1 คือ nearest/final word ของ segment ที่ไม่ใช่ transition เรียงตามเวลา ไม่ใช่การนับคำถูก accepted count คือจำนวน segment ที่ accepted ไม่ใช่ precision ค่า latency เป็นการสังเกต ไม่ใช่ controlled benchmark: PR6 ใช้ 4 threads และ warm-up ส่วนรอบ port ใช้ 2 threads ไม่มี synthetic warm-up และมีภาระเครื่องต่างเวลา จึงไม่สรุปว่า backend ใดเร็วกว่า

ตั้งค่า replay เหมือน PR6: conversation/en, LLM ปิด, use_face=True, face_question=False, face_baseline=None, user_bank=None, spell_only=False. Log เดิมไม่มีค่าบางฟิลด์ จึงเป็น assumptions ที่บันทึก ไม่ใช่ exact reconstruction ของ original web settings ไม่มีการประมาณ landmark ใหม่

## รายไฟล์: PR6 ONNX เทียบ port Torch

คำ Top-1 และลำดับ status เหมือนกันครบ 53 ไฟล์ ด้านล่างแสดงจำนวน accepted ต่อไฟล์; JSON/CSV ราย segment และ Top-5 อยู่ใน outputs ที่ ignore ไว้ในเครื่อง (ไม่ commit cache/recordings/models)

|ไฟล์|ONNX accepted|Torch accepted|Top-1 sequence|
|---|---:|---:|---|
|20261026_130052_394.json.gz|0|0|ตรงกัน|
|20261026_130122_467.json.gz|0|0|ตรงกัน|
|20261026_130156_724.json.gz|1|1|ตรงกัน|
|20261026_130232_559.json.gz|0|0|ตรงกัน|
|20261026_130335_216.json.gz|0|0|ตรงกัน|
|20261026_130402_269.json.gz|2|2|ตรงกัน|
|20261026_130410_748.json.gz|4|4|ตรงกัน|
|20261026_130425_474.json.gz|0|0|ตรงกัน|
|20261026_130441_909.json.gz|1|1|ตรงกัน|
|20261026_130443_828.json.gz|0|0|ตรงกัน|
|20261026_130500_698.json.gz|1|1|ตรงกัน|
|20261026_130510_689.json.gz|0|0|ตรงกัน|
|20261026_130606_070.json.gz|0|0|ตรงกัน|
|20261026_130608_150.json.gz|0|0|ตรงกัน|
|20261026_130621_676.json.gz|0|0|ตรงกัน|
|20261026_130656_575.json.gz|0|0|ตรงกัน|
|20261026_130732_044.json.gz|0|0|ตรงกัน|
|20261026_131316_123.json.gz|0|0|ตรงกัน|
|20261026_131328_322.json.gz|0|0|ตรงกัน|
|20261026_154031_283.json.gz|0|0|ตรงกัน|
|20261026_154113_334.json.gz|0|0|ตรงกัน|
|20261026_154114_904.json.gz|0|0|ตรงกัน|
|20261026_154214_329.json.gz|1|1|ตรงกัน|
|20261026_154217_636.json.gz|0|0|ตรงกัน|
|20261026_154224_466.json.gz|1|1|ตรงกัน|
|20261026_154237_563.json.gz|0|0|ตรงกัน|
|20261026_154344_238.json.gz|0|0|ตรงกัน|
|20261026_154423_617.json.gz|0|0|ตรงกัน|
|20261026_154502_778.json.gz|2|2|ตรงกัน|
|20261026_154503_039.json.gz|0|0|ตรงกัน|
|20261026_182605_054.json.gz|1|1|ตรงกัน|
|20261026_182647_858.json.gz|1|1|ตรงกัน|
|20261026_182736_966.json.gz|0|0|ตรงกัน|
|20261026_182857_487.json.gz|0|0|ตรงกัน|
|20261026_182915_271.json.gz|0|0|ตรงกัน|
|20261026_183512_002.json.gz|0|0|ตรงกัน|
|20261026_183521_582.json.gz|0|0|ตรงกัน|
|20261026_183532_291.json.gz|0|0|ตรงกัน|
|20261026_183551_680.json.gz|1|1|ตรงกัน|
|20261027_074947_285.json.gz|1|1|ตรงกัน|
|20261027_075012_046.json.gz|1|1|ตรงกัน|
|20261027_075021_560.json.gz|1|1|ตรงกัน|
|20261027_075035_536.json.gz|1|1|ตรงกัน|
|20261027_075047_839.json.gz|1|1|ตรงกัน|
|20261027_075116_961.json.gz|0|0|ตรงกัน|
|20261027_075125_714.json.gz|1|1|ตรงกัน|
|20261027_075140_168.json.gz|0|0|ตรงกัน|
|20261027_075149_137.json.gz|1|1|ตรงกัน|
|20261027_075216_456.json.gz|1|1|ตรงกัน|
|20261027_075223_813.json.gz|1|1|ตรงกัน|
|20261027_075238_515.json.gz|1|1|ตรงกัน|
|20261027_075305_836.json.gz|1|1|ตรงกัน|
|20261027_075318_107.json.gz|1|1|ตรงกัน|

## ความเสี่ยงและขั้นต่อไป

1. ข้อมูลเป็น Live 53 requests โดยไม่มีเฉลยครบ ไม่ควรใช้เลือก threshold หรือประกาศ accuracy; FPS ~13–17 ต่างจากข้อมูลจูน 25 fps การ replay ผ่าน schema resampling ไม่ได้พิสูจน์ว่า tracking กล้องสดดีขึ้น
2. ความมั่นใจ/Top-5 candidate numeric score อาจต่างเล็กน้อยระหว่าง Torch/ONNX แม้ final decisions ตรงกัน การทดสอบนี้ไม่ประกาศ bitwise floating-point parity
3. v7.2 model rollback ใช้ model assets เดิมได้ แต่ code เป็น port ใหม่ ไม่ใช่การย้อน implementation ทุกส่วน ต้อง checkout branch เดิมเพื่อ exact code rollback
4. Jack-model ต้อง provision แยกแบบ read-only ก่อนใช้ ไม่รวม assets ใน PR; ไม่มีการแตะ models เดิม
5. Isolated Docker ต้องตรวจ port 18080 ว่างและทดสอบ response จริงก่อนสลับ live ไม่มีการรัน Docker โดย agent
6. Branch นี้เป็น draft สำหรับรีวิว ไม่ merge เอง ไม่ push boom/main ไม่เปลี่ยน production

คำสั่ง cutover/rollback และ isolated port deployment อยู่ใน [CUTOVER.md](CUTOVER.md). ค่าเริ่มต้นยังเป็น models/ เดิม การเลือก Jack-model ทำด้วย THAISLM_MODELS เพียงตัวเดียว ไม่มีการเลือก threshold ใหม่จาก 53 recordings นี้
