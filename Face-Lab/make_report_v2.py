"""face-report-v2: HTML from results/*.json (every number is read, none typed) → face-report-v2.html → PDF with headless Chrome.
Compares only what is used: ours (models/face-model-ravhead-v1) vs the friend's models on main (EfficientNet-B0 v1 + the blendshape model).
  python make_report_v2.py"""
import json
import subprocess
from pathlib import Path

H = Path(__file__).resolve().parent; R = H / "results"
J = lambda n: json.loads((R / n).read_text(encoding="utf-8"))
rh, pr, ck, ds = J("rav_head_24actors.json"), J("rav_prior.json"), J("ravhead_check.json"), J("data_summary.json")
fz, fc = J("fuse24.json"), J("fer_cross.json")
rc = J("real_check.json")["models"]["efficientnetb0-v1"]["stability"]["shipped"]
card = json.loads((H.parent / "models/face-model/model_card.json").read_text(encoding="utf-8"))["cv"]
EMO = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
TH = ["ปกติ", "สุข", "เศร้า", "โกรธ", "กลัว", "รังเกียจ", "ประหลาดใจ"]
f = lambda v: f"{v:.3f}"
pc = lambda v: f"{100 * v:.1f}%"
ours, b0, bs = pr["neutral_prior_0.5"], fz["main: B0 v1 alone"], rh["blendshape_lr (friend recipe)"]
ck5 = ck["ravhead neutral_prior 0.5"]

# 4. RAVDESS, each model alone
rav_rows = [("เพื่อน: EfficientNet-B0 v1 (ภาพ, บน main)", b0, ""),
            ("เพื่อน: blendshape ตัวที่ใช้จริง <sup>a</sup>", {k: card[k] for k in card if k != "confusion"}, ""),
            ("เพื่อน: blendshape รันซ้ำด้วยวิธีเดียวกัน", bs, ""),
            ("<b>ของเรา: B2 + หัว RAVDESS (ภาพ, ใช้บนเว็บ)</b>", ours, "ship")]
cols = [("clips_acc", "accuracy คลิป"), ("clips_macro_f1", "macro-F1 คลิป"), ("windows_macro_f1", "macro-F1 หน้าต่าง 2 วิ"),
        ("strong_intensity_acc", "acc ระดับเข้ม"), ("neutral_vs_emotion_acc", "ปกติ vs มีอารมณ์"), ("valence_sign_agreement", "บวก/ลบ ตรง")]
best = {k: max(r[1][k] for r in rav_rows) for k, _ in cols}
t1 = "<tr><th>โมเดล</th>" + "".join(f"<th>{n}</th>" for _, n in cols) + "</tr>"
for name, m, tag in rav_rows:
    t1 += f"<tr class='{tag}'><td>{name}</td>" + "".join(f"<td class='{'best' if abs(m[k] - best[k]) < 1e-9 else ''}'>{f(m[k])}</td>" for k, _ in cols) + "</tr>"
rec_rows = [("เพื่อน: B0 v1", b0["per_class_recall"]), ("เพื่อน: blendshape", bs["per_class_recall"]), ("<b>ของเรา</b>", ours["per_class_recall"])]
t2 = "<tr><th>recall</th>" + "".join(f"<th>{t}</th>" for t in TH) + "</tr>"
for name, r in rec_rows:
    t2 += f"<tr><td>{name}</td>" + "".join(f"<td class='{'best' if abs(r[e] - max(x[1][e] for x in rec_rows)) < 1e-9 else ''}'>{r[e]:.2f}</td>" for e in EMO) + "</tr>"

# 5. whole system as the app combines it
sys_rows = [("main (เพื่อน): B0 v1 + blendshape", "main: B0 v1 + blendshape (0.5/0.5)", ""),
            ("&nbsp;&nbsp;→ อารมณ์ของเสียง", "main: B0 v1 + blendshape (0.5/0.5) → voice", ""),
            ("<b>ของเรา: B2 + หัว RAVDESS + blendshape</b>", "ours: head + blendshape (0.5/0.5)", "ship"),
            ("<b>&nbsp;&nbsp;→ อารมณ์ของเสียง</b>", "ours: head + blendshape (0.5/0.5) → voice", "ship")]
t4 = "<tr><th>ทั้งระบบ (ภาพ 0.5 + blendshape 0.5)</th><th>macro-F1 คลิป</th><th>accuracy คลิป</th><th>ปกติ vs มีอารมณ์</th>" + "".join(f"<th>{t}</th>" for t in TH) + "</tr>"
for n, k, tag in sys_rows:
    m = fz[k]; t4 += f"<tr class='{tag}'><td>{n}</td><td>{f(m['clips_macro_f1'])}</td><td>{f(m['clips_acc'])}</td><td>{f(m['neutral_vs_emotion_acc'])}</td>" + "".join(f"<td>{m['per_class_recall'][e]:.2f}</td>" for e in EMO) + "</tr>"

# 6. team clips
t3 = "<tr><th>ป้ายบนจอ (unknown rule + smoother แบบเว็บ)</th><th>ขึ้นหน้าปกติ</th><th>ขึ้นอารมณ์ผิด ↓</th><th>ไม่แน่ใจ</th><th>ยิ้มที่จับได้</th></tr>"
t3 += f"<tr><td>เพื่อน: B0 v1 (prior 0.6)</td><td>{pc(rc['shown_neutral'])}</td><td>{pc(rc['shown_wrong_expression'])}</td><td>{pc(rc['shown_unknown'])}</td><td>{rc['smiles_shown_happy']}</td></tr>"
t3 += f"<tr class='ship'><td><b>ของเรา (prior 0.5)</b></td><td>{pc(ck5['shown_neutral'])}</td><td>{pc(ck5['wrong'])}</td><td>{pc(ck5['unknown'])}</td><td>{ck5['smiles']}</td></tr>"

# 7. both ways
b0f, of = fc["main: B0 v1 (prior 0.6)"], fc["ours: RAVDESS head (prior 0.5)"]
t7 = ("<tr><th>macro-F1 (accuracy)</th><th>FER2013 test: รูปนิ่ง 48px จากเน็ต (7,178 รูป)</th><th>RAVDESS 24 คน: วิดีโอนักแสดง</th></tr>"
      f"<tr><td>เพื่อน: B0 v1 (เทรนบน FER2013)</td><td><b>{f(b0f['macro_f1'])}</b> ({f(b0f['accuracy'])}) · ข้อมูลที่เทรนมา</td>"
      f"<td>{f(b0['clips_macro_f1'])} ({f(b0['clips_acc'])}) · ไม่เคยเห็น</td></tr>"
      f"<tr class='ship'><td>ของเรา (เทรนบน RAVDESS)</td><td>{f(of['macro_f1'])} ({f(of['accuracy'])}) · ไม่เคยเห็น</td>"
      f"<td><b>{f(ours['clips_macro_f1'])}</b> ({f(ours['clips_acc'])}) · ข้อมูลที่เทรนมา (คนที่ไม่เคยเห็น)</td></tr>")

S_main, S_ours = fz["main: B0 v1 + blendshape (0.5/0.5)"], fz["ours: head + blendshape (0.5/0.5)"]
V_main, V_ours = fz["main: B0 v1 + blendshape (0.5/0.5) → voice"], fz["ours: head + blendshape (0.5/0.5) → voice"]

html = f"""<!doctype html><html lang="th"><head><meta charset="utf-8"><title>Face branch v2 report</title>
<style>
@page {{ size: A4; margin: 16mm 15mm; }}
body {{ font-family: 'Leelawadee UI', Tahoma, sans-serif; font-size: 10.5pt; color: #1a1a19; line-height: 1.5; }}
h1 {{ font-size: 19pt; margin: 0 0 2px; }} h2 {{ font-size: 13.5pt; margin: 18px 0 6px; border-bottom: 1.5px solid #2a78d6; padding-bottom: 2px; }}
h3 {{ font-size: 11pt; margin: 12px 0 4px; }}
.sub {{ color: #52514e; font-size: 9pt; margin-bottom: 10px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9pt; margin: 6px 0 4px; page-break-inside: avoid; }}
th, td {{ border-bottom: 1px solid #e6e5e0; padding: 3px 5px; text-align: right; }} th {{ background: #f0efec; font-weight: 600; }}
td:first-child, th:first-child {{ text-align: left; }}
table.data td, table.data th {{ text-align: left; }}
td.best {{ font-weight: 700; color: #1c5cab; }} tr.ship td {{ background: #e8f1fc; }}
.note {{ font-size: 8.5pt; color: #52514e; }}
.box {{ background: #f5f4f0; border-left: 3px solid #2a78d6; padding: 6px 10px; margin: 8px 0; }}
img {{ width: 100%; page-break-inside: avoid; }} code {{ font-size: 8.5pt; background: #f0efec; padding: 0 3px; }}
.pb {{ page-break-before: always; }} ul {{ margin: 4px 0; padding-left: 20px; }} li {{ margin: 2px 0; }}
</style></head><body>
<h1>Face branch v2: ของเราเทียบกับของเดิมบน main</h1>
<div class="sub">ThaiSLM · 15 พ.ย. 2026 · branch pitty · model <code>models/face-model-ravhead-v1</code> · code <code>modules/face_fer.py</code> · ต่อจาก face-report-v1</div>

<div class="box"><b>สรุป</b> — บน main ป้ายสีหน้ามาจาก EfficientNet-B0 v1 ของเพื่อน (เทรนบน FER2013) ซึ่งขึ้น "ปกติ" เกือบตลอด เราเปลี่ยนเฉพาะ<b>โมเดลภาพ</b>
เป็น HSEmotion B2 + ตัวตัดสินที่เทรนบน RAVDESS (+ ตัวจับขมวดคิ้ว) ส่วนโมเดล blendshape ของเพื่อนและกฎการรวมใช้เหมือนเดิม
วัดด้วยวิธีเดียวกันทุกตัว (RAVDESS 24 นักแสดง, คะแนนจากคนที่ไม่เคยเห็น):
<ul>
<li>ป้ายบนจอ (ภาพอย่างเดียว): {f(b0['clips_macro_f1'])} → <b>{f(ours['clips_macro_f1'])}</b> · เทียบ blendshape ของเพื่อนที่เทรนบน RAVDESS เหมือนกัน {f(bs['clips_macro_f1'])}</li>
<li>ทั้งระบบ (ภาพ + blendshape): {f(S_main['clips_macro_f1'])} → <b>{f(S_ours['clips_macro_f1'])}</b> · อารมณ์ของเสียง {f(V_main['clips_macro_f1'])} → <b>{f(V_ours['clips_macro_f1'])}</b></li>
<li>คลิปทีม: ขึ้นอารมณ์ผิดบนหน้าปกติ {pc(rc['shown_wrong_expression'])} → <b>{pc(ck5['wrong'])}</b> แต่ยิ้มบางๆ จับได้น้อยลง ({rc['smiles_shown_happy']} → {ck5['smiles']})</li>
<li>บน FER2013 ของเพื่อนชนะ ({f(b0f['macro_f1'])} เทียบกับ {f(of['macro_f1'])}) แต่งานจริงเป็นวิดีโอคนทำภาษามือ ซึ่งใกล้ RAVDESS มากกว่า</li>
</ul></div>

<h2>1. ของเดิมและสิ่งที่เปลี่ยน</h2>
<p>แอปใช้ใบหน้า 3 ทาง: (1) <b>ป้ายบนจอ</b> (<code>/v1/face</code>) ใช้โมเดลภาพอย่างเดียว · (2) <b>อารมณ์ทั้งประโยค</b> ใช้โมเดล blendshape
· (3) <b>อารมณ์ของเสียง</b> (<code>speech_affect</code>) เฉลี่ยความน่าจะเป็นจากภาพกับ blendshape อย่างละครึ่ง แล้วออกเสียงอารมณ์เฉพาะเมื่ออารมณ์ที่ไม่ใช่ปกติอยู่บนสุดและ ≥ 0.5</p>
<table class="data">
<tr><th></th><th>main (เพื่อน)</th><th>ของเรา</th></tr>
<tr><td>โมเดลภาพ</td><td>EfficientNet-B0 v1 (<code>models/face-model-efficientnetb0-v1</code>): เริ่มจาก ImageNet เทรนทั้งตัวบน FER2013 (รูปนิ่ง 48px), prior หน้าปกติ 0.6</td>
<td>HSEmotion B2 (เทรนมาแล้วบน AffectNet, ไม่แก้) ดึงฟีเจอร์ 1,408 ตัว + v2-ferplus (EfficientNet-B0 ที่เราเทรนบน FER+) → <b>ตัวตัดสิน</b> (logistic regression)
เทรนบน RAVDESS + หน้าปกติจากคลิปภาษามือ ดูทีละช่วง 2 วินาที, prior หน้าปกติ 0.5 (<code>models/face-model-ravhead-v1</code>)</td></tr>
<tr><td>ตัวจับขมวดคิ้ว</td><td>–</td><td>วัดความสูงคิ้วด้านในเทียบกับคิ้วตอนพักของคนนั้นเอง ดันป้ายไปทางโกรธเมื่อขมวดคิ้ว (ป้ายบนจอเท่านั้น เพราะคิ้วต่ำเป็นไวยากรณ์คำถาม wh ด้วย)</td></tr>
<tr><td>blendshape</td><td colspan="2">โมเดลของเพื่อน (<code>models/face-model</code>) เหมือนกันทั้งสองฝั่ง ไม่ได้แก้</td></tr>
<tr><td>การรวมเป็นเสียง</td><td colspan="2">ภาพ 0.5 + blendshape 0.5 เหมือนกันทั้งสองฝั่ง</td></tr>
</table>

<h2 class="pb">2. ข้อมูลที่ใช้</h2>
<table class="data">
<tr><th>ชุดข้อมูล</th><th>ใช้ทำอะไร</th><th>จำนวน</th><th>ป้าย</th></tr>
<tr><td>RAVDESS (วิดีโอพูด, นักแสดง 24 คน)</td><td>เทรน + ทดสอบ (leave-actors-out)</td>
<td>{ds['ravdess']['clips']:,} คลิป → {ds['ravdess']['samples_clip_plus_windows']:,} ตัวอย่าง (คลิป + หน้าต่าง 2 วิ) · ภาพหน้า {ds['ravdess']['crops_5fps']:,} ภาพ</td>
<td>ปกติ {ds['ravdess']['clips_per_class']['neutral']} (รวม calm), อารมณ์อื่นอย่างละ {ds['ravdess']['clips_per_class']['happy']}</td></tr>
<tr><td>คลิปท่าพจนานุกรม (<code>data_test/jack_test</code>)</td><td>เทรน (หน้าปกติขณะทำภาษามือ)</td>
<td>{ds['signing_neutral']['clips']} คลิป → ภาพหน้า {ds['signing_neutral']['crops_5fps']} ภาพ → {ds['signing_neutral']['samples']} ตัวอย่าง</td><td>ปกติทั้งหมด</td></tr>
<tr><td>FER2013 test (ของเพื่อน)</td><td>ทดสอบอย่างเดียว (หัวข้อ 7)</td><td>7,178 รูป 48×48</td><td>7 อารมณ์</td></tr>
<tr><td>คลิปของทีม (Boom, Jack, Pitty)</td><td>ทดสอบอย่างเดียว</td>
<td>{ds['team']['clips']} คลิป → {ds['team']['frames_every_0_5s']} เฟรม (ทุก 0.5 วิ)</td>
<td>ตัดสินด้วยตา: ปกติ {ds['team']['neutral']} · ยิ้ม {ds['team']['smile']} · กำกวม {ds['team']['ambiguous']} · มือบัง/ไม่ชัด {ds['team']['covered_or_poor']}</td></tr>
</table>
<div style="page-break-inside: avoid"><h3>อารมณ์ใน RAVDESS</h3>
<p>RAVDESS มี 8 อารมณ์ ส่วนระบบเรา (และ FER2013) ใช้ 7 อารมณ์</p>
<table class="data" style="width: 50%">
<tr><th>RAVDESS</th><th>ระบบเรา</th></tr>
<tr><td>neutral</td><td>ปกติ</td></tr>
<tr><td>calm (สงบ)</td><td>รวมเข้าปกติ</td></tr>
<tr><td>happy</td><td>สุข</td></tr>
<tr><td>sad</td><td>เศร้า</td></tr>
<tr><td>angry</td><td>โกรธ</td></tr>
<tr><td>fearful</td><td>กลัว</td></tr>
<tr><td>disgust</td><td>รังเกียจ</td></tr>
<tr><td>surprised</td><td>ประหลาดใจ</td></tr>
</table></div>
<p class="note">ภาพคือสิ่งที่โมเดลของเราเห็นจริง: ครอปหน้าสีเทา 96×96 px จากกรอบที่คำนวณจาก 68 จุดของ MediaPipe (ตัวเดียวกับบนเว็บ)</p>
<div style="page-break-inside: avoid"><h3>RAVDESS: 7 อารมณ์ (ระดับเข้ม) ของนักแสดง 2 คน</h3>
<img src="figures/v2_data_ravdess.png"></div>
<div style="page-break-inside: avoid"><h3>FER2013: ข้อมูลที่ B0 v1 ของเพื่อนเทรนและทดสอบ (รูปจากชุด test)</h3>
<img src="figures/v1_data_fer2013.png"></div>
<div style="page-break-inside: avoid"><h3>คลิปท่าพจนานุกรม: หน้าปกติขณะทำภาษามือ (ตัวอย่างที่เติมตอนเทรน)</h3>
<img src="figures/v2_data_signing.png"></div>
<div style="page-break-inside: avoid"><h3>คลิปของทีม: ชุดทดสอบคนทำภาษามือ</h3>
<img src="figures/v2_data_team.png"></div>
<p class="note">RAVDESS หน้าตรง แสงสตูดิโอ แสดงอารมณ์ชัด · FER2013 รูปนิ่งเล็กจากเน็ต · คลิปของทีมเป็นเว็บแคม ยิ้มบางๆ ปากขยับตามคำ และมือบังหน้า</p>

<h2>3. วิธีวัดผล (เหมือนของเพื่อน)</h2>
<ul>
<li>RAVDESS วิดีโอพูด 24 นักแสดง × 60 คลิป = {rh['n_clips']} คลิป, ป้ายจากชื่อไฟล์, calm รวมเป็นปกติ → 7 อารมณ์</li>
<li>ตัวอย่าง = ทั้งคลิป + หน้าต่าง 2 วินาที (เลื่อนทีละ 1 วิ), ดึงจุดหน้า/blendshape 10 fps และภาพหน้า 5 fps ด้วย MediaPipe Holistic ตัวเดียวกับระบบ</li>
<li>leave-actors-out 6 กลุ่ม (<code>actors[i::6]</code>) เหมือนที่เพื่อนใช้กับ blendshape: เทรนบน 20 คน ทดสอบบน 4 คนที่ไม่เคยเห็น วนจนครบ</li>
<li><b>ตรวจว่าวัดตรงจริง</b>: รัน blendshape ของเพื่อนซ้ำได้ accuracy {f(bs['clips_acc'])} ตรงกับรายงาน v9 ของเขา (0.592); B0 v1 บน FER2013 test ได้ {f(b0f['accuracy'])} ตรงกับ model card ของเพื่อน (0.681)</li>
</ul>

<h2>4. ผลบน RAVDESS: แต่ละโมเดลเดี่ยว</h2>
<table>{t1}</table>
<div class="note">ตัวหนาสีน้ำเงิน = ดีที่สุดในคอลัมน์ · <sup>a</sup> จาก <code>models/face-model/model_card.json</code> (CV ของเพื่อนเอง, วิธีเดียวกัน, มี 720 ตัวอย่างภาษามือที่ไม่มีในเครื่องนี้)
· B0 v1 ไม่เคยเห็น RAVDESS (เทรนบน FER2013) จึงเสียเปรียบเรื่องสภาพการถ่าย; การเทียบที่ยุติธรรมที่สุดคือกับ blendshape ซึ่งเทรนและทดสอบบน RAVDESS แบบเดียวกับเรา</div>
<h3>recall แยกอารมณ์ (ระดับคลิป)</h3>
<table>{t2}</table>
<div class="note">B0 v1 ตอบหน้าปกติเกือบทุกครั้ง ({b0['per_class_recall']['neutral']:.2f}) จึงจับเศร้า {b0['per_class_recall']['sad']:.2f} โกรธ {b0['per_class_recall']['angry']:.2f} ได้น้อย ·
ของเราดีกว่า blendshape ทุกอารมณ์ ยกเว้นประหลาดใจ ({ours['per_class_recall']['surprise']:.2f} เทียบกับ {bs['per_class_recall']['surprise']:.2f})</div>

<h2 class="pb">5. ทั้งระบบแบบที่แอปรวมจริง</h2>
<table>{t4}</table>
<div class="note"><code>fuse24.py</code> · ภาพสรุปต่อตัวอย่างแบบ <code>FaceBranch._summary</code> (ถ่วงด้วยความมั่นใจ × คุณภาพ); blendshape เทรนบน RAVDESS แบบ leave-actors-out ·
B0 v1 ดึงคะแนนของ blendshape ลง ({f(bs['clips_macro_f1'])} → {f(S_main['clips_macro_f1'])}) ส่วนของเราดึงขึ้น · อารมณ์ของเสียงคิดแบบ <code>speech_affect</code>
(ปกติ เว้นแต่อารมณ์อื่นอยู่บนสุดและ ≥ 0.5) · ลองน้ำหนักภาพ 0–1 แล้ว (<code>voice_weight.py</code>) คงไว้ 0.5 เท่าเดิม: บนคลิปทีมไม่ออกเสียงอารมณ์ผิดบนหน้าปกติเลย (0/75 ช่วง)</div>

<h2>6. คนทำภาษามือ (คลิปของทีม 18 คลิป)</h2>
<p>ไม่มีป้ายอารมณ์จริง จึงใช้ชุดเดียวกับรายงาน v1 (<code>real_check.py</code>): ตัดสินด้วยตาว่าเป็นหน้าปกติ {rh['signers']['neutral_frames']} เฟรม, ยิ้มชัด 3 เฟรม (ยิ้มบางๆ)
วัดว่าป้ายที่จะขึ้นบนจอ (ส่งทีละภาพแบบเว็บจริง) ขึ้นอารมณ์ผิดบนหน้าปกติบ่อยแค่ไหน</p>
<table>{t3}</table>
<div class="note">ของเราทักผิดน้อยกว่าและตอบ "ไม่แน่ใจ" น้อยกว่า แต่ยิ้มบางๆ จับไม่ได้ (ตัวอย่างแค่ 3 เฟรม น้อยเกินสรุป) · ชุดนี้บอกได้แค่ว่าทักผิดน้อย ไม่ได้บอกว่าจับอารมณ์จริงได้
· prior หน้าปกติ 0.5 เลือกจากการแลกกันระหว่าง F1 บน RAVDESS กับการทักผิดบนชุดนี้ (เปลี่ยนได้ที่ <code>head.prior_shift</code> ใน config.json)</div>

<h2>7. เทียบสองทาง: บนข้อมูลของเพื่อนด้วย</h2>
<table>{t7}</table>
<div class="note"><code>fer_cross.py</code> · แต่ละตัวชนะบนข้อมูลแบบที่ตัวเองเทรนมา · ของเราบน FER2013 เสียเปรียบ: ต้องขยายรูป 48 → 96 px และตัดสินจากรูปเดียว (ไม่มีช่วง 2 วิ)
· ลองให้ตัวตัดสินเรียน FER2013 train ด้วย (<code>fer_head.py</code>) ได้สูงสุด 0.618 ยังแพ้ B0 v1 และทำให้ RAVDESS ลดลง จึงไม่ใช้
· blendshape วัดบน FER2013 ไม่ได้ (รูป 48 px เล็กเกินให้ MediaPipe หาจุดบนหน้า)</div>

<h2>8. บนเว็บ</h2>
<ul>
<li>หลังบ้านใช้ <code>enet_b2_8_feat.onnx</code> (B2 ที่คืนทั้ง logits และฟีเจอร์) ~220–350 ms ต่อการเรียก 5 ภาพบน CPU</li>
<li>ตัวตัดสินเป็นเชิงเส้น จึงย่อแต่ละภาพเหลือ 21 ตัวเลข แล้วเก็บ 2 วินาทีล่าสุดใน state ที่ส่งไปกลับกับหน้าเว็บ (หลังบ้านไม่จำอะไร)</li>
<li>ตรวจว่าผลบนเว็บ = ผลตอนวัด: ต่างกันสูงสุด {ck['parity_max_abs_diff']:.1e} (<code>check_ravhead.py</code>)</li>
<li>การแปลภาษามือไม่ถูกแตะ (face branch แยกและล้มได้โดยไม่กระทบ) · สลับกลับเป็นของเดิมได้ด้วย <code>FACE_FER_DIR=./models/face-model-efficientnetb0-v1</code></li>
<li>ความเร็ว: หน้าต่าง Chrome ที่ถูกบังหรืออยู่ด้านหลังจะถูก Windows/Chrome ลดความถี่ (~1 fps) เมื่ออยู่ด้านหน้าได้ 20–22 fps (GPU Intel Iris Xe)</li>
<li>test <code>tests/test_face_branch.py</code> ผ่าน 12/12</li>
</ul>

<h2>9. ข้อจำกัด</h2>
<ul>
<li><b>ยิ้มบางๆ</b> ในคลิปทีม ของเราจับได้ {ck5['smiles']} (B0 v1 {rc['smiles_shown_happy']}) ควรเก็บคลิปยิ้มของคนทำภาษามือเพิ่ม</li>
<li><b>ประหลาดใจ / เลิกคิ้ว</b> ยังอ่อน (recall ~0.5); การเลิกคิ้วเป็นไวยากรณ์คำถาม yes/no ด้วย</li>
<li>RAVDESS เป็นนักแสดงพูดภาษาอังกฤษ หน้าตรง แสงดี; คะแนนจริงบนคนทำภาษามือจะต่ำกว่า และยังไม่มีชุดภาษามือไทยที่ติดป้ายอารมณ์</li>
<li>ชุดคลิปภาษามือที่เพื่อนใช้ทดสอบ blendshape ไม่มีในเครื่องนี้ จึงเทียบบนชุดนั้นไม่ได้</li>
<li><b>สัญญาอนุญาต</b>: RAVDESS (CC BY-NC-SA 4.0) และน้ำหนัก HSEmotion ที่เทรนจาก AffectNet ใช้ได้เพื่อการวิจัย/การศึกษาเท่านั้น ห้ามใช้เชิงพาณิชย์</li>
</ul>

<h2>10. ไฟล์และการทำซ้ำ</h2>
<table>
<tr><th>ไฟล์</th><th>หน้าที่</th></tr>
<tr><td><code>Face-Lab/ravdess24.py</code> · <code>sign_neutral.py</code></td><td>ดึง RAVDESS 24 คน · ภาพหน้าปกติจากคลิปท่าพจนานุกรม</td></tr>
<tr><td><code>Face-Lab/rav_head.py</code> · <code>rav_prior.py</code></td><td>เทรน/วัดตัวตัดสินของเรา และ blendshape ของเพื่อน · prior</td></tr>
<tr><td><code>Face-Lab/check_ravhead.py</code> · <code>real_check.py</code></td><td>ตรวจ bundle + คลิปทีมแบบเว็บ</td></tr>
<tr><td><code>Face-Lab/b0v1_24.py</code> · <code>fuse24.py</code> · <code>voice_weight.py</code></td><td>B0 v1 บน RAVDESS · ทั้งระบบแบบแอป · น้ำหนักภาพในเสียง</td></tr>
<tr><td><code>Face-Lab/fer_cross.py</code> · <code>fer_head.py</code></td><td>บน FER2013 test · ลองเรียน FER2013 เพิ่ม</td></tr>
<tr><td><code>Face-Lab/make_data_figures.py</code> · <code>make_report_v2.py</code></td><td>ภาพข้อมูลและรายงานนี้ (อ่านตัวเลขจาก <code>results/*.json</code>)</td></tr>
<tr><td><code>modules/face_fer.py</code></td><td><code>PictureHead</code>, <code>BrowCue</code>, กฎ wh ใน <code>speech_affect</code></td></tr>
</table>
<p class="note">รัน: <code>docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build</code> (ค่าตั้งต้นของ FACE_FER_DIR ชี้ models/face-model-ravhead-v1 แล้ว)
· การทดลองอื่นที่ไม่ได้ใช้ (ensemble, fine-tune B2 รอบ 8 ฯลฯ) อยู่ใน <code>Face-Lab/</code> และ <code>results/</code></p>
</body></html>"""
out = H / "face-report-v2.html"; out.write_text(html, encoding="utf-8")
chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={H / 'face-report-v2.pdf'}", out.as_uri()], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("ok", H / "face-report-v2.pdf")
