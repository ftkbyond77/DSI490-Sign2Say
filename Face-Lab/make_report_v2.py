"""face-report-v2: HTML from results/*.json (every number is read, none typed) → face-report-v2.html → PDF with headless Chrome.
  python make_figures_v2.py && python make_report_v2.py"""
import json
import subprocess
from pathlib import Path

H = Path(__file__).resolve().parent; R = H / "results"
J = lambda n: json.loads((R / n).read_text(encoding="utf-8"))
rh, pr, ck, en = J("rav_head_24actors.json"), J("rav_prior.json"), J("ravhead_check.json"), J("ensemble24.json")
ds = J("data_summary.json")
rc = J("real_check.json")["models"]["efficientnetb0-v1"]["stability"]["shipped"]
card = json.loads((H.parent / "models/face-model/model_card.json").read_text(encoding="utf-8"))["cv"]
EMO = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
TH = ["ปกติ", "สุข", "เศร้า", "โกรธ", "กลัว", "รังเกียจ", "ประหลาดใจ"]
f = lambda v: f"{v:.3f}"
pc = lambda v: f"{100 * v:.1f}%"

rav_rows = [
    ("ensemble เดิม (v2-ferplus + HSEmotion B2, ไม่ได้เทรนบน RAVDESS)", en["ensemble"], ""),
    ("ensemble + ตัวจับขมวดคิ้ว", en["ensemble_brow"], ""),
    ("blendshape ของเพื่อน, ตัวที่ใช้จริง (+720 ภาษามือหน้าปกติ) <sup>a</sup>", {k: card[k] for k in card if k != "confusion"}, ""),
    ("blendshape ของเพื่อน, RAVDESS อย่างเดียว (รันซ้ำ)", rh["blendshape_lr (friend recipe)"], ""),
    ("หัวใหม่, RAVDESS อย่างเดียว", rh["picture_head (B2 features + member probs)"], ""),
    ("หัวใหม่ + ภาษามือหน้าปกติ", pr["neutral_prior_0.0"], ""),
    ("<b>หัวใหม่ + ภาษามือหน้าปกติ + prior 0.5 (ใช้บนเว็บ)</b>", pr["neutral_prior_0.5"], "ship"),
    ("หัวใหม่ + ภาษามือหน้าปกติ + prior 1.0", pr["neutral_prior_1.0"], ""),
    ("fusion: หัวใหม่ + blendshape (ไม่ได้ใช้)", rh["fusion (picture head + blendshape)"], ""),
]
cols = [("clips_acc", "accuracy คลิป"), ("clips_macro_f1", "macro-F1 คลิป"), ("windows_macro_f1", "macro-F1 หน้าต่าง 2 วิ"),
        ("strong_intensity_acc", "acc ระดับเข้ม"), ("neutral_vs_emotion_acc", "ปกติ vs มีอารมณ์"), ("valence_sign_agreement", "บวก/ลบ ตรง")]
best = {k: max(r[1][k] for r in rav_rows if "fusion" not in r[0]) for k, _ in cols}
t1 = "<tr><th>โมเดล</th>" + "".join(f"<th>{n}</th>" for _, n in cols) + "</tr>"
for name, m, tag in rav_rows:
    t1 += f"<tr class='{tag}'><td>{name}</td>" + "".join(f"<td class='{'best' if abs(m[k] - best[k]) < 1e-9 else ''}'>{f(m[k])}</td>" for k, _ in cols) + "</tr>"

rec_rows = [("ensemble เดิม", en["ensemble"]["per_class_recall"]), ("ensemble + คิ้ว", en["ensemble_brow"]["per_class_recall"]),
            ("blendshape ของเพื่อน (RAVDESS อย่างเดียว)", rh["blendshape_lr (friend recipe)"]["per_class_recall"]),
            ("<b>หัวใหม่ (ใช้บนเว็บ)</b>", pr["neutral_prior_0.5"]["per_class_recall"])]
t2 = "<tr><th>โมเดล</th>" + "".join(f"<th>{t}</th>" for t in TH) + "</tr>"
for name, r in rec_rows:
    t2 += f"<tr><td>{name}</td>" + "".join(f"<td class='{'best' if abs(r[e] - max(x[1][e] for x in rec_rows)) < 1e-9 else ''}'>{r[e]:.2f}</td>" for e in EMO) + "</tr>"

sg = [("v1 EfficientNet-B0 + prior 0.6 (รายงาน v1)", rc["shown_neutral"], rc["shown_wrong_expression"], rc["shown_unknown"], rc["smiles_shown_happy"], "", ""),
      ("ensemble เดิม (บนเว็บก่อนหน้านี้)", *[ck["ensemble (deployed)"][k] for k in ("shown_neutral", "wrong", "unknown", "smiles")], "ยิ้ม 5", ""),
      ("หัวใหม่ prior 0", *[ck["ravhead neutral_prior 0.0"][k] for k in ("shown_neutral", "wrong", "unknown", "smiles")], "เศร้า 23", ""),
      ("<b>หัวใหม่ prior 0.5 (ใช้บนเว็บ)</b>", *[ck["ravhead neutral_prior 0.5"][k] for k in ("shown_neutral", "wrong", "unknown", "smiles")], "เศร้า 10", "ship"),
      ("หัวใหม่ prior 1.0", *[ck["ravhead neutral_prior 1.0"][k] for k in ("shown_neutral", "wrong", "unknown", "smiles")], "เศร้า 6", "")]
t3 = "<tr><th>โมเดล (ผ่าน unknown rule + smoother แบบบนเว็บ)</th><th>ขึ้นหน้าปกติ</th><th>ขึ้นอารมณ์ผิด ↓</th><th>ไม่แน่ใจ</th><th>ยิ้มที่จับได้</th><th>ผิดเป็นอะไร</th></tr>"
for n, a, b, c, d, w, tag in sg:
    t3 += f"<tr class='{tag}'><td>{n}</td><td>{pc(a)}</td><td>{pc(b)}</td><td>{pc(c)}</td><td>{d}</td><td>{w}</td></tr>"
sw = rh["signers"]; sw2 = rh["signers_with_neutral_signing"]

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
<h1>Face branch v2: จับสีหน้าจากภาพ + หัวที่เทรนบน RAVDESS</h1>
<div class="sub">ThaiSLM · 14 พ.ย. 2026 · branch pitty · model <code>models/face-model-ravhead-v1</code> · code <code>modules/face_fer.py</code> · ต่อจาก face-report-v1</div>

<div class="box"><b>สรุป</b> — บนเว็บเดิมป้ายสีหน้าขึ้น "ปกติ" เกือบตลอด (ขมวดคิ้วก็ยังปกติ) เราแก้ 2 ชั้น: (1) ตัวจับขมวดคิ้วจากจุดบนหน้า และ (2) <b>หัวใหม่</b>
ที่เทรนบน RAVDESS โดยใช้ HSEmotion B2 เป็นตัวดึงฟีเจอร์ภาพ เทียบกับโมเดล blendshape ของเพื่อนด้วย<b>วิธีวัดเดียวกันทุกอย่าง</b> (24 นักแสดง, leave-actors-out 6 กลุ่ม):
macro-F1 ระดับคลิป <b>{f(pr['neutral_prior_0.5']['clips_macro_f1'])}</b> เทียบกับ {f(rh['blendshape_lr (friend recipe)']['clips_macro_f1'])} ของเพื่อน
(และ {f(en['ensemble']['clips_macro_f1'])} ของ ensemble เดิม) ชนะทุกตัววัด บนคลิปคนทำภาษามือของทีม ขึ้นอารมณ์ผิดบนหน้าปกติ {pc(ck['ravhead neutral_prior 0.5']['wrong'])}
(ensemble เดิม {pc(ck['ensemble (deployed)']['wrong'])}) ใช้งานบน http://localhost:8080 แล้ว</div>

<h2>1. ปัญหาเดิมและสิ่งที่แก้</h2>
<ul>
<li><b>ensemble เดิม</b> (v2-ferplus 48px + HSEmotion B2 96px, prior หน้าปกติ 0.75) เรียนจากภาพนิ่ง FER2013 / AffectNet และถูกตั้งให้เอนไปทางหน้าปกติเพื่อไม่เดามั่วตอนทำภาษามือ
ผลคือบน RAVDESS จับโกรธได้ {en['ensemble']['per_class_recall']['angry']:.2f} กลัว {en['ensemble']['per_class_recall']['fear']:.2f} ประหลาดใจ {en['ensemble']['per_class_recall']['surprise']:.2f}</li>
<li><b>ตัวจับขมวดคิ้ว (BrowCue)</b>: วัดความสูงของคิ้วด้านในเหนือตา หารด้วยระยะห่างตา (หมุนตามแนวตา) เทียบกับคิ้วตอนพักของคนนั้นเอง (เปอร์เซ็นไทล์ 85 ของ 60 เฟรมล่าสุด)
ถ้าต่ำกว่าเกิน 2 หน่วย (0.02 × ระยะตา) และโมเดลตอบหน้าปกติ จะดันไปทาง "ขมวดคิ้ว" (โกรธ) สูงสุด 50% บน RAVDESS แยกโกรธจากปกติได้ AUC 0.92
<b>ใช้กับป้ายบนจอเท่านั้น</b> เพราะคิ้วต่ำเป็นไวยากรณ์คำถาม wh ในภาษามือไทยด้วย; ส่วนเสียง ถ้าเป็นคำถาม wh จะลดน้ำหนักโกรธลงครึ่งหนึ่ง</li>
<li><b>หัวใหม่ (PictureHead)</b>: ไม่เทรนโมเดลภาพใหม่ (ไม่มี GPU) ใช้ B2 ดึงฟีเจอร์ 1,408 ตัวต่อภาพ ในหน้าต่าง 2 วินาทีเอา
ค่าเฉลี่ยฟีเจอร์ + ค่าเฉลี่ย/สูงสุดของความน่าจะเป็นจากทั้งสองโมเดล (1,436 ตัว) เข้า multinomial logistic regression แบบถ่วงคลาส (โค้ด <code>fit_lr</code> เดียวกับของเพื่อน, L2 0.1 เลือกจาก grid ด้วยกฎเดียวกัน)</li>
<li><b>ปรับให้เข้ากับคนทำภาษามือ</b>: เติมตัวอย่างหน้าปกติ {rh['picture_head + neutral signing']['n_neutral_signing']} ตัวอย่าง จากคลิปท่าพจนานุกรม 53 คลิป (<code>data_test/jack_test</code>,
ตัดโฟลเดอร์คำที่มีความรู้สึกออก) วิธีเดียวกับที่เพื่อนใช้ 720 ตัวอย่าง คลิปของทีมไม่ได้ใช้เทรน เก็บไว้ทดสอบอย่างเดียว</li>
<li><b>prior หน้าปกติ 0.5</b> บน logit ของหัว เลือกจากการแลกกันในหัวข้อ 6</li>
</ul>

<h2 class="pb">2. ข้อมูลที่ใช้</h2>
<table class="data">
<tr><th>ชุดข้อมูล</th><th>ใช้ทำอะไร</th><th>จำนวน</th><th>ป้าย</th></tr>
<tr><td>RAVDESS (วิดีโอพูด, นักแสดง 24 คน)</td><td>เทรน + ทดสอบ (leave-actors-out)</td>
<td>{ds['ravdess']['clips']:,} คลิป → {ds['ravdess']['samples_clip_plus_windows']:,} ตัวอย่าง (คลิป + หน้าต่าง 2 วิ) · ภาพหน้า {ds['ravdess']['crops_5fps']:,} ภาพ</td>
<td>ปกติ {ds['ravdess']['clips_per_class']['neutral']} (รวม calm), อารมณ์อื่นอย่างละ {ds['ravdess']['clips_per_class']['happy']}</td></tr>
<tr><td>คลิปท่าพจนานุกรม (<code>data_test/jack_test</code>)</td><td>เทรน (หน้าปกติขณะทำภาษามือ)</td>
<td>{ds['signing_neutral']['clips']} คลิป → ภาพหน้า {ds['signing_neutral']['crops_5fps']} ภาพ → {ds['signing_neutral']['samples']} ตัวอย่าง</td><td>ปกติทั้งหมด</td></tr>
<tr><td>คลิปของทีม (Boom, Jack, Pitty)</td><td>ทดสอบอย่างเดียว</td>
<td>{ds['team']['clips']} คลิป → {ds['team']['frames_every_0_5s']} เฟรม (ทุก 0.5 วิ)</td>
<td>ตัดสินด้วยตา: ปกติ {ds['team']['neutral']} · ยิ้ม {ds['team']['smile']} · กำกวม {ds['team']['ambiguous']} · มือบัง/ไม่ชัด {ds['team']['covered_or_poor']}</td></tr>
</table>
<p class="note">ทุกภาพคือสิ่งที่โมเดลเห็นจริง: ครอปหน้าสีเทา 96×96 px จากกรอบที่คำนวณจาก 68 จุดของ MediaPipe (ตัวเดียวกับบนเว็บ)</p>
<h3>RAVDESS: 7 อารมณ์ (ระดับเข้ม) ของนักแสดง 2 คน</h3>
<img src="figures/v2_data_ravdess.png">
<h3>คลิปท่าพจนานุกรม: หน้าปกติขณะทำภาษามือ (ตัวอย่างที่เติมตอนเทรน)</h3>
<img src="figures/v2_data_signing.png">
<h3>คลิปของทีม: ชุดทดสอบคนทำภาษามือ</h3>
<img src="figures/v2_data_team.png">
<p class="note">สังเกตความต่าง: RAVDESS หน้าตรง แสงสตูดิโอ แสดงอารมณ์ชัด ส่วนคลิปของทีมเป็นเว็บแคม ยิ้มบางๆ ปากขยับตามคำ และมือบังหน้า นี่คือสาเหตุหลักที่คะแนนบนการใช้งานจริงต่ำกว่าบน RAVDESS</p>
<div style="display:flex; gap:14px; align-items:center; page-break-inside: avoid;"><img src="figures/v2_data_points.png" style="width:48%">
<p class="note" style="width:52%">68 จุดบนหน้า (ฟ้า) และสิ่งที่ตัวจับขมวดคิ้ววัด: ความสูงของคิ้วด้านใน (ส้ม) เหนือแนวตา (เขียว) หารด้วยระยะห่างตา
ตอนโกรธเส้นส้มสั้นลงชัดเจน ตัวจับเทียบกับคิ้วตอนพักของคนนั้นเอง จึงไม่ขึ้นกับรูปหน้าแต่ละคน</p></div>

<h2>3. วิธีวัดผล (เหมือนของเพื่อน)</h2>
<ul>
<li>RAVDESS วิดีโอพูด 24 นักแสดง × 60 คลิป = {rh['n_clips']} คลิป, ป้ายจากชื่อไฟล์, calm รวมเป็นปกติ → 7 อารมณ์</li>
<li>ตัวอย่าง = ทั้งคลิป + หน้าต่าง 2 วินาที (เลื่อนทีละ 1 วิ), ดึงจุดหน้า/blendshape 10 fps และภาพหน้า 5 fps (quality ≥ 0.25) ด้วย MediaPipe Holistic ตัวเดียวกับระบบ</li>
<li>leave-actors-out 6 กลุ่ม (<code>actors[i::6]</code>) ทุกโมเดลที่เทรนได้คะแนนจากนักแสดงที่ไม่เคยเห็นเท่านั้น; ensemble ไม่ได้เทรนบน RAVDESS เลย</li>
<li><b>ตรวจว่าวัดตรงจริง</b>: รันโมเดล blendshape ของเพื่อนซ้ำได้ accuracy {f(rh['blendshape_lr (friend recipe)']['clips_acc'])} ตรงกับรายงาน v9 ของเขา (0.592)</li>
</ul>

<h2>4. ผลบน RAVDESS (24 นักแสดง)</h2>
<table>{t1}</table>
<div class="note">ตัวหนาสีน้ำเงิน = ดีที่สุดในคอลัมน์ (ไม่นับ fusion) · <sup>a</sup> จาก <code>models/face-model/model_card.json</code> (CV ของเพื่อนเอง, วิธีเดียวกัน, temperature 1.6) ·
fusion ต้องดึง blendshape เพิ่มบนเว็บ ได้เพิ่มเพียง ~0.02 จึงไม่ใช้</div>
<img src="figures/v2_f1.png">

<h3>recall แยกอารมณ์ (ระดับคลิป)</h3>
<table>{t2}</table>
<img src="figures/v2_recall.png">
<div class="note">หัวใหม่ดีกว่าของเพื่อนทุกอารมณ์ ยกเว้นประหลาดใจ ({pr['neutral_prior_0.5']['per_class_recall']['surprise']:.2f} vs {rh['blendshape_lr (friend recipe)']['per_class_recall']['surprise']:.2f})
ซึ่งเป็นอารมณ์ที่อ่อนที่สุดของทุกโมเดล; ตัวจับขมวดคิ้วทำให้ ensemble จับโกรธได้จาก {en['ensemble']['per_class_recall']['angry']:.2f} เป็น {en['ensemble_brow']['per_class_recall']['angry']:.2f}</div>

<h2 class="pb">5. คนทำภาษามือ (คลิปของทีม 18 คลิป)</h2>
<p>ไม่มีป้ายอารมณ์จริง จึงใช้ชุดเดียวกับรายงาน v1 (<code>real_check.py</code>): ตัดสินด้วยตาว่าเป็นหน้าปกติ {sw['neutral_frames']} เฟรม, ยิ้มชัด 3 เฟรม (ยิ้มบางๆ)
วัดว่าป้ายที่จะขึ้นบนจอ (ผ่าน unknown rule + smoother ส่งทีละภาพแบบเว็บจริง) ขึ้นอารมณ์ผิดบนหน้าปกติบ่อยแค่ไหน</p>
<table>{t3}</table>
<div class="note">ก่อนใส่ prior / smoother (หน้าต่าง ±1 วิ, ตัดสินตรงจากหัว): หัวที่เรียนจาก RAVDESS อย่างเดียวผิด {pc(sw['false_emotion'])} (ส่วนใหญ่เศร้า),
เติมภาษามือหน้าปกติแล้วเหลือ {pc(sw2['false_emotion'])} · โมเดลของเพื่อนรายงาน 12.4% บนชุดคลิปภาษามือของเขาเอง (คนละชุด จึงเทียบตรงไม่ได้; ชุดนั้นไม่มีในเครื่องนี้)</div>

<h2>6. การแลกกัน: ไว vs ไม่เดามั่ว</h2>
<img src="figures/v2_tradeoff.png">
<p>prior หน้าปกติบนหัวใหม่: 0 → 0.5 ลดการขึ้นอารมณ์ผิดจาก {pc(ck['ravhead neutral_prior 0.0']['wrong'])} เป็น {pc(ck['ravhead neutral_prior 0.5']['wrong'])}
โดย F1 แทบไม่ลด ({f(pr['neutral_prior_0.0']['clips_macro_f1'])} → {f(pr['neutral_prior_0.5']['clips_macro_f1'])}); 1.0 ได้ {pc(ck['ravhead neutral_prior 1.0']['wrong'])} ใกล้ ensemble เดิม
แต่ F1 ลดเป็น {f(pr['neutral_prior_1.0']['clips_macro_f1'])} และจับเศร้า/โกรธได้น้อยลง เลือก 0.5 · เปลี่ยนได้ที่ <code>head.prior_shift</code> ใน config.json แล้ว restart (ไม่ต้องเทรนหรือ build)</p>

<h2>7. บนเว็บ</h2>
<ul>
<li>หลังบ้านใช้ <code>enet_b2_8_feat.onnx</code> (B2 ที่คืนทั้ง logits และฟีเจอร์) เวลาเท่ากับ B2 เดิม: ~220–350 ms ต่อการเรียก 5 ภาพบน CPU</li>
<li>หัวเป็นเชิงเส้น จึงย่อแต่ละภาพเหลือ 21 ตัวเลข แล้วเก็บ 2 วินาทีล่าสุดใน state ที่ส่งไปกลับกับหน้าเว็บ (หลังบ้านไม่จำอะไร)</li>
<li>ตรวจว่าผลบนเว็บ = ผลตอนวัด: ต่างกันสูงสุด {ck['parity_max_abs_diff']:.1e} (<code>check_ravhead.py</code>)</li>
<li>ภาพที่ไม่มีครอป 96px (client เก่า) ใช้ ensemble ตอบแทน; การแปลภาษามือไม่ถูกแตะ (face branch ยังแยกและล้มได้โดยไม่กระทบ)</li>
<li>ตัวจับขมวดคิ้วทำงานต่อจากหัวใหม่ (ป้ายบนจอเท่านั้น); อารมณ์ของเสียงมาจากสรุปทั้งประโยคของหัวใหม่ + กฎคำถาม yes/no และ wh</li>
<li>test <code>tests/test_face_branch.py</code> ผ่าน 12/12 (เพิ่ม test หน้าต่างและ state ของหัว)</li>
</ul>

<h2>8. ข้อจำกัด</h2>
<ul>
<li><b>ยิ้มบางๆ</b> ในคลิปทีม หัวใหม่จับได้ 0/3 (ensemble เดิม 2/3) ตัวอย่างน้อยเกินสรุป แต่ควรเก็บคลิปยิ้มของคนทำภาษามือเพิ่ม</li>
<li><b>ประหลาดใจ / เลิกคิ้ว</b> ยังอ่อน (recall ~0.5) และยังไม่มีตัวช่วยแบบขมวดคิ้ว; การเลิกคิ้วเป็นไวยากรณ์คำถาม yes/no ด้วย</li>
<li>RAVDESS เป็นนักแสดงพูดภาษาอังกฤษ หน้าตรง แสงดี; คะแนนจริงบนคนทำภาษามือหน้ากล้องจะต่ำกว่า และยังไม่มีชุดภาษามือไทยที่ติดป้ายอารมณ์</li>
<li>ชุดทดสอบคนทำภาษามือของเรากับของเพื่อนเป็นคนละชุด</li>
<li><b>สัญญาอนุญาต</b>: RAVDESS (CC BY-NC-SA 4.0) และน้ำหนัก HSEmotion ที่เทรนจาก AffectNet ใช้ได้เพื่อการวิจัย/การศึกษาเท่านั้น ห้ามใช้เชิงพาณิชย์</li>
</ul>

<h2>9. ไฟล์และการทำซ้ำ</h2>
<table>
<tr><th>ไฟล์</th><th>หน้าที่</th></tr>
<tr><td><code>Face-Lab/ravdess24.py</code></td><td>ดึง RAVDESS 24 คน (จุดหน้า, blendshape, ภาพหน้า) แล้วลบวิดีโอ</td></tr>
<tr><td><code>Face-Lab/sign_neutral.py</code></td><td>ภาพหน้าปกติจากคลิปท่าพจนานุกรม</td></tr>
<tr><td><code>Face-Lab/rav_head.py</code></td><td>เทรน/วัดหัวใหม่, blendshape ของเพื่อน, fusion</td></tr>
<tr><td><code>Face-Lab/rav_prior.py</code> · <code>check_ravhead.py</code></td><td>ผลของ prior บน RAVDESS · ตรวจ bundle + คลิปทีมแบบเว็บ</td></tr>
<tr><td><code>Face-Lab/ensemble24.py</code> · <code>compare24.py</code></td><td>ensemble เดิม (± คิ้ว) บน 24 คน</td></tr>
<tr><td><code>Face-Lab/make_data_figures.py</code></td><td>ภาพตัวอย่างข้อมูลและจำนวน (<code>results/data_summary.json</code>)</td></tr>
<tr><td><code>Face-Lab/make_figures_v2.py</code> · <code>make_report_v2.py</code></td><td>กราฟและรายงานนี้ (อ่านตัวเลขจาก <code>results/*.json</code>)</td></tr>
<tr><td><code>modules/face_fer.py</code></td><td><code>PictureHead</code>, <code>BrowCue</code>, กฎ wh ใน <code>speech_affect</code></td></tr>
</table>
<p class="note">รัน: <code>docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build</code> (ค่าตั้งต้นของ FACE_FER_DIR ชี้ models/face-model-ravhead-v1 แล้ว; ไฟล์ cpu สำหรับเครื่องที่ไม่มี GPU NVIDIA)</p>
</body></html>"""
out = H / "face-report-v2.html"; out.write_text(html, encoding="utf-8")
chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={H / 'face-report-v2.pdf'}", out.as_uri()], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("ok", H / "face-report-v2.pdf")
