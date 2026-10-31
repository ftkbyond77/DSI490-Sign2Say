import json,hashlib
from pathlib import Path
O=Path('outputs/hybrid-pipeline');B=Path('/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say/New data/หนังสือ')
def variant(id,book,page,pdf,region,hands,kind,location,trajectory,repetition,active,y,dist=None,checks=None,note=''):
 return dict(id=id,ref_book_page=f'เล่ม {book} หน้า {page}',source=dict(file=f'ThaiSignLanguageBook_{book}.pdf',printed_page=page,pdf_page=pdf,region=region),num_hands=hands,bimanual_type=kind,primary_location=location,movement_trajectory=trajectory,repetition=repetition,active_hand=active,wrist_y_min=y[0],wrist_y_max=y[1],target_distance_lr_wrist_max=dist,checks=checks or {},interpretation_note=note)
def wai(id,page,pdf):return variant(id,1,page,pdf,'upper',2,'symmetric','chest_torso','static',False,'both',[-.5,.65],.5,note='ภาพประนมมือ; ไม่มีลูกศรการเคลื่อนในภาพนี้ ไม่แปลว่าข้อมือต้องนิ่งตลอดคลิป')
entries=[
 dict(concept='สวัสดี',variants=[wai('wai',1,11),variant('salute',1,1,11,'lower',1,'single','head_face','linear',False,'either',[-.9,.3],note='มือเดียวเคลื่อนจากใกล้หน้าผากออกไป; ไม่บังคับข้างตามรูป')]),
 dict(concept='ขอบคุณ',variants=[wai('wai',2,12),variant('outward',1,2,12,'lower',1,'single','chest_torso','linear',False,'either',[-.3,.9],note='ภาพมือสองตำแหน่งแสดงลำดับการเคลื่อนของมือเดียว ไม่ใช่สองมือพร้อมกัน')]),
 dict(concept='ขอโทษ',variants=[wai('wai',3,13),variant('rub_circle',1,3,13,'lower',2,'asymmetric','chest_torso','circling',True,'dominant',[-.2,1.1],.9,{'circling_min':.35,'speed_balance_max':.65},'มือหนึ่งวนเหนือ/บนอีกมือ; ค่าพิกัดเป็น proxy ของภาพ ไม่ตรวจสัมผัสจริง')]),
 dict(concept='ภาษามือ',variants=[variant('two_circles',1,7,17,'upper',2,'symmetric','chest_torso','circling',True,'both',[-.5,1.1],None,{'circling_min':.35},'สองมือวนในพื้นที่หน้าลำตัวและมีลำดับ 1–3; ไม่บังคับว่ามือต้องชิดกันหรือมีความเร็วเท่ากันทุกเฟรม')]),
 dict(concept='เพื่อน',variants=[variant('swap_contact',1,84,95,'lower',2,'asymmetric','chest_torso','alternating',None,'both',[0,1.2],.95,{'alternating_min':.4},'ภาพ 1–2 สลับบทบาทนิ้วและมือ; ไม่ยืนยันว่าต้องทำซ้ำหลายรอบ การสลับ Y ข้อมือเป็นเพียง proxy ซึ่งอาจไม่เกิดชัด')]),
 dict(concept='ข้าว',variants=[variant('finger_motion',3,46,59,'upper',1,'single','chest_torso','linear',True,'either',[-.1,1.1],note='ภาพเน้นนิ้วขยับพร้อมลูกศรซ้ำ ไม่ใช่ท่ากินที่ปาก; ไม่ใช้วิถีข้อมือตัดสินการเคลื่อนนิ้ว')])]
meta=dict(schema_version=1,scope='six requested core concepts; not exhaustive across all six books',status='experimental_image_interpretation_not_expert_validated',coordinate_system='isotropic 2D / shoulder width; origin mid-shoulder; y positive downward',numeric_threshold_source='engineering hypotheses fixed before this dev rerun; NOT measurements or standards from the books',manual_weight=.5,bonus_range=[-.2,.15],missing_hand_policy='if either observed rate < .50, disable all bimanual constraints; no number-of-hands inference from detector presence',trajectory_limitations='2D wrist proxies do not capture finger articulation, contact, palm orientation or depth; repetition metadata is not scored',sources=[dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [B/'ThaiSignLanguageBook_1.pdf',B/'ThaiSignLanguageBook_3.pdf']],concepts=entries)
(O/'manual_heuristics_tsl_detailed.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
lines=['# กฎที่ถอดจากภาพในหนังสือ (ชุดทดลอง)', '', 'เลขหน้าเล่มและเลขหน้า PDF ระบุแยกกัน ตรวจภาพจริงของคำหลัก 6 คำในเล่ม 1 และ 3 ไม่ได้อ้างว่าตรวจครบทุกคำในเล่ม 1–6 ค่าตัวเลขเป็นสมมติฐานทางวิศวกรรม ไม่ใช่ค่ามาตรฐานที่หนังสือระบุ การจัดหมวดเป็นการตีความภาพที่ยังไม่ได้ตรวจโดยผู้เชี่ยวชาญ', '', '| concept / variant | ref_book_page (PDF) | num_hands | bimanual_type | primary_location | movement_trajectory | repetition | landmark_heuristic_rule |','|---|---|---|---|---|---|---|---|']
for e in entries:
 for v in e['variants']:
  rule=f"Y ข้อมือ [{v['wrist_y_min']}, {v['wrist_y_max']}] / shoulder width"
  if v['target_distance_lr_wrist_max'] is not None:rule+=f"; dist(L,R) ≤ {v['target_distance_lr_wrist_max']}"
  if v['checks']:rule+='; '+json.dumps(v['checks'])
  lines.append(f"| {e['concept']} / {v['id']} | {v['ref_book_page']} (PDF {v['source']['pdf_page']}) | {v['num_hands']} | {v['bimanual_type']} | {v['primary_location']} | {v['movement_trajectory']} | {json.dumps(v['repetition'])} | {rule} |")
lines+=['','`null` หมายถึงภาพไม่ยืนยันเรื่องทำซ้ำ จึงไม่เดาเป็น false','', '## หมายเหตุราย variant','']
for e in entries:
 for v in e['variants']:lines.append(f"- {e['concept']} / {v['id']}: {v['interpretation_note']}")
(O/'BOOK_RULES.md').write_text('\n'.join(lines)+'\n')
