from annotate import add
U='ระบุไม่ได้ชัดจากภาพ'
def record(book,text):
 groups={}
 for line in text.strip().splitlines():
  if not line.strip():continue
  p,w,h,shape,loc,motion=line.split('|');groups.setdefault(int(p),[]).append([w,int(h) if h.isdigit() else h,shape,U,loc,motion,'ทำตามลำดับภาพ/ลูกศร; ไม่กำหนดจำนวนครั้งที่ภาพไม่ระบุ',None])
 for p,es in groups.items():add(book,p,'vocabulary',es)
WORDS={'open':'แบมือ','spread':'กางนิ้ว','fist':'กำมือ','point':'นิ้วชี้เหยียด','two':'นิ้วชี้–กลางเหยียด','pinch':'นิ้วโป้งจรดนิ้วอื่น/หยิบ','curved':'นิ้วโค้ง','flat':'นิ้วเหยียดชิด','hang':'มือห้อย/งอข้อมือ','thumb':'นิ้วโป้งยื่น','L':'นิ้วโป้งและชี้กางเป็นมุม','specific':'รูปนิ้วเฉพาะตามภาพขยาย','support':'อีกมือรอง','stack':'สองมือซ้อนกัน','cross':'สองมือไขว้กัน','indexpinky':'นิ้วชี้และก้อยเหยียด','unknown':'ภาพไม่ชัดพอระบุรูปนิ้ว','front':'หน้าลำตัว','chest':'หน้าอก','mouth':'ปาก','chin':'คาง','nose':'จมูก','cheek':'แก้ม','eye':'รอบตา','ear':'ข้างหู','head':'ศีรษะ','forehead':'หน้าผาก','side':'ข้างลำตัว','shoulder':'ข้างไหล่','arm':'ท่อนแขน','elbow':'ข้อศอก','wrist':'ข้อมือ','hand':'มืออีกข้าง','waist':'เอว','abdomen':'หน้าท้อง','leg':'ขา','foot':'เท้า','neck':'คอ','back':'หลัง','palm':'ฝ่ามืออีกข้าง','fingers':'นิ้วมืออีกข้าง','up':'ยกขึ้น','down':'เลื่อนลง','out':'เคลื่อนออก','in':'เคลื่อนเข้า','apart':'แยกสองมือออก','together':'เคลื่อนสองมือเข้าหากัน','circle':'วนตามลูกศร','turn':'หมุน/พลิกมือ','shake':'สั่น/แกว่งสั้นซ้ำ','tap':'แตะ/เคาะตามสัญลักษณ์','rub':'ถู/ปัดผ่าน','sweep':'กวาดตามแนวลูกศร','point_at':'ชี้ตำแหน่งที่แสดง','hold':'ภาพไม่ระบุการเคลื่อนชัด','change':'เปลี่ยนรูปมือตามลำดับภาพ','openclose':'กาง–หุบนิ้ว/มือ','bend':'งอนิ้วตามภาพ','alternate':'ขยับสลับสองข้าง','trace':'ลากตามแนวตำแหน่งที่แสดง','pull':'ดึงออกตามลูกศร','flick':'ดีด/สะบัดนิ้วตามลูกศร','pour':'เอียงมือคล้ายเทตามภาพ','roll':'ม้วน/หมุนมือเข้าหากัน'}
def trans(s):
 return ' → '.join(' + '.join(WORDS.get(x,x) for x in t.split('+')) for t in s.split('>'))
def coded(book,text):
 groups={}
 for line in text.strip().splitlines():
  if not line.strip():continue
  p,w,h,shape,loc,motion=line.split('|');groups.setdefault(int(p),[]).append([w,int(h) if h.isdigit() else h,trans(shape),U,trans(loc),trans(motion),'ตามภาพจากจังหวะแรกไปจังหวะถัดไป; ไม่กำหนดจำนวนครั้งเพิ่ม',None])
 for p,es in groups.items():add(book,p,'vocabulary',es)
