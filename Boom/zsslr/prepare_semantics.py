"""Reproducible dictionary semantics: conservative evidence, grouped splits, real BERT CLS."""
from pathlib import Path
import json, re, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data'
# field, label, positive phrase pattern: project-defined schema, not a paper reproduction.
SPEC=[
('handshape','กำหมัด',r'กำมือ|กำหมัด'),('handshape','แบมือ',r'แบมือ'),
('handshape','ชี้นิ้วชี้',r'นิ้วชี้เหยียด|เหยียดนิ้วชี้'),('handshape','นิ้วชี้กลางเหยียด',r'นิ้วชี้[–\-และ]+กลาง|นิ้วชี้และนิ้วกลาง'),
('handshape','จีบ',r'จีบ'),('handshape','หยิบ',r'หยิบ'),('handshape','นิ้วโค้ง',r'นิ้วโค้ง'),
('handshape','นิ้วเหยียดชิด',r'นิ้วเหยียดชิด'),('handshape','กางนิ้ว',r'กางนิ้ว'),
('handshape','นิ้วโป้งยื่น',r'นิ้วโป้งยื่น|นิ้วโป้งเหยียด'),('handshape','รูปแอล',r'กางเป็นมุม|รูป L'),
('handshape','ชี้และก้อยเหยียด',r'นิ้วชี้และก้อยเหยียด'),('handshape','โป้งจรดนิ้วอื่น',r'นิ้วโป้งจรด'),
('handshape','งอข้อมือ',r'งอข้อมือ|มือห้อย'),('handshape','นิ้วไขว้',r'นิ้วไขว้|ไขว้นิ้ว'),
('handshape','รูปวงโอ',r'รูปวง|รูปโอ|รูป O'),('handshape','นิ้วกลางเหยียด',r'(?<!และ)นิ้วกลางเหยียด'),
('handshape','นิ้วก้อยเหยียด',r'นิ้วก้อยเหยียด'),('handshape','โป้งงอ',r'นิ้วโป้งงอ'),
('handshape','นิ้วงอ',r'งอนิ้ว|นิ้ว.*งอ'),
('location','ศีรษะหน้าผาก',r'ศีรษะ|หน้าผาก'),('location','ตา',r'รอบตา|ดวงตา|ข้างตา|คิ้ว'),
('location','จมูก',r'จมูก'),('location','ปาก',r'ปาก'),('location','คาง',r'คาง'),
('location','แก้มหู',r'แก้ม|หู'),('location','คอ',r'คอ'),('location','ไหล่',r'ไหล่'),
('location','หน้าอก',r'หน้าอก'),('location','ท้องเอว',r'หน้าท้อง|เอว'),
('location','หน้าลำตัว',r'หน้าลำตัว'),('location','แขนหรือมืออีกข้าง',r'ท่อนแขน|ข้อศอก|ข้อมือ|มืออีกข้าง|ฝ่ามืออีกข้าง'),
('movement','นิ่ง',r'อยู่นิ่ง|ค้างท่า'),('movement','ขึ้น',r'ยกขึ้น|เลื่อนขึ้น|เคลื่อนขึ้น'),
('movement','ลง',r'เลื่อนลง|เคลื่อนลง|ลดลง'),('movement','ออกข้าง',r'ออกข้าง|ไปด้านข้าง'),
('movement','เข้าตัว',r'เข้าหาตัว|เข้าตัว'),('movement','ออกจากตัว',r'ออกจากตัว|ไปข้างหน้า'),
('movement','วน',r'วน|ควง'),('movement','สลับ',r'สลับ'),('movement','แตะเคาะ',r'แตะ|เคาะ'),
('movement','ถูปัด',r'ถู|ปัด'),('movement','หมุนพลิก',r'หมุน|พลิก'),('movement','กางหุบ',r'กาง.*หุบ|หุบ.*กาง'),
('num_hands','มือเดียว',r'^1$'),('num_hands','สองมือ',r'^2$'),
('handshape','สองมือประกบ',r'ประกบ.*มือ|ประกบ.*ฝ่ามือ|ประนม'),
('handshape','สองมือซ้อน',r'สองมือซ้อน|มือ.*รอง'),
('movement','สองมือแยกออก',r'แยก.*มือ|แยกออก'),('movement','ทำซ้ำ',r'ซ้ำ'),
('palm_orientation','ฝ่ามือเข้าตัว',r'หันเข้าตัว'),('palm_orientation','ฝ่ามือออก',r'หันออก'),
('palm_orientation','หงายหรือคว่ำ',r'หงาย|คว่ำ')]

def parse_entry(row: dict) -> tuple[np.ndarray,np.ndarray,list[dict]]:
    """Positive evidence only; unknown stays zero with known-mask zero, not false."""
    values=np.zeros(53,dtype=np.uint8); known=values.copy(); evidence=[]
    for i,(field,label,pattern) in enumerate(SPEC):
        text=str(row.get(field,''))
        if re.search(r'ไม่ชัด|ไม่ได้ชัด|ระบุไม่ได้|ไม่ระบุ|ไม่มีลูกศร|ไม่ใช่',text):
            continue
        match=re.search(pattern,text)
        if match:
            values[i]=known[i]=1
            evidence.append({'index':i,'field':field,'matched':match.group()})
    if str(row.get('num_hands')) in ('1','2'):
        known[44:46]=1
    return values,known,evidence

def canonical(name: str) -> str:
    """Keep variant labels together; final vocabulary alignment still requires review."""
    return re.split(r'\s+—\s+',name)[0].strip()

def prepare() -> None:
    """Write aligned row-level arrays and deterministic class-group split manifests."""
    OUT.mkdir(exist_ok=True)
    rows=json.loads((ROOT.parent/'book-catalog/reviewed_entries.json').read_text())
    rows=sorted(rows,key=lambda r:(r['book'],r['pdf_page'],r['variant']))
    schema=[{'index':i,'name':label,'field':field,'positive_regex':pattern} for i,(field,label,pattern) in enumerate(SPEC)]
    assert len(schema)==53
    values=[]; masks=[]; classes=[]
    for i,r in enumerate(rows):
        a,m,e=parse_entry(r); values.append(a); masks.append(m)
        description='; '.join(f'{f}: {r[f]}' for f in ('handshape','palm_orientation','location','movement','sequence') if r.get(f))
        classes.append({'row_id':i,'concept':r['concept'],'split_group':canonical(r['concept']),
                        'description':description,'book':r['book'],'pdf_page':r['pdf_page'],'evidence':e})
    # Merge identical aliases conservatively using shared slash-delimited labels.
    parent={c['split_group']:c['split_group'] for c in classes}
    def find(x: str) -> str:
        while parent[x]!=x: x=parent[x]
        return x
    aliases={}
    for c in classes:
        g=c['split_group']
        for alias in g.split('/'):
            alias=alias.strip()
            if alias in aliases: parent[find(g)]=find(aliases[alias])
            else: aliases[alias]=g
    for c in classes: c['split_group']=find(c['split_group'])
    groups=sorted({c['split_group'] for c in classes},key=lambda x:hashlib.sha256(('seed42:'+x).encode()).hexdigest())
    n=len(groups); unseen=set(groups[:round(.2*n)]); val=set(groups[round(.2*n):round(.3*n)])
    splits={k:[] for k in ('train_seen','validation_unseen','test_unseen')}
    for c in classes:
        s='test_unseen' if c['split_group'] in unseen else 'validation_unseen' if c['split_group'] in val else 'train_seen'
        c['split']=s; splits[s].append(c['row_id'])
    for name,obj in [('attribute_schema.json',{'version':'project-53-v1','note':'Custom schema; multi-hot presence across described sequence. Last group includes coarse palm orientation. Unknown is not negative.','attributes':schema}),('classes.json',classes),('splits.json',{'seed':42,'status':'provisional_dictionary_only_requires_video_mapping','ids_are':'row_id prototype IDs; group variants for word-level evaluation','splits':splits})]:
        (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
    np.save(OUT/'attributes_matrix.npy',values); np.save(OUT/'attributes_known_mask.npy',masks)
    (OUT/'descriptions.json').write_text(json.dumps([c['description'] for c in classes],ensure_ascii=False,indent=2))
    report={'rows':len(rows),'groups':n,'split_rows':{k:len(v) for k,v in splits.items()},'known_fraction':float(np.mean(masks)),'positive_counts':np.sum(values,axis=0).tolist(),'training_ready':False}
    (OUT/'preparation_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': prepare()
