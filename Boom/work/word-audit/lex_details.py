from pathlib import Path
import json,pandas as pd
R=Path('/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say');O=Path('outputs/word-audit')
lex=json.loads((O/'lexicon.json').read_text())[:19]; meta=pd.read_csv(R/'metadata/metadata_word.csv');concepts=json.load(open(R/'models/concepts.json'))
alias={'ทีวี':['โทรทัศน์'],'อาจารย์':['ครู'],'ตาย':['เสียชีวิต'],'ทั้งหมด':['ทั้งหมด','ทุก'],'รถ':['รถยนต์'],'แดด':['แสงแดด'],'ล่าม':['ล่ามภาษามือ'],'พัน':['หนึ่งพัน'],'ราตรีสวัสดิ์':['ฝันดี'],'หรือเปล่า':['หรือไม่'],'ไหม':['มั้ย'],'กับ':['ด้วยกัน']}
for r in lex:
 w=r['word']; labels=meta.label.fillna('').astype(str);exact=meta[labels.eq(w)];candidates=[c for c in concepts if c==w or any(c==a for a in alias.get(w,[]))]
 hits=meta[labels.str.startswith(w+'(')|labels.str.startswith(w+' (')|labels.eq(w)]
 r['metadata_exact']=exact[['uid','label','rel_path']].to_dict('records');r['metadata_annotated']=hits[['uid','label','rel_path']].head(8).to_dict('records');r['candidate_concepts_review_only']=candidates
print(json.dumps(lex,ensure_ascii=False,indent=2));(O/'lexicon_19_review.json').write_text(json.dumps(lex,ensure_ascii=False,indent=2))
