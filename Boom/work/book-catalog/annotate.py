import json
from pathlib import Path
O=Path('outputs/book-catalog');F=O/'reviewed_entries.json';P=O/'page_reviews.json'
def add(book,page,kind,entries=(),note=''):
 rows=json.loads(F.read_text()) if F.exists() else []
 rows=[r for r in rows if not(r['book']==book and r['pdf_page']==page)]
 # Per-entry: concept, hands, shape, orientation, location, movement, sequence, printed page
 for i,e in enumerate(entries):
  fields=['concept','num_hands','handshape','palm_orientation','location','movement','sequence','printed_page']
  row=dict(zip(fields,e));row.update(book=book,pdf_page=page,variant=i+1,review_status='image_reviewed',source=f'ThaiSignLanguageBook_{book}.pdf',uncertainty='ภาพสแกน 2D; รายละเอียดที่ไม่ได้ระบุไม่ควรอนุมาน',image=f'pages/b{book}p{page:03}.jpg');rows.append(row)
 F.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 reviews=json.loads(P.read_text()) if P.exists() else {};reviews[f'b{book}p{page:03}']=dict(kind=kind,note=note,review_status='image_reviewed',entries=len(entries));P.write_text(json.dumps(reviews,ensure_ascii=False,indent=2))
