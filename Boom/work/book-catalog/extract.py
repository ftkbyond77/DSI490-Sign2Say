from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,json,os
from pypdf import PdfReader
from PIL import Image
R=Path('/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say/New data/หนังสือ');O=Path('outputs/book-catalog');W=Path('work/book-catalog');manifest=[]
for b in range(1,7):
 src=R/f'ThaiSignLanguageBook_{b}.pdf';n=len(PdfReader(src).pages)
 for p in range(1,n+1):manifest.append(dict(book=b,pdf_page=p,source=str(src),id=f'b{b}p{p:03}',review_status='pending'))
(O/'page_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
def extract(row):
 ident=row['id'];png=W/(ident+'.png');img=O/'pages'/(ident+'.jpg');txt=W/(ident+'.txt')
 if not txt.exists():
  subprocess.run(['pdftoppm','-f',str(row['pdf_page']),'-l',str(row['pdf_page']),'-scale-to','1800','-singlefile','-png',row['source'],str(png.with_suffix(''))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  Image.open(png).convert('RGB').save(img,quality=85)
  subprocess.run(['/opt/homebrew/bin/tesseract',str(png),str(txt.with_suffix('')),'--tessdata-dir',str(W/'tessdata'),'-l','tha+eng','--psm','11'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env={**os.environ,'OMP_THREAD_LIMIT':'1'})
  png.unlink()
 return ident
with ThreadPoolExecutor(max_workers=3) as ex:
 for i,r in enumerate(ex.map(extract,manifest)):
  if i%20==0:print(i+1,'/',len(manifest),r,flush=True)
print('Complete',flush=True)
