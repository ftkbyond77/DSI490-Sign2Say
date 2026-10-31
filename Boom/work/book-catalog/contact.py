from PIL import Image,ImageDraw
from pathlib import Path
import sys
b,a,z=map(int,sys.argv[1:]);ps=[]
for p in range(a,z+1):
 f=Path(f'outputs/book-catalog/pages/b{b}p{p:03}.jpg')
 if not f.exists():f=Path(f'work/book-rules/book{b}-{p:03}.png')
 if f.exists():ps.append((p,f))
w=500;h=700;out=Image.new('RGB',(w*3,h*((len(ps)+2)//3)),'#dedede');d=ImageDraw.Draw(out)
for i,(p,f) in enumerate(ps):
 im=Image.open(f);im.thumbnail((w,h-28));x=i%3*w;y=i//3*h;out.paste(im,(x,y+28));d.text((x+8,y+4),f'BOOK {b} PDF PAGE {p}',fill='black')
p=Path(f'work/book-catalog/contact-b{b}-{a}-{z}.jpg');out.save(p,quality=90);print(p)
