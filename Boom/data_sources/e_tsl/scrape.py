"""Extract the public e-TSL catalogue. Cached metadata by default; no bank mutation."""
from __future__ import annotations
import csv,json,html,time,urllib.request,urllib.parse
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];BASE='http://e-tsl.eng.chula.ac.th/'
TARGETS=['ไป','กลับ','เที่ยว','หนึ่ง','ภาษามือ','วิ่ง','กิน','ดู','ข้าว','ไม่มี','คุณ','ยาย','ญาติ','ป้า']
def write_csv(path:Path,rows:list[dict])->None:
    """Export UTF-8 BOM for Thai spreadsheet compatibility."""
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def fetch(refresh:bool=False)->dict:
    """Use the same search endpoint as browsex.php; one request for the full list."""
    path=HERE/'searchx.php.json'
    if refresh or not path.exists():
        req=urllib.request.Request(BASE+'searchx.php',data=urllib.parse.urlencode({'keywords':''}).encode(),headers={'User-Agent':'Sign2Say-research-catalog/1.0'})
        with urllib.request.urlopen(req,timeout=60) as r:data=json.load(r)
        if str(data.get('status'))!='0':raise RuntimeError('API returned unsuccessful status')
        path.write_text(json.dumps(data,ensure_ascii=False))
    data=json.loads(path.read_text())
    if int(data['count'])!=len(data['results']):raise ValueError('Incomplete API response')
    return data
def build(refresh:bool=False)->dict:
    """Normalize entries and variants; preserve source IDs and unresolved semantics."""
    data=fetch(refresh);concepts=json.loads((ROOT/'models/concepts.json').read_text());lookup={html.unescape(x).strip():i for i,x in enumerate(concepts)}
    rows=[];signs=[]
    for r in data['results']:
        word=r['wrd_word_th'];syn=[x['syn_thaiword'] for x in r.get('wrd_synonyms',[]) if x['syn_thaiword'] not in ('','-')]
        rows.append(dict(id=r['wrd_id'],word=word,label=word,source_label=word,label_source='e_tsl.wrd_word_th',english=r.get('wrd_word_en',''),meaning=r.get('wrd_meaning_th',''),category=' | '.join(x['cat_name_th'] for x in r.get('wrd_categories',[])),synonyms=' | '.join(syn),sign_variants=len(r.get('wrd_signs',[])),exact_concept_id=lookup.get(html.unescape(word).strip(),''),page_url=BASE+'browsex.php',video_url=' | '.join(BASE+'resources/Video/Video_'+s['sgn_video_normal']['vid_id']+'_0.mp4' for s in r.get('wrd_signs',[]) if s.get('sgn_video_normal'))))
        for s in r.get('wrd_signs',[]):
            vid=(s.get('sgn_video_normal') or {}).get('vid_id');roots=[x['rot_id'] for x in s.get('sgn_roots',[])]
            signs.append(dict(word_id=r['wrd_id'],word=word,label=word,source_label=word,label_source='e_tsl.wrd_word_th',sign_id=s['sgn_id'],video_id=vid,video_url=BASE+f'resources/Video/Video_{vid}_0.mp4' if vid else '',video_view=0,root_ids=' | '.join(roots),root_image_urls=' | '.join(BASE+f'resources/Root/Root_{i}.jpg' for i in roots),sign_image_url=BASE+f'resources/Sign/{s["sgn_id"]}.jpg' if str(s.get('sgn_has_image'))=='1' else '',url_status='derived from observed player code; not individually checked',root_meaning='uninterpreted; not automatic handshape labels'))
    write_csv(HERE/'words.csv',rows);write_csv(HERE/'sign_variants.csv',signs)
    (HERE/'words.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    target=[]
    for word in TARGETS:
        for r in rows:
            if word==r['word'] or word in r['word']:
                target.append(dict(target=word,match='exact' if word==r['word'] else 'substring_review_required',**r))
    write_csv(HERE/'priority_words.csv',target)
    summary=dict(entries=len(rows),unique_word_strings=len({r['word'] for r in rows}),sign_variants=len(signs),variants_with_video=sum(bool(s['video_url']) for s in signs),unique_video_urls=len({s['video_url'] for s in signs if s['video_url']}),exact_project_concept_matches=sum(r['exact_concept_id']!='' for r in rows),exact_target_matches=[w for w in TARGETS if any(r['word']==w for r in rows)],notes=['word overlap is not proof of duplicate videos','No video downloaded or bank modified','Root IDs are source symbols, not validated 53-dimensional attributes','Alphabet/number entries and lexical variants require review','No license inferred from public access'])
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary
def check_priority(limit:int=12)->list[dict]:
    """HEAD-check a small exact-match priority sample; no bulk video download."""
    rows=list(csv.DictReader((HERE/'sign_variants.csv').open(encoding='utf-8-sig')));out=[]
    for r in [r for r in rows if r['word'] in TARGETS and r['video_url']][:limit]:
        try:
            req=urllib.request.Request(r['video_url'],method='HEAD')
            with urllib.request.urlopen(req,timeout=20) as response:
                out.append(dict(word=r['word'],sign_id=r['sign_id'],video_url=r['video_url'],status=response.status,content_type=response.headers.get('Content-Type'),bytes=response.headers.get('Content-Length'),checked_utc=datetime.now(timezone.utc).isoformat()))
        except Exception as e:out.append(dict(word=r['word'],sign_id=r['sign_id'],video_url=r['video_url'],status=str(e),content_type=None,bytes=None,checked_utc=datetime.now(timezone.utc).isoformat()))
        time.sleep(1)
    write_csv(HERE/'priority_url_checks.csv',out);return out
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--refresh',action='store_true');p.add_argument('--check-priority',action='store_true');a=p.parse_args()
    print(json.dumps(build(a.refresh),ensure_ascii=False,indent=2),flush=True)
    if a.check_priority:print(json.dumps(check_priority(),ensure_ascii=False,indent=2),flush=True)
