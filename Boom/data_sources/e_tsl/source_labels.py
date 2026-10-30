"""Keep exact website labels separate from search terms and model mappings."""
from pathlib import Path
import csv,json,html
import scrape
HERE=Path(__file__).resolve().parent

def build()->dict:
    """Validate source names and regenerate review pages plus local clip manifest."""
    raw=json.loads((HERE/'searchx.php.json').read_text())['results']
    names={str(r['wrd_id']):r['wrd_word_th'] for r in raw}
    signs={str(s['sgn_id']):(str(r['wrd_id']),r['wrd_word_th']) for r in raw for s in r.get('wrd_signs',[])}
    for filename,idfield in [('words.csv','id'),('sign_variants.csv','word_id'),('priority_words.csv','id')]:
        with (HERE/filename).open(encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):assert r['label']==r['source_label']==names[r[idfield]],filename
    priority=list(csv.DictReader((HERE/'priority_words.csv').open(encoding='utf-8-sig')))
    e=html.escape;body=''
    for r in priority:
        links=' '.join(f'<a href="{e(u,quote=True)}" target="_blank">วิดีโอ {i+1}</a>' for i,u in enumerate(r['video_url'].split(' | ')) if u)
        body+=f'<tr><td><b>{e(r["source_label"])}</b></td><td>{e(r["id"])}</td><td>{links}</td><td>{e(r["target"])}</td><td>{e(r["match"])}</td></tr>'
    style='<style>body{font:17px sans-serif;margin:30px}td,th{padding:12px;border-bottom:1px solid #ddd}a{margin-right:15px}video{max-width:100%}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:20px}article{padding:16px;background:#f3f5fa;border-radius:12px}</style>'
    (HERE/'priority_review.html').write_text('<!doctype html><meta charset="utf-8"><title>ชื่อจากเว็บ e-TSL</title>'+style+'<h1>ชื่อจากเว็บ = label ของคลิป</h1><p>เก็บชื่อเต็มตามเว็บ ไม่มีการย่อเป็นคำค้น เช่น “เดินไปข้างหน้า” ไม่เปลี่ยนเป็น “ไป”</p><p><a href="local_review.html">ดูคลิปที่ดาวน์โหลดแล้ว</a></p><table><tr><th>ชื่อจากเว็บ / label</th><th>Word ID</th><th>วิดีโอ</th><th>คำค้น (ไม่ใช่ label)</th><th>วิธีค้นพบ</th></tr>'+body+'</table>')
    audit=list(csv.DictReader((HERE/'priority_download_audit.csv').open(encoding='utf-8-sig')));clips=[];cards=[]
    for r in audit:
        wid,label=signs[r['sign_id']]
        r.update(word=label,label=label,source_label=label,label_source='e_tsl.wrd_word_th',word_id=wid)
        clips.append(dict(clip_id='etsl:'+r['sign_id'],word_id=wid,sign_id=r['sign_id'],label=label,source_label=label,label_source='e_tsl.wrd_word_th',path=r['path'],sha256=r['sha256'],source_url=r['video_url'],mapped_concept_id=None,mapping_status='not_reviewed',split='unassigned',bank_added=False))
        rel=Path(r['path']).relative_to(HERE)
        cards.append(f'<article><h2>{e(label)}</h2><p>ชื่อจากเว็บ · word {e(wid)} · sign {e(r["sign_id"])}</p><video controls preload="metadata" width="400" src="{e(str(rel),quote=True)}"></video><p>Model concept: ยังไม่จับคู่</p></article>')
    scrape.write_csv(HERE/'priority_download_audit.csv',audit)
    scrape.write_csv(HERE/'clip_manifest.csv',clips)
    (HERE/'clip_manifest.json').write_text(json.dumps(clips,ensure_ascii=False,indent=2))
    (HERE/'local_review.html').write_text('<!doctype html><meta charset="utf-8"><title>คลิปและชื่อจากเว็บ</title>'+style+'<h1>คลิปจริง — ใช้ชื่อจากเว็บเป็น label</h1><p>ไม่เปลี่ยนชื่อเป็นคำค้น; ยังไม่จับคู่ concept หรือเพิ่ม bank</p><div class="grid">'+''.join(cards)+'</div>')
    result=dict(source_entries_validated=len(names),local_clips_validated=len(clips),label_policy='exact wrd_word_th',bank_modified=False)
    (HERE/'label_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));return result
if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
