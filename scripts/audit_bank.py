"""Read-only bank/prep audit; NumPy + stdlib, no GPU, network or recognition changes.

python scripts/audit_bank.py --eda ../eda --out artifacts/reports/prep-audit
python scripts/audit_bank.py --build-review review.json --out artifacts/experiments/review-01
The second command writes filtered deployment AND train-only evaluation banks to
an empty experiment directory. It never overwrites source banks or changes labels.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def unique_index(rows, key):
    groups = defaultdict(list)
    for row in rows:
        value = key(row)
        if value:
            groups[value].append(row)
    # Ambiguous keys are deliberately not joined.
    return {k: v[0] for k, v in groups.items() if len(v) == 1}


def raw_key(value):
    value = str(value).replace('\\', '/')
    for prefix in ('raw_data/', 'keypoint_samples/'):
        if prefix in value:
            return 'raw_data/' + value.split(prefix, 1)[1]
    return value


def load_bank(path, concepts):
    with np.load(path, allow_pickle=False) as z:
        b = {k: z[k] for k in z.files}
    required = {'Z', 'c', 'clip', 'dataset', 'signer'}
    if not required <= b.keys():
        raise ValueError(f'Missing bank arrays: {required - b.keys()}')
    n = len(b['c'])
    if b['Z'].ndim != 2 or any(v.ndim == 0 or len(v) != n for v in b.values()):
        raise ValueError('Bank arrays are not row aligned')
    if b['c'].ndim != 1 or b['c'].dtype.kind not in 'iu':
        raise ValueError('Concept indices must be a one-dimensional integer array')
    if not isinstance(concepts, list) or len(concepts) != len(set(concepts)):
        raise ValueError('Concepts must be a unique list')
    return b


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def audit(root, eda, out):
    out.mkdir(parents=True, exist_ok=True)
    cp = root / 'models/concepts.json'; bp = root / 'models/bank.npz'
    concepts = json.loads(cp.read_text()); b = load_bank(bp, concepts)
    conversation = set(json.loads((root/'models/vocab_conversation.json').read_text())['concepts'])
    inputs = [bp, cp, root/'models/vocab_conversation.json', root/'metadata/metadata_master.csv',
              eda/'eda_per_clip/per_clip_summary.csv', eda/'eda_duplicates/all_actual_sha256_matches.csv',
              eda/'eda_signers/clip_signer_manifest.csv', eda/'all_video_properties.csv',
              eda/'keypoint_samples/manifest.csv', eda/'duplicate_review/labels.csv', root/'artifacts/eval/bank_eval.npz']
    meta = read_csv(inputs[3]); midx = unique_index(meta, lambda x: x['uid'])
    eda_rows = read_csv(inputs[4]); eidx = unique_index(eda_rows, lambda x: raw_key(x['video_file']))
    duplicates = read_csv(inputs[5]); didx = unique_index(duplicates, lambda x: raw_key(x['s3_key']))
    signers = unique_index(read_csv(inputs[6]), lambda x: x['file'])
    props = unique_index(read_csv(inputs[7]), lambda x: raw_key(x['s3_key']))
    samples = unique_index(read_csv(inputs[8]), lambda x: raw_key(x['s3_key']))
    local_duplicates = unique_index(read_csv(inputs[9]), lambda x: x['file'])
    dgroups = defaultdict(list)
    for row in duplicates:
        if row['sha256_actual']:
            dgroups[row['sha256_actual']].append(row)
    mhash = defaultdict(list)
    for row in meta:
        if row.get('sha256'):
            mhash[row['sha256']].append(row)
    rows = []; dup_to_bank = defaultdict(list)
    clip_counts = Counter(map(str,b['clip']))
    for i in range(len(b['c'])):
        clip, ds, signer = (str(b[k][i]) for k in ('clip','dataset','signer'))
        ci = int(b['c'][i]); valid = 0 <= ci < len(concepts)
        concept = concepts[ci] if valid else None
        m = midx.get(clip)
        join = 'uid_exact' if m else 'unmatched'
        if m is None and clip.startswith('agent:yt:'):
            m = midx.get('yt:' + clip[len('agent:yt:'):]);join='youtube_id_exact' if m else 'unmatched'
        key = raw_key(m['rel_path']) if m else None
        er = eidx.get(key); dr = didx.get(key); pr = props.get(key)
        sm = signers.get(Path(key).name) if ds == 'ttrs' and key else None
        sample = samples.get(key)
        video = eda / sample['local_path'] if sample else None
        local_hash_verified = False
        if dr and key:
            ld = local_duplicates.get(Path(key).name)
            local = eda/'duplicate_review'/ld['group']/ld['file'] if ld else None
            if local and local.exists() and digest(local) == dr['sha256_actual']:
                video = local
                local_hash_verified = True
        flags=[]
        if not valid: flags.append('invalid_concept_index')
        if not np.isfinite(b['Z'][i]).all(): flags.append('nonfinite_embedding')
        if np.isfinite(b['Z'][i]).all() and np.linalg.norm(b['Z'][i].astype(np.float32)) < 1e-8: flags.append('zero_embedding')
        if clip_counts[clip] > 1: flags.append('duplicate_clip_id')
        if not m: flags.append('metadata_unmatched')
        if not er: flags.append('quality_unmeasured')
        elif er.get('status') != 'ok': flags.append('eda_measurement_failed')
        else:
            if float(er['zero_hands_pct']) > 50: flags.append('sampled_hands_missing_over_50pct')
            if float(er['face_detected_pct']) <= 10: flags.append('sampled_face_low_not_signer_proof')
        if pr and pr.get('orientation') == 'portrait': flags.append('portrait_review_rotation')
        if dr: flags.append('actual_sha256_duplicate'); dup_to_bank[dr['sha256_actual']].append(i)
        if m and m.get('sha256') and len(mhash[m['sha256']]) > 1: flags.append('metadata_hash_duplicate_unverified')
        if sm and sm.get('signer_status') == 'unknown': flags.append('eda_signer_unknown')
        rows.append(dict(bank_row=i,clip_id=clip,concept_index=ci,concept=concept,dataset=ds,
            signer_label=signer,signer_verified=False,in_conversation=concept in conversation,
            metadata_join=join,raw_path=key,source_label=m.get('label') if m else None,
            source_url=m.get('source_page') if m else None,metadata_signer=m.get('signer_creator') if m else None,
            actual_sha256=dr['sha256_actual'] if dr else None,local_hash_verified=local_hash_verified,
            eda=er,video_properties=pr,eda_signer=sm,
            local_video=str(video.resolve()) if video and video.exists() else None,
            span_start_frame=None,span_end_frame=None,span_quality=None,interpolated_fraction=None,
            flags=flags,status='needs_review'))
    conflicts=[]
    for sha, ids in dup_to_bank.items():
        labels=sorted({rows[i]['concept'] for i in ids if rows[i]['concept'] is not None})
        if len(labels)>1:
            for i in ids: rows[i]['flags'].append('duplicate_concept_conflict')
            conflicts.append(dict(sha256=sha,concepts=labels,clip_ids=[rows[i]['clip_id'] for i in ids]))
    fatal={'invalid_concept_index','nonfinite_embedding','zero_embedding','duplicate_clip_id'}
    for r in rows:
        if fatal.intersection(r['flags']):r['status']='invalid_structure'
    # Coverage counts are instance and signer-label counts, not verified people.
    by_word=defaultdict(list)
    for r in rows:
        if r['concept'] is not None: by_word[r['concept']].append(r)
    words=[]
    for word in sorted(conversation):
        rr=by_word[word]
        words.append(dict(word=word,instances=len(rr),signer_labels=len({x['signer_label'] for x in rr}),
            datasets=dict(Counter(x['dataset'] for x in rr)),quality_measured_instances=sum(x['eda'] is not None for x in rr),
            conflict_instances=sum('duplicate_concept_conflict' in x['flags'] for x in rr),
            clip_ids=[x['clip_id'] for x in rr],collection_priority='missing' if not rr else 'high' if len(rr)<=3 else 'normal'))
    words.sort(key=lambda x:(x['instances'],x['word']))
    flag_counts=dict(Counter(f for r in rows for f in r['flags']))
    summary=dict(bank_instances=len(rows),represented_concepts=len(by_word),conversation_words=len(words),
        conversation_single=sum(w['instances']==1 for w in words),conversation_at_most_three=sum(w['instances']<=3 for w in words),
        metadata_matched=sum(r['metadata_join']!='unmatched' for r in rows),
        bank_rows_with_sampled_eda=sum(r['eda'] is not None for r in rows),
        duplicate_rows_hash_verified_locally=sum(r['local_hash_verified'] for r in rows),
        bank_rows_with_local_video=sum(r['local_video'] is not None for r in rows),
        measured_eda_clips_total=len(eda_rows),actual_duplicate_groups_total=len(dgroups),
        actual_duplicate_groups_in_bank=len(dup_to_bank),conflicting_concept_groups_in_bank=len(conflicts),
        flags=flag_counts)
    fingerprints={str(p.resolve()):digest(p) for p in inputs if p.exists()}
    missing=[str(p.resolve()) for p in inputs if not p.exists()]
    # Conflict first, then sparse conversation examples; deterministic order.
    priority=lambda r:(0 if 'duplicate_concept_conflict' in r['flags'] else 1 if r['in_conversation'] and len(by_word[r['concept']])<=3 else 2, len(by_word.get(r['concept'],[])),r['clip_id'])
    queue=sorted(rows,key=priority)[:200]
    review=dict(bank_sha256=digest(bp),concepts_sha256=digest(cp),eval_bank_sha256=digest(inputs[10]) if inputs[10].exists() else None,decisions=[dict(clip_id=r['clip_id'],action='pending',reason='',flags=r['flags']) for r in queue])
    write_json(out/'summary.json',dict(summary=summary,input_sha256=fingerprints,missing_inputs=missing,conflicts=conflicts))
    with (out/'bank_audit.jsonl').open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    write_json(out/'word_coverage.json',words);write_json(out/'review.json',review)
    report='''# ผลตรวจ bank และข้อมูลเตรียม Sign2Say

รายงานนี้อ่าน bank/metadata/ผล EDA เดิม ไม่ได้รัน detector ใหม่ ไม่ได้แก้ recognition หรือ bank ที่ใช้งานจริง

'''
    report+='\n'.join(f'- {k}: {v}' for k,v in summary.items() if k!='flags')
    report+='''

## วิธีตีความ

- EDA เป็นการสุ่มทั้งคลิปด้วยตัวตรวจมือ/ใบหน้าแยกกัน ไม่ใช่ Holistic production และไม่ใช่คะแนนเฉพาะช่วงคำ
- ไม่พบ EDA หมายถึงยังไม่รู้คุณภาพ ไม่ใช่คุณภาพดีหรือเสีย
- signer_labels คือรหัสที่ bank บันทึก ไม่ใช่จำนวนคนที่ยืนยันด้วยคนแล้ว
- label ต่างกันบนวิดีโอ hash เดียวกันเป็นรายการให้ตรวจ อาจเป็นคำพ้อง/ความหมายหลายระดับ ไม่ตัดสินว่าป้ายใดผิดอัตโนมัติ
- ไม่มี span/pose ต้นทางครบ จึงรายงานช่วงคำ คุณภาพมือรายช่วง และสัดส่วนจุดเติมเป็น null
- จับคู่ metadata ด้วย uid หรือ YouTube ID เท่านั้น; จับคู่ EDA ด้วยเส้นทาง raw ที่ normalize จาก manifest ไม่ใช้ความคล้ายของชื่อคำ
- review.json เป็นคิว 200 ตัวอย่างแรกตามลำดับความเสี่ยงและความบางของคำ ไม่ใช่ตัวอย่างสุ่มสำหรับประมาณอัตรา noise ทั้งชุด

## คำที่ควรเพิ่มตัวอย่างก่อน (ตัวอย่าง 30 คำแรก)

| คำ | ตัวอย่าง | รหัสผู้ทำท่า | มี EDA |
|---|---:|---:|---:|
'''
    report+='\n'.join(f"| {w['word']} | {w['instances']} | {w['signer_labels']} | {w['quality_measured_instances']} |" for w in words[:30])
    report+='\n\n## กลุ่มไฟล์เหมือนกันแต่ concept ต่างกัน\n\n'
    report+='\n'.join('- '+', '.join(g['concepts'])+' — '+', '.join(g['clip_ids']) for g in conflicts) or 'ไม่พบ'
    report+='''

## ทดลองกรอง bank โดยไม่แตะตัวใช้งานจริง

1. ตรวจคลิป/ป้าย/ช่วงคำ แล้วแก้ `review.json`: action เป็น `exclude` พร้อม reason เฉพาะรายการที่ต้องการกันออก; `pending` และ `keep` ไม่เปลี่ยน bank
2. รัน `python scripts/audit_bank.py --build-review <review.json> --out <โฟลเดอร์ทดลองใหม่>`
3. ได้ bank.npz และ bank_eval.npz ที่กรอง clip ID เดียวกันจากแต่ละต้นฉบับ ไม่คัด deployment bank ไปใช้แทน evaluation bank
4. comparison.json แสดงจำนวนที่เอาออก คำที่หายไป และ fingerprint; ไม่ได้ยืนยันว่าความแม่นยำดีขึ้น
5. ใช้ชุด validation เดิมที่ไม่รั่วเข้า train-only bank เปรียบเทียบ baseline กับ bank_eval.npz ทดลอง ผ่าน SignPipeline(bank_file=...) ก่อน deploy

เครื่องมือนี้ยังไม่แก้ label หรือตัด span/re-embed เพราะต้องยืนยันข้อมูลต้นทางและ encoder ที่ตรงกับ bank ก่อน
'''
    (out/'report.md').write_text(report,encoding='utf-8')
    h='<meta charset="utf-8"><title>Sign2Say prep audit</title><style>body{font-family:Thonburi,sans-serif;max-width:1200px;margin:36px auto;color:#203247}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccd;padding:8px;text-align:left}th{background:#eaf3f7}input{padding:10px;width:70%}pre{white-space:pre-wrap}</style><h1>Sign2Say: รายการตรวจ prep</h1><p>ข้อมูลไม่ครบไม่เท่ากับข้อมูลเสีย — EDA เป็นการสุ่มทั้งคลิป ไม่ใช่คะแนนช่วงคำ</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><h2>คิวตรวจ 200 ตัวอย่าง</h2><input id="q" placeholder="ค้นหาคำ / clip ID / flags"><table><thead><tr><th>คำ / คลิป</th><th>แหล่ง / รหัสคน</th><th>เหตุผลตรวจ</th><th>คลิป</th></tr></thead><tbody>'
    for r in queue:
        link='<a href="'+html.escape(Path(r['local_video']).as_uri(),quote=True)+'">วิดีโอในเครื่อง</a>' if r['local_video'] else 'ยังไม่มีไฟล์ในเครื่องที่จับคู่ได้'
        h+='<tr><td>'+html.escape(str(r['concept']))+'<br>'+html.escape(r['clip_id'])+'</td><td>'+html.escape(r['dataset']+' / '+r['signer_label'])+'</td><td>'+html.escape(', '.join(r['flags']))+'</td><td>'+link+'</td></tr>'
    h+='</tbody></table><script>document.querySelector("#q").oninput=e=>document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase()));</script>'
    (out/'report.html').write_text(h,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def build_review(root, review_path, out):
    bp=root/'models/bank.npz'; cp=root/'models/concepts.json'; ep=root/'artifacts/eval/bank_eval.npz'
    review=json.loads(review_path.read_text());concepts=json.loads(cp.read_text())
    if review.get('bank_sha256')!=digest(bp) or review.get('concepts_sha256')!=digest(cp):
        raise ValueError('Review fingerprints do not match current bank/concepts; re-audit first')
    if review.get('eval_bank_sha256') != digest(ep):
        raise ValueError('Evaluation bank fingerprint changed; re-audit first')
    banks={name:load_bank(path,concepts) for name,path in [('bank',bp),('bank_eval',ep)]}
    deploy_index = dict(zip(map(str, banks['bank']['clip']), map(int, banks['bank']['c'])))
    for clip, ci in zip(banks['bank_eval']['clip'], banks['bank_eval']['c']):
        if str(clip) in deploy_index and deploy_index[str(clip)] != int(ci):
            raise ValueError('Evaluation/deployment concept indices disagree for '+str(clip))
    clips=set(map(str,banks['bank']['clip']));seen=set();exclude=set()
    for d in review['decisions']:
        clip=d['clip_id'];action=d['action']
        if clip not in clips or clip in seen:raise ValueError('Unknown or duplicate review clip: '+clip)
        seen.add(clip)
        if action not in ('pending','keep','exclude'):raise ValueError('Unsupported action: '+action)
        if action=='exclude':
            if not d.get('reason','').strip():raise ValueError('Exclusion requires reason: '+clip)
            exclude.add(clip)
    if not exclude:raise ValueError('No reviewed exclusions; refusing to label an unchanged copy as an experiment')
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty experiment directory')
    if out.resolve() in (bp.parent.resolve(),ep.parent.resolve()):raise ValueError('Cannot write to source bank directories')
    out.mkdir(parents=True,exist_ok=True);report={'excluded_clip_ids':sorted(exclude),'review_sha256':digest(review_path),'sources':{str(p):digest(p) for p in [bp,cp,ep]},'banks':{}}
    filtered={}
    for name,b in banks.items():
        keep=~np.isin(b['clip'],list(exclude))
        if not keep.any():raise ValueError('Refusing empty bank: '+name)
        old=set(map(int,b['c']));new=set(map(int,b['c'][keep]));lost=sorted(old-new)
        if any(c<0 or c>=len(concepts) for c in old):raise ValueError('Invalid source concept IDs')
        report['banks'][name]={'before':len(keep),'after':int(keep.sum()),'removed':int((~keep).sum()),'lost_concepts':[concepts[c] for c in lost]}
        filtered[name]={k:v[keep] for k,v in b.items()}
    for name,b in filtered.items():np.savez(out/(name+'.npz'),**b)
    write_json(out/'comparison.json',report);write_json(out/'review_applied.json',review)
    print(json.dumps(report['banks'],ensure_ascii=False,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--eda',type=Path)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--build-review',type=Path)
    a=ap.parse_args();root=a.root.resolve()
    if a.build_review:build_review(root,a.build_review,a.out)
    else:audit(root,(a.eda or root.parent/'eda').resolve(),a.out)


if __name__=='__main__':main()
