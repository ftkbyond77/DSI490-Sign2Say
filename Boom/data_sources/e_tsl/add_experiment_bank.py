"""Append the 12 source-labelled e-TSL clips to an isolated experiment bank."""
from pathlib import Path
import os,sys,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
os.environ.update(MPLBACKEND='Agg',MPLCONFIGDIR=str(HERE/'.mplconfig'),ORT_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.path[:0]=[str(ROOT/'Boom/phase2.1/02_hybrid_score/runtime_packages'),str(ROOT)]
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def run():
    from modules.mediapipe_pose import extract,to_standard
    from modules.schema import canonical_hands
    from modules.parts import prep,to_iso
    from modules.runtime import make_backend,Bank
    dest=HERE/'experiment_bank';dest.mkdir(exist_ok=True)
    output=dest/'bank_etsl12.npz'
    if output.exists():raise RuntimeError('Experiment bank already exists; refusing duplicate append')
    source=ROOT/'models/bank.npz';before=sha(source)
    enc=ROOT/'models/onnx/encoder.int8.onnx'
    if not enc.exists():enc=ROOT/'models/onnx/encoder.onnx'
    hashes={str(p):sha(p) for p in [source,enc,ROOT/'models/concepts.json',ROOT/'models/mediapipe/holistic_landmarker.task',ROOT/'modules/mediapipe_pose.py',ROOT/'modules/parts.py']}
    items=json.loads((HERE/'clip_manifest.json').read_text());concepts=json.loads((ROOT/'models/concepts.json').read_text())
    assert len(items)==12 and len({r['clip_id'] for r in items})==12
    with np.load(source,allow_pickle=False) as f:bank={k:f[k].copy() for k in f.files}
    assert set(bank)=={'Z','c','clip','dataset','signer'}
    assert not set(r['clip_id'] for r in items)&set(bank['clip'])
    be=make_backend(ROOT/'models','onnx','cpu');poses=dest/'poses';poses.mkdir(exist_ok=True)
    zs=[];audit=[]
    for r in items:
        assert r['label']==r['source_label'] and r['label'] in concepts
        path=Path(r['path']);assert sha(path)==r['sha256']
        rawfile=poses/(r['sign_id']+'_raw.npz')
        if rawfile.exists():
            with np.load(rawfile) as z:raw={k:z[k] for k in z.files}
        else:
            raw=extract(path,fps=25.,normalize=False,model_path=ROOT/'models/mediapipe/holistic_landmarker.task')
            if raw is None:raise RuntimeError('No landmarks: '+r['clip_id'])
            np.savez_compressed(rawfile,**raw)
        d=to_standard(raw,src=r['clip_id']);kp,sc,mirror=canonical_hands(d['kp'],d['sc'],tuple(d['hw']))
        if len(kp)<8 or not np.isfinite(kp).all():raise RuntimeError('Unusable pose: '+r['clip_id'])
        z=be.embed([prep(to_iso(kp,tuple(d['hw'])),sc)])[0];norm=np.linalg.norm(z)
        if not np.isfinite(z).all() or norm<1e-8:raise RuntimeError('Invalid embedding')
        zs.append(z/norm)
        audit.append(dict(clip_id=r['clip_id'],label=r['label'],concept_id=concepts.index(r['label']),frames=len(kp),span=[0,len(kp)],span_policy='whole isolated dictionary clip; no manually annotated boundaries',mirrored=bool(mirror),sha256=r['sha256'],source_url=r['source_url'],signer='unknown:etsl',signer_note='not a verified identity; do not count as new independent signer',bank_added=True))
        (dest/'progress.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));print('EMBEDDED',len(audit),r['label'],flush=True)
    n=len(bank['c']);bank['Z']=np.concatenate([bank['Z'],np.asarray(zs,dtype=bank['Z'].dtype)]);bank['c']=np.concatenate([bank['c'],np.array([r['concept_id'] for r in audit],dtype=bank['c'].dtype)])
    for key,new in [('clip',[r['clip_id'] for r in items]),('dataset',['etsl']*12),('signer',['unknown:etsl']*12)]:bank[key]=np.concatenate([bank[key].astype(str),np.array(new,dtype=str)])
    temp=dest/'bank_etsl12.pending.npz';np.savez(temp,**bank)
    with np.load(temp,allow_pickle=False) as check,np.load(source,allow_pickle=False) as original:
        for k in original.files:assert np.array_equal(check[k][:n],original[k]),'Original prefix changed'
        assert all(len(check[k])==n+12 for k in check.files)
        Bank(check['Z'],check['c'],len(concepts))
    assert all(sha(Path(p))==v for p,v in hashes.items()),'Source artifact changed'
    temp.replace(output)
    report=dict(status='complete',base_bank=str(source),base_rows=n,new_rows=12,final_rows=n+12,output=str(output),output_sha256=sha(output),hashes=hashes,original_bank_unchanged=sha(source)==before,labels_preserved=True,encoder_precision=be.precision,production_enabled=False,eval_bank_modified=False,identity_overlap='not verified',clips=audit)
    (dest/'BANK_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['clips','hashes']},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':run()
