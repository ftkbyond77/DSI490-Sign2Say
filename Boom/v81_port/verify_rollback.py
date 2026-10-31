"""One real cache request verifies legacy-model compatibility, not accuracy."""
import argparse,gzip,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from modules.pipeline import SignPipeline
    from webapp.api.app import _public
    pipe=SignPipeline(a.models,backend='onnx',device='cpu',use_llm=False,use_face=True)
    with gzip.open(a.request,'rt') as f:r=json.load(f)
    raw=pipe.landmarks_from_browser(r['frames'],(r['height'],r['width']))
    assert len(pipe.standardise(raw))==5
    assert len(pipe.standardise_with_metadata(raw))==6
    out=_public(pipe.run_landmarks(raw,alphabet=r['alphabet']))
    for k in ['sentence','segments','face','timings_ms','model_version','mirrored_to_right_hand','hand_canonicalization']:assert k in out,k
    a.output.write_text(json.dumps(dict(version=out['model_version'],keys=list(out),n_segments=len(out['segments']),request=a.request.name),ensure_ascii=False,indent=2))
    print('Legacy model response contract OK:',out['model_version'])
if __name__=='__main__':main()
