"""Compare PR6-format real replay CSVs; no labels or accuracy claims."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def read(path):
    with path.open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    by={r['file']:r for r in rows}
    if len(rows)!=53 or len(by)!=53:raise ValueError('Expected 53 unique files')
    return by


def tokens(row):
    return [s for s in json.loads(row['new_segments_json']) if s['status']!='transition']


def compare(a,b):
    if a.keys()!=b.keys():raise ValueError('Request sets differ')
    rows=[]
    for f in sorted(a):
        x,y=tokens(a[f]),tokens(b[f])
        wx,wy=[s['word'] for s in x],[s['word'] for s in y]
        ax,ay=[s['word'] for s in x if s['status']=='accepted'],[s['word'] for s in y if s['status']=='accepted']
        rows.append(dict(file=f,onnx_top1=' | '.join(wx),torch_top1=' | '.join(wy),top1_same=wx==wy,
             onnx_accepted=len(ax),torch_accepted=len(ay),accepted_words_same=ax==ay,
             status_same=[s['status'] for s in x]==[s['status'] for s in y],
             onnx_ms=float(a[f]['new_latency_ms']),torch_ms=float(b[f]['new_latency_ms'])))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx',type=Path,required=True);p.add_argument('--torch',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();rows=compare(read(args.onnx),read(args.torch))
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'per_file.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(n=53,top1_same=sum(r['top1_same'] for r in rows),accepted_words_same=sum(r['accepted_words_same'] for r in rows),
       accepted_count_same=sum(r['onnx_accepted']==r['torch_accepted'] for r in rows),status_same=sum(r['status_same'] for r in rows),
       onnx_accepted_total=sum(r['onnx_accepted'] for r in rows),torch_accepted_total=sum(r['torch_accepted'] for r in rows),
       onnx_median_ms=statistics.median(r['onnx_ms'] for r in rows),torch_median_ms=statistics.median(r['torch_ms'] for r in rows))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
