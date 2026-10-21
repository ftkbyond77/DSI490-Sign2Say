"""Historical tune-only sensitivity; ordered lexical alignment is a proxy, not verified segment labels."""
from pathlib import Path
import json,hashlib
O=Path('outputs/improvement-experiment');p=Path('/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say/artifacts/reports/phrases.json');d=json.load(open(p))['conversation']['videos']
tune={k:v for k,v in d.items() if v['split']=='tune'}
def lcs(a,b):
 dp=[[0]*(len(b)+1) for _ in range(len(a)+1)]
 for i,x in enumerate(a,1):
  for j,y in enumerate(b,1):dp[i][j]=dp[i-1][j-1]+1 if x==y else max(dp[i-1][j],dp[i][j-1])
 return dp[-1][-1]
rows=[]
for t in [i/100 for i in range(5,100,5)]:
 accepted=correct=0
 for v in tune.values():
  words=[s['top5'][0] for s in v['segments'] if s.get('top5') and s['p']>=t and s.get('status')!='transition']
  accepted+=len(words);correct+=lcs(words,v['ref'])
 rows.append(dict(threshold=t,accepted=accepted,ordered_matches=correct,unmatched=accepted-correct,precision_proxy=correct/accepted if accepted else None))
result=dict(source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),tune_videos=len(tune),test_videos_used=0,rows=rows,decision='retain production 0.8; do not fit from historical lexical-proxy scores',limitations=['Historical model predictions, not freshly rerun modified model','LCS matches words in order; no manually verified segment alignment','No new-signer test set supplied','No inference of calibrated precision from six dictionary smoke clips'])
(O/'threshold_tune_sensitivity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
