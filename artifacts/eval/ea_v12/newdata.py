import pandas as pd, json, collections, sys, re, numpy as np
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"D:/SLM-labs/SLM-Labs-SignDINO"); S = Path(sys.argv[1])
R = pd.read_csv(ROOT / 'cloud_s3/sign-ราชบัณฑิต/manifest.csv'); E = pd.read_csv(ROOT / 'cloud_s3/sign-etsl-chula/manifest.csv')
T = json.load(open(ROOT / 'cloud_s3/data_prep/manifest/tiers.json', encoding='utf-8'))
print({k: (type(v).__name__, (list(v)[:3] if hasattr(v, '__iter__') and not isinstance(v, str) else v)) for k, v in T.items() if k != 'doc'})
core = set(T['core']); ext = set(T['extended']); thin = set(T['thin']); words = T['words']
W = set(words) if not isinstance(words, dict) else set(words.keys())
print('vocab', len(W), 'core', len(core), 'ext', len(ext), 'thin', len(thin))
if isinstance(words, dict):
    k0 = next(iter(words)); print('word entry', k0, words[k0])
C = pd.read_parquet(ROOT / 'cloud_s3/data_prep/shards/clips.parquet')
print('clips cols', list(C.columns)); print(C.groupby('dataset').agg(clips=('clip_id', 'size'), signers=('signer', 'nunique'), concepts=('concept', 'nunique')).to_string())
allc = set(C.concept.dropna().unique()); print('all concepts in corpus', len(allc))
cnt = C[C.kind == 'isolated'].groupby('concept').agg(clips=('clip_id', 'size'), signers=('signer', 'nunique'))
rw = set(R[R.kind == 'word'].word); rnew = set(R[(R.kind == 'word') & (R.status == 'new')].word)
ew_all = E[E.type.isin(['word', 'phrase', 'sentence'])]; ew = set(ew_all.word_th)
syn = collections.defaultdict(set)
for r in E.itertuples():
    if isinstance(r.synonyms, str) and r.synonyms.strip() not in ('-', ''):
        for s in re.split(r'[|,]', r.synonyms):
            if s.strip(): syn[s.strip()].add(r.word_th)
def rep(name, ws):
    print(f"{name:34s} n={len(ws):5d} | in vocab328={len(ws & W):4d} core={len(ws & core):3d} ext={len(ws & ext):4d} thin={len(ws & thin):4d} | in corpus(any)={len(ws & allc):5d} new-to-corpus={len(ws - allc):5d}")
rep('RID words (all 200)', rw); rep('RID words (new 183)', rnew); rep('eTSL word/phrase/sentence', ew); rep('eTSL + synonyms', ew | set(syn)); rep('union RID+eTSL', rw | ew)
u = (rw | ew)
print('vocab words NOT covered by either:', len(W - u), '| core not covered', sorted(core - u))
print('thin words covered by new data:', len(thin & u), '/', len(thin))
focus = ['กิน', 'คุณ', 'ยาย', 'ไป', 'อะไร', 'เหงา', 'น้อง', 'น้า', 'เที่ยว', 'ข้าว', 'อา', 'ย่า', 'กลับ', 'ญาติ', 'ดู', 'ฉัน', 'มา', 'ส่ง', 'ลูก', 'ปลา', 'น้ำ', 'แฟน', 'ต้อง', 'สวัสดี']
for w in focus:
    n_e = int((ew_all.word_th == w).sum()); print(f"  {w:8s} RID={'Y' if w in rw else '-'} eTSL={n_e} (forms) corpus clips/signers={tuple(cnt.loc[w]) if w in cnt.index else (0,0)} RIDsent={int(R[(R.kind=='sentence')&(R.word==w)].shape[0])}")
# signer gain: words in vocab with <5 signers that gain +1 / +2 sources
sg = cnt.reindex(sorted(W)).fillna(0)
sg['rid'] = [w in rw for w in sg.index]; sg['etsl'] = [w in ew for w in sg.index]; sg['gain'] = sg.rid.astype(int) + sg.etsl.astype(int)
print('vocab words by current signer count x gain'); print(pd.crosstab(pd.cut(sg.signers, [-1, 1, 2, 4, 9, 999], labels=['<=1', '2', '3-4', '5-9', '>=10']), sg.gain).to_string())
sg.to_csv(S / 'vocab_gain.csv', encoding='utf-8-sig')
# daily conversation vocabulary
try:
    dc = json.load(open(ROOT / 'vocab/daily_conversation.json', encoding='utf-8')); print('daily keys', list(dc.keys())[:8] if isinstance(dc, dict) else type(dc))
except Exception as e: print(e)
# sentences: vocabulary words inside RID captions (longest-match substring)
vs = sorted(W, key=len, reverse=True)
def toks(c):
    out = []; c = str(c)
    for w in vs:
        if len(w) >= 2 and w in c: out.append(w); c = c.replace(w, ' ')
    return out
Rs = R[R.kind == 'sentence'].copy(); Rs['vw'] = Rs.caption.map(toks); Rs['nv'] = Rs.vw.map(len)
print('RID sentences', len(Rs), 'with >=1 vocab word', int((Rs.nv >= 1).sum()), '>=2', int((Rs.nv >= 2).sum()), '>=3', int((Rs.nv >= 3).sum()), 'entry word in vocab', int(Rs.word.isin(W).sum()), 'Q/A dialogs', int(Rs.caption.str.contains('ถาม').sum()))
vc = collections.Counter(w for l in Rs.vw for w in l); print('distinct vocab words appearing in RID sentences', len(vc), 'top', vc.most_common(25))
print('RID classifier entries', R[R.kind == 'classifier'].word.nunique(), 'examples', R[R.kind == 'classifier'].caption.head(6).tolist())
print('RID groups', R.groupby('group').word.nunique().to_dict())
print('eTSL letters', E[E.type == 'letter'].word_th.nunique(), 'multi-sign entries (n_signs>1)', int((E.n_signs > 1).sum()), 'words with >=2 variants', int((E[E.type=='word'].groupby('word_th').size() >= 2).sum()))
print(E[E.type == 'word'].groupby('word_th').size().sort_values(ascending=False).head(12).to_dict())
