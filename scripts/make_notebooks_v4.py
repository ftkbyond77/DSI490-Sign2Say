"""Generate + execute notebooks/v4/*.ipynb (kernel: hugging). Each notebook calls modules/ and reads artifacts;
07 runs the full pipeline live on data_test/."""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "v4"

HEAD = """import sys, json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
ROOT = Path.cwd()
while not (ROOT / 'modules').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Tahoma', 'Leelawadee UI', 'DejaVu Sans']
from IPython.display import display, Image, Audio, Markdown
from modules.utils import ART, read_json
REP, V4, MAN4 = ART / 'reports', ART / 'v4', ART / 'manifests_v4'
pd.set_option('display.max_colwidth', 80)"""

BOOKS = {
"00_data_cleaning_schema": [
("md", "# 00 · Data cleaning & Standard Schema v4\nLabels without manual annotation: TTRS gloss · YouTube **title/filename** · **speech in the clip** (YouTube th-orig captions with word timestamps, Thai Whisper where missing). Hard-to-use sources are excluded with a recorded reason."),
("code", HEAD),
("code", "clips = pd.read_parquet(MAN4 / 'clips_v4.parquet')\nclips.groupby(['source','subset']).agg(clips=('clip_id','size'), usable=('usable_for', lambda s: (s!='').sum()))"),
("code", "clips.exclude_reason.value_counts(dropna=False).to_frame('clips')"),
("code", "clips[clips.label_source=='title'][['title','label','duration_s']].head(30)"),
("code", "sw = pd.read_parquet(MAN4 / 'speech_words.parquet')\nprint('speech hits:', len(sw), 'distinct lexicon words:', sw.word.nunique(), '| sources:', sw.source.value_counts().to_dict())\nsw.word.value_counts().head(25).to_frame('mentions')"),
("code", "words = ['ฉัน','ปลอบใจ','เพื่อน','ร้องไห้','ไป','กิน','ข้าว','ด้วยกัน','ไหม']\nttrs = clips[clips.source=='ttrs']\npd.DataFrame([dict(word=w, ttrs_clips=int((ttrs.label==w).sum()), youtube_speech_mentions=int((sw.word==w).sum()), youtube_videos=int(sw[sw.word==w].clip_id.nunique())) for w in words])"),
],
"01_pose_wholebody": [
("md", "# 01 · Whole-body pose (RTMW-X, 133 keypoints)\nSigner box from the v1 signer lock, expanded to the upper body so hands get more pixels. Hands = 21 keypoints each."),
("code", HEAD),
("code", "from modules.data import decode_frames\nimport cv2\ndef draw(stem):\n    d = np.load(ART/'pose_v4'/f'test_{stem}_125.npz'); kp, sc = d['kp'], d['sc']\n    fr = np.concatenate(list(decode_frames(ROOT/'data_test'/f'{stem}.mp4', fps=12.5))); H, W = fr.shape[1:3]\n    fig, ax = plt.subplots(1, 6, figsize=(18, 4))\n    for a, i in zip(ax, np.linspace(10, len(fr)-10, 6).astype(int)):\n        a.imshow(fr[i][..., ::-1]); ok = sc[i] > 0.3\n        a.scatter(kp[i, 91:133, 0][ok[91:133]]*W, kp[i, 91:133, 1][ok[91:133]]*H, s=4, c='lime')\n        a.scatter(kp[i, 23:91, 0][ok[23:91]]*W, kp[i, 23:91, 1][ok[23:91]]*H, s=1, c='cyan')\n        a.scatter(kp[i, :11, 0][ok[:11]]*W, kp[i, :11, 1][ok[:11]]*H, s=8, c='red'); a.set_title(f't={i/12.5:.1f}s'); a.axis('off')\n    plt.suptitle(stem); plt.show()\n    print('mean hand confidence L/R:', sc[:, 91:112].mean().round(3), sc[:, 112:133].mean().round(3))\nfor stem in ['ฉันปลอบเพื่อนร้องไห้', 'ไปทานข้าวด้วยกันมั้ย']:\n    draw(stem)"),
("code", "import glob\nfiles = glob.glob(str(ART/'pose_v4'/'ttrs_*.npz'))\nconf = [float(np.load(f)['sc'][:, 91:133].astype(np.float32).mean()) for f in files[::25]]\nprint(len(files), 'TTRS pose files'); plt.hist(conf, bins=30); plt.title('TTRS mean hand-keypoint confidence (sample)'); plt.show()"),
],
"02_encoder_pretrained_finetune": [
("md", "# 02 · Encoder: pretrained Uni-Sign (pose) vs v1 · fine-tuning adapter\n**E1** = a new signer performs a dictionary sign; rank of the correct sign in the gallery."),
("code", HEAD),
("code", "pilot = read_json(REP/'v4_eval_enc_pilot.json'); note = pilot.pop('_note', '')\nprint(note)\npd.DataFrame(pilot).T.sort_values('MRR', ascending=False)"),
("code", "ev = read_json(REP/'v4_eval.json')\ndisplay(pd.DataFrame({k: v for k, v in ev.items() if isinstance(v, dict)}).T)\nif 'E5|per_clip' in ev: display(pd.DataFrame(ev['E5|per_clip']))"),
("code", "ad = read_json(REP/'v4_adapter.json')\nprint('adapter selected:', ad['selected'], '| rule:', ad['rule'])\ndisplay(pd.DataFrame({'zero-shot': {p: ad['zero_shot'][p]['MRR'] for p in ('val','test')}, 'adapter': {p: ad['adapter'][p]['MRR'] for p in ('val','test')}}).rename_axis('MRR'))\nh = pd.DataFrame(ad['hist']); plt.plot(h.step, h.MRR, marker='o'); plt.xlabel('step'); plt.ylabel('val MRR'); plt.title('adapter fine-tuning'); plt.show()"),
],
"03_vocabulary_bank_mining": [
("md", "# 03 · Vocabulary bank & weakly-supervised mining\nSpoken word at time t → candidate windows → accepted only if the pose encoder ranks that word in the top-K of the whole TTRS vocabulary."),
("code", HEAD),
("code", "mr = read_json(REP/'v4_mining.json'); print({k: v for k, v in mr.items() if k != 'per_video'})\nm = pd.read_parquet(V4/'mined.parquet'); m.word.value_counts().head(30).to_frame('mined instances')"),
("code", "b = np.load(V4/'bank.npz')\nbank = pd.DataFrame(dict(word=b['words'], source=b['source'], signer=b['signer']))\nprint('bank instances:', len(bank), '| vocabulary words:', bank.word.nunique())\ndisplay(bank.source.value_counts().to_frame('instances'))\nsig = bank.groupby('word').signer.nunique(); print('words with ≥2 signers:', int((sig>=2).sum()))"),
("code", "for w in ['ฉัน','ปลอบใจ','เพื่อน','ร้องไห้','กิน','ข้าว','ด้วยกัน','ไป','ไหม']:\n    g = bank[bank.word==w]; print(f'{w:10s} instances={len(g):2d} sources={g.source.value_counts().to_dict()} signers={g.signer.nunique()}')"),
("code", "p = ROOT/'result_reporting'/'v4_mined_examples.jpg'\ndisplay(Image(filename=str(p))) if p.exists() else print('no example sheet')"),
],
"04_segmentation": [
("md", "# 04 · Continuous segmentation — which frames are a word?\nTrained on synthetic sentences built from isolated TTRS signs (same signer, 1–2.5× faster, interpolated transitions, compact signing space). Validation uses a **held-out signer**."),
("code", HEAD),
("code", "seg = read_json(REP/'v4_segmentation.json')\nprint('held-out signer:', seg['val_signer']); display(pd.DataFrame({'learned': seg['val_learned'], 'heuristic': seg['val_heuristic']}))\nh = pd.DataFrame(seg['hist']); fig, ax = plt.subplots(1, 2, figsize=(10,3)); ax[0].plot(h.step, h.loss); ax[0].set_title('loss'); ax[1].plot(h.step, h.boundary_f1, label='boundary F1'); ax[1].plot(h.step, h.frame_acc, label='frame acc'); ax[1].legend(); plt.show()"),
("code", "import torch\nfrom modules.segment import Segmenter, decode_segments, sequence_features\nfrom modules.unisign import UniSignEncoder\nfrom modules.recognize import load_pose\nenc = UniSignEncoder('wlasl').eval(); model = Segmenter().cuda(); model.load_state_dict(torch.load(V4/'segmenter.pt')); model.eval()\ntext_events = {'ไปทานข้าวด้วยกันมั้ย': [2.32, 3.08, 3.68, 4.24, 4.72]}  # on-screen text changes (evaluation only)\nfor stem in ['ฉันปลอบเพื่อนร้องไห้', 'ไปทานข้าวด้วยกันมั้ย']:\n    kp, sc, hw, fps = load_pose(ART/'pose_v4'/f'test_{stem}_125.npz')\n    x = sequence_features(enc, kp, sc, hw)\n    with torch.no_grad(): prob = model(torch.from_numpy(x)[None].cuda()).softmax(-1)[0].cpu().numpy()\n    segs = decode_segments(prob); t = np.arange(len(prob))/12.5\n    plt.figure(figsize=(12, 2.6)); plt.plot(t, prob[:,1]+prob[:,2], label='P(sign)'); plt.plot(t, prob[:,1], label='P(begin)')\n    for a, b in segs: plt.axvspan(a/12.5, b/12.5, color='orange', alpha=0.25)\n    for e in text_events.get(stem, []): plt.axvline(e, color='k', ls='--', lw=0.8)\n    plt.title(f'{stem}: {len(segs)} segments ' + str([(round(a/12.5,2), round(b/12.5,2)) for a,b in segs])); plt.legend(); plt.show()"),
],
"04b_spotting_decoder": [
("md", """# 04b · Recognition-driven spotting (the decoder actually used on data_test)
The tagger above sees fluent real signing as one long *inside* run (no internal begin peaks), so the old decoder could only cut equal chunks. Spotting scores every 0.5–2.4 s window against the vocabulary and a semi-Markov DP picks non-overlapping words.

**Tuning/eval data:** synthetic sentences of held-out signer S_ttrs_00 with a bank that contains *no* clip of that signer (78% of that signer's words are therefore out-of-vocabulary, like real use). λ/γ/min_len tuned on split *tune*, reported on split *test*. data_test is never used for tuning."""),
("code", HEAD),
("code", """sp = read_json(REP/'v4_spotting.json')
display(pd.DataFrame(sp['results']).T[['seg_f1','boundary_f1','count_err','words_per_sentence_pred','words_per_sentence_gt','word_recall_top1','word_recall_top5','word_precision_top1','WER']].round(3))
print({k: v for k, v in sp['config'].items() if k != 'tiers_empirical'})"""),
("code", """g = pd.DataFrame(sp['grid']); cfg = sp['config']
g0 = g[(g.beta == 0.0) & (g.min_len == cfg['min_len'])]
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
for a, col in zip(ax, ['objective', 'seg_f1', 'words_per_sentence_pred']):
    pv = g0.pivot(index='lam', columns='gap_cost', values=col); im = a.imshow(pv.values, aspect='auto', cmap='viridis')
    a.set_xticks(range(len(pv.columns))); a.set_xticklabels(pv.columns); a.set_yticks(range(len(pv.index))); a.set_yticklabels(pv.index)
    a.set_xlabel('gap cost γ'); a.set_ylabel('λ'); a.set_title(f"{col} (tune, β=0, min_len={cfg['min_len']})"); plt.colorbar(im, ax=a)
plt.show()
display(g.sort_values('objective', ascending=False).groupby('beta').head(5)[['beta','min_len','lam','gap_cost','seg_f1','boundary_f1','count_err','words_per_sentence_pred','word_recall_top5','objective']].round(3))"""),
("code", """print(open(REP/'v4_spot_agg_check.txt', encoding='utf-8').read())"""),
],
"05_recognition_calibration": [
("md", "# 05 · Open-set recognition & calibration\nA segment is **accepted** only if its top-1 score/margin pass thresholds calibrated for top-1 precision; **uncertain** if the correct word is likely in the top-5; otherwise **unknown** — reported with the nearest known word instead of a guess."),
("code", HEAD),
("code", "cal = read_json(REP/'v4_calibration.json'); cal"),
("md", "Spotted segments use the same score/margin rule. Measured on held-out-signer synthetic sentences (bank without that signer) — the tiers are only weakly informative:"),
("code", "print(open(REP/'v4_spot_tier_check.txt', encoding='utf-8').read())\nprint(open(REP/'v4_spot_agg_check.txt', encoding='utf-8').read())"),
],
"06_face_model": [
("md", "# 06 · Face model (separate from the Sign model)\nEmotion from a ViT FER model on the face crop · non-manual cues from the 68 face landmarks relative to the signer's own baseline."),
("code", HEAD),
("code", "from modules.face_v4 import FaceModel, nmm, landmark_signals\nfrom modules.data import decode_frames\nfm = FaceModel()\nfor stem in ['ฉันปลอบเพื่อนร้องไห้', 'ไปทานข้าวด้วยกันมั้ย']:\n    d = np.load(ART/'pose_v4'/f'test_{stem}_125.npz'); kp, sc, hw = d['kp'], d['sc'], tuple(d['hw'])\n    fr = np.concatenate(list(decode_frames(ROOT/'data_test'/f'{stem}.mp4', fps=12.5)))\n    emo = fm.emotions(fr, kp, sc); cues = nmm(kp, sc, hw)\n    print(stem, '| emotion:', emo['emotion'], emo['probs'], '| strongest non-neutral:', emo['strongest_non_neutral'])\n    print('   NMM:', cues)\n    s = landmark_signals(kp, sc, hw); t = np.arange(len(kp))/12.5\n    fig, ax = plt.subplots(1, 2, figsize=(12, 2.5)); ax[0].plot(t, s['brow'], label='brow height'); ax[0].plot(t, s['mouth'], label='mouth open'); ax[0].legend()\n    pf = emo['per_frame']; ax[1].stackplot(np.array(emo['frames'])/12.5, pf.T, labels=['angry','disgust','fear','happy','neutral','sad','surprise']); ax[1].legend(fontsize=6, loc='upper left'); plt.suptitle(stem); plt.show()"),
],
"07_inference_data_test": [
("md", "# 07 · End-to-end inference on real-world `data_test/`\nPose → tagger (active region) → **recognition-driven spotting** (04b) → open-set status (E1 calibration) → Face model → LLM fusion (OpenAI, constrained to the Sign model's candidates) → **deterministic evidence guard** (a sentence is shown only if ≥1 sign is accepted and ≥ half of the signs are resolved). **Audio and on-screen text of the test videos are never used.**"),
("code", HEAD),
("code", "from modules.pipeline_v4 import SignPipelineV4\npipe = SignPipelineV4()"),
("code", "results = {}\nfor stem in ['ฉันปลอบเพื่อนร้องไห้', 'ไปทานข้าวด้วยกันมั้ย']:\n    out = ROOT/'result_reporting'/'inference_v4_notebook'/stem\n    r = pipe.run(ROOT/'data_test'/f'{stem}.mp4', out, tts=True); results[stem] = r\n    display(Markdown(f'## {stem}'))\n    display(pd.DataFrame([dict(t0=s['t0'], t1=s['t1'], status=s['status'], nearest=s['nearest'], score=round(s['score'],3), margin=round(s['margin'],3), top5=', '.join(c['word'] for c in s['candidates'])) for s in r['segments']]))\n    display(Image(filename=str(out/'segments.jpg')))\n    f = r['fusion']; display(Markdown(f\"**Evidence (Sign model only):** {f.get('evidence_gloss')}  \\n**Sentence:** {f.get('sentence_th') or '— withheld —'}  \\n**Guard:** {f.get('evidence_guard')} · tentative LLM reading (not shown to users): {f.get('tentative_sentence_th')}  \\n**Meaning:** {f.get('meaning_th')}  \\n**Emotion:** {f.get('emotion_th')}  \\n**Chosen:** {f.get('chosen')} · confidence {f.get('confidence')} · {f.get('provider')}  \\n**Notes:** {f.get('notes')}\"))\n    print('face:', r['face'].get('emotion'), r['face'].get('nmm', {}).get('question_yesno'), '| timings:', r['timings_ms'])\n    if (out/'audio.wav').exists(): display(Audio(filename=str(out/'audio.wav')))"),
],
"08_evaluation_summary": [
("md", "# 08 · Evaluation summary (v4)"),
("code", HEAD),
("code", "inf = read_json(REP/'v4_infer_final.json')\nfor k, v in inf.items():\n    display(Markdown(f'### {k}'))\n    print('reference:', v['ref'], '| segments:', v['n_segments'], 'vs', v['n_ref'], 'signs')\n    print('accepted words:', v['accepted_words'], '| WER(accepted):', round(v['WER_accepted'],2), '| WER(LLM chosen):', v['WER_llm_chosen'])\n    print('reference word inside some segment top-5:', v['ref_in_top5_any_segment'], '| best rank:', v['best_rank'])\n    print('sentence:', v['sentence'], '| chrF:', v['chrF'])"),
("code", "rows = []\nfor tag, name in (('tagger_llm', 'tagger decoder (equal chunks)'), ('final', 'spotting + guard (final)')):\n    d = read_json(REP/f'v4_infer_{tag}.json')\n    for k, v in d.items():\n        rows.append(dict(decoder=name, video=k, n_segments=v['n_segments'], n_ref=v['n_ref'], ref_in_top5=sum(v['ref_in_top5_any_segment'].values()), best_rank=v['best_rank'], accepted=v['accepted_words'], sentence=v['sentence'], tentative=v.get('tentative_sentence')))\npd.DataFrame(rows)"),
("code", "lm = read_json(REP/'v4_lm_eval.json')['summary']; lm1 = read_json(REP/'v4_lm_eval_prompt1.json')['summary']\npd.DataFrame({'prompt v1 (pick from top-3)': lm1, 'prompt v2 + evidence guard (final)': lm}).T"),
("code", "print(open(ROOT/'result_reporting'/'result_v4.md', encoding='utf-8').read()[:6000])"),
],
}


def build(execute=True, only=None):
    NB.mkdir(parents=True, exist_ok=True)
    for name, cells in BOOKS.items():
        if only and name not in only:
            continue
        nb = nbf.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "hugging", "display_name": "hugging", "language": "python"}
        nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in cells]
        path = NB / f"{name}.ipynb"
        if execute:
            from nbclient import NotebookClient
            client = NotebookClient(nb, timeout=1800, kernel_name="hugging", resources={"metadata": {"path": str(NB)}}, allow_errors=True)
            client.execute()
        nbf.write(nb, path)
        errs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
        print(name, "executed" if execute else "written", "errors:", len(errs))


if __name__ == "__main__":
    build(execute="--no-exec" not in sys.argv, only=[a for a in sys.argv[1:] if not a.startswith("--")] or None)
