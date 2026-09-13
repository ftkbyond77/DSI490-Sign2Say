"""Generate the blueprint's notebooks 00–08. Each notebook only calls modules/ and reads artifacts/reports —
no pipeline logic lives in notebooks."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
HEAD = """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from modules.utils import ART, read_json
REP = ART / 'reports'"""

BOOKS = {
    "00_explore_raw_data": [
        ("md", "# 00 · Explore & inspect raw data\nManifests (Standard Data Schema), lexeme statistics, signer inventory (E1, E2, E8)."),
        ("code", HEAD),
        ("code", "from modules.data import build_manifests\nclips, lex = build_manifests()\nclips.groupby(['source','subset']).agg(n=('clip_id','size'), hours=('duration_s', lambda x: x.sum()/3600))"),
        ("code", "lex.n_instances.value_counts().sort_index()"),
        ("code", "read_json(REP / 'signers.json')"),
    ],
    "01_schema_articulators_cache": [
        ("md", "# 01 · Articulators & embedding cache\nPose → signer lock → crops (native resolution) → frozen DINOv2-S. Run the full job with `python scripts/cli.py cache --set ttrs`."),
        ("code", HEAD),
        ("code", "from modules.articulators import Articulators, active_mask\nfrom modules.data import decode_frames\nart = Articulators(device='cpu', threads=4)\nframes = np.concatenate(list(decode_frames(ROOT / 'data_test' / 'ไปทานข้าวด้วยกันมั้ย.mp4')))\nK, V = [], []\nfor f in frames:\n    kp, g, v = art.step(f); K.append(kp); V.append(v)\nK = np.stack(K); print('valid-rate', np.mean(V, 0), 'active', active_mask(K, mode='stream').mean())"),
        ("code", "import matplotlib.pyplot as plt\ng = art.geometry(K[40][:, :2], K[40][:, 2])[0]\nc = art.crops(frames[40], g)\nplt.figure(figsize=(9, 3))\nfor i in range(3):\n    plt.subplot(1, 3, i + 1); plt.imshow(c[i][..., ::-1]); plt.axis('off')"),
    ],
    "02_prepare_datasets": [
        ("md", "# 02 · Prepare datasets\nLeakage-safe splits: every clip of a val/test sign is excluded from SSL and ISLR training; data_test/ is sha256-blocked."),
        ("code", HEAD),
        ("code", "sp = read_json(ART / 'manifests' / 'splits.json')\n{k: (len(v) if isinstance(v, list) else v) for k, v in sp.items()}"),
    ],
    "03_ssl_pretrain": [
        ("md", "# 03 · Temporal SSL pre-training\n`python scripts/experiments.py ssl --tag ssl_v1` (DINO + iBOT + KoLeo, Gram anchoring in stage 2)."),
        ("code", HEAD),
        ("code", "import matplotlib.pyplot as plt\nfor p in sorted(REP.glob('ssl_*.json')):\n    h = [r for r in read_json(p)['hist'] if 'loss' in r]\n    plt.plot([r['step'] for r in h], [r['loss'] for r in h], label=p.stem)\nplt.legend(); plt.xlabel('step'); plt.ylabel('loss')"),
        ("code", "{p.stem: read_json(p)['knn_val'] for p in sorted(REP.glob('ssl_*.json'))}"),
    ],
    "04_train_heads_islr_face": [
        ("md", "# 04 · ISLR prototype head (+ face / NMM)\n`python scripts/experiments.py islr --init ssl_v1 --tag islr_v1`"),
        ("code", HEAD),
        ("code", "rows = []\nfor p in sorted(REP.glob('islr_*.json')):\n    j = read_json(p)\n    rows.append(dict(tag=p.stem, **{f'val_{k}': v for k, v in j['val'].items() if k.startswith('R@')}, **{f'test_{k}': v for k, v in j['test'].items() if k.startswith('R@')}))\npd.DataFrame(rows)"),
    ],
    "05_continuous_spotting_llm": [
        ("md", "# 05 · Continuous spotting + language layer\nDecoder grid on pseudo-continuous val sentences, best config evaluated on test sentences."),
        ("code", HEAD),
        ("code", "{p.stem: read_json(p)['best'] for p in sorted(REP.glob('continuous_*.json'))}"),
        ("code", "from modules.language import RuleComposer\nRuleComposer().compose([dict(lemma='ไป', conf=.6), dict(lemma='กิน', conf=.5), dict(lemma='ข้าว', conf=.7)], dict(question=True))"),
    ],
    "06_evaluation": [
        ("md", "# 06 · Evaluation\nAll results collected in `result_reporting/result_v1.md`."),
        ("code", HEAD),
        ("code", "pd.DataFrame({k: v for k, v in read_json(REP / 'baselines.json').items() if isinstance(v, dict)}).T[['R@1','R@5','R@10','R@1_small','R@5_small','nn_same_signer']]"),
    ],
    "07_tts": [
        ("md", "# 07 · TTS\nLocal MMS-TTS Thai (offline). edge-tts / OpenAI TTS behind the same interface."),
        ("code", HEAD),
        ("code", "from modules.tts import LocalMMSTTS, SpeechStyle\nimport IPython.display as ipd\ntts = LocalMMSTTS()\nwav, sr = tts.synthesize('ไปกินข้าวด้วยกันไหม', SpeechStyle.from_affect('happy', 0.7))\nipd.Audio(wav, rate=sr)"),
    ],
    "08_inference_realworld": [
        ("md", "# 08 · Inference on real-world data (`data_test/`)\n`python scripts/cli.py infer --video data_test/<file>.mp4 --tag <tag>`"),
        ("code", HEAD),
        ("code", "rw = sorted(REP.glob('realworld_*.json'))\nread_json(rw[-1]) if rw else 'run scripts/experiments.py realworld first'"),
    ],
}


def main():
    NB.mkdir(exist_ok=True)
    for name, cells in BOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "hugging", "display_name": "hugging", "language": "python"}
        nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
        nbf.write(nb, NB / f"{name}.ipynb")
    print("wrote", len(BOOKS), "notebooks to", NB)


if __name__ == "__main__":
    main()
