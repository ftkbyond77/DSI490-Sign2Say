"""Round 4 (not blind): measured PSOLA templates vs the old ones, and interjections. 4 voices x 1 sentence x 7 variants.
 run measure_templates.py first.   HF_HUB_OFFLINE=1 caffeinate -i python -u round4.py   ->  out4/index.html
"""
import html, json, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent; OUT = HERE / "out4"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE)); from speech_mix import MixSpeech
mix = MixSpeech(HERE.parent.parent / "models" / "Speech-model-v4-TTS", refs=HERE / "refs_final.csv", policy=HERE / "policy.json", device="cpu")
import tts as T                                                # MixSpeech put the model folder on sys.path
NEW = {k: tuple(v) for k, v in json.loads((HERE / "templates_measured.json").read_text(encoding="utf-8"))["templates"].items()}
OLD = {k: T.TEMPLATE[k] for k in NEW}
mix.pol["overrides"] = {}
SENT = [x.strip() for x in open(HERE / "sentences.txt", encoding="utf-8") if x.strip()][4]
# (name, emotion, method, template set, prefix). C = reference already carries the emotion, so the measured template goes on A (neutral reference)
VAR = [("neutral", "neutral", "N", None, ""), ("happy_C_old", "happy", "C", OLD, ""), ("happy_A_new", "happy", "A", NEW, ""), ("sad_A_old", "sad", "A", OLD, ""),
       ("sad_A_new", "sad", "A", NEW, ""), ("happy_hahaha", "happy", "C", OLD, "ฮ่าๆ "), ("surprise_oho", "surprise", "N", None, "โอ้โห "), ("sad_sigh", "sad", "A", OLD, "เฮ้อ ")]
for spk in ("female", "male", "female_2", "male_2"):
    for name, emo, m, tpl, pre in VAR:
        f = OUT / f"{spk}_{name}.wav"
        if f.exists(): continue
        if tpl: T.TEMPLATE.update(tpl)
        mix.pol["methods"][emo] = m
        wav, sr = mix.say(pre + SENT, voice=spk, emotion=None if emo == "neutral" else emo, intensity=0.5, seed=0)
        mix.save(f, wav, sr); print(f.name, mix.last, flush=True)
rows = ["<meta charset=utf-8><style>body{font-family:sans-serif;margin:1em}td{padding:.3em;border-bottom:1px solid #ddd}</style><h2>รอบ 4: ค่า PSOLA ที่วัดจากข้อมูลไทย (new) เทียบค่าเดิม (old) + เสียงอุทาน</h2>"
        f"<p>{html.escape(SENT)}</p><table><tr><th>เสียง</th>" + "".join(f"<th>{n}</th>" for n, *_ in VAR) + "</tr>"]
for spk in ("female", "male", "female_2", "male_2"):
    rows.append(f"<tr><td>{spk}</td>" + "".join(f"<td><audio controls preload=none src='{spk}_{n}.wav' style='width:140px'></audio></td>" for n, *_ in VAR) + "</tr>")
(OUT / "index.html").write_text("\n".join(rows) + "</table>", encoding="utf-8"); print("->", OUT / "index.html")
