# ThaiSLM Reverse — Thai / English text or speech → Thai Sign Language avatar

The reverse direction of ThaiSLM: type or say a sentence (Thai or English) and a 3D VRM avatar signs it in Thai Sign Language
(TSL), half-body, in the browser. Results, blueprint and every experiment: `result_reporting/result_v2.md` (+ `.pdf`; v1:
`result_v1.md`). Comparison images (real signer above, avatar below): `result_reporting/image/`.

```bash
docker compose up --build          # at the repository root → http://localhost:8081 (this part) · http://localhost:8080 (forward part)
```

## How it works (one line per stage)

| stage | what | where |
|---|---|---|
| speech → text | Whisper fine-tuned on Thai (biodatlab): **medium on a GPU, small on a CPU** (auto), CTranslate2 int8; text appears while speaking (incremental recognition every second); impossible transcripts (invented greetings on half a second of audio) are flagged, not signed; correction **suggestions** the person accepts | `service/speech/`, `t2s/text/asr_correct.py` |
| English → Thai | NLLB-200 distilled 600M (CTranslate2 int8) — TSL is derived from the Thai, never from English order | `service/speech/`, `t2s/text/english.py` |
| Thai → TSL structure | tokenise → closed-class / dictionary roles → **homograph senses from context** (ตา eye ≠ grandfather) → ordering in the chosen **Grammar Format**: `TSL51` (rules from the TSL51 corpus) or `ปกติ` (same signs, Thai word order) | `t2s/text/` (`grammar.py`, `senses.py`) |
| words → signs | dictionary lemma / reviewed synonym / curated colloquial variant → compound split → fingerspelling (recorded letters) → text card. **No sign is ever invented.** | `t2s/text/translate.py`, `curated.py` |
| IR | one JSON per sentence (`t2s-2`): TSL order + grammar mode, sentence type, per sign: lexeme (concept · sense · variant), clip, **provenance** (origin, dataset, clip, signer, quality, licence, reconstruction), two separate confidences; versions of lexicon / library / rules / rig; the recogniser's own text | `t2s/ir.py`, `t2s/text/lexemes.py` |
| signs → motion | 9,734 recorded signs retargeted offline from MediaPipe to the VRM skeleton; **palm-contact reconstruction** (wai), **face-contact depth** correction; strokes untouched, joined by minimum-jerk transitions with anticipatory handshape | `t2s/motion/`, `models/motion/` |
| face / head | lexical (each clip's own) + **scoped sentence marks from data**: WH brows from the 2nd sign, yes/no forward tilt on the last sign, negation head-shake on the negation sign; no topic marking (no evidence) | `t2s/motion/nmm.py`, `t2s/eval/nmm_data.py` |
| avatar | three.js + three-vrm, 25 fps timeline interpolated at display rate; `.vrma` export for Unity (UniVRM) | `service/web/`, `t2s/motion/vrma.py` |
| characters | every `*.vrm` in `vrm/` (VRM 0.x or 1.0, any exporter) is selectable; checked on arrival (rig, absent bones folded, face channels available — shown in the picker tooltip); motion fitted per body (face anchors scale with the head, torso anchors with shoulder width); a model without an embedded thumbnail gets a rendered headshot once | `t2s/avatar.py`, `t2s/motion/fit.py`, `rig.py` |

## Rebuild the models (laptop CPU, ≈ 45 min, $0 cloud)

```bash
python main.py audit && python main.py select && python main.py library && python main.py lexicon
python main.py timing && python main.py avatar
python main.py test && python main.py eval all
```
Data is read from `../cloud_s3/` (the S3 lake mirror). Words the dictionaries lack can be harvested: `python main.py harvest WORD …`
(YouTube lessons → MediaPipe → QC → eye check in `t2s/data/harvest_decisions.csv`), then `python main.py add-clips` appends
them to the library without a full rebuild.

## GPU for speech (optional)

The speech service uses a GPU when one is visible (`SPEECH_DEVICE=auto`) and then switches to Whisper-th **medium**
(real-speech CER 3.8 % vs 6.1 %, ≈ 0.7 s per short utterance on an RTX 2050). Build the image with the CUDA libraries and give it
the GPU:
```bash
docker compose -f docker-compose.yml -f reverse_text2sign/service/speech/compose.gpu.yml up --build
```
(The GPU numbers were measured outside Docker; this container variant has not been run yet — see result_v2.md §7.2.)

## Development (without Docker)

```bash
pip install fastapi==0.115.6 uvicorn==0.32.1 httpx==0.28.1 python-multipart==0.0.20 pytest   # on top of the forward requirements
uvicorn --app-dir service/api app:app --port 8091                 # API (reads models/)
cp models/avatar.vrm service/web/public/ && npm --prefix service/web ci && npm --prefix service/web run dev   # :5181, proxies /api
python main.py test
```
English input in development: set `T2S_MT_DIR` to a CTranslate2 NLLB folder, or run the `reverse-speech` container (`T2S_SPEECH_URL`).
`T2S_ASR_CORRECT=suggest|auto|off` (default `suggest`).

## Layout

| path | what |
|---|---|
| `main.py` | the only CLI |
| `t2s/data/` | audit (QC tiers), exemplar selection, harvest; eye-check decision files (`youtube_fill.csv`, `harvest_decisions.csv`) |
| `t2s/text/` | lexicon, lexemes (concept / sense / variant / provenance), grammar (two modes), senses (homographs), curated colloquial table, translator, ASR correction, English MT client |
| `t2s/motion/` | rig, cleaning (palm contact), retargeting (face depth), library, composer, non-manual overlays, body fit, codec, VRMA |
| `t2s/eval/` | language (held-out), motion quality, non-manual evidence, ASR, aligned real-vs-avatar images, timing, renders, report data |
| `service/` | `api` (FastAPI) · `speech` (ASR + MT) · `web` (Vite / three-vrm, nginx) — wired in the root `docker-compose.yml` |
| `models/` | deployable bundle: `motion/` (q.bin · m.bin · index.json), `lexicon.json`, `timing.json`, `avatar.vrm` |
| `tests/` | pytest (language, motion, API) + `conversation.tsv` (everyday sentences, dev) + `language_heldout.tsv` (frozen held-out set) |
| `vrm/` | the characters (`default.vrm` = the library avatar). Add or remove any `.vrm` — no rebuild or restart; check each model's licence (shown in the picker tooltip; e.g. Model 2 disallows commercial use, Models 3–4 are personal non-profit) |
| `data/` | derived, git-ignored (audit tables, harvest, evaluation audio and runs) |

Licensing: TTRS (public research use), th-sl.com (rights reserved; research use approved by the project owner, not harvested by
this part), TSL51 (dataset licence), YouTube lessons (publishers' rights, research use only). Every sign's terms are in its
provenance (IR and the page's provenance box).
