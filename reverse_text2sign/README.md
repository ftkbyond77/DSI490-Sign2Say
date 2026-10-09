# ThaiSLM Reverse — Thai / English text or speech → Thai Sign Language avatar

The reverse direction of ThaiSLM: type or say a sentence (Thai or English) and a 3D VRM avatar signs it in Thai Sign Language
(TSL), half-body, in the browser. Results, blueprint and every experiment: `result_reporting/result_v1.md` (+ `.pdf`).

```bash
docker compose up --build          # at the repository root → http://localhost:8081 (this part) · http://localhost:8080 (forward part)
```

## How it works (one line per stage)

| stage | what | where |
|---|---|---|
| speech → text | Whisper-small fine-tuned on Thai (biodatlab) / Whisper base.en, CTranslate2 int8, CPU | `service/speech/` |
| English → Thai | NLLB-200 distilled 600M (CTranslate2 int8) — TSL is derived from the Thai, never from English order | `service/speech/`, `t2s/text/english.py` |
| Thai → TSL structure | tokenise (pythainlp + dictionary-aware repair) → closed-class / dictionary roles → TSL rules (time first, object-topic, head-final verb chain, WH last, negation after the predicate, possessor first) | `t2s/text/` |
| words → signs | dictionary lemma / reviewed synonym / curated colloquial variant → compound split → fingerspelling (recorded letters) → text card. **No sign is ever invented.** | `t2s/text/translate.py`, `curated.py` |
| IR | one JSON per sentence: TSL order, sentence type, per sign: clip, hand, location, movement, face, two separate confidences (translation · motion) | `t2s/ir.py` |
| signs → motion | 9,734 recorded signs (TTRS, th-sl.com, TSL51, eye-checked YouTube) retargeted offline from MediaPipe to the VRM skeleton; strokes joined by minimum-jerk transitions timed from real signers; recorded phrases for exact conversational matches | `t2s/motion/`, `models/motion/` |
| face | the signer's own blendshapes per sign + sentence-level brow raise for questions (data: TSL51) | `t2s/motion/retarget.py`, `nmm.py` |
| avatar | three.js + three-vrm, 25 fps timeline interpolated at display rate; `.vrma` export for Unity (UniVRM) | `service/web/`, `t2s/motion/vrma.py` |
| characters | every `*.vrm` in `vrm/` is selectable (top right of the stage): `default.vrm` = Default, the rest Model 1…n by file name; each motion is fitted to the chosen body (arm IK anchored to face / chest landmarks), so all characters sign the same places | `t2s/avatar.py`, `t2s/motion/fit.py` |

## Rebuild the models (laptop CPU, ≈ 30 min, $0 cloud)

```bash
python main.py audit && python main.py select && python main.py library && python main.py lexicon
python main.py timing && python main.py avatar
python main.py test && python main.py eval all
```
Data is read from `../cloud_s3/` (the S3 lake mirror: `python ../main.py data pull raw_data` / `data_prep`). Words the dictionaries
lack can be harvested: `python main.py harvest WORD …` (YouTube lessons → MediaPipe → QC → eye check in `t2s/data/harvest_decisions.csv`).

## Development (without Docker)

```bash
pip install fastapi==0.115.6 uvicorn==0.32.1 httpx==0.28.1 python-multipart==0.0.20 pytest   # on top of the forward requirements
uvicorn --app-dir service/api app:app --port 8091                 # API (reads models/)
cp models/avatar.vrm service/web/public/ && npm --prefix service/web ci && npm --prefix service/web run dev   # :5181, proxies /api
python main.py test
```
English input in development: set `T2S_MT_DIR` to a CTranslate2 NLLB folder, or run the `reverse-speech` container (`T2S_SPEECH_URL`).

## Layout

| path | what |
|---|---|
| `main.py` | the only CLI |
| `t2s/data/` | audit (QC tiers), exemplar selection, harvest; eye-check decision files (`youtube_fill.csv`, `harvest_decisions.csv`) |
| `t2s/text/` | lexicon, grammar, curated colloquial table, translator, English MT client |
| `t2s/motion/` | rig, cleaning, retargeting, library, composer, non-manual overlays, codec, VRMA |
| `t2s/eval/` | timing model, conversation coverage, back-translation, renders, report data |
| `service/` | `api` (FastAPI) · `speech` (ASR + MT) · `web` (Vite / three-vrm, nginx) — wired in the root `docker-compose.yml` |
| `models/` | deployable bundle: `motion/` (q.bin · m.bin · index.json), `lexicon.json`, `timing.json`, `avatar.vrm` |
| `tests/` | pytest (language, motion, API) + `conversation.tsv` (Thai / English everyday sentences) |
| `vrm/` | the characters (`default.vrm` = the library avatar, VRoid AvatarSample_C). Add or remove any `.vrm` (VRM 0.x / 1.0, standard humanoid) — no rebuild or restart; the API makes an optimised copy + thumbnail in the `reverse-avatars` volume. Check each model's VRoid Hub licence (e.g. Model 2 disallows commercial use) |
| `data/` | derived, git-ignored (audit tables, contact sheets, harvest, evaluation renders) |

Licensing: TTRS (public research use), th-sl.com (rights reserved; research use approved by the project owner, not harvested by
this part), TSL51 (dataset licence), YouTube lessons (publishers' rights, research use only).
