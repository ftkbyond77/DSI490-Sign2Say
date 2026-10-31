# baseline-model — Uni-Sign + Thai retrieval (reference for v10 and later)

The original **Uni-Sign** pose encoder (Li et al., ICLR 2025; checkpoint `ZechengLi19/Uni-Sign` → `wlasl_pose_only_islr.pth`:
part-wise ST-GCN + mT5-base encoder, 90.3 M params) used as a frozen feature extractor, with nearest-neighbour retrieval against a bank of
~19k embedded Thai clips (9.5k concepts). Two variants:

| variant | what |
|---|---|
| `zeroshot` | frozen encoder + bank Z0 — "the baseline as published" |
| `adapted` | + residual MLP adapter (768-1024-768, CosFace, **train signers only**, epoch picked on val signers) + bank Z1 — "fitted to our data" |

Not included on purpose: tagger, spotting model, calibration, language layer (v8–v10). Segmentation = hands-raised runs cut at the median sign
length; same MediaPipe landmarks and scorer as `python main.py test`.

| file | |
|---|---|
| `unisign_wlasl_enc.pt`, `mt5_config.json` | frozen encoder (fp16, 184 MB, Git LFS) |
| `adapter.pt`, `bank.npz` | adapter weights; embeddings Z0 / Z1 of every clip + concept / role / dataset labels |
| `baseline.py` | `Baseline(variant, vocab)` · `recognise_video / recognise_landmarks` · `score_video` |
| `build.py` · `make_figures.py` · `run_v10_reference.py` · `evalsets.py` | rebuild · architecture figures · v10 reference results · evaluation videos |
| `figures/` | fig1 Uni-Sign architecture · fig2 one ST-GCN branch · fig3 baseline pipeline vs v10 · fig4 input parts · results |
| `results/` | `adapter_history.json`, `v10_reference.json` |

Results and usage: `baseline-model.ipynb` (repo root). Never tuned on data_test.
