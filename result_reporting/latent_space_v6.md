# ThaiSLM — Latent space v6 (what the encoder learned, and where it still fails)

Encoder: Uni-Sign pose branch (CSL-News → WLASL) fine-tuned on all v6 data (`enc-all-0241-us-central1`, best step 2,700, EMA
weights). All numbers are on **held-out signers**, bank = training clips, open vocabulary of ~9.9k concepts, unless stated.
Source: `artifacts/reports/isolated.json`, `data_ablation.json`, `baseline_v5.json`; figures in `figures_v6/`.

## 1. Is the space organised by sign?

![t-SNE](figures_v6/latent_tsne.png)

| held-out queries (val + test), 5 nearest neighbours | zero-shot Uni-Sign | v6 |
|---|---|---|
| share with the same **concept** | 0.403 | **0.448** |
| share with the same signer | 0.335 | 0.318 |
| share with the same dataset / extractor | 0.764 / 0.876 | 0.757 / 0.885 |
| cos(same concept, other signer) — TSL-ONE-S | 0.604 | **0.695** |
| cos(other concept, same signer) | 0.210 | **0.055** |
| cos(other concept, other signer) | 0.196 | 0.051 |

Same-concept clips moved together (+0.09 cosine) and the "same signer" pull vanished (0.21 → 0.055, now equal to the other-signer
baseline): **the space is organised by sign, not by person.** In the t-SNE the same concept from TTRS (o), th_sl (+), TSL51 (△) and
YouTube (◇) forms one cluster after fine-tuning; zero-shot clusters of กระโดดสูง / กัมพูชา / กอล์ฟ / กุมภาพันธ์ overlapped and
separate in v6. (The 5-NN concept share is lower than v5's 0.84 because the v6 neighbourhood contains 9.9k concepts, 5k of them
from one-signer dictionary entries.)

## 2. Does fine-tuning learn TSL, or lean on the ASL / CSL prior? (prompt_7 "uncertain point")

| check | result | reading |
|---|---|---|
| Thai fingerspelling (TSL-ONE-S consonants / vowels — handshapes ASL/CSL do not have), 5 unseen signers | R@1 0.823 → **0.900** | as large as for lexical signs (0.782 → 0.885): the model learns Thai-specific forms |
| representation change (linear CKA zero-shot vs v6, 5k clips) | **0.60** | the geometry moved substantially, not a thin head on a frozen prior |
| encoder benchmark: remove the ASL step (CSL-only init) | val 0.5005 → 0.4424 | the ASL prior **helps** overall … |
| encoder benchmark: no prior at all (scratch) | val 0.5005 → 0.342 | … and pretraining is essential with this much data |
| inherited confusions: v6 errors that are the zero-shot model's same wrong answer | **38 %** | … but part of the prior's confusions survive fine-tuning |
| zero-shot errors fixed / correct answers broken (pooled test queries) | 10.2 % / 10.8 % | on the pooled test set the gains and losses are of similar size; the gains are on multi-signer data (TTRS +7, TSL-ONE-S +9, th_sl +2 points), the losses on the single TSL51 researcher |

Conclusion: the fast convergence (v5 best at step 400; v6 at 2,700 with ~2× the data) is mostly good transfer, not "learning
nothing new"; the measurable risk is the 38 % of errors that are inherited ASL/CSL confusions.

![hubness and prior](figures_v6/hubness_prior_bias.png)

## 3. Hubness

| | zero-shot | v6 |
|---|---|---|
| k-occurrence skew of nearest-concept assignments (held-out queries) | 4.01 | **3.63** |
| share of errors absorbed by the 10 most frequent wrong answers | 4.2 % | 4.8 % |

v5's latent report counted 24 % of errors in the top-10 hubs on a different denominator; on v6's held-out queries the hub
concentration is low and fine-tuning slightly reduced the skew. CSLS (score − α·hubness, hubness = mean of each concept's 10
highest scores over 11k unlabeled broadcast windows) was implemented and swept (figure, left): it helps RTMW dictionary
queries a little (th_sl val +0.011, TTRS val +0.008 MRR at α 0.5–0.75) and **hurts TSL-ONE-S strongly (−0.04 at α 0.5)** — the
concepts it discounts are the everyday words seen from many signers. On conversational sentences it lowered word recall, so the
final decoder does **not** use it. The real-video false positives (เด็ก, ศูนย์ on data_test) behave like hubs of the *deployment*
domain (phone video), not of the training domain — the reference set for CSLS would need real conversational video.

## 4. The extractor gap (RTMW vs MediaPipe)

| TSL51 researcher (same 547 videos, both extractors) | zero-shot | v6 |
|---|---|---|
| open-vocabulary R@1 — MediaPipe / RTMW | 0.608 / 0.372 | 0.620 / 0.342 |
| closed 51-sign R@1 — MediaPipe / RTMW | 0.900 / 0.865 | 0.912 / 0.861 |
| cos(same clip MP, same clip RTMW) vs cos(random pair) | 0.785 vs 0.172 | 0.730 vs 0.050 |
| null-vs-sign AUC — MediaPipe / RTMW | 0.839 / 0.861 | 0.895 / **0.951** |

The relative separation of a clip's two extractions from random clips improved (gap 0.61 → 0.68), and RTMW null detection is
much better, but the open-vocabulary RTMW score of this one signer did not improve: in a 9.9k-concept bank, the TSL51 words'
nearest RTMW neighbours now include many visually close dictionary signs. An extractor-adversarial loss (gradient reversal) did
not change this (0.340). Because every video source is now RTMW, the gap matters less for real use than in v5 (TSL51 is the only
remaining MediaPipe-vs-RTMW pair).

## 5. Bank depth — still the strongest lever

![depth](figures_v6/recall_by_depth.png)

| training signers of the query's concept | 1 | 2 | 3 | 4–5 | 6–9 | ≥ 10 |
|---|---|---|---|---|---|---|
| R@1 (held-out test signers) | ≈ 0.13 | ≈ 0.15 | ≈ 0.48 | ≈ 0.44 | ≈ 0.46 | **0.88** |

A word becomes reliable around 3 signers and excellent with ~10. v6 raised the number of concepts with ≥ 2 signers from 605 to
1,969 and with ≥ 3 to 567; the data_test words that crossed that line (ร้องไห้, ดื่ม, น้ำ) are now found. Bank depth is **not** used
as a prior or calibration feature, though: in this corpus it also tells *which dataset* a word comes from (TSL-ONE-S's 184 words
have 29 signers each), and using it pushed those words onto real video (`result_v6.md` §5.3).

## 6. What would change the space most (evidence-based)

1. More signers per everyday word (3+), recorded on phones in conversational style — §5 shows the step from 2 to 3 signers is the
   largest single gain.
2. Real conversational RTMW video as the hubness / calibration reference instead of broadcasts (§3).
3. A Thai self-supervised stage with a harder task (the v6 SSL view-matching stayed at accuracy 1.00 — too easy to shape the space).
