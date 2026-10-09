# ThaiSLM — Latent space v8 (a Thai-only encoder: what it learned, what it no longer inherits)

All numbers: the same held-out queries, the same train-only bank, the same v8 labels, for every encoder
(`artifacts/reports/isolated_v8.json`, `python main.py train-encoder --report …`, `lab.ipynb` §3).
Encoders: **zero-shot** = Uni-Sign (CSL-News → WLASL) without Thai training; **v7** = Uni-Sign fine-tuned on Thai; **v8** = three
small encoders trained from scratch on Thai data only (ensemble).

## 1. The ASL/CSL prior — measured, then removed

v7 reported that 39 % of its errors were "inherited" from the zero-shot model, and asked whether that was a prior bias or just
signs that look alike. A model that never saw ASL/CSL answers it: if the overlap were only look-alike signs, a Thai-only model would
share the same wrong answers just as often.

| | v7 | v8a (one member) | v8c | **v8 ensemble** |
|---|---|---|---|---|
| test errors (same 2,000 held-out queries) | 1,839 | 1,853 | 1,830 | **1,780** |
| errors that are the zero-shot model's same wrong answer | **41 %** | 17 % | 16 % | **20 %** |
| … per query | 0.261 | 0.107 | 0.100 | 0.120 |

The overlap halves, so most of v7's inherited errors **were** the prior. What remains (~1 in 5) is shared because those signs
genuinely look alike in pose space. v8 removes the bias without paying for it in errors.

## 2. Is the space organised by sign, not by signer?

| 5-NN neighbourhood | zero-shot | v7 | **v8** |
|---|---|---|---|
| same concept | 0.278 | 0.354 | **0.392** |
| same signer | 0.306 | 0.254 | 0.258 |
| same dataset | 0.709 | 0.650 | 0.668 |

v8 groups clips by sign more than v7 (+0.04). Signer and dataset are about as strong as in v7: each source still records its own
signing style, which is why one-source words stay hard.

## 3. Signers per word decides recognition — v8 uses depth better

| R@1 by people signing the word in the bank | 1 | 2 | 3–4 | 5–9 | 10+ |
|---|---|---|---|---|---|
| zero-shot | 0.116 | 0.113 | 0.366 | 0.436 | 0.789 |
| v7 | **0.159** | 0.217 | 0.422 | 0.414 | 0.855 |
| **v8** | 0.130 | **0.255** | **0.530** | **0.543** | **0.859** |

v8 is better for every word that at least two people sign (+4 to +13 points), and worse only for one-signer words, where the
pre-training's generic motion knowledge helped v7. That is why the v8 "daily" vocabulary keeps only words with ≥ 2 signers (plus the
curated conversation list).

## 4. Continuous windows — robustness to boundaries

The first v8 encoder was brittle when a word window cut from a real sentence included part of the neighbouring signs (word recall
on the unseen signer's sentence windows fell from 0.56 to 0.48 when the boundaries moved ±25 %; v7 0.52 → 0.52 on the tune
sentences). Training with **co-articulation context** (other signs glued before / after, v8c) and averaging three members fixed it:

| tune sentence windows, conversation vocabulary, R@1 | aligned | boundaries moved ±25 % |
|---|---|---|
| v7 | 0.520 | 0.520 |
| v8a | 0.549 | 0.480 |
| v8c (+ context) | 0.484 | 0.458 |
| **v8 ensemble** | **0.597** | **0.568** |

## 5. Not-a-sign

Null-vs-sign AUC on the TSL51 researcher: zero-shot 0.82, v7 0.70, v8 0.74. As in v7, continuous spotting decides "not a sign"
with the tagger and the transition filter, not with this score.

## 6. What would change the space most

More people per everyday word. The depth curve (§3) is still the strongest effect in the data: a word recorded by 3–4 people is
recognised four times as often as a one-person word. The YouTube harvest of v7 plus the v8 audit gives 525 clean clips of 23 new
people; the next gain is more such people for the 1,366 daily words, not a larger model.
