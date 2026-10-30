# v7.2 — grammar audit / filters / confusing words

No retraining; production configuration and banks unchanged. Experiments are opt-in. Current local model is v7.2, so supplied v8 results are not treated as reproduced results.

## A. Grammar leakage

{'grammar_n': 62, 'reconstructed_n': 62, 'exact_reconstruction': True, 'recording_provenance': 'Original grammar omits clip IDs; this reconstructs current script/manifest, not a historical execution log', 'test_metadata_only': True, 'source_script': 'scripts/eval.py:cmd_grammar', 'tune_test_clip_id_overlap': 0, 'tune_test_word_sequence_overlap': 7, 'tune_test_role_template_overlap': 9}

```
        split  n_recordings  same_word_sequence  same_role_template  recordings_in_reconstructed_input
sentence_test           126                 126                 126                                126
 tagger_train           126                 126                 126                                126
```
The exact reconstruction includes test reference sentences. This is text-label leakage, not proof of video training leakage. Recording membership is reconstructed from the current script and manifest; original grammar has no clip-ID provenance.
CV holds out entire role-templates and all their tune recordings. Selected CV score is still a model-selection estimate, not an unbiased final score. Test was already reported by the user and must not be described as a never-seen holdout.

## B. Full pipeline comparisons

### boom_1x

```
                  mode  macro_WER  S  D  I  accepted  accepted_correct  count_error
             accept070      1.714 27  0 21         4               NaN        3.500
           filters_all      1.672 28  0 19         4               NaN        3.167
grammar_clean_selected      1.714 27  0 21         3               NaN        3.500
           grammar_off      1.803 27  0 23         3               NaN        3.833
      grammar_original      1.714 27  0 21         3               NaN        3.500
               hand030      1.672 28  0 19         3               NaN        3.167
             p_sign055      1.714 27  0 21         3               NaN        3.500
```
```
                  mode   delta  ci_low  ci_high  improved  unchanged  worsened    conclusion                                    note
             accept070  0.0000  0.0000   0.0000         0          6         0 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
           filters_all -0.0417 -0.1250   0.0000         1          5         0 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
grammar_clean_selected  0.0000  0.0000   0.0000         0          6         0 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
           grammar_off  0.0889 -0.0556   0.2778         1          3         2 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
      grammar_original  0.0000  0.0000   0.0000         0          6         0 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
               hand030 -0.0417 -0.1250   0.0000         1          5         0 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
             p_sign055  0.0000  0.0000   0.0000         0          6         0 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
```
### test_2x

```
                  mode  macro_WER   S  D  I  accepted  accepted_correct  count_error  accepted_precision_ordered
           filters_all      0.372 118 48  6       129               124        0.429                       0.961
grammar_clean_selected      0.385 109 65  5        78                74        0.556                       0.949
           grammar_off      0.401 115 63  5        78                74        0.540                       0.949
      grammar_original      0.386 108 65  5        78                74        0.556                       0.949
```
```
                  mode   delta  ci_low  ci_high  improved  unchanged  worsened    conclusion                                    note
           filters_all -0.0139 -0.0258  -0.0046         6        120         0      WER ลดลง clip bootstrap; not independent signers
grammar_clean_selected -0.0007 -0.0232   0.0205        10        104        12 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
           grammar_off  0.0152 -0.0060   0.0377         8        104        14 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
      grammar_original  0.0000  0.0000   0.0000         0        126         0 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
```
### tune_1x

```
                  mode  macro_WER   S  D  I  accepted  accepted_correct  count_error
             accept070      0.402 153 12  8       119               109        0.159
           filters_all      0.405 158  6 10       119               111        0.127
grammar_clean_selected      0.388 144 12  8        77                71        0.159
           grammar_off      0.442 170 11  8        77                71        0.151
      grammar_original      0.402 153 12  8        77                71        0.159
               hand030      0.405 153 13  8        77                71        0.167
             p_sign055      0.402 158  5 10        77                71        0.119
```
```
                  mode   delta  ci_low  ci_high  improved  unchanged  worsened    conclusion                                    note
             accept070  0.0000  0.0000   0.0000         0        126         0 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
           filters_all  0.0026  0.0000   0.0079         0        125         1 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
grammar_clean_selected -0.0142 -0.0393   0.0115        18         99         9 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
           grammar_off  0.0397  0.0165   0.0632         5        100        21 WER เพิ่มขึ้น clip bootstrap; not independent signers
      grammar_original  0.0000  0.0000   0.0000         0        126         0 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
               hand030  0.0026  0.0000   0.0079         0        125         1 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
             p_sign055  0.0000  0.0000   0.0000         0        126         0 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
```
### tune_2x

```
                  mode  macro_WER   S  D  I  accepted  accepted_correct  count_error
             accept070      0.385 113 49  1       110               109        0.397
           filters_all      0.375 125 33  1       110               109        0.270
grammar_clean_selected      0.376 108 49  1        66                65        0.397
           grammar_off      0.381 112 49  1        66                65        0.397
      grammar_original      0.385 113 49  1        66                65        0.397
               hand030      0.388 112 51  1        66                65        0.413
             p_sign055      0.370 126 30  1        66                65        0.246
```
```
                  mode   delta  ci_low  ci_high  improved  unchanged  worsened    conclusion                                    note
             accept070  0.0000  0.0000   0.0000         0        126         0 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
           filters_all -0.0099 -0.0251   0.0026         4        121         1 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
grammar_clean_selected -0.0090 -0.0286   0.0085         8        114         4 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
           grammar_off -0.0037 -0.0217   0.0148         7        113         6 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
      grammar_original  0.0000  0.0000   0.0000         0        126         0 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
               hand030  0.0026  0.0000   0.0079         0        125         1 ไม่มีนัยสำคัญ clip bootstrap; not independent signers
             p_sign055 -0.0152 -0.0304  -0.0026         5        121         0      WER ลดลง clip bootstrap; not independent signers
```

Timing uses shared real neural caches: not a fresh CPU latency benchmark. Boom WER uses filename text/newmm; TSL51 uses ordered gloss labels. Do not pool them. Hand-filter visibility uses standardised confidence, which can include interpolated points. Accepted correctness is ordered alignment, not human temporal verification.

## C. Confusing words

```
          pair  tune_pipeline_confusions  n_queries  true_mean  wrong_mean  wrong_max_beats_true                                                       cause                                                evidence
     ไป → กลับ                         8          9      0.528       0.685                     8 unresolved: variant/label/similar pose/automatic boundaries pair_cosines + nearest_neighbours + visual_review_queue
เที่ยว → หนึ่ง                         3          3      0.237       0.270                     2 unresolved: variant/label/similar pose/automatic boundaries pair_cosines + nearest_neighbours + visual_review_queue
ภาษามือ → วิ่ง                         6         24      0.287       0.143                     6 unresolved: variant/label/similar pose/automatic boundaries pair_cosines + nearest_neighbours + visual_review_queue
      กิน → ดู                         6         51      0.202       0.168                    12 unresolved: variant/label/similar pose/automatic boundaries pair_cosines + nearest_neighbours + visual_review_queue
  ข้าว → ไม่มี                         3         24      0.218       0.109                     6 unresolved: variant/label/similar pose/automatic boundaries pair_cosines + nearest_neighbours + visual_review_queue
  ยาย → ที่ไหน                         0         18      0.234       0.074                     0 unresolved: variant/label/similar pose/automatic boundaries pair_cosines + nearest_neighbours + visual_review_queue
    ยาย → ญาติ                        14         18      0.234       0.272                    12 unresolved: variant/label/similar pose/automatic boundaries pair_cosines + nearest_neighbours + visual_review_queue
   คุณ → หนึ่ง                         6         24      0.174       0.170                     9 unresolved: variant/label/similar pose/automatic boundaries pair_cosines + nearest_neighbours + visual_review_queue
     คุณ → ป้า                         3         24      0.174       0.136                    13 unresolved: variant/label/similar pose/automatic boundaries pair_cosines + nearest_neighbours + visual_review_queue
```
Query segments use automatic alignment; a cosine mismatch alone cannot distinguish segmentation error, variant, label error and a real visual similarity. Missing aligned query words are not silently counted as failures. Production bank neighbours may include same-source sentence exemplars; use eval bank for primary diagnosis. Review exact video/pose paths in review_priority.csv and review.html. No exemplar removed.