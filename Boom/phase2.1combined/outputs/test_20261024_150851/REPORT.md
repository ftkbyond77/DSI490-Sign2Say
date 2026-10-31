# Boom Phase 2.2 — frozen test

Real pipeline, current baseline; no selection on these test results. Grammar held fixed with previously identified reference leakage: exploratory evidence, no production promotion.

 speed     case  clips    WER   S  D  I  n_ref  recall1  recall5  count_error  accepted  correct  inference_s  accepted_precision  corpus_WER  delta_WER
   1.0        A    126 0.4325 152  8 42    450   0.6548   0.8333       0.3968        78       65       1.8805              0.8333      0.4489     0.0007
   1.0 Combined    126 0.4312 153  6 42    450   0.6561   0.8426       0.3810       139      122       1.7822              0.8777      0.4467    -0.0007
   1.0 baseline    126 0.4319 149 14 39    450   0.6488   0.8221       0.4048        78       65       1.9631              0.8333      0.4489     0.0000
   1.5        A    126 0.3743 138 22 12    450   0.6515   0.8280       0.2540        81       74       1.0130              0.9136      0.3822     0.0007
   1.5 Combined    126 0.3724 138 21 12    450   0.6534   0.8280       0.2460       137      129       0.9712              0.9416      0.3800    -0.0013
   1.5 baseline    126 0.3737 128 34 10    450   0.6475   0.8161       0.3175        81       74       0.9704              0.9136      0.3822     0.0000
   2.0        A    126 0.3690 118 47  6    450   0.6429   0.7903       0.4206        78       74       0.8516              0.9487      0.3800    -0.0165
   2.0 Combined    126 0.3710 115 50  6    450   0.6409   0.7844       0.4444       128      123       0.8554              0.9609      0.3800    -0.0146
   2.0 baseline    126 0.3856 108 65  5    450   0.6243   0.7619       0.5556        78       74       0.8819              0.9487      0.3956     0.0000

    case  complete  mean_delta  worst_delta  minimum_precision
       A      True     -0.0051       0.0007             0.8333
Combined      True     -0.0055      -0.0007             0.8777
baseline      True      0.0000       0.0000             0.8333

Latency excludes landmark extraction and model load; no inference memoization. Recall@5 uses the top-1 edit alignment (not presence anywhere). Segment F1 is unavailable in this run.


## Combined versus baseline and A

 speed  control  delta_WER   ci_low  ci_high  improved  unchanged  worsened    conclusion
   1.0 baseline   -0.00066 -0.01389  0.01190         3        120         3 ไม่มีนัยสำคัญ
   1.0        A   -0.00132 -0.01323  0.00926         2        122         2 ไม่มีนัยสำคัญ
   1.5 baseline   -0.00132 -0.00992  0.00728         3        121         2 ไม่มีนัยสำคัญ
   1.5        A   -0.00198 -0.00595  0.00000         1        125         0 ไม่มีนัยสำคัญ
   2.0 baseline   -0.01455 -0.03042  0.00000         9        115         2 ไม่มีนัยสำคัญ
   2.0        A    0.00198 -0.01058  0.01720         3        121         2 ไม่มีนัยสำคัญ

Frozen test comparison; clip-level bootstrap, not signer-clustered. Test previously evaluated and original grammar leakage persists: not an uncontaminated holdout. No retuning or production promotion.
