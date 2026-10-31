# Boom Phase 2.2 — tune only

Real pipeline, current baseline; no test/Eval-Test selection. Grammar held fixed with previously identified reference leakage: exploratory evidence, no production promotion.

 speed     case  clips    WER   S  D  I  n_ref  recall1  recall5  count_error  accepted  correct  inference_s  accepted_precision  corpus_WER  delta_WER
   1.0        A    126 0.4022 158  5 10    417   0.6183   0.8872       0.1190        77       71       2.1688              0.9221      0.4149     0.0000
   1.0 Combined    126 0.4003 158  5  9    417   0.6183   0.8892       0.1111       118      110       2.2373              0.9322      0.4125    -0.0020
   1.0 baseline    126 0.4022 153 12  8    417   0.6143   0.8772       0.1587        77       71       2.3998              0.9221      0.4149     0.0000
   1.5        A    126 0.3692 137 19  1    417   0.6328   0.8507       0.1587        73       72       1.7009              0.9863      0.3765    -0.0040
   1.5 Combined    126 0.3692 138 18  1    417   0.6328   0.8533       0.1508       116      115       1.6262              0.9914      0.3765    -0.0040
   1.5 baseline    126 0.3731 121 37  1    417   0.6288   0.8352       0.3016        73       72       1.6475              0.9863      0.3813     0.0000
   2.0        A    126 0.3698 126 30  1    417   0.6321   0.8238       0.2460        66       65       2.0372              0.9848      0.3765    -0.0152
   2.0 Combined    126 0.3672 125 30  1    417   0.6348   0.8238       0.2460       109      108       1.9792              0.9908      0.3741    -0.0179
   2.0 baseline    126 0.3851 113 49  1    417   0.6169   0.7892       0.3968        66       65       2.2118              0.9848      0.3909     0.0000

    case  complete  mean_delta  worst_delta  minimum_precision  pass_gate
       A      True     -0.0064        0.000             0.9221       True
Combined      True     -0.0079       -0.002             0.9322       True
baseline      True      0.0000        0.000             0.9221      False

Latency excludes landmark extraction and model load; no inference memoization. Recall@5 uses the top-1 edit alignment (not presence anywhere). Segment F1 is unavailable in this run.


## Combined versus baseline and A

 speed  control  delta_WER   ci_low  ci_high  improved  unchanged  worsened      conclusion
   1.0 baseline   -0.00198 -0.00595  0.00000         1        125         0   ไม่มีนัยสำคัญ
   1.0        A   -0.00198 -0.00595  0.00000         1        125         0   ไม่มีนัยสำคัญ
   1.5 baseline   -0.00397 -0.00992  0.00000         2        124         0   ไม่มีนัยสำคัญ
   1.5        A    0.00000  0.00000  0.00000         0        126         0   ไม่มีนัยสำคัญ
   2.0 baseline   -0.01786 -0.03373 -0.00529         6        120         0 WER ลดลงบน tune
   2.0        A   -0.00265 -0.00794  0.00000         1        125         0   ไม่มีนัยสำคัญ

Exploratory post-selection tune confidence intervals; clip-level, not signer-clustered. No test or Eval-Test inference. Production not promoted.
