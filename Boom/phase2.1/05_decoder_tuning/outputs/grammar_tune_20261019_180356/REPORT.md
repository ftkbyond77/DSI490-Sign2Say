# Grammar grid — TSL51 tune only

         mode  macro_WER   S  D  I  n_ref  collapsed  micro_WER  delta_WER  delta_S  delta_D  eligible_for_review
g0.03_n3_w0.5     0.4022 153 12  8    417          1     0.4149     0.0000      0.0      0.0                False
g0.03_n3_w1.0     0.3996 152 12  8    417          1     0.4125    -0.0026     -1.0      0.0                 True
g0.03_n5_w0.5     0.4102 157 11  8    417          0     0.4221     0.0079      4.0     -1.0                False
g0.03_n5_w1.0     0.4102 157 11  8    417          0     0.4221     0.0079      4.0     -1.0                False
g0.06_n3_w0.5     0.3907 148 12  8    417          1     0.4029    -0.0115     -5.0      0.0                 True
g0.06_n3_w1.0     0.3892 147 12  8    417          1     0.4005    -0.0131     -6.0      0.0                 True
g0.06_n5_w0.5     0.3967 151 11  8    417          0     0.4077    -0.0056     -2.0     -1.0                 True
g0.06_n5_w1.0     0.3951 150 11  8    417          0     0.4053    -0.0071     -3.0     -1.0                 True
g0.10_n3_w0.5     0.3639 135 12  8    417          1     0.3717    -0.0384    -18.0      0.0                 True
g0.10_n3_w1.0     0.3570 132 12  8    417          1     0.3645    -0.0452    -21.0      0.0                 True
g0.10_n5_w0.5     0.3646 136 11  8    417          0     0.3717    -0.0377    -17.0     -1.0                 True
g0.10_n5_w1.0     0.3603 134 11  8    417          0     0.3669    -0.0419    -19.0     -1.0                 True
          off     0.4419 170 11  8    417          0     0.4532     0.0397     17.0     -1.0                False

All baseline replays checked against full pipeline. No production changes. S/D are Levenshtein alignment counts; changes do not necessarily mean changed segmentation. Visual-score loss is a review flag, not proof of incorrect hand motion. Manual segment review pending; test remains unopened. Boom must be reported separately.
