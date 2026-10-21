# Decoder ablation — development diagnostics

No production configs changed. CSLS is blocked without a verified hub. WER uses filename text tokenization; segment F1 needs timed labels.

```
           video                  mode  WER  substitutions  deletions  insertions  n_pred  inference_s
ฉันแอบชอบคุณ.MOV              baseline 2.00              4          0           4       8        3.221
ฉันแอบชอบคุณ.MOV combined_without_csls 1.75              3          0           4       8        6.741
```
