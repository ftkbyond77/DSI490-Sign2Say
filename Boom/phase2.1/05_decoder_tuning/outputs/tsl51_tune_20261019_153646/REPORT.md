# TSL51 tune — baseline vs combined without CSLS

126 real cached pose sequences; evaluation bank; test split not opened. Presence Recall follows pitty. Segment F1 and accepted correctness unavailable without human temporal labels.

```
                 mode  n_clips  macro_WER  micro_WER  recall1  recall5   S  D  I  count_error  mean_inference_s
             baseline      126      0.402      0.415    0.618    0.895 153 12  8        0.159             1.712
combined_without_csls      126      0.385      0.396    0.636    0.874 136 20  9        0.230             2.796
```

```json
{
  "improved": 27,
  "unchanged": 79,
  "worsened": 20,
  "mean_delta": -0.016798941798941796,
  "ci95": [
    -0.05145833333333333,
    0.017331349206349163
  ],
  "caveat": "clip-level bootstrap, not signer-clustered; tune evaluation only"
}
```
