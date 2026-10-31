# Decoder tuning — real six-video development sweep

These videos were inspected to propose changes. Results are development diagnostics, not an independent held-out evaluation.

WER compares Thai filename sentences against concatenated retrieval output using the same newmm tokenizer. WER can exceed 1.0 when insertion errors are numerous. Timed gloss boundaries are not verified.

All variants share extracted landmarks per clip. Inference latency excludes extraction/model load and is measured once in fixed order; do not treat timing as a rigorous benchmark.

Production config/model/bank hashes were verified unchanged. No configuration is automatically promoted.

## Aggregate results
```
                 mode  clips  macro_WER  micro_WER  delta_macro_WER  improved  unchanged  worsened  substitutions  deletions  insertions  total_predicted_segments  mean_count_error  mean_inference_s  median_inference_s
             baseline      6      1.714      1.655            0.000         0          6         0             27          0          21                        51             3.667             3.037               3.066
          activity060      6      1.672      1.621           -0.042         1          5         0             27          0          20                        50             3.500             3.449               3.661
            min_len16      6      1.714      1.655            0.000         0          6         0             27          0          21                        51             3.667             3.069               3.199
        exact_rescore      6      1.678      1.621           -0.036         3          1         2             26          0          21                        51             3.667             5.952               6.035
        post_frames12      6      1.714      1.655            0.000         0          6         0             27          0          21                        51             3.667             2.970               3.017
        post_psign070      6      1.714      1.655            0.000         0          6         0             27          0          21                        51             3.667             3.252               3.603
           grammar070      6      1.714      1.655            0.000         0          6         0             27          0          21                        51             3.667             3.217               3.470
              stride2      6      1.714      1.655            0.000         0          6         0             27          0          21                        51             3.667             3.822               4.108
combined_without_csls      6      1.636      1.586           -0.078         3          2         1             26          0          20                        50             3.500             5.975               6.472
```

## Per-video results
```
                       video                  mode   WER  substitutions  deletions  insertions  n_pred  inference_s                                                               prediction
      ฉันยังลืมคุณไม่ได้.MOV              baseline 1.333              6          0           2       8        2.878                         ยาย | ไป | คุณ | ชอบ | ญาติ | เป็น | อย่า | เกิด
      ฉันยังลืมคุณไม่ได้.MOV           activity060 1.333              6          0           2       8        2.520                         ยาย | ไป | คุณ | ชอบ | ญาติ | เป็น | อย่า | เกิด
      ฉันยังลืมคุณไม่ได้.MOV             min_len16 1.333              6          0           2       8        2.219                         ยาย | ไป | คุณ | ชอบ | ญาติ | เป็น | อย่า | เกิด
      ฉันยังลืมคุณไม่ได้.MOV         exact_rescore 1.167              5          0           2       8        5.359                         ถูก | ลูก | อย่า | ฉัน | ถูก | ตลาด | อย่า | ได้
      ฉันยังลืมคุณไม่ได้.MOV         post_frames12 1.333              6          0           2       8        2.547                         ยาย | ไป | คุณ | ชอบ | ญาติ | เป็น | อย่า | เกิด
      ฉันยังลืมคุณไม่ได้.MOV         post_psign070 1.333              6          0           2       8        2.598                         ยาย | ไป | คุณ | ชอบ | ญาติ | เป็น | อย่า | เกิด
      ฉันยังลืมคุณไม่ได้.MOV            grammar070 1.333              6          0           2       8        2.562                         ยาย | ไป | คุณ | ชอบ | ญาติ | เป็น | อย่า | เกิด
      ฉันยังลืมคุณไม่ได้.MOV               stride2 1.333              6          0           2       8        2.890                         แฟน | ไป | คุณ | ชอบ | ญาติ | เป็น | อย่า | เกิด
      ฉันยังลืมคุณไม่ได้.MOV combined_without_csls 1.167              5          0           2       8        4.657                         ถูก | ลูก | อย่า | ฉัน | ถูก | ตลาด | อย่า | ได้
            ฉันแอบชอบคุณ.MOV              baseline 2.000              4          0           4       8        2.446                 แฟน | แต่งงาน | เกิด | เป็น | แฟน | แต่งงาน | ข้าว | ถาม
            ฉันแอบชอบคุณ.MOV           activity060 2.000              4          0           4       8        3.521                 แฟน | แต่งงาน | เกิด | เป็น | แฟน | แต่งงาน | ข้าว | ถาม
            ฉันแอบชอบคุณ.MOV             min_len16 2.000              4          0           4       8        2.698                 แฟน | แต่งงาน | เกิด | เป็น | แฟน | แต่งงาน | ข้าว | ถาม
            ฉันแอบชอบคุณ.MOV         exact_rescore 1.750              3          0           4       8        4.647                  แฟน | แต่งงาน | เหงา | คุณ | โสด | แต่งงาน | หมา | เป็น
            ฉันแอบชอบคุณ.MOV         post_frames12 2.000              4          0           4       8        2.224                 แฟน | แต่งงาน | เกิด | เป็น | แฟน | แต่งงาน | ข้าว | ถาม
            ฉันแอบชอบคุณ.MOV         post_psign070 2.000              4          0           4       8        1.972                 แฟน | แต่งงาน | เกิด | เป็น | แฟน | แต่งงาน | ข้าว | ถาม
            ฉันแอบชอบคุณ.MOV            grammar070 2.000              4          0           4       8        1.996                 แฟน | แต่งงาน | เกิด | เป็น | แฟน | แต่งงาน | ข้าว | ถาม
            ฉันแอบชอบคุณ.MOV               stride2 2.000              4          0           4       8        2.294                  แฟน | แต่งงาน | เกิด | เป็น | แฟน | ขนมปัง | ข้าว | ถาม
            ฉันแอบชอบคุณ.MOV combined_without_csls 1.750              3          0           4       8        3.764                  แฟน | แต่งงาน | เหงา | คุณ | โสด | แต่งงาน | หมา | เป็น
    ตอนนี้กำลังเรียนภาษา.MOV              baseline 1.750              2          0           5       9        2.072   เกิด | ตอนนี้ | เรียน | กรุงเทพ | โสด | วันนี้ | เรียน | กรุงเทพ | คุณ
    ตอนนี้กำลังเรียนภาษา.MOV           activity060 1.500              2          0           4       8        1.960             เกิด | ตอนนี้ | เรียน | กรุงเทพ | โสด | วันนี้ | เรียน | คุณ
    ตอนนี้กำลังเรียนภาษา.MOV             min_len16 1.750              2          0           5       9        1.979   เกิด | ตอนนี้ | เรียน | กรุงเทพ | โสด | วันนี้ | เรียน | กรุงเทพ | คุณ
    ตอนนี้กำลังเรียนภาษา.MOV         exact_rescore 2.000              3          0           5       9        3.642 เกิด | วันนี้ | เรียน | กรุงเทพ | อยู่ | ตอนนี้ | เรียน | ภาษามือ | อยู่
    ตอนนี้กำลังเรียนภาษา.MOV         post_frames12 1.750              2          0           5       9        1.922   เกิด | ตอนนี้ | เรียน | กรุงเทพ | โสด | วันนี้ | เรียน | กรุงเทพ | คุณ
    ตอนนี้กำลังเรียนภาษา.MOV         post_psign070 1.750              2          0           5       9        3.641   เกิด | ตอนนี้ | เรียน | กรุงเทพ | โสด | วันนี้ | เรียน | กรุงเทพ | คุณ
    ตอนนี้กำลังเรียนภาษา.MOV            grammar070 1.750              2          0           5       9        4.110   เกิด | ตอนนี้ | เรียน | กรุงเทพ | โสด | วันนี้ | เรียน | กรุงเทพ | คุณ
    ตอนนี้กำลังเรียนภาษา.MOV               stride2 1.750              2          0           5       9        4.238   เกิด | ตอนนี้ | เรียน | กรุงเทพ | คุณ | ตอนนี้ | เรียน | กรุงเทพ | คุณ
    ตอนนี้กำลังเรียนภาษา.MOV combined_without_csls 1.750              3          0           4       8        6.204           ข้าว | วันนี้ | เรียน | กรุงเทพ | อยู่ | วันนี้ | เรียน | อยู่
     มีอะไรให้ฉันช่วยไหม.MOV              baseline 1.500              6          0           3       9        3.838                เกิด | ข้าว | ยาย | ถูก | กิน | เกิด | สิบ | ขนมปัง | ยาย
     มีอะไรให้ฉันช่วยไหม.MOV           activity060 1.500              6          0           3       9        4.958                เกิด | ข้าว | ยาย | ถูก | กิน | เกิด | สิบ | ขนมปัง | ยาย
     มีอะไรให้ฉันช่วยไหม.MOV             min_len16 1.500              6          0           3       9        4.052                เกิด | ข้าว | ยาย | ถูก | กิน | เกิด | สิบ | ขนมปัง | ยาย
     มีอะไรให้ฉันช่วยไหม.MOV         exact_rescore 1.500              6          0           3       9        7.612                 ฉัน | ข้าว | ตลาด | ถูก | ข้าว | เหงา | ตลาด | กอด | ถูก
     มีอะไรให้ฉันช่วยไหม.MOV         post_frames12 1.500              6          0           3       9        3.671                เกิด | ข้าว | ยาย | ถูก | กิน | เกิด | สิบ | ขนมปัง | ยาย
     มีอะไรให้ฉันช่วยไหม.MOV         post_psign070 1.500              6          0           3       9        3.985                เกิด | ข้าว | ยาย | ถูก | กิน | เกิด | สิบ | ขนมปัง | ยาย
     มีอะไรให้ฉันช่วยไหม.MOV            grammar070 1.500              6          0           3       9        3.491                เกิด | ข้าว | ยาย | ถูก | กิน | เกิด | สิบ | ขนมปัง | ยาย
     มีอะไรให้ฉันช่วยไหม.MOV               stride2 1.500              6          0           3       9        5.238                เกิด | ข้าว | ยาย | ถูก | กิน | เกิด | สิบ | ขนมปัง | ยาย
     มีอะไรให้ฉันช่วยไหม.MOV combined_without_csls 1.500              6          0           3       9        7.007                 ฉัน | ข้าว | ตลาด | ถูก | ข้าว | เหงา | ตลาด | กอด | ถูก
เราจะเป็นเพื่อนกันตลอดไป.MOV              baseline 1.200              5          0           1       8        3.733                 ยาย | แฟน | เพื่อน | กิน | แต่งงาน | อะไร | เพื่อน | กิน
เราจะเป็นเพื่อนกันตลอดไป.MOV           activity060 1.200              5          0           1       8        3.801                 ยาย | แฟน | เพื่อน | กิน | แต่งงาน | อะไร | เพื่อน | กิน
เราจะเป็นเพื่อนกันตลอดไป.MOV             min_len16 1.200              5          0           1       8        3.700                 ยาย | แฟน | เพื่อน | กิน | แต่งงาน | อะไร | เพื่อน | กิน
เราจะเป็นเพื่อนกันตลอดไป.MOV         exact_rescore 1.400              5          0           2       8        7.742             น้องสาว | แฟน | เพื่อน | สวัสดี | ตลาด | เป็น | เพื่อน | กิน
เราจะเป็นเพื่อนกันตลอดไป.MOV         post_frames12 1.200              5          0           1       8        3.966                 ยาย | แฟน | เพื่อน | กิน | แต่งงาน | อะไร | เพื่อน | กิน
เราจะเป็นเพื่อนกันตลอดไป.MOV         post_psign070 1.200              5          0           1       8        3.751                 ยาย | แฟน | เพื่อน | กิน | แต่งงาน | อะไร | เพื่อน | กิน
เราจะเป็นเพื่อนกันตลอดไป.MOV            grammar070 1.200              5          0           1       8        3.449                 ยาย | แฟน | เพื่อน | กิน | แต่งงาน | อะไร | เพื่อน | กิน
เราจะเป็นเพื่อนกันตลอดไป.MOV               stride2 1.200              5          0           1       8        4.293                 ยาย | แฟน | เพื่อน | กิน | แต่งงาน | อะไร | เพื่อน | กิน
เราจะเป็นเพื่อนกันตลอดไป.MOV combined_without_csls 1.400              5          0           2       8        7.477          น้องสาว | แฟน | เพื่อน | สวัสดี | วันหยุด | เป็น | เพื่อน | กิน
         เล่นเกมส์กันไหม.MOV              baseline 2.500              4          0           6       9        3.253              ญาติ | เกิด | ถูก | ญาติ | เกิด | ถูก | ญาติ | เกิด | ไม่ดี
         เล่นเกมส์กันไหม.MOV           activity060 2.500              4          0           6       9        3.932              ญาติ | เกิด | ถูก | ญาติ | เกิด | ถูก | ญาติ | เกิด | ไม่ดี
         เล่นเกมส์กันไหม.MOV             min_len16 2.500              4          0           6       9        3.768              ญาติ | เกิด | ถูก | ญาติ | เกิด | ถูก | ญาติ | เกิด | ไม่ดี
         เล่นเกมส์กันไหม.MOV         exact_rescore 2.250              4          0           5       9        6.710              ยาย | เกิด | ถูก | ญาติ | ได้ | ถูก | ญาติ | สบายดี | ไม่ดี
         เล่นเกมส์กันไหม.MOV         post_frames12 2.500              4          0           6       9        3.488              ญาติ | เกิด | ถูก | ญาติ | เกิด | ถูก | ญาติ | เกิด | ไม่ดี
         เล่นเกมส์กันไหม.MOV         post_psign070 2.500              4          0           6       9        3.564              ญาติ | เกิด | ถูก | ญาติ | เกิด | ถูก | ญาติ | เกิด | ไม่ดี
         เล่นเกมส์กันไหม.MOV            grammar070 2.500              4          0           6       9        3.694              ญาติ | เกิด | ถูก | ญาติ | เกิด | ถูก | ญาติ | เกิด | ไม่ดี
         เล่นเกมส์กันไหม.MOV               stride2 2.500              4          0           6       9        3.977              ญาติ | เกิด | ถูก | ญาติ | ข้าว | ถูก | ญาติ | เกิด | ไม่ดี
         เล่นเกมส์กันไหม.MOV combined_without_csls 2.250              4          0           5       9        6.740              ยาย | เกิด | ถูก | ญาติ | ได้ | ถูก | ญาติ | สบายดี | ไม่ดี
```

## Skipped cases
```json
{
  "csls015": "missing hub; do not silently run alpha with no effect"
}
```

## Interpretation limits
One-factor ablations isolate configuration changes, but both pre/post-processing and recalibration assumptions can affect the result. The combined case omits CSLS explicitly. Grammar weights may improve text agreement without validating visual recognition. Candidate changes are not a substitute for human boundary annotation.

## Vocabulary constraint
The conversation vocabulary contains ฉัน, ชอบ, คุณ but not แอบ or แอบชอบ. A closed-set decoder cannot output that missing word. Vocabulary/bank changes require a separate controlled experiment.

## Next decision
Review the winning and losing clips and annotate the four words ฉัน/แอบ/ชอบ/คุณ in time. Confirm configurations on separate development data and evaluate the final fixed configuration on a new held-out set. Do not infer that a lower count alone is better segmentation.