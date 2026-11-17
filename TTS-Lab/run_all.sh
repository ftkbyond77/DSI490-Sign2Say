#!/bin/bash
# รันทุกอย่างต่อเนื่อง (~35-60 นาทีบน CPU): รอบ 3 ก่อน แล้วค่อย EmoSteer. ต่อจากที่ค้างได้ (ข้ามไฟล์ที่ทำเสร็จ)
set -e; cd "$(dirname "$0")"
source .venv-tts/bin/activate; export HF_HUB_OFFLINE=1
( cd emo_ref && caffeinate -i python -u gen3.py --out out 2>&1 | tee -a gen3.log )
( cd emo_steer && caffeinate -i python -u emo_steer.py --preflight 2>&1 | tee -a steer.log && caffeinate -i python -u emo_steer.py 2>&1 | tee -a steer.log )
open emo_ref/out/play3.html emo_steer/out/index.html
