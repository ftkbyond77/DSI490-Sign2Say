#!/bin/bash
# วัดค่าจากข้อมูลไทย แล้วสร้างเสียงรอบ 4 (~15 นาที) แล้วเปิดหน้าฟัง
set -e; cd "$(dirname "$0")/emo_ref"
source ../.venv-tts/bin/activate; export HF_HUB_OFFLINE=1
[ -f templates_measured.json ] || python -u measure_templates.py
caffeinate -i python -u round4.py && open out4/index.html
