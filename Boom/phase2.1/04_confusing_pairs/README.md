# THSL confusing-pair experiment

Open `confusing_pairs_thsl.ipynb`. Its 19 cells call `pair_analysis.py`, production data manifests, and the existing real handshape tracker.

The five pairs are provisional alignment errors, each observed once, not verified frequent recognition errors. There are 12 unique THSL clips across nine concepts in this selection. Raw 3D hand landmarks are available; local source RGB video is not found at checked manifest paths. Review skeletal evidence now and acquire source videos before detailed visual labeling.

Default `PAIR_ID='pair01'` opens ชอบ–ข้าว. Human review is kept separately in `manual_review.json` and never overwritten on rerun. No fine-tuning is performed automatically. Data counts do not duplicate MediaPipe/RTMW representations of one recording.

Validation: all nine code cells executed using actual project data through the Anaconda display interpreter and project venv worker. Tables and representative skeletal figures are embedded. A Jupyter kernel could not be launched inside the tool sandbox because local socket binding is restricted; cells were executed directly in order instead. User Jupyter reruns remain supported by the notebook.
