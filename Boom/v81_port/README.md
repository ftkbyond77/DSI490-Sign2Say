# v8.1 selective port (draft; no production cutover)

Start with [REPORT.md](REPORT.md) and [CUTOVER.md](CUTOVER.md).

`replay_v81.py` is derived from draft PR #6 (`9c2a85b`) and adds an explicit `--ported` mode. Its default staged-input ONNX path stays available. Ported mode imports **this worktree** and uses caller-specified read-only model/cache paths; it does not patch either input. Each new output directory is created once and reruns into existing output are refused.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say/Boom/venv-sign2say/bin/python -B \
Boom/v81_replay/replay_v81.py --ported --backend torch --threads 2 \
--models /absolute/path/to/models/Jack-model \
--cache /absolute/path/to/frozen/live_requests \
--output Boom/v81_port/outputs/torch_new_run
```

Replace `torch` with `onnx` and use a fresh output directory to compare. No inference should be restarted merely to view the completed results. Compare saved CSVs with `compare_backends.py --onnx <CSV> --torch <CSV> --out <directory>`.

Raw replay output, model assets, private recordings, runtime packages and logs are excluded from this PR. The report contains the aggregate and per-file decision comparisons. No training, TSL51 test or Eval-Test was run.
