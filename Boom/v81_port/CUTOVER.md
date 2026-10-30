# v8.1 cutover — prepared only, not executed

The live checkout/containers were not changed. Run these commands yourself only after reviewing the draft PR. Do not copy commands into a shell attached to another checkout. `models/` remains v7.2; Jack-model assets are NOT in this PR. Use an existing absolute model path, mounted read-only. Model paths refer to the HOST; the container always sees `/app/models`.

## Switch the existing live project (manual)

The live checkout currently has staged/unstaged work. Save/commit that work deliberately before switching; do not force checkout, discard, reset or clean it. This command block will stop if checkout fails. The branch must be fetched locally first if on another machine. Existing compose project name is `thaislm`; confirm that matches your deployment.

```bash
cd /Users/thawaphorn/Pee4/Term1/DSI490/DSI490-Sign2Say && \
git checkout boom-v81-port && \
THAISLM_MODELS=/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-jack/models/Jack-model docker compose up -d --build && \
docker compose exec sign-api grep -c handshape /app/modules/pipeline.py
# Expect grep count > 0. This confirms code only, not the selected model.
# Open http://localhost:8080, sign สวัสดี / ขอบคุณ three times each.
# In each translate response verify sentence, segments, face, timings_ms,
# model_version="v8.1", and mirrored_to_right_hand.
# Inspect request cache result.model_version and result.hand_canonicalization.
```

If this experimental branch is already attached to the isolated worktree, `git checkout` in the live checkout will safely refuse. First finish review and release that branch from the isolated worktree (`git -C /Users/thawaphorn/Documents/Codex/boom-v81-port switch --detach`) yourself; this does not touch the running live containers. Do not use `--ignore-other-worktrees`.

Model/config rollback with one command (same port code, v7.2 assets):

```bash
THAISLM_MODELS=./models docker compose up -d --build
```

This reverts model/config selection, **not** the implementation. For exact pre-port code rollback, after safely saving work, run `git checkout boom-live-diag && THAISLM_MODELS=./models docker compose up -d --build`. No automatic production promotion is implied by this PR.

`THAISLM_MODELS` selects the host model directory. Unset it to retain the old default `./models`; the older `THAISLM_MODELS_DIR` variable remains a fallback for compatibility. Shell variables apply to that invocation only: use the same value on later recreate commands (or deliberately set it in your local, uncommitted `.env`).

## Start an isolated deployment instead (manual, recommended first)

From the experimental worktree, with no changes to the existing compose project:

```bash
cd /Users/thawaphorn/Documents/Codex/boom-v81-port && \
THAISLM_MODELS=/Users/thawaphorn/Pee4/Term1/DSI490/DSI490-jack/models/Jack-model \
V81_PORT=18080 docker compose -p thaislm-v81-port -f Boom/v81_port/compose.isolated.yml up -d --build
```

Open http://localhost:18080. This standalone compose file runs forward services only, writes separate `Boom/v81_port/outputs/isolated_requests`, and does not publish port 8080/8081 or use existing containers. It does not enable external LLM calls by default. Check port 18080 availability yourself before running. No Docker commands have been executed by the agent.
