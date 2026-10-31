# Masked hand pilot (experimental)

This is an opt-in research pilot, not a replacement for GCP/script/job_entry.py.
Serving code, production encoder and banks are untouched. It trains the existing
encoder using SupCon plus an optional hand reconstruction loss. This is **not** a
replication of SignBERT or the production CosFace+SupCon baseline.

## Input contract

Supply a small prepared NPZ cache with `allow_pickle=False` compatible arrays:

- `body`, `left`, `right`, `face_all`: float32 `[N,T,V,3]`, V=9,21,21,18 respectively.
- `valid`: boolean `[N,T]`, false on padding.
- `observed_left`, `observed_right`: boolean `[N,T,21]`. True only for actual detections,
  never inferred from interpolated confidence. Carry these masks through all transforms;
  mark resampled positions conservatively if either contributing endpoint was unobserved.
- `labels`: integer `[N]`; `clip_id`: string `[N]` for original clip identity;
  `role`: string `[N]` with train or val. Never split augmented views of a clip across roles.
- At least two training concepts and two training views per concept; separate validation
  clips whose labels are represented in the training gallery. Validate signer/source/hash
  separation outside this pilot; distinct IDs alone do not establish an independent split.

Normalization must match modules.parts.prepare_parts. Targets are prepared coordinates
before the new mask. A target needs an observed joint AND wrist because hand coordinates
are wrist-relative. Existing missing/low-confidence/padded targets are excluded.
This pilot does not generate this cache or invent detection provenance for old shards.

## Small controlled runs

Run from the repository root using a PyTorch/Transformers environment:

```sh
python experiments/masked_hand_pilot.py --cache /absolute/cache.npz --models models --out /absolute/runs/A --mt5-layers 8 --weight 0 --steps 20
python experiments/masked_hand_pilot.py --cache /absolute/cache.npz --models models --out /absolute/runs/B --mt5-layers 2 --weight 0 --steps 20
python experiments/masked_hand_pilot.py --cache /absolute/cache.npz --models models --out /absolute/runs/C --mt5-layers 8 --weight 0.1 --steps 20
```

Default CPU, seed 0. No cloud launch. Weight 0 skips auxiliary forward. The full small
cache is used per batch; keep it small to avoid memory exhaustion. Twenty steps is a
smoke test, not enough evidence of generalization. Outputs use fresh directories and
experimental weight filenames, deliberately not `models/encoder.pt`.

`validation_r1_before/after` is nearest-training-view accuracy on the supplied validation
clips, not bank production accuracy or end-to-end continuous signing accuracy. Run the
full production baseline and evaluation before accepting/deploying a change. No checkpoint
resume is implemented. The reconstruction head is training-only.

## Verification

```sh
python -m unittest discover -s tests -p test_masked_hand.py -v
```

Tests check missing-data/padding exclusion, wrist provenance, alignment failures, finite
zero loss without targets, non-mutation, and gradient flow. They do not measure recognition.
