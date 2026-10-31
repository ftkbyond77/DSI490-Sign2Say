# Dictionary preparation v1

## Files
- attribute_schema.json: fixed 0-based indices, Thai names, source fields and matching expressions
- attributes_matrix.npy: uint8 (473,53), **draft evidence matrix**
- attributes_known_mask.npy: uint8 (473,53); 1=evidence available, 0=unknown
- classes.json: exact aligned row order, source pages, split groups and matched evidence
- descriptions.json: physical descriptions (no word name) in the same row order
- text_embeddings.npy: float32 (473,768), actual frozen multilingual BERT raw final-layer CLS
- embedding_metadata.json: revision, pooling, truncation and source hash
- splits.json: fixed seed 42, train_seen / validation_unseen / test_unseen

Schema is project-defined, not a recovered official 53-attribute schema from a paper. Multi-hot presence across the described sequence; not a per-frame label, not left/right-specific. Groups: handshape 0–19, location 20–31, movement 32–43, hand count/relationship/repetition 44–49, coarse palm orientation 50–52. Channel 52 combines up/down as a coarse presence category: expand/version schema if the task requires distinguishing them. Different handshapes/movements can coexist in a multi-stage sign.

Zero with known_mask=0 means UNKNOWN, never confirmed absence. Only explicit hand counts currently provide negative evidence for the opposite count. Regex hits are weak labels requiring human review; they do not validate the source image. This conservatism avoids assuming a gesture is stationary because a book has no arrow.

**Do not feed this draft directly to the original fully observed SemanticEncoder as gold attributes.** Complete/review missing values first, or explicitly run a documented weak-label ablation using zero-filled unknowns and report that approximation. The original model has no unknown mask input. Many dimensions have no positive evidence in this subset; see preparation_report.json.

Rows are dictionary prototypes, including alternate signs and sentence examples, NOT 473 verified independent word classes. variants separated by an em dash and shared slash aliases are grouped for splitting. Other synonyms/inflections require manual vocabulary alignment. For word-level evaluation, merge alternative prototypes under a canonical word ID and aggregate their scores consistently; this code does not silently union incompatible alternate-sign attributes.

Split is a **provisional dictionary split**, independent of video availability: 333 train-seen rows, 47 validation-unseen, 93 test-unseen. Video manifest and vocabulary mapping must be checked before real training. Keep all clips of test-unseen words out of training; validation-unseen only tunes hyperparameters/gamma. Create held-out video subsets of train-seen words for validation/test GZSL seen metrics; this dictionary-only split does not itself provide those videos. No clip3 or production weights were read.

## Regenerate
From parent zsslr directory:
```
../work/demo-venv/bin/python prepare_semantics.py
../work/demo-venv/bin/python embed_text.py
../work/demo-venv/bin/python -m unittest -v test_preparation
```
BERT script expects pinned cached model snapshot documented in embedding_metadata.json. All computation is local; book descriptions are not sent to an API.
