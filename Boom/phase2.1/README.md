# Phase 2.1 — Continuous TSL improvement experiments

Each folder contains executable notebooks, raw predictions, and a report produced from real evaluation data.

| Folder | Concept | Decision metric |
| --- | --- | --- |
| `01_tagger` | Train fluent-ratio taggers A35, A50, A65 against A25. | TSL51 WER, held-cut WER, fluent WER, segment F1. |
| `02_hybrid_score` | Fuse Top-K retrieval evidence and validated landmark attributes. | AI-only vs hybrid WER/Recall and changed Top-1 words. |
| `03_grammar_context` | Apply sentence context only to uncertain, close-score candidates. | WER and number of unsupported inserted words. |
| `shared` | Locked manifests, evaluation protocol, raw result summaries. | Reproducibility across all experiments. |

Production model files in `models/` are never overwritten by an experiment. A variant is promoted only after its held-out evaluation is recorded here.
