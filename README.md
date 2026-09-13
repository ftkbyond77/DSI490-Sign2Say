# SIGN_DATA — Thai Sign Language raw video collection

Raw `.mp4` video for Thai Sign Language, collected with its labels intact.
No pose extraction, no `.npy`, no pre-computed features — the files here are
the videos as published, ready to feed a SignDINO-style self-supervised
pipeline that does its own frame handling.

---

## Current state (2026-09-13)

**5,331 clips · 25h 35m of video · 11.8 GB · zero corrupt, zero missing.**

| | clips | footage |
|---|---|---|
| `word_level/ttrs_dictionary` | 5,043 | 6h 11m — full Thai gloss, definition, POS, category |
| `word_level/youtube_word` | 186 | ~11h — vocabulary and fingerspelling lessons |
| `sentence_level/youtube_sentence` | 57 | ~3h |
| `continuous/youtube_bigsign` + `youtube_continuous` | 45 | ~6h |

**TTRS is complete** — all 5,043 dictionary entries, zero failures.

**What actually happened with YouTube (corrected from the 2026-09-10 note):**
the original mass failure was misdiagnosed as an IP-wide bot block. The real
cause was that **yt-dlp had no JavaScript runtime available** to solve
YouTube's challenge scripts, so many videos were mis-reported as "not
available" or "Please sign in" regardless of IP. That's now fixed:

- Installed **Deno** (`winget install DenoLand.Deno`), yt-dlp's default
  challenge-solver runtime.
- Added `--remote-components ejs:github` to the harvester so yt-dlp is
  permitted to fetch the solver script (a separate opt-in gate).

Verified: every video that previously failed with "not available" now
resolves cleanly.

**A separate, genuine, volume-triggered IP throttle does still exist.** On the
2026-09-13 resume, after ~110 fresh successful downloads in about 35 minutes,
YouTube began answering the real `"Sign in to confirm you're not a bot"`
message. The circuit breaker (tightened this session to key on that exact
phrase, not the broader phrases that also cover legitimate per-video age/
member gates) tripped after 8 consecutive hits and stopped cleanly — no queue
was burned; the ~806 remaining items were left completely untouched, not even
counted as an attempt.

This throttle appears to be about sustained volume in a short window, not a
standing ban: the 2026-09-10 block took about 3 days to clear on its own, and
this session's resume worked cleanly for ~35 minutes before retripping.
**Hammering it with another immediate retry is likely to just retrip
quickly** — a longer gap between attempts, and/or gentler settings, are the
practical way through it:

```bash
python scripts/harvest.py --sources youtube --fetch-only --workers 1 --host-interval 8
```

Consider spreading remaining collection across several short sessions on
different days rather than one long batch. `--yt-cookies-from-browser` would
likely bypass this too, at the account-risk cost already noted below.

---

## Layout

```
raw_data/
  word_level/
    ttrs_dictionary/      one isolated lexical sign per clip, Thai gloss attached
    youtube_word/         vocabulary lessons and fingerspelling series
  sentence_level/
    youtube_sentence/     single utterances, phrases, conversation lessons
  continuous/
    youtube_bigsign/      Thai PBS "Big Sign" — signer fills the whole frame
    youtube_continuous/   interpreted news, stories, songs, vlogs
    parliament/           government / ministry briefings with an interpreter
  _quarantine/            downloaded but failed decode validation — inspect, don't train

metadata/
  metadata_master.csv     every clip, one row each
  metadata_word.csv       the same, split by level
  metadata_sentence.csv
  metadata_continuous.csv
  failures.csv            what did not come down, and why
  sources_manifest.csv    provenance and access status per source

reports/
  collection_report.pdf   the collection report

state/harvest.db          SQLite catalogue — the resume point
logs/harvest.log          full run log
scripts/                  the harvester
```

Beside every video sits a `.json` sidecar with the complete upstream record,
and — for YouTube material — any `.vtt` subtitle tracks the publisher had.
For continuous clips those caption tracks are effectively the sentence-level
annotation.

---

## The metadata CSV

`metadata/metadata_master.csv` is UTF-8 with BOM, so Excel on a Thai Windows
machine opens it without mojibake. `pandas.read_csv(path)` handles it as-is.

Key columns:

| column | meaning |
|---|---|
| `uid` | stable primary key, `ttrs:<id>` or `yt:<id>` |
| `level` | `word` \| `sentence` \| `continuous` |
| `label` | the Thai gloss (TTRS) or the publisher's title (YouTube) |
| `label_type` | `gloss` = curated per-clip gloss; `title` = weak, topic-level |
| `rel_path` | path from the project root to the `.mp4` |
| `sha256` | content hash — dedup and integrity |
| `duration_sec`, `width`, `height`, `fps`, `vcodec` | decode profile from ffprobe |
| `category_th`, `parts_of_speech_th`, `definition_th`, `synonyms_th` | TTRS lexical fields |
| `signer_gender_th`, `signer_creator`, `reference_book_th` | TTRS signer/provenance |
| `channel`, `upload_date`, `subtitle_files` | YouTube fields |
| `source_page` | where the clip came from, for citation |

```python
import pandas as pd
df = pd.read_csv("metadata/metadata_master.csv")
words = df[(df.level == "word") & (df.label_type == "gloss")]   # clean supervision
ssl_pool = df                                                    # everything, for SSL
```

---

## Running it again

The catalogue is idempotent: re-running never re-downloads a clip that is
already on disk and verified.

```bash
python scripts/harvest.py --status                      # where things stand
python scripts/harvest.py                               # discover + fetch everything
python scripts/harvest.py --sources ttrs_dictionary     # one source
python scripts/harvest.py --sources youtube --yt-keys thaipbs_bigsign
python scripts/harvest.py --discover-only               # catalogue, no downloads
python scripts/harvest.py --limit 50                    # cap new downloads per source
python scripts/make_report.py                           # regenerate CSVs + PDF
```

Useful flags: `--workers N` (default 4), `--max-height 720`, `--max-seconds`,
`--host-interval` to slow every request down further.

Ctrl-C is safe. In-flight downloads are discarded, everything already verified
is kept, and the next run picks up from there.

---

## How it stays polite and stable

- **Per-host rate limiting.** One request per configured interval plus jitter,
  shared across all worker threads — raising `--workers` never raises the
  request rate at any single host.
- **Exponential backoff with full jitter** on 408/425/429/5xx/timeouts, over a
  window that doubles per attempt. A `Retry-After` header always wins.
- **Adaptive braking.** A 429 or 503 adds a decaying per-host cooldown, so a
  strained server automatically gets more room.
- **Atomic writes.** Downloads stream to `.part` and are renamed into place only
  after content-length, sha256 and an ffprobe decode check all pass.
- **Content deduplication** by sha256, so the same clip reached through two
  listings is stored once.
- **Quarantine, not deletion,** for files that download but fail validation.

---

## Before you train on this

- **Collection did not clear the licensing.** These clips came from public
  endpoints for research use. They are not public domain. Redistribution, and
  model release trained on them, is a separate conversation with each rights
  holder.
- **th-sl.com (NADT) was deliberately excluded.** The National Association of
  the Deaf in Thailand's database is the richest Thai word-level source, but its
  `robots.txt` carries an express machine-readable reservation against AI
  training (`Content-Signal: ai-train=no`) and disallows this agent by name.
  Requesting a research licence from the association directly is the
  highest-value next step for this project.
- **Titles are weak labels.** YouTube rows name the topic, not each sign. Treat
  them as coarse supervision; the TTRS rows carry true per-clip glosses.
- **Continuous clips may need cropping.** Interpreted broadcasts often put the
  signer in a corner inset. Big Sign material does not — the signer fills the
  frame — which makes it the better continuous subset to start from.
