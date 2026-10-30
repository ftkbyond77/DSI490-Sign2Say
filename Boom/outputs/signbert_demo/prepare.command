#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"
export MPLCONFIGDIR="$PWD/.matplotlib"
exec /Users/thawaphorn/Documents/Codex/2026-09-17/good-id-word-category-video-url/work/demo-venv/bin/python prepare.py
