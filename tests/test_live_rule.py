"""modules/live_rule.py == the browser's handsUp / handsRaised (webapp/web/lib/holistic.ts), frame by frame, on 20 saved live requests.
node evaluates the TypeScript functions themselves (type annotations stripped), so a change on either side breaks this test."""
import gzip
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.live_rule import flags, frames_to_arrays, live_utts  # noqa: E402

REQ = ROOT / "cache" / "live_requests"
TS = ROOT / "webapp" / "web" / "lib" / "holistic.ts"
pytestmark = pytest.mark.skipif(not REQ.exists() or shutil.which("node") is None, reason="no saved requests / node")


def _ts_functions():
    src = TS.read_text(encoding="utf-8")
    out = []
    for name in ("handsUp", "handsRaised"):
        m = re.search(r"export function " + name + r"\(f: LmFrame\): boolean \{.*?\n\}", src, re.S)
        assert m, name
        js = m.group(0).replace("export ", "").replace("(f: LmFrame): boolean", "(f)").replace("(hand: number[][])", "(hand)").replace("(i: number)", "(i)")
        out.append(js)
    raised = re.search(r"export const RAISED_SW = ([0-9.]+);", src).group(1)
    return f"const RAISED_SW = {raised};\n" + "\n".join(out)


def _requests(n=20):
    files = sorted(REQ.glob("*.json.gz"))
    pick = [files[i] for i in np.linspace(0, len(files) - 1, n).astype(int)]
    return [json.load(gzip.open(f, "rt", encoding="utf-8"))["frames"] for f in pick]


def test_flags_match_typescript(tmp_path):
    reqs = _requests()
    (tmp_path / "frames.json").write_text(json.dumps(reqs), encoding="utf-8")
    js = _ts_functions() + """
const fs = require('fs'); const R = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
console.log(JSON.stringify(R.map(fr => fr.map(f => [handsUp(f) ? 1 : 0, handsRaised(f) ? 1 : 0]))));
"""
    (tmp_path / "t.js").write_text(js, encoding="utf-8")
    ts = json.loads(subprocess.run(["node", str(tmp_path / "t.js"), str(tmp_path / "frames.json")], capture_output=True, text=True, check=True).stdout)
    n = agree = 0
    for fr, tsf in zip(reqs, ts):
        up, raised, _, _ = flags(frames_to_arrays(fr))
        tsf = np.asarray(tsf, bool)
        agree += int((up == tsf[:, 0]).sum() + (raised == tsf[:, 1]).sum()); n += 2 * len(fr)
    assert n > 1000 and agree == n, f"{n - agree} of {n} flags differ"


def test_live_rule_defaults_and_options():
    fr = _requests(5)[2]
    raw = frames_to_arrays(fr)
    u0, _ = live_utts(raw)
    u1, _ = live_utts(raw, wrist_grace_ms=800, continue_ms=1200, cap_cut=True)      # runs; options only change cut points
    for u in u0 + u1:
        assert np.all(np.diff(u) == 1)
