"""The evaluation videos (cached MediaPipe landmarks + reference words). data_test = the 5 videos of `python main.py test`;
iPhone = the owner's own recordings (extra). Never used for any choice in this folder."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from modules.lexicon import DATA_TEST_REFS, IPHONE_TEST_REFS

DATA_TEST = {k: (ROOT / "cache" / "data_test_mp" / f"{k}.npz", v) for k, v in DATA_TEST_REFS.items()}
IPHONE = {f"{g}_{k}": (ROOT / "cache" / "iphone_test_mp" / f"{g}_{k}.npz", v) for g, d in IPHONE_TEST_REFS.items() for k, v in d.items()}
VIDEOS = DATA_TEST


def team(person):
    """data_test/team-drive/Eval-Test/<person>: cached landmarks + reference words (list built by scripts/modeling_compare.py extract)."""
    import json
    items = json.loads((ROOT / "cache" / "modeling_compare" / "team_items.json").read_text(encoding="utf-8"))
    return {it["id"]: (Path(it["mp"]), it["ref"], Path(it["video"])) for it in items if it["person"] == person}
