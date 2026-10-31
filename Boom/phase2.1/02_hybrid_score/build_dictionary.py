"""Build normalized attributes from reviewed book rows for the Hybrid experiment."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from modules.symbolic import build_dictionary

out = Path(__file__).with_name("dictionary_attributes.json")
concepts = json.loads((ROOT / "models/concepts.json").read_text(encoding="utf-8"))
rows = ROOT / "Boom/book-catalog/reviewed_entries.json"
dictionary = build_dictionary(rows, concepts)
out.write_text(json.dumps(dictionary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{out}: {len(dictionary)} concepts, {sum(map(len, dictionary.values()))} variants")
