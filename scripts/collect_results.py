"""Collect artifacts/reports/*.json into markdown tables (printed; pasted/merged into result_reporting/result_v1.md)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "artifacts" / "reports"


def load(name):
    p = REP / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def islr_table(tags):
    rows = ["| model | split | E1 R@1 | E1 R@5 | E1 R@10 | E1 median rank / gallery | E2 R@1 (small) | E2 R@5 (small) |",
            "|---|---|---|---|---|---|---|---|"]
    for t in tags:
        j = load(f"islr_{t}.json")
        if not j:
            continue
        for part in ("val", "test"):
            r = j[part]
            rows.append(f"| {t} | {part} | {f3(r['R@1'])} ±{r['R@1_ci95']:.2f} | {f3(r['R@5'])} ±{r['R@5_ci95']:.2f} | "
                        f"{f3(r['R@10'])} | {r['median_rank']:.0f} / {r['gallery_size']} | {f3(r['E2_R@1_small'])} | "
                        f"{f3(r['E2_R@5_small'])} |")
    return "\n".join(rows)


def main():
    out = open(sys.argv[1], "w", encoding="utf-8") if len(sys.argv) > 1 else sys.stdout
    b = load("baselines.json")
    if b:
        print("## Baselines\n", file=out)
        print("| variant | split | E1 R@1 | E1 R@5 | E1 median rank | E2 R@1 small | E2 R@5 small | NN same signer |", file=out)
        print("|---|---|---|---|---|---|---|---|", file=out)
        for k, r in b.items():
            if not isinstance(r, dict):
                continue
            name, part = k.split("|")
            print(f"| {name} | {part} | {f3(r['E1_R@1'])} | {f3(r['E1_R@5'])} | {r['E1_median_rank']:.0f} | "
                  f"{f3(r['R@1_small'])} | {f3(r['R@5_small'])} | {r['nn_same_signer']:.2f} |", file=out)
    tags = [p.stem[5:] for p in sorted(REP.glob("islr_*.json"))]
    print("\n## ISLR\n", file=out)
    print(islr_table(tags), file=out)
    for p in sorted(REP.glob("evalckpt_*.json")):
        print(f"\n## {p.stem}\n", file=out)
        print("| key | R@1 | R@5 | R@10 | median rank |", file=out)
        print("|---|---|---|---|---|", file=out)
        for k, r in json.loads(p.read_text(encoding="utf-8")).items():
            if "|E1|" in k:
                print(f"| {k} | {f3(r['R@1'])} | {f3(r['R@5'])} | {f3(r['R@10'])} | {r['median_rank']:.0f} |", file=out)
            else:
                print(f"| {k} (E2 small) | {f3(r['R@1_small'])} | {f3(r['R@5_small'])} | {f3(r['R@10_small'])} | |", file=out)


if __name__ == "__main__":
    main()
