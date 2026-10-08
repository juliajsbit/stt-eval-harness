"""
Put several scored runs side by side, one row per model.

Each run is a directory written by run_eval.py (it holds scores.json). The table
shows overall WER with its noise band, WER on each side of the language switch,
and how many mixed clips came back in one script only.

Usage:
  python eval/compare.py results/codeswitch/small results/codeswitch/large-v3
"""
import argparse, json
from pathlib import Path


def row(run: Path) -> dict:
    s = json.loads((run / "scores.json").read_text())
    return {"run": run.name, "wer": s["overall_wer"], "ci": s["overall_wer_ci95"],
            "by_script": s.get("wer_by_script", {}), "collapsed": len(s.get("collapsed", {})),
            "n": s["n"], "by_slice": s["wer_by_slice"]}


def table(rows: list[dict]) -> str:
    slices = sorted({k for r in rows for k in r["by_slice"]})
    head = ["Run", "WER (95% CI)", "Latin WER", "Cyrillic WER", "Collapsed"] + [f"{s} WER" for s in slices]
    L = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for r in rows:
        bs = r["by_script"]
        cells = [r["run"], f"{r['wer']:.3f} ({r['ci'][0]:.2f}-{r['ci'][1]:.2f})",
                 f"{bs['latin']:.3f}" if "latin" in bs else "-",
                 f"{bs['cyrillic']:.3f}" if "cyrillic" in bs else "-",
                 f"{r['collapsed']}/{r['n']}"]
        cells += [f"{r['by_slice'][s]:.3f}" if s in r["by_slice"] else "-" for s in slices]
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Compare scored runs side by side.")
    ap.add_argument("runs", nargs="+", help="directories written by run_eval.py --out")
    ap.add_argument("--out", help="also write the table to this markdown file")
    args = ap.parse_args()

    md = table([row(Path(r)) for r in args.runs])
    print(md)
    if args.out:
        Path(args.out).write_text(md + "\n")


if __name__ == "__main__":
    main()
