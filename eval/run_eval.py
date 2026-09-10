"""
Speech-to-text (STT) evaluation runner.

Scores a set of predictions (id -> hypothesis transcript) against a golden set
(audio + reference transcript + slice tag) and reports:

  - WER (Word Error Rate)      = (S + I + D) / N_reference_words. Lower is better.
                                 S/I/D = substitutions / insertions / deletions.
  - CER (Character Error Rate) = same idea at the character level. Catches small
                                 slips (plurals, endings) that WER rounds over.
  - a runaway / hallucination flag: hypothesis much longer than reference, which
                                 is how STT models "invent" words or loop.
  - a bootstrap 95% confidence interval on the overall WER. This is the number
    that separates a real regression from run-to-run noise: if the new WER sits
    inside the baseline's interval, it is noise, not a regression.

Design notes (why it is built this way):
  * Corpus-level aggregation. Overall WER = total edits / total reference words,
    NOT the mean of per-clip WERs. Averaging per-clip WER over-weights short
    clips and hides real error mass. Same for slices.
  * Predict/score split. Transcription (slow, costs money) is done once by
    transcribe.py and cached as a predictions file. Scoring reads that cache, so
    you can re-score instantly without re-running the model. Reproducible and cheap.
  * Everything is normalized the same way on both sides before scoring, so casing
    and punctuation do not inflate the error rate.
"""
from __future__ import annotations
import argparse, json, random, statistics
from pathlib import Path
import jiwer

ROOT = Path(__file__).resolve().parents[1]

# Text normalization applied identically to reference and hypothesis before
# scoring. Real STT eval always normalizes first (lowercase, drop punctuation).
_WORDS = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def load_json(p):
    with open(p) as f:
        return json.load(f)


def runaway_flag(reference: str, hypothesis: str, ratio: float = 1.5) -> bool:
    """Flag likely runaway / hallucination: hypothesis much longer than reference."""
    r, h = len(reference.split()), len(hypothesis.split())
    return h > max(3, r * ratio)


def per_item_counts(reference: str, hypothesis: str) -> dict:
    """Edit counts for one clip, using jiwer's word-level alignment."""
    if not hypothesis:
        n = len(reference.split())
        return {"S": 0, "I": 0, "D": n, "N": n, "errors": n}
    out = jiwer.process_words(reference, hypothesis,
                              reference_transform=_WORDS, hypothesis_transform=_WORDS)
    N = out.substitutions + out.deletions + out.hits  # reference-word count
    errors = out.substitutions + out.insertions + out.deletions
    return {"S": out.substitutions, "I": out.insertions, "D": out.deletions,
            "N": N, "errors": errors}


def corpus_wer(items: list[dict]) -> float:
    tot_err = sum(i["errors"] for i in items)
    tot_n = sum(i["N"] for i in items)
    return round(tot_err / tot_n, 4) if tot_n else 0.0


def bootstrap_ci(items: list[dict], n_boot: int = 1000, seed: int = 0) -> list[float]:
    """95% confidence interval on corpus WER by resampling clips with replacement.
    Wide interval = few/variable clips; a WER move inside it is noise, not signal."""
    if not items:
        return [0.0, 0.0]
    rng = random.Random(seed)
    boots = []
    k = len(items)
    for _ in range(n_boot):
        sample = [items[rng.randrange(k)] for _ in range(k)]
        boots.append(corpus_wer(sample))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return [round(lo, 4), round(hi, 4)]


def score(golden: dict, preds: dict) -> tuple[list, dict]:
    rows = []
    for e in golden["entries"]:
        ref, hyp = e["reference"], preds.get(e["id"], "")
        c = per_item_counts(ref, hyp)
        wer = round(c["errors"] / c["N"], 4) if c["N"] else 0.0
        cer = jiwer.cer(ref, hyp) if hyp else 1.0
        rows.append({"id": e["id"], "slice": e["slice"], "wer": wer, "cer": round(cer, 4),
                     "runaway": runaway_flag(ref, hyp), **c,
                     "reference": ref, "hypothesis": hyp})

    by_slice = {}
    for r in rows:
        by_slice.setdefault(r["slice"], []).append(r)
    wer_by_slice = {s: corpus_wer(v) for s, v in sorted(by_slice.items())}

    agg = {
        "n": len(rows),
        "overall_wer": corpus_wer(rows),
        "overall_wer_ci95": bootstrap_ci(rows),
        "overall_cer": round(statistics.mean(r["cer"] for r in rows), 4),
        "wer_by_slice": wer_by_slice,
        "errors": {"S": sum(r["S"] for r in rows), "I": sum(r["I"] for r in rows),
                   "D": sum(r["D"] for r in rows), "N": sum(r["N"] for r in rows)},
        "runaways": [r["id"] for r in rows if r["runaway"]],
    }
    return rows, agg


def write_report(rows, agg, out: Path):
    ci = agg["overall_wer_ci95"]
    e = agg["errors"]
    L = ["# STT Evaluation Report", "",
         f"- Clips: **{agg['n']}**",
         f"- Overall **WER {agg['overall_wer']:.3f}** (95% CI {ci[0]:.3f}-{ci[1]:.3f}, lower is better)",
         f"- Overall **CER {agg['overall_cer']:.3f}**",
         f"- Errors: {e['S']} sub, {e['I']} ins, {e['D']} del over {e['N']} reference words",
         f"- Runaway / hallucination flags: **{len(agg['runaways'])}** {agg['runaways'] or ''}",
         "", "## WER by slice", "", "| Slice | WER |", "| --- | --- |"]
    for s, v in agg["wer_by_slice"].items():
        L.append(f"| {s} | {v:.3f} |")
    L += ["", "## Per-clip", "", "| ID | Slice | WER | CER | S/I/D | Runaway |",
          "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        L.append(f"| {r['id']} | {r['slice']} | {r['wer']:.3f} | {r['cer']:.3f} | "
                 f"{r['S']}/{r['I']}/{r['D']} | {'yes' if r['runaway'] else ''} |")
    out.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Score STT predictions against a golden set.")
    ap.add_argument("--golden", default=str(ROOT / "data/golden.json"))
    ap.add_argument("--preds", required=True, help="JSON cache: {id: hypothesis}")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    golden, preds = load_json(args.golden), load_json(args.preds)
    rows, agg = score(golden, preds)

    outdir = Path(args.out); outdir.mkdir(exist_ok=True)
    (outdir / "scores.json").write_text(json.dumps(agg, indent=2) + "\n")
    write_report(rows, agg, outdir / "report.md")

    ci = agg["overall_wer_ci95"]
    print(f"Overall WER {agg['overall_wer']:.3f}  (95% CI {ci[0]:.3f}-{ci[1]:.3f})   CER {agg['overall_cer']:.3f}")
    for s, v in agg["wer_by_slice"].items():
        print(f"  {s:8s} WER {v:.3f}")
    if agg["runaways"]:
        print("  runaway flags:", agg["runaways"])
    print(f"Wrote {outdir/'scores.json'} and {outdir/'report.md'}")


if __name__ == "__main__":
    main()
