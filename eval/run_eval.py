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
  - a hallucination-on-silence check for clips with an empty reference. WER is
    undefined there (no reference words to divide by), so these clips are held
    out of the rate and counted separately.

Design notes (why it is built this way):
  * Corpus-level aggregation. Overall WER = total edits / total reference words,
    NOT the mean of per-clip WERs. Averaging per-clip WER over-weights short
    clips and hides real error mass. Same for slices, and same for CER at the
    character level.
  * Predict/score split. Transcription (slow, costs money) is done once by
    transcribe.py and cached as a predictions file. Scoring reads that cache, so
    you can re-score instantly without re-running the model. Reproducible and cheap.
  * Everything is normalized the same way on both sides before scoring, so casing
    and punctuation do not inflate the error rate.
  * Code-switching (one speaker mixing languages in one sentence) gets two extra
    views. WER by script splits the errors between Latin and Cyrillic reference
    words, so you can see which side of the switch breaks. The collapse flag
    marks a mixed clip whose transcript came back in one script only - the model
    translated or transliterated half of the speech instead of writing it down.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import jiwer

ROOT = Path(__file__).resolve().parents[1]

# Text normalization applied identically to reference and hypothesis before
# scoring. Real STT eval always normalizes first (lowercase, drop punctuation).
# Russian "ё" is folded into "е": most writers and models drop the dots, and
# that is spelling, not hearing.
_WORDS = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.SubstituteRegexes({"ё": "е"}),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

_CHARS = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.SubstituteRegexes({"ё": "е"}),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars(),
])


def load_json(p):
    with open(p) as f:
        return json.load(f)


def runaway_flag(reference: str, hypothesis: str, ratio: float = 1.5) -> bool:
    """Flag likely runaway / hallucination: hypothesis much longer than reference."""
    r, h = len(reference.split()), len(hypothesis.split())
    return h > max(3, r * ratio)


def per_item_counts(reference: str, hypothesis: str) -> dict:
    """Word- and character-level edit counts for one clip.

    Character counts are returned alongside the word counts so CER can be
    aggregated corpus-wide exactly like WER, instead of as a mean of per-clip
    rates - the same over-weighting problem applies at the character level.
    """
    chars = _char_counts(reference, hypothesis)
    if not hypothesis:
        ref_words = _WORDS(reference)[0]
        n = len(ref_words)
        by_script = {}
        for w in ref_words:
            s = by_script.setdefault(script_of(w), {"errors": 0, "N": 0})
            s["errors"] += 1
            s["N"] += 1
        return {"S": 0, "I": 0, "D": n, "N": n, "errors": n, **chars, "by_script": by_script}
    out = jiwer.process_words(reference, hypothesis,
                              reference_transform=_WORDS, hypothesis_transform=_WORDS)
    N = out.substitutions + out.deletions + out.hits  # reference-word count
    errors = out.substitutions + out.insertions + out.deletions
    return {"S": out.substitutions, "I": out.insertions, "D": out.deletions,
            "N": N, "errors": errors, **chars, "by_script": _script_counts(out)}


def script_of(word: str) -> str:
    """Which writing system a word is in: 'cyrillic', 'latin' or 'other'."""
    if any("Ѐ" <= ch <= "ӿ" for ch in word):
        return "cyrillic"
    if any(ch.isascii() and ch.isalpha() for ch in word):
        return "latin"
    return "other"


def _script_counts(out) -> dict:
    """Attribute each word error to the script of the word it hit.

    Substitutions and deletions belong to the reference word that was missed.
    Insertions have no reference word, so they go to the script of the inserted
    word - a model that adds an English translation shows up on the Latin side.
    """
    ref, hyp = out.references[0], out.hypotheses[0]
    by_script = {}

    def bucket(word):
        return by_script.setdefault(script_of(word), {"errors": 0, "N": 0})

    for chunk in out.alignments[0]:
        if chunk.type == "insert":
            for w in hyp[chunk.hyp_start_idx:chunk.hyp_end_idx]:
                bucket(w)["errors"] += 1
            continue
        for w in ref[chunk.ref_start_idx:chunk.ref_end_idx]:
            b = bucket(w)
            b["N"] += 1
            if chunk.type != "equal":
                b["errors"] += 1
    return by_script


def collapsed_to(reference: str, hypothesis: str) -> str | None:
    """The code-switching failure: mixed speech in, one language out.

    If the reference mixes Latin and Cyrillic words but the hypothesis is in a
    single script, the model translated or transliterated part of what was said.
    Returns the script that survived, or None if nothing collapsed.
    """
    mixed = {"latin", "cyrillic"}
    ref_scripts = {script_of(w) for w in _WORDS(reference)[0]} & mixed
    if ref_scripts != mixed or not hypothesis.strip():
        return None
    hyp_scripts = {script_of(w) for w in _WORDS(hypothesis)[0]} & mixed
    return hyp_scripts.pop() if len(hyp_scripts) == 1 else None


def _char_counts(reference: str, hypothesis: str) -> dict:
    """Character edit counts for one clip, normalized the same way as words."""
    if not hypothesis:
        n = len(_CHARS(reference)[0])
        return {"char_errors": n, "char_N": n}
    out = jiwer.process_characters(reference, hypothesis,
                                   reference_transform=_CHARS, hypothesis_transform=_CHARS)
    return {"char_errors": out.substitutions + out.insertions + out.deletions,
            "char_N": out.substitutions + out.deletions + out.hits}


def corpus_wer(items: list[dict]) -> float:
    tot_err = sum(i["errors"] for i in items)
    tot_n = sum(i["N"] for i in items)
    return round(tot_err / tot_n, 4) if tot_n else 0.0


def wer_by_script(items: list[dict]) -> dict:
    """Corpus WER per script, aggregated the same way as the overall rate."""
    totals = {}
    for i in items:
        for s, c in i.get("by_script", {}).items():
            t = totals.setdefault(s, {"errors": 0, "N": 0})
            t["errors"] += c["errors"]
            t["N"] += c["N"]
    return {s: corpus_wer([t]) for s, t in sorted(totals.items()) if t["N"]}


def corpus_cer(items: list[dict]) -> float:
    tot_err = sum(i["char_errors"] for i in items)
    tot_n = sum(i["char_N"] for i in items)
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


def silence_report(rows: list[dict]) -> dict:
    """Hallucination check for clips that contain no speech.

    WER cannot score these: an empty reference has no words to divide by, so a
    model that invents a sentence over silence still scores WER 0. The check is
    therefore not a rate but a count - any word emitted here is invented.
    """
    return {
        "n": len(rows),
        "hallucinated": [r["id"] for r in rows if r["emitted_words"]],
        "emitted_words": sum(r["emitted_words"] for r in rows),
    }


def score(golden: dict, preds: dict) -> tuple[list, dict]:
    rows, silence_rows = [], []
    for e in golden["entries"]:
        ref, hyp = e["reference"], preds.get(e["id"], "")

        # Silence clips are held out of WER/CER and scored by their own check,
        # so an empty denominator cannot quietly absorb invented words.
        if not ref.strip():
            silence_rows.append({"id": e["id"], "slice": e["slice"], "hypothesis": hyp,
                                 "emitted_words": len(hyp.split())})
            continue

        c = per_item_counts(ref, hyp)
        wer = round(c["errors"] / c["N"], 4) if c["N"] else 0.0
        cer = round(c["char_errors"] / c["char_N"], 4) if c["char_N"] else 0.0
        rows.append({"id": e["id"], "slice": e["slice"], "wer": wer, "cer": cer,
                     "runaway": runaway_flag(ref, hyp),
                     "collapsed_to": collapsed_to(ref, hyp), **c,
                     "reference": ref, "hypothesis": hyp})

    by_slice = {}
    for r in rows:
        by_slice.setdefault(r["slice"], []).append(r)
    wer_by_slice = {s: corpus_wer(v) for s, v in sorted(by_slice.items())}

    agg = {
        "n": len(rows),
        "overall_wer": corpus_wer(rows),
        "overall_wer_ci95": bootstrap_ci(rows),
        "overall_cer": corpus_cer(rows),
        "wer_by_slice": wer_by_slice,
        "errors": {"S": sum(r["S"] for r in rows), "I": sum(r["I"] for r in rows),
                   "D": sum(r["D"] for r in rows), "N": sum(r["N"] for r in rows)},
        "runaways": [r["id"] for r in rows if r["runaway"]],
        "silence": silence_report(silence_rows),
        "wer_by_script": wer_by_script(rows),
        "collapsed": {r["id"]: r["collapsed_to"] for r in rows if r["collapsed_to"]},
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
         f"- Runaway / hallucination flags: **{len(agg['runaways'])}** {agg['runaways'] or ''}"]

    s = agg["silence"]
    if s["n"]:
        L.append(f"- Silence clips: **{s['n']}**, hallucinated on "
                 f"**{len(s['hallucinated'])}** {s['hallucinated'] or ''} "
                 f"({s['emitted_words']} invented words, not counted in WER)")

    L += ["", "## WER by slice", "", "| Slice | WER |", "| --- | --- |"]
    for s, v in agg["wer_by_slice"].items():
        L.append(f"| {s} | {v:.3f} |")

    # Only a corpus with more than one script has a code-switching story to tell.
    if len(agg["wer_by_script"]) > 1:
        L += ["", "## Code-switching", "", "| Script | WER |", "| --- | --- |"]
        for s, v in agg["wer_by_script"].items():
            L.append(f"| {s} | {v:.3f} |")
        L += ["", f"Mixed clips collapsed to one script: **{len(agg['collapsed'])}**"]
        for cid, s in agg["collapsed"].items():
            L.append(f"- {cid}: came back in {s} only")
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

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "scores.json").write_text(json.dumps(agg, indent=2) + "\n")
    write_report(rows, agg, outdir / "report.md")

    ci = agg["overall_wer_ci95"]
    print(f"Overall WER {agg['overall_wer']:.3f}  (95% CI {ci[0]:.3f}-{ci[1]:.3f})   CER {agg['overall_cer']:.3f}")
    for s, v in agg["wer_by_slice"].items():
        print(f"  {s:8s} WER {v:.3f}")
    if agg["runaways"]:
        print("  runaway flags:", agg["runaways"])
    s = agg["silence"]
    if s["hallucinated"]:
        print(f"  hallucinated on silence: {s['hallucinated']} ({s['emitted_words']} invented words)")
    if len(agg["wer_by_script"]) > 1:
        for sc, v in agg["wer_by_script"].items():
            print(f"  {sc:8s} words WER {v:.3f}")
        if agg["collapsed"]:
            print(f"  collapsed to one script: {agg['collapsed']}")
    print(f"Wrote {outdir/'scores.json'} and {outdir/'report.md'}")


if __name__ == "__main__":
    main()
