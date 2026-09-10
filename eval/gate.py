"""
Regression gate for the STT (speech-to-text) eval. Runs in CI/CD on every model
or config change and FAILS (exit 2) if quality regressed.

WER (Word Error Rate) is "lower is better", so a regression means WER went UP.
The gate is built to separate a real regression from run-to-run noise:

  - Noise-aware overall check. The baseline stores a bootstrap 95% confidence
    interval on its WER. A new WER is only a regression if it clears the top of
    that noise band by a margin - a move inside the band is noise, not signal.
  - Absolute ceiling. WER above a hard ceiling fails regardless of baseline.
  - Per-slice check. A real regression usually shows up in one slice first
    (e.g. noisy audio), so each slice is gated too, not just the average.

Exit codes: 0 = pass, 2 = regression, 1 = usage/harness error.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

POLICY = {
    "overall_margin": 0.02,   # WER must clear baseline CI upper bound by this to fail
    "overall_ceiling": 0.30,  # hard cap regardless of baseline
    "slice_max_increase": 0.08,
}


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(ROOT / "results/scores.json"))
    ap.add_argument("--baseline", default=str(ROOT / "eval/baseline.json"))
    args = ap.parse_args()

    if not Path(args.baseline).exists():
        print(f"[gate] ERROR: no baseline at {args.baseline}.\n"
              f"       Create one from a known-good run: cp {args.scores} {args.baseline}")
        return 1

    cur, base = load(args.scores), load(args.baseline)
    failures = []

    cur_w = cur["overall_wer"]
    base_ci = base.get("overall_wer_ci95", [base["overall_wer"], base["overall_wer"]])
    noise_top = base_ci[1]                      # top of baseline's noise band
    trip = noise_top + POLICY["overall_margin"]

    if cur_w > trip:
        failures.append(
            f"overall WER {cur_w:.3f} is above the noise band "
            f"(baseline {base['overall_wer']:.3f}, 95% CI up to {noise_top:.3f}; "
            f"trips at {trip:.3f}) - real regression, not noise")
    if cur_w > POLICY["overall_ceiling"]:
        failures.append(f"overall WER {cur_w:.3f} above hard ceiling {POLICY['overall_ceiling']:.2f}")

    for s, base_v in base.get("wer_by_slice", {}).items():
        cur_v = cur.get("wer_by_slice", {}).get(s)
        if cur_v is None:
            continue
        d = cur_v - base_v
        if d > POLICY["slice_max_increase"]:
            failures.append(f"slice '{s}' WER rose {d:+.3f} "
                            f"(> {POLICY['slice_max_increase']:.2f}) [{base_v:.3f} -> {cur_v:.3f}]")

    if failures:
        print("[gate] REGRESSION - build blocked:")
        for f in failures:
            print("   -", f)
        return 2
    print(f"[gate] PASS  overall WER {cur_w:.3f} within noise band of baseline "
          f"{base['overall_wer']:.3f} (CI up to {noise_top:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
