"""
Tests for the regression gate.

The gate is the part that blocks a release, so its contract is the exit code:
0 pass, 2 regression, 1 harness error. These run it as a subprocess, the same
way CI does, so the contract is tested rather than the internals.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "eval" / "gate.py"

BASELINE = {
    "overall_wer": 0.180,
    "overall_wer_ci95": [0.120, 0.240],
    "wer_by_slice": {"clean": 0.10, "noisy": 0.25},
}


def run_gate(tmp_path, scores, baseline=BASELINE):
    scores_file = tmp_path / "scores.json"
    scores_file.write_text(json.dumps(scores))
    args = [sys.executable, str(GATE), "--scores", str(scores_file)]

    if baseline is not None:
        baseline_file = tmp_path / "baseline.json"
        baseline_file.write_text(json.dumps(baseline))
        args += ["--baseline", str(baseline_file)]
    else:
        args += ["--baseline", str(tmp_path / "missing.json")]

    return subprocess.run(args, capture_output=True, text=True)


def test_a_wobble_inside_the_noise_band_passes(tmp_path):
    """WER moved up but stayed inside the baseline's confidence interval.

    This is the case that makes a gate usable day to day. A gate that fires on
    every wobble gets muted by the team, and then it protects nothing.
    """
    result = run_gate(tmp_path, {
        "overall_wer": 0.230,
        "wer_by_slice": {"clean": 0.10, "noisy": 0.25},
    })
    assert result.returncode == 0


def test_wer_clearing_the_noise_band_blocks_the_build(tmp_path):
    """0.30 is above the band's top (0.240) plus the 0.02 margin - a real regression."""
    result = run_gate(tmp_path, {
        "overall_wer": 0.300,
        "wer_by_slice": {"clean": 0.10, "noisy": 0.25},
    })
    assert result.returncode == 2
    assert "overall WER" in result.stdout


def test_one_bad_slice_blocks_the_build_even_when_overall_looks_fine(tmp_path):
    """The reason slices are gated separately.

    Overall WER sits inside the noise band, so an average-only gate would pass
    this. Meanwhile noisy audio went from 0.25 to 0.40 - the exact shape of a
    regression that hits one condition first.
    """
    result = run_gate(tmp_path, {
        "overall_wer": 0.200,
        "wer_by_slice": {"clean": 0.10, "noisy": 0.40},
    })
    assert result.returncode == 2
    assert "noisy" in result.stdout


def test_the_hard_ceiling_fails_regardless_of_the_baseline(tmp_path):
    """Guards against a baseline that drifted upward over time.

    If every release is allowed a little more error than the last, the baseline
    itself rots. The ceiling is an absolute floor on quality.
    """
    result = run_gate(tmp_path, {
        "overall_wer": 0.500,
        "wer_by_slice": {"clean": 0.10, "noisy": 0.25},
    }, baseline={"overall_wer": 0.450, "overall_wer_ci95": [0.400, 0.480], "wer_by_slice": {}})
    assert result.returncode == 2
    assert "ceiling" in result.stdout


def test_a_new_hallucination_on_silence_blocks_the_build(tmp_path):
    """The case WER cannot see.

    Overall WER is identical to the baseline - inventing words over silence never
    moves it. Without a separate count this ships unnoticed.
    """
    result = run_gate(tmp_path, {
        "overall_wer": 0.180,
        "wer_by_slice": {"clean": 0.10, "noisy": 0.25},
        "silence": {"n": 1, "hallucinated": ["silence-001"], "emitted_words": 2},
    }, baseline={**BASELINE, "silence": {"n": 1, "hallucinated": [], "emitted_words": 0}})

    assert result.returncode == 2
    assert "silence" in result.stdout


def test_a_missing_baseline_is_a_harness_error_not_a_pass(tmp_path):
    """Exit 1, not 0.

    A missing baseline means the gate cannot judge anything. Reporting that as a
    pass would let a silently broken gate wave every change through.
    """
    result = run_gate(tmp_path, {"overall_wer": 0.180, "wer_by_slice": {}}, baseline=None)
    assert result.returncode == 1
