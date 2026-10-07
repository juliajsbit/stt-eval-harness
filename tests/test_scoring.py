"""
Tests for the scoring math.

An eval harness is a measuring instrument: if it is wrong, every number it
produces is wrong and nobody can tell. Each test below pins down one decision
that would otherwise silently skew the numbers.
"""
from eval.run_eval import (
    bootstrap_ci,
    corpus_cer,
    corpus_wer,
    per_item_counts,
    runaway_flag,
    score,
)


def test_corpus_wer_is_not_the_mean_of_per_clip_wer():
    """The aggregation choice, pinned down.

    One short clip fully wrong (2 words) and one long clip perfect (18 words).
    Mean of per-clip WER says 0.50 - as if half the corpus were wrong.
    Corpus WER says 0.10, which is the true share of misheard words.
    Averaging per-clip WER over-weights short clips, so it is not used here.
    """
    items = [
        {"errors": 2, "N": 2},
        {"errors": 0, "N": 18},
    ]
    assert corpus_wer(items) == 0.1


def test_cer_is_aggregated_corpus_wide_like_wer():
    """CER gets the same treatment as WER, for the same reason.

    A two-character clip that is fully wrong and a long clip that is perfect:
    the mean of per-clip CER would report 0.50, the corpus rate reports the true
    share of misread characters.
    """
    items = [
        {"char_errors": 2, "char_N": 2},
        {"char_errors": 0, "char_N": 38},
    ]
    assert corpus_cer(items) == 0.05


def test_empty_hypothesis_counts_every_reference_word_as_a_deletion():
    """A backend returning nothing must score as total failure, not as a pass.

    A crashed or silent backend returns an empty string. If that were scored as
    0 errors, a broken integration would look like a perfect run.
    """
    counts = per_item_counts("the shipment arrives on friday", "")
    assert counts["D"] == 5
    assert counts["N"] == 5
    assert counts["errors"] == 5


def test_normalization_ignores_casing_and_punctuation():
    """Formatting is not a hearing error.

    Both sides are normalized identically before scoring, so casing and
    punctuation cannot inflate WER.
    """
    counts = per_item_counts("can you hear me now", "Can you hear me now?")
    assert counts["errors"] == 0


def test_runaway_flag_catches_a_hypothesis_far_longer_than_the_reference():
    """Runaway decoding - the model loops or invents words."""
    reference = "the meeting is at three thirty"
    assert runaway_flag(reference, reference + " " + reference + " " + reference)
    assert not runaway_flag(reference, "the meeting is at three forty")


def test_bootstrap_ci_is_deterministic_and_brackets_the_point_estimate():
    """A noise band that moves between runs cannot gate a release.

    The interval is seeded, so the same scores always produce the same band,
    and the band contains the WER it was computed from.
    """
    items = [{"errors": e, "N": 10} for e in (0, 1, 2, 3, 0, 1, 4, 2)]
    first = bootstrap_ci(items)
    assert first == bootstrap_ci(items)

    low, high = first
    assert low <= corpus_wer(items) <= high


def test_words_invented_over_silence_do_not_move_wer():
    """Why silence needs a check of its own.

    Same corpus scored twice - once with the model silent on the silence clip,
    once with it inventing a word. WER is identical both times, because an empty
    reference contributes no words to divide by. The metric is blind here, so the
    hallucination is reported as a count instead.
    """
    golden = {
        "entries": [
            {"id": "clean-001", "slice": "clean", "reference": "the balance is ready"},
            {"id": "silence-001", "slice": "silence", "reference": ""},
        ]
    }
    quiet = {"clean-001": "the balance is ready", "silence-001": ""}
    hallucinating = {"clean-001": "the balance is ready", "silence-001": "thank you"}

    _, quiet_agg = score(golden, quiet)
    _, loud_agg = score(golden, hallucinating)

    assert quiet_agg["overall_wer"] == loud_agg["overall_wer"]
    assert quiet_agg["silence"]["hallucinated"] == []
    assert loud_agg["silence"]["hallucinated"] == ["silence-001"]
    assert loud_agg["silence"]["emitted_words"] == 2


def test_silence_clips_stay_out_of_the_scored_rows():
    """A clip with no reference must not land in the per-clip WER table."""
    golden = {
        "entries": [
            {"id": "clean-001", "slice": "clean", "reference": "the balance is ready"},
            {"id": "silence-001", "slice": "silence", "reference": ""},
        ]
    }
    rows, agg = score(golden, {"clean-001": "the balance is ready", "silence-001": "you"})

    assert [r["id"] for r in rows] == ["clean-001"]
    assert "silence" not in agg["wer_by_slice"]
    assert agg["silence"]["n"] == 1


def test_a_broken_slice_is_visible_even_when_the_overall_number_looks_fine():
    """Why slices exist: one condition can fail while the average stays calm."""
    golden = {
        "entries": [
            {"id": "clean-001", "slice": "clean", "reference": "the balance is ready"},
            {"id": "clean-002", "slice": "clean", "reference": "the meeting is tomorrow"},
            {"id": "clean-003", "slice": "clean", "reference": "please send the report"},
            {"id": "numbers-001", "slice": "numbers", "reference": "call four one five"},
        ]
    }
    preds = {
        "clean-001": "the balance is ready",
        "clean-002": "the meeting is tomorrow",
        "clean-003": "please send the report",
        "numbers-001": "call 415",
    }

    _, agg = score(golden, preds)

    assert agg["overall_wer"] < 0.25
    assert agg["wer_by_slice"]["clean"] == 0.0
    assert agg["wer_by_slice"]["numbers"] > 0.5
