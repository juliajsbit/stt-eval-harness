"""
Tests for the code-switching views: WER by script and the collapse flag.

Mixed-language speech fails in ways one overall number hides. A model can get
every English word right and drop every Russian one, or write the whole clip in
one language. These tests pin down that the harness sees both.
"""
from eval.run_eval import collapsed_to, per_item_counts, score, script_of


def test_script_of_tells_latin_from_cyrillic():
    assert script_of("deploy") == "latin"
    assert script_of("давай") == "cyrillic"
    assert script_of("відправила") == "cyrillic"
    assert script_of("42") == "other"


def test_errors_land_on_the_side_of_the_switch_that_broke():
    """English words right, Russian word wrong: only the Cyrillic side moves."""
    counts = per_item_counts("давай сделаем deploy", "давай сделали deploy")
    assert counts["by_script"]["cyrillic"] == {"errors": 1, "N": 2}
    assert counts["by_script"]["latin"] == {"errors": 0, "N": 1}


def test_an_inserted_translation_counts_against_its_own_script():
    """An insertion has no reference word, so it goes to the inserted word's script."""
    counts = per_item_counts("купи хлеб", "купи хлеб bread")
    assert counts["by_script"]["latin"]["errors"] == 1
    assert counts["by_script"]["cyrillic"]["errors"] == 0


def test_empty_hypothesis_fails_every_script():
    counts = per_item_counts("давай сделаем deploy", "")
    assert counts["by_script"]["cyrillic"] == {"errors": 2, "N": 2}
    assert counts["by_script"]["latin"] == {"errors": 1, "N": 1}


def test_yo_and_ye_are_the_same_letter_for_scoring():
    """Dropping the dots on "ё" is spelling, not a hearing error."""
    assert per_item_counts("ещё раз", "еще раз")["errors"] == 0


def test_collapse_flag_catches_mixed_speech_written_in_one_script():
    ref = "can you pick up хлеб and молоко"
    assert collapsed_to(ref, "can you pick up bread and milk") == "latin"
    assert collapsed_to(ref, "can you pick up khleb and moloko") == "latin"
    assert collapsed_to(ref, "can you pick up хлеб and молоко") is None


def test_collapse_flag_ignores_single_language_clips_and_empty_output():
    assert collapsed_to("the meeting is tomorrow", "the meeting is tomorrow") is None
    assert collapsed_to("купи хлеб and milk", "") is None


def test_a_translated_clip_is_visible_even_when_overall_wer_looks_moderate():
    """Why the by-script view exists.

    One clip comes back translated into English. Overall WER is moderate because
    the English half is right, but the Cyrillic side is fully wrong and the clip
    is flagged as collapsed.
    """
    golden = {"entries": [
        {"id": "cs-001", "slice": "en-ru", "reference": "can you pick up хлеб and молоко on the way home"},
    ]}
    preds = {"cs-001": "can you pick up bread and milk on the way home"}

    _, agg = score(golden, preds)

    assert agg["overall_wer"] < 0.25
    assert agg["wer_by_script"]["latin"] == 0.0
    assert agg["wer_by_script"]["cyrillic"] == 1.0
    assert agg["collapsed"] == {"cs-001": "latin"}
