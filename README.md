# STT Eval Harness

A small, self-contained evaluation harness for speech-to-text (STT) models:
it measures transcription quality, separates a real regression from run-to-run
noise, and blocks a release when quality drops - the same discipline a test suite
gives normal code, applied to a non-deterministic model.

It runs end to end on a Mac with no dataset download and no API key: it synthesizes
audio from text, transcribes it with an open model, and scores it. Swap in real
audio (LibriSpeech, Common Voice) or the Deepgram API and nothing else changes.

## What it measures

- **WER (Word Error Rate)** = (substitutions + insertions + deletions) / reference
  words. The headline STT accuracy metric. Lower is better.
- **CER (Character Error Rate)** - same idea at the character level, catches small
  slips (endings, plurals) that WER rounds over.
- **WER by slice** - clean / noisy / accent / numbers. Most regressions show up in
  one slice before the average moves, so slices are where you catch things early.
- **Runaway / hallucination flag** - hypothesis much longer than the reference,
  which is how STT models invent words or loop.
- **Bootstrap 95% confidence interval on WER** - the noise band. If a new WER lands
  inside the baseline's interval, it is noise, not a regression.
- **Hallucination on silence** - clips with no speech in them. WER cannot score
  these (an empty reference has no words to divide by), so they are held out of the
  rate and counted separately.

## Quickstart

```bash
python3.12 -m venv venv && ./venv/bin/pip install -r requirements.txt

./venv/bin/python eval/make_audio.py                                  # text -> audio (macOS `say`)
./venv/bin/python eval/transcribe.py --backend faster-whisper --model tiny.en
./venv/bin/python eval/run_eval.py --preds results/preds_faster-whisper.json
./venv/bin/python eval/gate.py                                        # 0 pass, 2 regression

./venv/bin/python -m pytest tests/ -q                                 # test the harness itself
```

Run the exact model a customer would, with the Deepgram API:

```bash
DEEPGRAM_API_KEY=... ./venv/bin/python eval/transcribe.py --backend deepgram --model nova-2
```

## How it is built (and why)

- **Predict/score split.** Transcription is slow and can cost money, so it runs
  once (`transcribe.py`) and caches the hypotheses. Scoring (`run_eval.py`) reads
  the cache, so you can re-score instantly and reproducibly without re-running the
  model.
- **Corpus-level WER**, not the mean of per-clip WERs. Averaging per-clip WER
  over-weights short clips and hides real error mass. Overall WER = total edits /
  total reference words. Same for slices.
- **Noise-aware gate.** The baseline stores a bootstrap confidence interval. The
  gate fails only when the new WER clears the top of that band by a margin, or a
  slice jumps, or WER passes a hard ceiling. That is how it tells a real regression
  from noise instead of firing on every wobble.
- **Normalization** is applied identically to reference and hypothesis before
  scoring, so casing and punctuation do not inflate the error rate.

## Findings from the harness

**1. The `numbers` slice fails on formatting, not hearing.** It scored WER 0.71 while
clean/noisy/accent were near zero. The model transcribed "four one five ..." as
"415-...", and "twelve percent" as "12%". This is the classic STT eval trap: without
**inverse text normalization** (numbers, dates, currency) WER is dominated by
formatting rather than accuracy. The slice made it obvious in one number - an average
alone would have hidden it. Next step is a normalization pass before scoring and a
separate "formatting" metric.

**2. The model invents words over silence, and WER cannot see it.** On three seconds
of digital silence, `tiny.en` returns "You". Scoring the corpus with and without that
hallucination gives the *identical* overall WER of 0.179 - an empty reference has no
words to divide by, so the invented word has nothing to inflate. This is why silence
clips are pulled out of the rate and gated on their own count instead.

That second one is an **open finding**, not a fixed bug: the baseline records the
hallucination as today's reality so the gate stays usable, and the gate blocks any
change that hallucinates on *more* clips. To watch it fire, set
`silence.hallucinated` to `[]` in `eval/baseline.json` and run `eval/gate.py` - it
exits 2 while overall WER never moves.

## Code-switching slice (in progress)

Many people mix languages inside one sentence: "давай сначала сделаем deploy а
потом проверим logs". STT models often handle this badly, and in ways one WER
number hides. A model can translate the Russian half into English or write it in
Latin letters. The overall rate barely moves, because the English half is right.

`data/golden_codeswitch.json` holds 12 such sentences (Russian + English,
English + Russian, Ukrainian + English), recorded in a real voice rather than
synthesized. The harness adds two views for them:

- **WER by script** - errors split between Latin and Cyrillic reference words, so
  you see which side of the switch breaks.
- **Collapse flag** - a mixed clip whose transcript came back in one script only.
  In a test, one translated clip moved overall WER to just 0.019, and the flag
  still caught it.

```bash
./venv/bin/python eval/transcribe.py --backend faster-whisper --model large-v3 --language auto \
    --golden data/golden_codeswitch.json --out results/codeswitch/preds_large-v3.json
./venv/bin/python eval/run_eval.py --golden data/golden_codeswitch.json \
    --preds results/codeswitch/preds_large-v3.json --out results/codeswitch/large-v3
./venv/bin/python eval/compare.py results/codeswitch/*/
```

Recordings and model results are next.

## Applying this to production

The metrics change per surface but the machine is the same:
- STT accuracy: WER by slice (accent, noise, domain, length), with number/date
  normalization.
- Streaming: latency and time-to-first-byte alongside accuracy.
- TTS (text-to-speech): MOS (Mean Opinion Score) or a calibrated judge for naturalness.
- Production: canaries on live traffic for runaways, and every real failure folded
  back into the golden set so the eval gets stronger over time.

## Layout

```
data/golden.json         golden set: audio path + reference transcript + slice
data/golden_codeswitch.json  mixed-language set, real recorded voice
eval/make_audio.py       synthesize audio from references (macOS `say`)
eval/transcribe.py       STT backends (faster-whisper, deepgram, openai) -> predictions cache
eval/run_eval.py         WER / CER / slices / bootstrap CI / runaway / by-script -> scores + report
eval/compare.py          several scored runs side by side
eval/gate.py             noise-aware regression gate (CI/CD), exit 2 on regression
eval/baseline.json       committed known-good scores
tests/                   pytest suite over the scoring math and the gate's exit codes
results/                 scores.json + report.md
```

## Testing the harness

The harness is a measuring instrument, so its own math is tested: that WER is
aggregated corpus-wide rather than averaged per clip, that an empty hypothesis counts
as total failure instead of a pass, that the noise band is deterministic, and that the
gate returns 2 on a real regression, 0 on a wobble inside the band, and 1 - not 0 -
when the baseline is missing. A gate that silently passes when it cannot judge is
worse than no gate.
