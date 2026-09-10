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

## Quickstart

```bash
python3.12 -m venv venv && ./venv/bin/pip install -r requirements.txt

./venv/bin/python eval/make_audio.py                                  # text -> audio (macOS `say`)
./venv/bin/python eval/transcribe.py --backend faster-whisper --model tiny.en
./venv/bin/python eval/run_eval.py --preds results/preds_faster-whisper.json
./venv/bin/python eval/gate.py                                        # 0 pass, 2 regression
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

## A real finding from the harness

On the generated set, the `numbers` slice scored WER 0.71 while clean/noisy/accent
were near zero. The model transcribed "four one five ..." as "415-...", and "twelve
percent" as "12%". The errors are formatting, not hearing. This is the classic STT
eval trap: without **inverse text normalization** (numbers, dates, currency) WER is
dominated by formatting, not accuracy. The slice made it obvious in one number - an
average alone would have hidden it. Next step is a normalization pass before scoring
and a separate "formatting" metric.

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
eval/make_audio.py       synthesize audio from references (macOS `say`)
eval/transcribe.py       STT backends (faster-whisper, deepgram) -> predictions cache
eval/run_eval.py         WER / CER / slices / bootstrap CI / runaway -> scores + report
eval/gate.py             noise-aware regression gate (CI/CD), exit 2 on regression
eval/baseline.json       committed known-good scores
results/                 scores.json + report.md
```
