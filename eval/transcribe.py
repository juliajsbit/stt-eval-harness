"""
Transcribe the golden set's audio with an STT backend and cache the results as a
predictions file (id -> hypothesis). This is the slow, possibly paid step; keeping
it separate from scoring (run_eval.py) is the predict/score split - transcribe
once, re-score for free.

Backends:
  faster-whisper : local, open, no API key. Good default for a self-contained run.
  deepgram       : Deepgram's API (needs DEEPGRAM_API_KEY). Use this to run the
                   exact model a customer would - "I ran Nova through my harness."
  openai         : OpenAI's transcription API (needs OPENAI_API_KEY).

Language: "en" by default. Pass --language auto for mixed-language speech, so the
model is not forced to hear everything as English. On Deepgram, auto means its
"multi" code-switching mode.

Usage:
  python eval/transcribe.py --backend faster-whisper --model tiny.en
  DEEPGRAM_API_KEY=... python eval/transcribe.py --backend deepgram --model nova-2
  python eval/transcribe.py --backend faster-whisper --model large-v3 --language auto \\
      --golden data/golden_codeswitch.json --out results/codeswitch/preds_large-v3.json
"""
import argparse, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def transcribe_faster_whisper(audio_paths: dict, model: str, language: str) -> dict:
    from faster_whisper import WhisperModel
    m = WhisperModel(model, device="cpu", compute_type="int8")
    lang = None if language == "auto" else language
    preds = {}
    for cid, path in audio_paths.items():
        segments, _ = m.transcribe(str(path), language=lang, beam_size=5)
        preds[cid] = " ".join(s.text.strip() for s in segments).strip()
        print(f"  {cid:12s} -> {preds[cid][:60]}")
    return preds


def transcribe_deepgram(audio_paths: dict, model: str, language: str) -> dict:
    from deepgram import DeepgramClient, PrerecordedOptions, FileSource
    dg = DeepgramClient(os.environ["DEEPGRAM_API_KEY"])
    lang = "multi" if language == "auto" else language
    opts = PrerecordedOptions(model=model, smart_format=True, language=lang)
    preds = {}
    for cid, path in audio_paths.items():
        with open(path, "rb") as f:
            src: FileSource = {"buffer": f.read()}
        resp = dg.listen.rest.v("1").transcribe_file(src, opts)
        preds[cid] = resp.results.channels[0].alternatives[0].transcript.strip()
        print(f"  {cid:12s} -> {preds[cid][:60]}")
    return preds


def transcribe_openai(audio_paths: dict, model: str, language: str) -> dict:
    from openai import OpenAI
    client = OpenAI()
    extra = {} if language == "auto" else {"language": language}
    preds = {}
    for cid, path in audio_paths.items():
        with open(path, "rb") as f:
            resp = client.audio.transcriptions.create(model=model, file=f, **extra)
        preds[cid] = resp.text.strip()
        print(f"  {cid:12s} -> {preds[cid][:60]}")
    return preds


BACKENDS = {"faster-whisper": transcribe_faster_whisper, "deepgram": transcribe_deepgram,
            "openai": transcribe_openai}


def main():
    ap = argparse.ArgumentParser(description="Transcribe golden audio into a predictions cache.")
    ap.add_argument("--backend", choices=BACKENDS, default="faster-whisper")
    ap.add_argument("--model", default="tiny.en",
                    help="e.g. tiny.en / large-v3 (whisper), nova-3 (deepgram), gpt-4o-transcribe (openai)")
    ap.add_argument("--language", default="en", help="language code, or 'auto' for mixed-language speech")
    ap.add_argument("--golden", default=str(ROOT / "data/golden.json"))
    ap.add_argument("--out", default=None, help="predictions JSON (default results/preds_<backend>.json)")
    args = ap.parse_args()

    golden = json.loads(Path(args.golden).read_text())
    audio_paths = {e["id"]: ROOT / e["audio"] for e in golden["entries"]}
    missing = [c for c, p in audio_paths.items() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing audio for {missing}. Run: python eval/make_audio.py")

    print(f"Transcribing {len(audio_paths)} clips with {args.backend} ({args.model}, language={args.language})...")
    preds = BACKENDS[args.backend](audio_paths, args.model, args.language)

    out = Path(args.out) if args.out else ROOT / "results" / f"preds_{args.backend}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preds, indent=2) + "\n")
    print(f"Wrote {out}")
    print(f"Now score it: python eval/run_eval.py --preds {out}")


if __name__ == "__main__":
    main()
