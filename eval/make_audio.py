"""
Generate real audio for the golden set from macOS text-to-speech (`say`).

This makes the harness self-contained: no dataset download, no API key, just a
reproducible audio -> STT -> WER pipeline you can run anywhere on a Mac. Different
voices stand in for different conditions (a rough proxy for accent/speaker slices).

For a real-world eval you would swap this for LibriSpeech or Common Voice clips -
the golden.json schema (audio path + reference + slice) is the same either way.

Output: 16 kHz mono WAV in data/audio/, the sample rate STT models expect.
"""
import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# One voice per slice, so slices are also different speakers (proxy for variety).
VOICE = {"clean": "Samantha", "noisy": "Daniel", "accent": "Moira",
         "numbers": "Alex", "default": "Samantha"}


def synth(text: str, out_wav: Path, voice: str):
    aiff = out_wav.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    # convert to 16 kHz mono WAV (standard STT input)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                    str(aiff), str(out_wav)], check=True)
    aiff.unlink(missing_ok=True)


def main():
    golden = json.loads((ROOT / "data/golden.json").read_text())
    audio_dir = ROOT / "data/audio"
    audio_dir.mkdir(exist_ok=True)
    for e in golden["entries"]:
        voice = VOICE.get(e["slice"], VOICE["default"])
        out = ROOT / e["audio"]
        out.parent.mkdir(parents=True, exist_ok=True)
        synth(e["reference"], out, voice)
        print(f"  {e['id']:12s} [{voice}] -> {out.relative_to(ROOT)}")
    print(f"Generated {len(golden['entries'])} clips in {audio_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
