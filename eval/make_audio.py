"""
Generate real audio for the golden set from macOS text-to-speech (`say`).

This makes the harness self-contained: no dataset download, no API key, just a
reproducible audio -> STT -> WER pipeline you can run anywhere on a Mac. Different
voices stand in for different conditions (a rough proxy for accent/speaker slices).

For a real-world eval you would swap this for LibriSpeech or Common Voice clips -
the golden.json schema (audio path + reference + slice) is the same either way.

Output: 16 kHz mono WAV in data/audio/, the sample rate STT models expect.
"""
import json, subprocess, wave
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


def synth_silence(out_wav: Path, seconds: float = 3.0, rate: int = 16000):
    """Write a clip of digital silence - the adversarial input for STT.

    There is nothing to transcribe, so any word the model returns is invented.
    This is a known failure mode of Whisper-family models, and it is invisible to
    WER: an empty reference has no words to divide by, so the clip needs its own
    check rather than a word error rate.
    """
    with wave.open(str(out_wav), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))


def main():
    golden = json.loads((ROOT / "data/golden.json").read_text())
    for e in golden["entries"]:
        out = ROOT / e["audio"]
        out.parent.mkdir(parents=True, exist_ok=True)
        if not e["reference"].strip():
            synth_silence(out)
            print(f"  {e['id']:12s} [silence] -> {out.relative_to(ROOT)}")
            continue
        voice = VOICE.get(e["slice"], VOICE["default"])
        synth(e["reference"], out, voice)
        print(f"  {e['id']:12s} [{voice}] -> {out.relative_to(ROOT)}")
    print(f"Generated {len(golden['entries'])} clips in {(ROOT / 'audio').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
