from datasets import load_dataset, Audio
from pathlib import Path
import soundfile as sf
import io

DATASET = "JacobLinCool/VoiceBank-DEMAND-16k"

OUTPUT = Path("datasets/anc/raw")

OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Downloading VoiceBank+DEMAND 16 kHz")
print("=" * 70)

# Load dataset WITHOUT auto-decoding audio
dataset = load_dataset(DATASET)

# CRITICAL FIX: Disable automatic audio decoding for both columns
# This prevents torchcodec from being triggered
dataset = dataset.cast_column("clean", Audio(decode=False))
dataset = dataset.cast_column("noisy", Audio(decode=False))

print()
print(dataset)

for split in dataset.keys():

    print()
    print(f"Processing split: {split}")

    split_dir = OUTPUT / split

    clean_dir = split_dir / "clean"
    noisy_dir = split_dir / "noisy"

    clean_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    noisy_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, sample in enumerate(dataset[split]):

        sample_id = sample["id"]

        clean = sample["clean"]
        noisy = sample["noisy"]

        clean_path = clean_dir / f"{sample_id}.wav"
        noisy_path = noisy_dir / f"{sample_id}.wav"

        # MANUAL DECODING: Read the audio from bytes using soundfile
        # This avoids torchcodec entirely
        clean_audio, clean_sr = sf.read(io.BytesIO(clean["bytes"]))
        noisy_audio, noisy_sr = sf.read(io.BytesIO(noisy["bytes"]))

        sf.write(
            clean_path,
            clean_audio,
            clean_sr,
        )

        sf.write(
            noisy_path,
            noisy_audio,
            noisy_sr,
        )

        if (index + 1) % 100 == 0:
            print(
                f"  {index + 1} files processed..."
            )

    print(
        f"Finished {split}: "
        f"{len(dataset[split])} pairs"
    )

print()
print("=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print()
print(f"Dataset stored in: {OUTPUT}")