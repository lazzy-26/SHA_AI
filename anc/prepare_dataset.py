from pathlib import Path
import hashlib

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16000

ROOT = Path(__file__).resolve().parents[1]

LIBRISPEECH_SOURCE = (
    ROOT
    / "datasets"
    / "anc"
    / "sources"
    / "librispeech"
)

MUSAN_SOURCE = (
    ROOT
    / "datasets"
    / "anc"
    / "sources"
    / "musan"
)

CLEAN_OUTPUT = (
    ROOT
    / "datasets"
    / "anc"
    / "clean"
)

NOISE_OUTPUT = (
    ROOT
    / "datasets"
    / "anc"
    / "noise"
)


def find_audio_files(folder):
    return sorted(
        path
        for path in folder.rglob("*")
        if path.suffix.lower() in {
            ".wav",
            ".flac",
        }
    )


def load_audio(path):
    audio, sample_rate = sf.read(
        path,
        always_2d=False,
    )

    if audio.ndim == 2:
        audio = np.mean(
            audio,
            axis=1,
        )

    audio = audio.astype(
        np.float32
    )

    if sample_rate != SAMPLE_RATE:
        audio = resample_poly(
            audio,
            SAMPLE_RATE,
            sample_rate,
        ).astype(np.float32)

    peak = np.max(
        np.abs(audio)
    )

    if peak > 1.0:
        audio = audio / peak

    return audio


def make_id(path):
    return hashlib.sha1(
        str(path).encode("utf-8")
    ).hexdigest()[:12]


def convert_files(
    files,
    destination,
    prefix,
):
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    converted = 0

    for index, source in enumerate(
        files,
        start=1,
    ):
        try:
            audio = load_audio(source)

            if len(audio) < SAMPLE_RATE // 2:
                continue

            filename = (
                f"{prefix}_"
                f"{make_id(source)}.wav"
            )

            output = (
                destination
                / filename
            )

            sf.write(
                output,
                audio,
                SAMPLE_RATE,
                subtype="PCM_16",
            )

            converted += 1

            if (
                index % 250 == 0
                or index == len(files)
            ):
                print(
                    f"{prefix}: "
                    f"{index}/{len(files)}"
                )

        except Exception as error:
            print(
                f"Skipped {source}: "
                f"{error}"
            )

    return converted


def main():
    print("=" * 70)
    print("S.H.A ANC DATASET PREPARATION")
    print("=" * 70)

    clean_sources = find_audio_files(
        LIBRISPEECH_SOURCE
    )

    musan_sources = find_audio_files(
        MUSAN_SOURCE
    )

    noise_sources = [
        path
        for path in musan_sources
        if "noise" in {
            part.lower()
            for part in path.parts
        }
    ]

    print(
        f"LibriSpeech files: "
        f"{len(clean_sources)}"
    )

    print(
        f"MUSAN noise files: "
        f"{len(noise_sources)}"
    )

    if not clean_sources:
        raise RuntimeError(
            "No LibriSpeech files found."
        )

    if not noise_sources:
        raise RuntimeError(
            "No MUSAN noise files found."
        )

    clean_count = convert_files(
        clean_sources,
        CLEAN_OUTPUT,
        "clean",
    )

    noise_count = convert_files(
        noise_sources,
        NOISE_OUTPUT,
        "noise",
    )

    print()
    print("=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print(
        f"Clean WAV files: "
        f"{clean_count}"
    )

    print(
        f"Noise WAV files: "
        f"{noise_count}"
    )

    print(
        f"Clean folder: "
        f"{CLEAN_OUTPUT}"
    )

    print(
        f"Noise folder: "
        f"{NOISE_OUTPUT}"
    )


if __name__ == "__main__":
    main()