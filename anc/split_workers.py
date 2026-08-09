from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

CLEAN_DIR = ROOT / "datasets" / "anc" / "clean"
NOISE_DIR = ROOT / "datasets" / "anc" / "noise"

OUTPUT_ROOT = ROOT / "datasets" / "anc" / "workers"

NUM_WORKERS = 3


def get_wav_files(folder):
    return sorted(folder.rglob("*.wav"))


def copy_files(files, destination):
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, source in enumerate(files, start=1):
        destination_file = destination / source.name

        shutil.copy2(
            source,
            destination_file,
        )

        if index % 500 == 0 or index == len(files):
            print(
                f"    {index}/{len(files)}"
            )


def main():
    clean_files = get_wav_files(CLEAN_DIR)
    noise_files = get_wav_files(NOISE_DIR)

    print("=" * 70)
    print("S.H.A ANC WORKER DATASET SPLITTER")
    print("=" * 70)

    print(
        f"Clean files : {len(clean_files)}"
    )

    print(
        f"Noise files : {len(noise_files)}"
    )

    if not clean_files:
        raise RuntimeError(
            "No clean WAV files found."
        )

    if not noise_files:
        raise RuntimeError(
            "No noise WAV files found."
        )

    for worker_id in range(1, NUM_WORKERS + 1):

        worker_root = (
            OUTPUT_ROOT
            / f"worker_{worker_id}"
        )

        worker_clean = (
            worker_root
            / "clean"
        )

        worker_noise = (
            worker_root
            / "noise"
        )

        # Every third clean file goes to this worker.
        clean_shard = clean_files[
            worker_id - 1::NUM_WORKERS
        ]

        print()
        print(
            f"Worker {worker_id}"
        )

        print(
            f"  Clean files: {len(clean_shard)}"
        )

        print(
            f"  Copying clean shard..."
        )

        copy_files(
            clean_shard,
            worker_clean,
        )

        print(
            f"  Copying {len(noise_files)} "
            f"noise files..."
        )

        copy_files(
            noise_files,
            worker_noise,
        )

    print()
    print("=" * 70)
    print("WORKER DATASETS READY")
    print("=" * 70)

    for worker_id in range(1, NUM_WORKERS + 1):
        path = (
            OUTPUT_ROOT
            / f"worker_{worker_id}"
        )

        clean_count = len(
            get_wav_files(path / "clean")
        )

        noise_count = len(
            get_wav_files(path / "noise")
        )

        print(
            f"Worker {worker_id}: "
            f"{clean_count} clean | "
            f"{noise_count} noise"
        )


if __name__ == "__main__":
    main()