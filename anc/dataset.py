import hashlib
import pickle
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset, DataLoader

from config import (
    AUDIO_CACHE_DIR,
    AUDIO_CACHE_MAX_MEMORY_FILES,
    BATCH_SIZE,
    DATALOADER_WORKERS,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    SEED,
    TRAIN_CLEAN_DIR,
    TRAIN_NOISY_DIR,
    TEST_CLEAN_DIR,
    TEST_NOISY_DIR,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# FILE DISCOVERY
# ============================================================

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".ogg",
}


def find_audio_files(directory):
    """
    Recursively find supported audio files.
    """

    directory = Path(directory)

    if not directory.exists():
        return []

    files = []

    for path in directory.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in AUDIO_EXTENSIONS
        ):
            files.append(path)

    files.sort()

    return files


# ============================================================
# RAW AUDIO LOADING
# ============================================================

def _load_audio_uncached(path):
    """
    Load audio from disk.

    This is the ONLY function that performs
    the actual audio disk read.
    """

    path = Path(path)

    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # MONO
    # --------------------------------------------------------

    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1,
        )

    # --------------------------------------------------------
    # REMOVE INVALID VALUES
    # --------------------------------------------------------

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # --------------------------------------------------------
    # RESAMPLE
    # --------------------------------------------------------

    if sample_rate != SAMPLE_RATE:

        audio = resample_poly(
            audio,
            SAMPLE_RATE,
            sample_rate,
        ).astype(np.float32)

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-8:

        audio = (
            audio / peak
        )

    return audio.astype(
        np.float32
    )


# ============================================================
# AUDIO CACHE
# ============================================================

class AudioCache:
    """
    Two-level audio cache.

    Level 1:
        RAM

    Level 2:
        Disk

    Level 3:
        Original WAV file
    """

    def __init__(
        self,
        max_size=AUDIO_CACHE_MAX_MEMORY_FILES,
        cache_dir=AUDIO_CACHE_DIR,
    ):

        self.cache = {}

        self.max_size = int(
            max_size
        )

        self.cache_dir = Path(
            cache_dir
        )

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.hits = 0
        self.misses = 0

    # --------------------------------------------------------
    # CACHE FILE NAME
    # --------------------------------------------------------

    def _cache_filename(self, path):

        path = Path(
            path
        ).resolve()

        try:

            stat = path.stat()

            signature = (
                f"{path}|"
                f"{stat.st_mtime_ns}|"
                f"{stat.st_size}"
            )

        except OSError:

            signature = str(path)

        digest = hashlib.md5(
            signature.encode(
                "utf-8"
            )
        ).hexdigest()

        return (
            self.cache_dir
            / f"{digest}.pkl"
        )

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    def get(self, path):

        path = str(
            Path(path).resolve()
        )

        # ----------------------------------------------------
        # RAM
        # ----------------------------------------------------

        if path in self.cache:

            self.hits += 1

            return self.cache[path]

        # ----------------------------------------------------
        # DISK
        # ----------------------------------------------------

        cache_file = (
            self._cache_filename(path)
        )

        if cache_file.exists():

            try:

                with cache_file.open(
                    "rb"
                ) as file:

                    audio = pickle.load(
                        file
                    )

                audio = np.asarray(
                    audio,
                    dtype=np.float32,
                )

                if (
                    len(self.cache)
                    < self.max_size
                ):

                    self.cache[
                        path
                    ] = audio

                self.hits += 1

                return audio

            except (
                OSError,
                EOFError,
                pickle.PickleError,
                ValueError,
            ):

                try:

                    cache_file.unlink()

                except OSError:

                    pass

        # ----------------------------------------------------
        # ORIGINAL FILE
        # ----------------------------------------------------

        audio = _load_audio_uncached(
            path
        )

        # ----------------------------------------------------
        # SAVE DISK CACHE
        # ----------------------------------------------------

        try:

            with cache_file.open(
                "wb"
            ) as file:

                pickle.dump(
                    audio,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        except OSError:

            pass

        # ----------------------------------------------------
        # SAVE RAM CACHE
        # ----------------------------------------------------

        if (
            len(self.cache)
            < self.max_size
        ):

            self.cache[
                path
            ] = audio

        self.misses += 1

        return audio

    # --------------------------------------------------------
    # STATS
    # --------------------------------------------------------

    def stats(self):

        total = (
            self.hits
            + self.misses
        )

        if total == 0:

            hit_rate = 0.0

        else:

            hit_rate = (
                self.hits
                / total
            )

        return (
            f"Cache hit rate: "
            f"{hit_rate:.1%} "
            f"({self.hits}/{total})"
        )


# Global cache
_audio_cache = AudioCache()


def load_audio(path):

    return _audio_cache.get(
        path
    )


# ============================================================
# FIXED-LENGTH SEGMENT
# ============================================================

def fixed_segment(audio):
    """
    Return exactly SEGMENT_SAMPLES samples.

    For prepared datasets we use a deterministic
    center crop/padding rather than generating
    a new random example every access.
    """

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    length = len(audio)

    # Exact
    if length == SEGMENT_SAMPLES:

        return audio.copy()

    # Longer
    if length > SEGMENT_SAMPLES:

        start = (
            length
            - SEGMENT_SAMPLES
        ) // 2

        return audio[
            start:
            start + SEGMENT_SAMPLES
        ].copy()

    # Shorter
    result = np.zeros(
        SEGMENT_SAMPLES,
        dtype=np.float32,
    )

    result[:length] = audio

    return result


# ============================================================
# PREPARED PAIRED DATASET
# ============================================================

class SHAPairedAudioDataset(Dataset):
    """
    Dataset using already-prepared pairs:

        clean/
        noisy/

    No MUSAN mixing happens during training.

    This significantly reduces CPU work.
    """

    def __init__(
        self,
        clean_files,
        noisy_files,
    ):

        self.clean_files = list(
            clean_files
        )

        self.noisy_files = list(
            noisy_files
        )

        if not self.clean_files:

            raise RuntimeError(
                "No clean audio files found."
            )

        if not self.noisy_files:

            raise RuntimeError(
                "No noisy audio files found."
            )

        if (
            len(self.clean_files)
            != len(self.noisy_files)
        ):

            raise RuntimeError(
                "Clean/noisy dataset size mismatch:\n"
                f"Clean: {len(self.clean_files)}\n"
                f"Noisy: {len(self.noisy_files)}"
            )

    def __len__(self):

        return len(
            self.clean_files
        )

    def __getitem__(
        self,
        index,
    ):

        clean_path = (
            self.clean_files[index]
        )

        noisy_path = (
            self.noisy_files[index]
        )

        clean_audio = load_audio(
            clean_path
        )

        noisy_audio = load_audio(
            noisy_path
        )

        clean = fixed_segment(
            clean_audio
        )

        noisy = fixed_segment(
            noisy_audio
        )

        noisy_tensor = (
            torch.from_numpy(
                noisy
            )
            .float()
            .unsqueeze(0)
        )

        clean_tensor = (
            torch.from_numpy(
                clean
            )
            .float()
            .unsqueeze(0)
        )

        return (
            noisy_tensor,
            clean_tensor,
        )


# ============================================================
# MATCH CLEAN/NOISY FILES
# ============================================================

def build_pairs(
    clean_files,
    noisy_files,
):
    """
    Match clean and noisy files.

    First attempts exact filename matching.

    If names don't match, falls back to
    sorted positional matching.
    """

    clean_files = list(
        clean_files
    )

    noisy_files = list(
        noisy_files
    )

    if not clean_files:
        return []

    if not noisy_files:
        return []

    noisy_by_name = {
        path.name: path
        for path in noisy_files
    }

    pairs = []

    # --------------------------------------------------------
    # Exact filename matching
    # --------------------------------------------------------

    for clean in clean_files:

        noisy = noisy_by_name.get(
            clean.name
        )

        if noisy is not None:

            pairs.append(
                (
                    clean,
                    noisy,
                )
            )

    # --------------------------------------------------------
    # If filenames don't match, use sorted order
    # --------------------------------------------------------

    if len(pairs) != len(
        clean_files
    ):

        clean_sorted = sorted(
            clean_files
        )

        noisy_sorted = sorted(
            noisy_files
        )

        count = min(
            len(clean_sorted),
            len(noisy_sorted),
        )

        pairs = list(
            zip(
                clean_sorted[:count],
                noisy_sorted[:count],
            )
        )

    return pairs


# ============================================================
# LOCAL LOADERS
# ============================================================

def get_local_loaders(
    participant_id="LOCAL",
    batch_size=BATCH_SIZE,
    max_samples=None,
):
    """
    Prepare the local participant dataset.

    Training:
        datasets/anc/raw/train/clean
        datasets/anc/raw/train/noisy

    Validation:
        datasets/anc/raw/test/clean
        datasets/anc/raw/test/noisy

    MUSAN is NOT required here.
    """

    print()
    print("=" * 70)
    print(
        f"[{participant_id}] PREPARING LOCAL DATASET"
    )
    print("=" * 70)

    print(
        f"[{participant_id}] "
        f"Train clean: {TRAIN_CLEAN_DIR}"
    )

    print(
        f"[{participant_id}] "
        f"Train noisy: {TRAIN_NOISY_DIR}"
    )

    print(
        f"[{participant_id}] "
        f"Test clean: {TEST_CLEAN_DIR}"
    )

    print(
        f"[{participant_id}] "
        f"Test noisy: {TEST_NOISY_DIR}"
    )

    # ========================================================
    # CHECK DIRECTORIES
    # ========================================================

    required_dirs = [
        TRAIN_CLEAN_DIR,
        TRAIN_NOISY_DIR,
        TEST_CLEAN_DIR,
        TEST_NOISY_DIR,
    ]

    for directory in required_dirs:

        if not directory.exists():

            raise FileNotFoundError(
                f"[{participant_id}] "
                f"Required dataset directory "
                f"does not exist:\n"
                f"{directory}"
            )

    # ========================================================
    # FIND TRAINING FILES
    # ========================================================

    train_clean_files = (
        find_audio_files(
            TRAIN_CLEAN_DIR
        )
    )

    train_noisy_files = (
        find_audio_files(
            TRAIN_NOISY_DIR
        )
    )

    # ========================================================
    # FIND TEST FILES
    # ========================================================

    test_clean_files = (
        find_audio_files(
            TEST_CLEAN_DIR
        )
    )

    test_noisy_files = (
        find_audio_files(
            TEST_NOISY_DIR
        )
    )

    print(
        f"[{participant_id}] "
        f"Train clean files: "
        f"{len(train_clean_files)}"
    )

    print(
        f"[{participant_id}] "
        f"Train noisy files: "
        f"{len(train_noisy_files)}"
    )

    print(
        f"[{participant_id}] "
        f"Test clean files: "
        f"{len(test_clean_files)}"
    )

    print(
        f"[{participant_id}] "
        f"Test noisy files: "
        f"{len(test_noisy_files)}"
    )

    # ========================================================
    # LIMIT TRAINING DATA
    # ========================================================

    if max_samples is not None:

        max_samples = int(
            max_samples
        )

        if max_samples < 1:

            raise ValueError(
                "max_samples must be at least 1."
            )

        if (
            max_samples
            < len(train_clean_files)
        ):

            rng = np.random.default_rng(
                SEED
            )

            indices = rng.choice(
                len(train_clean_files),
                size=max_samples,
                replace=False,
            )

            indices = sorted(
                indices.tolist()
            )

            train_clean_files = [
                train_clean_files[i]
                for i in indices
            ]

            # We need the corresponding noisy
            # files. Build the pairs before limiting
            # when possible.
            train_pairs = build_pairs(
                find_audio_files(
                    TRAIN_CLEAN_DIR
                ),
                find_audio_files(
                    TRAIN_NOISY_DIR
                ),
            )

            train_pairs = [
                train_pairs[i]
                for i in indices
                if i < len(train_pairs)
            ]

        else:

            train_pairs = build_pairs(
                train_clean_files,
                train_noisy_files,
            )

    else:

        train_pairs = build_pairs(
            train_clean_files,
            train_noisy_files,
        )

    # ========================================================
    # TEST PAIRS
    # ========================================================

    test_pairs = build_pairs(
        test_clean_files,
        test_noisy_files,
    )

    # ========================================================
    # CHECK PAIRS
    # ========================================================

    if not train_pairs:

        raise RuntimeError(
            f"[{participant_id}] "
            "No training clean/noisy pairs found."
        )

    if not test_pairs:

        raise RuntimeError(
            f"[{participant_id}] "
            "No validation clean/noisy pairs found."
        )

    # ========================================================
    # UNPACK
    # ========================================================

    train_clean = [
        pair[0]
        for pair in train_pairs
    ]

    train_noisy = [
        pair[1]
        for pair in train_pairs
    ]

    test_clean = [
        pair[0]
        for pair in test_pairs
    ]

    test_noisy = [
        pair[1]
        for pair in test_pairs
    ]

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = (
        SHAPairedAudioDataset(
            clean_files=train_clean,
            noisy_files=train_noisy,
        )
    )

    validation_dataset = (
        SHAPairedAudioDataset(
            clean_files=test_clean,
            noisy_files=test_noisy,
        )
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=DATALOADER_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(
            DATALOADER_WORKERS > 0
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(
            DATALOADER_WORKERS > 0
        ),
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        f"[{participant_id}] "
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"[{participant_id}] "
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    print(
        f"[{participant_id}] "
        f"Training batches: "
        f"{len(train_loader)}"
    )

    print(
        f"[{participant_id}] "
        f"Validation batches: "
        f"{len(validation_loader)}"
    )

    print(
        f"[{participant_id}] "
        f"DataLoader workers: "
        f"{DATALOADER_WORKERS}"
    )

    print(
        f"[{participant_id}] "
        f"Batch size: "
        f"{batch_size}"
    )

    print(
        f"[{participant_id}] "
        f"{_audio_cache.stats()}"
    )

    print()
    print(
        f"[{participant_id}] "
        "Prepared paired dataset READY."
    )
    print("=" * 70)

    return (
        train_loader,
        validation_loader,
    )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def get_worker_loader(
    worker_id,
    num_workers=2,
    batch_size=BATCH_SIZE,
):
    """
    Compatibility helper for older code.
    """

    train_loader, _ = (
        get_local_loaders(
            participant_id=(
                f"WORKER {worker_id}"
            ),
            batch_size=batch_size,
            max_samples=None,
        )
    )

    return train_loader