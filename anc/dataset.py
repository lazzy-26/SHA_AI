import hashlib
import pickle
import random
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
    DATASET_MULTIPLIER,
    DATALOADER_WORKERS,
    LIBRISPEECH_DIR,
    MUSAN_DIR,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    SEED,
    SNR_MAX_DB,
    SNR_MIN_DB,
    VALIDATION_RATIO,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(path)

    files.sort()

    return files


# ============================================================
# RAW AUDIO LOADING
# ============================================================

def _load_audio_uncached(path):
    """
    Load one audio file from disk.

    This is the ONLY function that performs the actual disk read.
    The cache calls this function when necessary.
    """

    path = Path(path)

    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )

    audio = np.asarray(audio, dtype=np.float32)

    # Convert stereo/multichannel to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Remove NaN / infinity
    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Resample
    if sample_rate != SAMPLE_RATE:
        audio = resample_poly(
            audio,
            SAMPLE_RATE,
            sample_rate,
        ).astype(np.float32)

    # Peak normalization
    peak = np.max(np.abs(audio))

    if peak > 1e-8:
        audio = audio / peak

    return audio.astype(np.float32)


# ============================================================
# AUDIO CACHE
# ============================================================

class AudioCache:
    """
    Two-level audio cache.

    Level 1:
        In-memory dictionary.

    Level 2:
        Pickled audio files on disk.

    IMPORTANT:
        This class NEVER calls load_audio().
        It calls _load_audio_uncached() to avoid recursion.
    """

    def __init__(
        self,
        max_size=AUDIO_CACHE_MAX_MEMORY_FILES,
        cache_dir=AUDIO_CACHE_DIR,
    ):
        self.cache = {}

        self.max_size = int(max_size)

        self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.hits = 0
        self.misses = 0

    def _cache_filename(self, path):
        """
        Generate a stable cache filename.

        File path + modification time + size are included so
        changing the source audio invalidates the old cache.
        """

        path = Path(path).resolve()

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
            signature.encode("utf-8")
        ).hexdigest()

        return self.cache_dir / f"{digest}.pkl"

    def get(self, path):
        """
        Get audio from memory cache, disk cache, or source file.
        """

        path = str(
            Path(path).resolve()
        )

        # ----------------------------------------------------
        # MEMORY CACHE
        # ----------------------------------------------------

        if path in self.cache:
            self.hits += 1
            return self.cache[path]

        # ----------------------------------------------------
        # DISK CACHE
        # ----------------------------------------------------

        cache_file = self._cache_filename(path)

        if cache_file.exists():

            try:
                with cache_file.open("rb") as file:
                    audio = pickle.load(file)

                audio = np.asarray(
                    audio,
                    dtype=np.float32,
                )

                if len(self.cache) < self.max_size:
                    self.cache[path] = audio

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
        # ACTUAL DISK AUDIO READ
        # ----------------------------------------------------

        audio = _load_audio_uncached(path)

        # ----------------------------------------------------
        # SAVE DISK CACHE
        # ----------------------------------------------------

        try:
            with cache_file.open("wb") as file:
                pickle.dump(
                    audio,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except OSError:
            pass

        # ----------------------------------------------------
        # SAVE MEMORY CACHE
        # ----------------------------------------------------

        if len(self.cache) < self.max_size:
            self.cache[path] = audio

        self.misses += 1

        return audio

    def stats(self):
        total = self.hits + self.misses

        if total == 0:
            hit_rate = 0.0
        else:
            hit_rate = self.hits / total

        return (
            f"Cache hit rate: "
            f"{hit_rate:.1%} "
            f"({self.hits}/{total})"
        )


# Global cache
_audio_cache = AudioCache()


def load_audio(path):
    """
    Public cached audio loader.
    """

    return _audio_cache.get(path)


# ============================================================
# AUDIO SEGMENTATION
# ============================================================

def random_segment(audio):
    """
    Return exactly SEGMENT_SAMPLES samples.
    """

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    length = len(audio)

    # Exact length
    if length == SEGMENT_SAMPLES:
        return audio.copy()

    # Longer than required -> random crop
    if length > SEGMENT_SAMPLES:

        start = random.randint(
            0,
            length - SEGMENT_SAMPLES,
        )

        return audio[
            start:start + SEGMENT_SAMPLES
        ].copy()

    # Shorter -> zero pad
    result = np.zeros(
        SEGMENT_SAMPLES,
        dtype=np.float32,
    )

    result[:length] = audio

    return result


# ============================================================
# NOISE PREPARATION
# ============================================================

def prepare_noise(noise):
    """
    Return a noise segment with exactly SEGMENT_SAMPLES samples.
    """

    noise = np.asarray(
        noise,
        dtype=np.float32,
    )

    if len(noise) == 0:
        return np.zeros(
            SEGMENT_SAMPLES,
            dtype=np.float32,
        )

    # If noise is long enough, random crop
    if len(noise) >= SEGMENT_SAMPLES:
        return random_segment(noise)

    # If short, repeat it
    repeats = (
        SEGMENT_SAMPLES // len(noise)
    ) + 1

    noise = np.tile(
        noise,
        repeats,
    )

    return random_segment(noise)


# ============================================================
# MIXING
# ============================================================

def mix_at_snr(
    clean,
    noise,
    snr_db,
):
    """
    Mix clean speech and noise at the requested SNR.
    """

    clean = np.asarray(
        clean,
        dtype=np.float32,
    )

    noise = np.asarray(
        noise,
        dtype=np.float32,
    )

    clean_power = np.mean(
        clean ** 2
    )

    noise_power = np.mean(
        noise ** 2
    )

    if noise_power < 1e-10:
        return clean.copy()

    # Desired noise power
    desired_noise_power = (
        clean_power
        / (10.0 ** (snr_db / 10.0))
    )

    scale = np.sqrt(
        desired_noise_power
        / (noise_power + 1e-12)
    )

    scaled_noise = noise * scale

    noisy = clean + scaled_noise

    # Prevent clipping
    peak = np.max(
        np.abs(noisy)
    )

    if peak > 1.0:
        noisy = noisy / peak

    return noisy.astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class SHAAudioDataset(Dataset):

    def __init__(
        self,
        clean_files,
        noise_files,
        multiplier=DATASET_MULTIPLIER,
    ):

        self.clean_files = list(clean_files)

        self.noise_files = list(noise_files)

        self.multiplier = max(
            int(multiplier),
            1,
        )

        if not self.clean_files:
            raise RuntimeError(
                "No clean audio files were supplied."
            )

        if not self.noise_files:
            raise RuntimeError(
                "No noise audio files were supplied."
            )

    def __len__(self):

        return (
            len(self.clean_files)
            * self.multiplier
        )

    def __getitem__(self, index):

        # Different generated sample can use
        # the same clean source.
        clean_index = (
            index
            % len(self.clean_files)
        )

        clean_path = self.clean_files[
            clean_index
        ]

        # Random noise
        noise_path = random.choice(
            self.noise_files
        )

        clean_audio = load_audio(
            clean_path
        )

        noise_audio = load_audio(
            noise_path
        )

        clean = random_segment(
            clean_audio
        )

        noise = prepare_noise(
            noise_audio
        )

        snr_db = random.uniform(
            SNR_MIN_DB,
            SNR_MAX_DB,
        )

        noisy = mix_at_snr(
            clean,
            noise,
            snr_db,
        )

        noisy_tensor = torch.from_numpy(
            noisy
        ).float().unsqueeze(0)

        clean_tensor = torch.from_numpy(
            clean
        ).float().unsqueeze(0)

        return (
            noisy_tensor,
            clean_tensor,
        )


# ============================================================
# DATASET SPLITTING
# ============================================================

def split_clean_files(
    clean_files,
    validation_ratio=VALIDATION_RATIO,
):
    """
    Deterministic train/validation split.
    """

    clean_files = list(clean_files)

    if len(clean_files) < 2:
        raise RuntimeError(
            "At least 2 clean audio files are required "
            "for train/validation splitting."
        )

    rng = random.Random(SEED)

    rng.shuffle(clean_files)

    validation_count = max(
        1,
        int(
            len(clean_files)
            * validation_ratio
        ),
    )

    validation_files = clean_files[
        :validation_count
    ]

    train_files = clean_files[
        validation_count:
    ]

    return (
        train_files,
        validation_files,
    )


# ============================================================
# LOCAL LOADERS
# ============================================================

def get_local_loaders(
    participant_id="LOCAL",
    batch_size=BATCH_SIZE,
    max_samples=None,
):
    """
    Prepare a local dataset for one participant.

    Every participant reads from its OWN PC:

        sources/librispeech
        sources/musan

    No network dataset is accessed.
    """

    print()
    print("-" * 70)
    print(
        f"[{participant_id}] PREPARING LOCAL AUDIO DATASET"
    )
    print("-" * 70)

    print(
        f"[{participant_id}] LibriSpeech: "
        f"{LIBRISPEECH_DIR}"
    )

    print(
        f"[{participant_id}] MUSAN: "
        f"{MUSAN_DIR}"
    )

    # --------------------------------------------------------
    # CHECK DIRECTORIES
    # --------------------------------------------------------

    if not LIBRISPEECH_DIR.exists():
        raise FileNotFoundError(
            f"[{participant_id}] LibriSpeech directory "
            f"does not exist:\n{LIBRISPEECH_DIR}"
        )

    if not MUSAN_DIR.exists():
        raise FileNotFoundError(
            f"[{participant_id}] MUSAN directory "
            f"does not exist:\n{MUSAN_DIR}"
        )

    # --------------------------------------------------------
    # FIND AUDIO
    # --------------------------------------------------------

    clean_files = find_audio_files(
        LIBRISPEECH_DIR
    )

    noise_files = find_audio_files(
        MUSAN_DIR
    )

    print(
        f"[{participant_id}] Clean files found: "
        f"{len(clean_files)}"
    )

    print(
        f"[{participant_id}] Noise files found: "
        f"{len(noise_files)}"
    )

    if not clean_files:
        raise RuntimeError(
            f"[{participant_id}] No LibriSpeech "
            "audio files were found."
        )

    if not noise_files:
        raise RuntimeError(
            f"[{participant_id}] No MUSAN "
            "audio files were found."
        )

    # --------------------------------------------------------
    # LIMIT DATASET FOR TESTING
    # --------------------------------------------------------

    if max_samples is not None:

        max_samples = int(
            max_samples
        )

        if max_samples < 2:
            raise ValueError(
                "max_samples must be at least 2."
            )

        if max_samples < len(clean_files):

            rng = random.Random(
                SEED
            )

            clean_files = rng.sample(
                clean_files,
                max_samples,
            )

            print(
                f"[{participant_id}] "
                f"Limited clean files to "
                f"{len(clean_files)}"
            )

    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    (
        train_clean_files,
        validation_clean_files,
    ) = split_clean_files(
        clean_files
    )

    print(
        f"[{participant_id}] Training files: "
        f"{len(train_clean_files)}"
    )

    print(
        f"[{participant_id}] Validation files: "
        f"{len(validation_clean_files)}"
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = SHAAudioDataset(
        clean_files=train_clean_files,
        noise_files=noise_files,
        multiplier=DATASET_MULTIPLIER,
    )

    validation_dataset = SHAAudioDataset(
        clean_files=validation_clean_files,
        noise_files=noise_files,
        multiplier=1,
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=DATALOADER_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print(
        f"[{participant_id}] Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"[{participant_id}] Validation samples: "
        f"{len(validation_dataset)}"
    )

    print(
        f"[{participant_id}] Training batches: "
        f"{len(train_loader)}"
    )

    print(
        f"[{participant_id}] Validation batches: "
        f"{len(validation_loader)}"
    )

    print(
        f"[{participant_id}] "
        f"{_audio_cache.stats()}"
    )

    print(
        f"[{participant_id}] Local dataset READY."
    )

    return (
        train_loader,
        validation_loader,
    )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def get_worker_loader(
    worker_id,
    num_workers=DEFAULT_WORKERS if "DEFAULT_WORKERS" in globals() else 2,
    batch_size=BATCH_SIZE,
):
    """
    Compatibility helper for older code.

    Returns the training loader only.
    """

    train_loader, _ = get_local_loaders(
        participant_id=f"WORKER {worker_id}",
        batch_size=batch_size,
        max_samples=None,
    )

    return train_loader