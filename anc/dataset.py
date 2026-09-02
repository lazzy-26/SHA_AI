import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from scipy.signal import resample_poly

from torch.utils.data import (
    DataLoader,
    Dataset,
)

from config import (
    BATCH_SIZE,
    DATALOADER_WORKERS,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    TRAIN_CLEAN_DIR,
    TRAIN_NOISY_DIR,
    VALIDATION_RATIO,
)


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac"}


def find_audio_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


@lru_cache(maxsize=512)
def load_audio_cached(path_string):
    path = Path(path_string)
    audio, sample_rate = sf.read(path, always_2d=False)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if sample_rate != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, sample_rate).astype(np.float32)

    if len(audio) > 0:
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak

    return audio.astype(np.float32)


def load_audio(path):
    return load_audio_cached(str(Path(path).resolve())).copy()


def crop_pair(clean, noisy, segment_samples, start=None):
    """
    Crops (or pads) clean/noisy to the same segment, using the SAME
    start offset for both since they're time-aligned recordings.
    """
    min_len = min(len(clean), len(noisy))
    clean, noisy = clean[:min_len], noisy[:min_len]

    if min_len >= segment_samples:
        if start is None:
            start = random.randint(0, min_len - segment_samples)
        else:
            start = start % (min_len - segment_samples + 1)
        return (
            clean[start:start + segment_samples],
            noisy[start:start + segment_samples],
        )

    padded_clean = np.zeros(segment_samples, dtype=np.float32)
    padded_noisy = np.zeros(segment_samples, dtype=np.float32)
    padded_clean[:min_len] = clean
    padded_noisy[:min_len] = noisy
    return padded_clean, padded_noisy


def pair_clean_noisy(clean_dir, noisy_dir):
    """
    Pairs clean/noisy files by matching filename (stem). Files that
    only exist on one side are dropped, with a warning.
    """
    clean_by_stem = {f.stem: f for f in find_audio_files(clean_dir)}
    noisy_by_stem = {f.stem: f for f in find_audio_files(noisy_dir)}

    common = sorted(set(clean_by_stem) & set(noisy_by_stem))

    only_clean = set(clean_by_stem) - set(noisy_by_stem)
    only_noisy = set(noisy_by_stem) - set(clean_by_stem)

    if only_clean:
        print(
            f"[WARNING] {len(only_clean)} clean files have no matching noisy "
            f"file, skipping (e.g. {sorted(only_clean)[:3]})"
        )

    if only_noisy:
        print(
            f"[WARNING] {len(only_noisy)} noisy files have no matching clean "
            f"file, skipping (e.g. {sorted(only_noisy)[:3]})"
        )

    return [(clean_by_stem[s], noisy_by_stem[s]) for s in common]


class VoiceBankDataset(Dataset):
    """
    Dataset of pre-paired (clean, noisy) audio files, matched by filename.
    """

    def __init__(
        self,
        clean_dir=None,
        noisy_dir=None,
        pairs=None,
        segment_samples=SEGMENT_SAMPLES,
        deterministic=False,
    ):
        if pairs is not None:
            self.pairs = list(pairs)
        elif clean_dir is not None and noisy_dir is not None:
            self.pairs = pair_clean_noisy(clean_dir, noisy_dir)
        else:
            raise ValueError(
                "Provide either 'pairs', or both 'clean_dir' and 'noisy_dir'."
            )

        if not self.pairs:
            raise RuntimeError("No matching clean/noisy file pairs found.")

        self.segment_samples = segment_samples
        self.deterministic = deterministic

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        clean_path, noisy_path = self.pairs[index]

        clean = load_audio(clean_path)
        noisy = load_audio(noisy_path)

        start = index if self.deterministic else None
        clean, noisy = crop_pair(clean, noisy, self.segment_samples, start=start)

        noisy_tensor = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0)
        clean_tensor = torch.from_numpy(clean.astype(np.float32)).unsqueeze(0)

        return noisy_tensor, clean_tensor


def split_pairs(pairs, validation_ratio=VALIDATION_RATIO):
    pairs = list(pairs)
    if len(pairs) < 2:
        return pairs, pairs

    rng = random.Random(42)
    rng.shuffle(pairs)

    validation_count = max(1, int(len(pairs) * validation_ratio))
    return pairs[validation_count:], pairs[:validation_count]


def get_local_loaders(
    participant_id,
    batch_size=BATCH_SIZE,
    max_samples=None,
):
    print()
    print("-" * 70)
    print(f"[{participant_id}] LOCAL DATASET")
    print("-" * 70)
    print(f"Train clean : {TRAIN_CLEAN_DIR}")
    print(f"Train noisy : {TRAIN_NOISY_DIR}")

    if max_samples is not None and max_samples > 0:
        print(f"MAX SAMPLES : {max_samples} (TESTING MODE)")

    if not TRAIN_CLEAN_DIR.exists():
        raise RuntimeError(
            f"[{participant_id}] Train clean directory does not exist:\n{TRAIN_CLEAN_DIR}"
        )
    if not TRAIN_NOISY_DIR.exists():
        raise RuntimeError(
            f"[{participant_id}] Train noisy directory does not exist:\n{TRAIN_NOISY_DIR}"
        )

    all_pairs = pair_clean_noisy(TRAIN_CLEAN_DIR, TRAIN_NOISY_DIR)

    if not all_pairs:
        raise RuntimeError(f"[{participant_id}] No matching clean/noisy pairs found.")

    training_pairs, validation_pairs = split_pairs(all_pairs)

    if max_samples is not None and max_samples > 0:
        val_samples = max(1, max_samples // 5)

        if len(training_pairs) > max_samples:
            training_pairs = training_pairs[:max_samples]
            print(f"[{participant_id}] Training pairs limited to: {len(training_pairs)}")

        if len(validation_pairs) > val_samples:
            validation_pairs = validation_pairs[:val_samples]
            print(f"[{participant_id}] Validation pairs limited to: {len(validation_pairs)}")

    training_dataset = VoiceBankDataset(pairs=training_pairs, deterministic=False)
    validation_dataset = VoiceBankDataset(pairs=validation_pairs, deterministic=True)

    loader_workers = max(int(DATALOADER_WORKERS), 0)

    train_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(loader_workers > 0),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(loader_workers > 0),
    )

    print(f"Total pairs : {len(all_pairs)}")
    print(f"Training    : {len(training_pairs)}")
    print(f"Validation  : {len(validation_pairs)}")
    print(f"DataLoader workers: {loader_workers}")
    print("-" * 70)

    return train_loader, validation_loader


def get_worker_loader(worker_id, num_workers=None, batch_size=BATCH_SIZE):
    train_loader, _ = get_local_loaders(
        participant_id=f"WORKER {worker_id}",
        batch_size=batch_size,
    )
    return train_loader