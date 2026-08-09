import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset

from config import (
    CLEAN_DIR,
    NOISE_DIR,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    SNR_MAX_DB,
    SNR_MIN_DB,
)


# ============================================================
# AUDIO FILE DISCOVERY
# ============================================================

def find_audio_files(folder: Path):
    """
    Find all supported audio files inside a directory.
    """
    return sorted(
        path
        for path in folder.rglob("*")
        if path.suffix.lower() in {
            ".wav",
            ".flac",
        }
    )


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(path):
    """
    Load an audio file, convert it to mono and resample it
    to the sample rate used by the S.H.A ANC model.
    """

    audio, sample_rate = sf.read(
        path,
        always_2d=False,
    )

    # Convert stereo/multi-channel audio to mono.
    if audio.ndim == 2:
        audio = np.mean(
            audio,
            axis=1,
        )

    audio = audio.astype(
        np.float32
    )

    # Resample when necessary.
    if sample_rate != SAMPLE_RATE:
        audio = resample_poly(
            audio,
            SAMPLE_RATE,
            sample_rate,
        ).astype(np.float32)

    # Prevent samples from exceeding the valid range.
    peak = np.max(
        np.abs(audio)
    )

    if peak > 1.0:
        audio /= peak

    return audio


# ============================================================
# RANDOM AUDIO SEGMENT
# ============================================================

def random_segment(audio):
    """
    Return one fixed-length training segment.

    Long files:
        Select a random segment.

    Short files:
        Zero-pad them to SEGMENT_SAMPLES.
    """

    if len(audio) >= SEGMENT_SAMPLES:
        start = random.randint(
            0,
            len(audio) - SEGMENT_SAMPLES,
        )

        return audio[
            start:start + SEGMENT_SAMPLES
        ]

    padded = np.zeros(
        SEGMENT_SAMPLES,
        dtype=np.float32,
    )

    padded[:len(audio)] = audio

    return padded


# ============================================================
# NOISE PREPARATION
# ============================================================

def prepare_noise(noise):
    """
    Prepare a noise recording to have exactly the same number
    of samples as the clean speech segment.
    """

    if len(noise) >= SEGMENT_SAMPLES:
        return random_segment(noise)

    # Repeat short noise recordings instead of padding them
    # mostly with silence.
    repeats = int(
        np.ceil(
            SEGMENT_SAMPLES
            / max(len(noise), 1)
        )
    )

    repeated = np.tile(
        noise,
        repeats,
    )

    return random_segment(
        repeated
    )


# ============================================================
# CLEAN + NOISE MIXING
# ============================================================

def mix_at_snr(
    clean,
    noise,
    snr_db,
):
    """
    Mix clean speech and noise at a randomly selected
    signal-to-noise ratio.
    """

    clean_power = (
        np.mean(clean ** 2)
        + 1e-8
    )

    noise_power = (
        np.mean(noise ** 2)
        + 1e-8
    )

    target_noise_power = (
        clean_power
        / (10 ** (snr_db / 10.0))
    )

    noise_scale = np.sqrt(
        target_noise_power
        / noise_power
    )

    scaled_noise = (
        noise * noise_scale
    )

    noisy = (
        clean
        + scaled_noise
    )

    # Avoid clipping while preserving the relationship
    # between the noisy signal and clean target.
    peak = np.max(
        np.abs(noisy)
    )

    if peak > 0.99:
        scale = (
            0.99
            / peak
        )

        noisy *= scale
        clean *= scale

    return (
        noisy.astype(np.float32),
        clean.astype(np.float32),
    )


# ============================================================
# S.H.A ANC DATASET
# ============================================================

class SHAAudioDataset(Dataset):
    """
    Dataset used by each S.H.A distributed training worker.

    Every clean speech recording can produce several different
    training examples because noise and SNR are selected
    randomly each time.
    """

    def __init__(
        self,
        clean_files,
        noise_files,
        multiplier=8,
    ):
        self.clean_files = list(
            clean_files
        )

        self.noise_files = list(
            noise_files
        )

        self.multiplier = multiplier

        if not self.clean_files:
            raise RuntimeError(
                "No clean speech files found."
            )

        if not self.noise_files:
            raise RuntimeError(
                "No noise files found."
            )

    def __len__(self):
        return (
            len(self.clean_files)
            * self.multiplier
        )

    def __getitem__(
        self,
        index,
    ):
        # Cycle through the worker's local clean speech files.
        clean_path = self.clean_files[
            index % len(self.clean_files)
        ]

        # Randomly choose a noise recording.
        noise_path = random.choice(
            self.noise_files
        )

        clean = load_audio(
            clean_path
        )

        noise = load_audio(
            noise_path
        )

        clean = random_segment(
            clean
        )

        noise = prepare_noise(
            noise
        )

        # Generate a random SNR for this training example.
        snr_db = random.uniform(
            SNR_MIN_DB,
            SNR_MAX_DB,
        )

        noisy, clean = mix_at_snr(
            clean,
            noise,
            snr_db,
        )

        noisy_tensor = (
            torch.from_numpy(noisy)
            .unsqueeze(0)
        )

        clean_tensor = (
            torch.from_numpy(clean)
            .unsqueeze(0)
        )

        return (
            noisy_tensor,
            clean_tensor,
        )


# ============================================================
# WORKER DATA LOADER
# ============================================================

def get_worker_loader(
    worker_id,
    num_workers,
    batch_size,
):
    """
    Load the physical dataset stored on this worker.

    IMPORTANT:

    The dataset has already been physically divided into
    three worker shards.

    Therefore we DO NOT perform:

        clean_files[worker_id - 1::num_workers]

    anymore.

    Each worker uses 100% of the clean files that exist in
    its own local datasets/anc/clean directory.
    """

    clean_files = find_audio_files(
        CLEAN_DIR
    )

    noise_files = find_audio_files(
        NOISE_DIR
    )

    if not clean_files:
        raise RuntimeError(
            f"No clean files found in "
            f"{CLEAN_DIR}"
        )

    if not noise_files:
        raise RuntimeError(
            f"No noise files found in "
            f"{NOISE_DIR}"
        )

    print()
    print(
        f"[Worker {worker_id}] "
        f"Dataset information"
    )

    print(
        f"[Worker {worker_id}] "
        f"Clean files : "
        f"{len(clean_files)}"
    )

    print(
        f"[Worker {worker_id}] "
        f"Noise files : "
        f"{len(noise_files)}"
    )

    print(
        f"[Worker {worker_id}] "
        f"Multiplier  : 8"
    )

    training_samples = (
        len(clean_files)
        * 8
    )

    print(
        f"[Worker {worker_id}] "
        f"Training examples per epoch: "
        f"{training_samples}"
    )

    print()

    dataset = SHAAudioDataset(
        clean_files=clean_files,
        noise_files=noise_files,
        multiplier=8,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    return loader