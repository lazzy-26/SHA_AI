import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from config import (
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    SNR_MIN_DB,
    SNR_MAX_DB,
)


def find_wav_files(folder: Path):
    return sorted(
        path
        for path in folder.rglob("*")
        if path.suffix.lower() == ".wav"
    )


def load_audio(path: Path):
    audio, sample_rate = sf.read(path, always_2d=False)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    if sample_rate != SAMPLE_RATE:
        audio = resample_poly(
            audio,
            SAMPLE_RATE,
            sample_rate,
        ).astype(np.float32)

    peak = np.max(np.abs(audio))

    if peak > 1.0:
        audio = audio / peak

    return audio


def random_segment(audio):
    if len(audio) >= SEGMENT_SAMPLES:
        start = random.randint(
            0,
            len(audio) - SEGMENT_SAMPLES,
        )

        return audio[
            start:start + SEGMENT_SAMPLES
        ]

    result = np.zeros(
        SEGMENT_SAMPLES,
        dtype=np.float32,
    )

    result[:len(audio)] = audio

    return result


def repeat_noise(noise):
    if len(noise) >= SEGMENT_SAMPLES:
        return random_segment(noise)

    repeats = int(
        np.ceil(
            SEGMENT_SAMPLES / max(len(noise), 1)
        )
    )

    noise = np.tile(noise, repeats)

    return random_segment(noise)


def mix_at_snr(clean, noise, snr_db):
    clean_power = (
        np.mean(clean ** 2) + 1e-8
    )

    noise_power = (
        np.mean(noise ** 2) + 1e-8
    )

    target_noise_power = (
        clean_power /
        (10 ** (snr_db / 10.0))
    )

    scale = np.sqrt(
        target_noise_power / noise_power
    )

    scaled_noise = noise * scale

    noisy = clean + scaled_noise

    max_value = np.max(np.abs(noisy))

    if max_value > 0.99:
        scale_down = 0.99 / max_value

        noisy = noisy * scale_down
        clean = clean * scale_down

    return noisy.astype(np.float32), clean.astype(np.float32)


class ANCDataset(Dataset):
    def __init__(
        self,
        clean_files,
        noise_files,
        length_multiplier=8,
    ):
        self.clean_files = list(clean_files)
        self.noise_files = list(noise_files)
        self.length_multiplier = length_multiplier

        if not self.clean_files:
            raise RuntimeError(
                "No clean WAV files found."
            )

        if not self.noise_files:
            raise RuntimeError(
                "No noise WAV files found."
            )

    def __len__(self):
        return (
            len(self.clean_files)
            * self.length_multiplier
        )

    def __getitem__(self, index):
        clean_path = self.clean_files[
            index % len(self.clean_files)
        ]

        noise_path = random.choice(
            self.noise_files
        )

        clean = random_segment(
            load_audio(clean_path)
        )

        noise = repeat_noise(
            load_audio(noise_path)
        )

        snr_db = random.uniform(
            SNR_MIN_DB,
            SNR_MAX_DB,
        )

        noisy, clean = mix_at_snr(
            clean,
            noise,
            snr_db,
        )

        noisy_tensor = torch.from_numpy(
            noisy
        ).unsqueeze(0)

        clean_tensor = torch.from_numpy(
            clean
        ).unsqueeze(0)

        return noisy_tensor, clean_tensor