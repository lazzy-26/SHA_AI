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
    DATASET_MULTIPLIER,
    LIBRISPEECH_DIR,
    MUSAN_DIR,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    SNR_MAX_DB,
    SNR_MIN_DB,
    VALIDATION_RATIO,
)


SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
}


def find_audio_files(
    folder: Path,
):
    folder = Path(folder)

    if not folder.exists():
        return []

    return sorted(
        path
        for path in folder.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_AUDIO_EXTENSIONS
        )
    )


@lru_cache(maxsize=512)
def load_audio_cached(
    path_string,
):
    path = Path(path_string)

    audio, sample_rate = sf.read(
        path,
        always_2d=False,
    )

    if audio.ndim == 2:
        audio = np.mean(
            audio,
            axis=1,
        )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if sample_rate != SAMPLE_RATE:
        audio = resample_poly(
            audio,
            SAMPLE_RATE,
            sample_rate,
        ).astype(
            np.float32
        )

    if len(audio) > 0:
        peak = float(
            np.max(
                np.abs(audio)
            )
        )

        if peak > 1.0:
            audio = (
                audio / peak
            )

    return audio.astype(
        np.float32
    )


def load_audio(path):
    return load_audio_cached(
        str(
            Path(path).resolve()
        )
    ).copy()


def random_segment(
    audio,
):
    if len(audio) == 0:
        return np.zeros(
            SEGMENT_SAMPLES,
            dtype=np.float32,
        )

    if len(audio) >= SEGMENT_SAMPLES:
        start = random.randint(
            0,
            len(audio)
            - SEGMENT_SAMPLES,
        )

        return audio[
            start:
            start + SEGMENT_SAMPLES
        ].astype(
            np.float32
        )

    padded = np.zeros(
        SEGMENT_SAMPLES,
        dtype=np.float32,
    )

    padded[
        :len(audio)
    ] = audio

    return padded


def prepare_noise(
    noise,
):
    if len(noise) == 0:
        return np.zeros(
            SEGMENT_SAMPLES,
            dtype=np.float32,
        )

    if len(noise) >= SEGMENT_SAMPLES:
        return random_segment(
            noise
        )

    repeats = int(
        np.ceil(
            SEGMENT_SAMPLES
            / len(noise)
        )
    )

    repeated = np.tile(
        noise,
        repeats,
    )

    return random_segment(
        repeated
    )


def mix_at_snr(
    clean,
    noise,
    snr_db,
):
    clean = clean.astype(
        np.float32
    )

    noise = noise.astype(
        np.float32
    )

    clean_power = (
        np.mean(
            clean ** 2
        )
        + 1e-8
    )

    noise_power = (
        np.mean(
            noise ** 2
        )
        + 1e-8
    )

    target_noise_power = (
        clean_power
        / (
            10.0
            ** (
                snr_db
                / 10.0
            )
        )
    )

    noise_scale = np.sqrt(
        target_noise_power
        / noise_power
    )

    scaled_noise = (
        noise
        * noise_scale
    )

    noisy = (
        clean
        + scaled_noise
    )

    peak = float(
        np.max(
            np.abs(noisy)
        )
    )

    if peak > 0.99:
        scale = (
            0.99 / peak
        )

        noisy *= scale
        clean *= scale

    return (
        noisy.astype(np.float32),
        clean.astype(np.float32),
    )


class SHAAudioDataset(
    Dataset
):

    def __init__(
        self,
        clean_files,
        noise_files,
        multiplier=DATASET_MULTIPLIER,
        deterministic=False,
    ):
        self.clean_files = list(
            clean_files
        )

        self.noise_files = list(
            noise_files
        )

        self.multiplier = max(
            int(multiplier),
            1,
        )

        self.deterministic = (
            deterministic
        )

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
        clean_index = (
            index
            % len(
                self.clean_files
            )
        )

        clean_path = (
            self.clean_files[
                clean_index
            ]
        )

        if self.deterministic:
            noise_index = (
                index
                % len(
                    self.noise_files
                )
            )

            noise_path = (
                self.noise_files[
                    noise_index
                ]
            )

            rng = random.Random(
                index + 12345
            )

            snr_db = rng.uniform(
                SNR_MIN_DB,
                SNR_MAX_DB,
            )

        else:
            noise_path = random.choice(
                self.noise_files
            )

            snr_db = random.uniform(
                SNR_MIN_DB,
                SNR_MAX_DB,
            )

        clean = load_audio(
            clean_path
        )

        noise = load_audio(
            noise_path
        )

        if self.deterministic:
            if len(clean) >= SEGMENT_SAMPLES:
                max_start = (
                    len(clean)
                    - SEGMENT_SAMPLES
                )

                start = (
                    index
                    % (
                        max_start + 1
                    )
                )

                clean = clean[
                    start:
                    start + SEGMENT_SAMPLES
                ]

            else:
                clean = random_segment(
                    clean
                )

        else:
            clean = random_segment(
                clean
            )

        noise = prepare_noise(
            noise
        )

        noisy, clean = mix_at_snr(
            clean,
            noise,
            snr_db,
        )

        noisy_tensor = (
            torch.from_numpy(
                noisy
            )
            .unsqueeze(0)
        )

        clean_tensor = (
            torch.from_numpy(
                clean
            )
            .unsqueeze(0)
        )

        return (
            noisy_tensor,
            clean_tensor,
        )


def split_files(
    clean_files,
    validation_ratio=VALIDATION_RATIO,
):
    files = list(
        clean_files
    )

    if len(files) < 2:
        return (
            files,
            files,
        )

    rng = random.Random(
        42
    )

    rng.shuffle(
        files
    )

    validation_count = max(
        1,
        int(
            len(files)
            * validation_ratio
        ),
    )

    validation_files = files[
        :validation_count
    ]

    training_files = files[
        validation_count:
    ]

    return (
        training_files,
        validation_files,
    )


def discover_local_files():
    clean_files = find_audio_files(
        LIBRISPEECH_DIR
    )

    noise_files = find_audio_files(
        MUSAN_DIR
    )

    return (
        clean_files,
        noise_files,
    )


def get_local_loaders(
    participant_id,
    batch_size=BATCH_SIZE,
):
    clean_dir = Path(
        LIBRISPEECH_DIR
    )

    noise_dir = Path(
        MUSAN_DIR
    )

    print()
    print(
        "-" * 70
    )

    print(
        f"[{participant_id}] LOCAL DATASET"
    )

    print(
        "-" * 70
    )

    print(
        f"LibriSpeech : "
        f"{clean_dir}"
    )

    print(
        f"MUSAN       : "
        f"{noise_dir}"
    )

    if not clean_dir.exists():
        raise RuntimeError(
            f"[{participant_id}] "
            f"LibriSpeech directory does not exist:\n"
            f"{clean_dir}"
        )

    if not noise_dir.exists():
        raise RuntimeError(
            f"[{participant_id}] "
            f"MUSAN directory does not exist:\n"
            f"{noise_dir}"
        )

    clean_files = find_audio_files(
        clean_dir
    )

    noise_files = find_audio_files(
        noise_dir
    )

    if not clean_files:
        raise RuntimeError(
            f"[{participant_id}] "
            "No LibriSpeech audio files found."
        )

    if not noise_files:
        raise RuntimeError(
            f"[{participant_id}] "
            "No MUSAN audio files found."
        )

    (
        training_files,
        validation_files,
    ) = split_files(
        clean_files
    )

    training_dataset = (
        SHAAudioDataset(
            clean_files=training_files,
            noise_files=noise_files,
            multiplier=DATASET_MULTIPLIER,
            deterministic=False,
        )
    )

    validation_dataset = (
        SHAAudioDataset(
            clean_files=validation_files,
            noise_files=noise_files,
            multiplier=1,
            deterministic=True,
        )
    )

    loader_workers = max(
        int(DATALOADER_WORKERS),
        0,
    )

    train_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(
            loader_workers > 0
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(
            loader_workers > 0
        ),
    )

    print(
        f"Clean files : "
        f"{len(clean_files)}"
    )

    print(
        f"Noise files : "
        f"{len(noise_files)}"
    )

    print(
        f"Training    : "
        f"{len(training_files)}"
    )

    print(
        f"Validation  : "
        f"{len(validation_files)}"
    )

    print(
        f"Multiplier  : "
        f"{DATASET_MULTIPLIER}"
    )

    print(
        f"Train examples: "
        f"{len(training_dataset)}"
    )

    print(
        f"Validation examples: "
        f"{len(validation_dataset)}"
    )

    print(
        f"DataLoader workers: "
        f"{loader_workers}"
    )

    print(
        "-" * 70
    )

    return (
        train_loader,
        validation_loader,
    )


def get_worker_loader(
    worker_id,
    num_workers=None,
    batch_size=BATCH_SIZE,
):
    (
        train_loader,
        _,
    ) = get_local_loaders(
        participant_id=f"WORKER {worker_id}",
        batch_size=batch_size,
    )

    return train_loader