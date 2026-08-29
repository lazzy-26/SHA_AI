import random
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
    DATASET_MULTIPLIER,
    LIBRISPEECH_DIR,
    MUSAN_DIR,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    SEED,
    SNR_MAX_DB,
    SNR_MIN_DB,
)


# ============================================================
# AUDIO FILE DISCOVERY
# ============================================================

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
}


def find_audio_files(folder):
    """
    Recursively discover supported audio files.
    """

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


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(path):
    """
    Load an audio file and convert it to:

        float32
        mono
        SAMPLE_RATE
    """

    path = Path(path)

    audio, sample_rate = sf.read(
        path,
        always_2d=False,
    )

    # --------------------------------------------------------
    # Convert stereo/multi-channel to mono
    # --------------------------------------------------------

    if audio.ndim == 2:

        audio = np.mean(
            audio,
            axis=1,
        )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # --------------------------------------------------------
    # Resample
    # --------------------------------------------------------

    if sample_rate != SAMPLE_RATE:

        audio = resample_poly(
            audio,
            SAMPLE_RATE,
            sample_rate,
        )

        audio = audio.astype(
            np.float32
        )

    # --------------------------------------------------------
    # Prevent excessive amplitude
    # --------------------------------------------------------

    if len(audio) > 0:

        peak = float(
            np.max(
                np.abs(audio)
            )
        )

        if peak > 1.0:

            audio = (
                audio
                / peak
            )

    return audio.astype(
        np.float32
    )


# ============================================================
# RANDOM SEGMENT
# ============================================================

def random_segment(audio):
    """
    Return exactly SEGMENT_SAMPLES samples.
    """

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


# ============================================================
# NOISE PREPARATION
# ============================================================

def prepare_noise(noise):

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


# ============================================================
# MIX CLEAN + NOISE
# ============================================================

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
            0.99
            / peak
        )

        noisy *= scale
        clean *= scale

    return (
        noisy.astype(
            np.float32
        ),
        clean.astype(
            np.float32
        ),
    )


# ============================================================
# DATASET
# ============================================================

class SHAAudioDataset(Dataset):

    def __init__(
        self,
        clean_files,
        noise_files,
        multiplier=DATASET_MULTIPLIER,
        validation=False,
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

        self.validation = (
            validation
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

        clean_path = (
            self.clean_files[
                index
                % len(
                    self.clean_files
                )
            ]
        )

        # ----------------------------------------------------
        # Deterministic validation
        # ----------------------------------------------------

        if self.validation:

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

            clean = load_audio(
                clean_path
            )

            noise = load_audio(
                noise_path
            )

            # Deterministic segment.
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

            noise = prepare_noise(
                noise
            )

            # Fixed validation SNR.
            snr_db = 5.0

        else:

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


# ============================================================
# SPLIT LOCAL DATASET
# ============================================================

def split_files(
    files,
    validation_ratio=0.2,
):
    """
    Split the LOCAL clean dataset.

    Every PC performs this independently.

    80% -> training
    20% -> validation
    """

    files = list(files)

    if len(files) < 2:

        return files, files

    rng = random.Random(
        SEED
    )

    shuffled = list(files)

    rng.shuffle(
        shuffled
    )

    validation_count = max(
        1,
        int(
            len(shuffled)
            * validation_ratio
        ),
    )

    validation_files = (
        shuffled[
            :validation_count
        ]
    )

    training_files = (
        shuffled[
            validation_count:
        ]
    )

    if not training_files:

        training_files = (
            validation_files
        )

    return (
        training_files,
        validation_files,
    )


# ============================================================
# LOCAL DATALOADERS
# ============================================================

def get_local_loaders(
    participant_id="HOST",
    batch_size=BATCH_SIZE,
):
    """
    Build training and validation loaders from:

        sources/librispeech
        sources/musan

    on the CURRENT machine.
    """

    clean_dir = Path(
        LIBRISPEECH_DIR
    )

    noise_dir = Path(
        MUSAN_DIR
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
        f"LibriSpeech : {clean_dir}"
    )

    print(
        f"MUSAN       : {noise_dir}"
    )

    print(
        f"Clean files : {len(clean_files)}"
    )

    print(
        f"Noise files : {len(noise_files)}"
    )

    print(
        f"Training    : {len(training_files)}"
    )

    print(
        f"Validation  : {len(validation_files)}"
    )

    print(
        f"Multiplier  : {DATASET_MULTIPLIER}"
    )

    print(
        "-" * 70
    )

    train_dataset = SHAAudioDataset(
        clean_files=training_files,
        noise_files=noise_files,
        multiplier=DATASET_MULTIPLIER,
        validation=False,
    )

    validation_dataset = SHAAudioDataset(
        clean_files=validation_files,
        noise_files=noise_files,
        multiplier=max(
            2,
            DATASET_MULTIPLIER // 2,
        ),
        validation=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return (
        train_loader,
        validation_loader,
    )