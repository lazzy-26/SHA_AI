import os
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# AUDIO
# ============================================================

SAMPLE_RATE = 16000

SEGMENT_SECONDS = 2

SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS


# ============================================================
# ANC DATASET
#
# Prepared dataset:
#
# datasets/
# └── anc/
#     ├── dataset/
#     │   └── musan/
#     │       └── noise/
#     │
#     └── raw/
#         ├── train/
#         │   ├── clean/
#         │   └── noisy/
#         │
#         └── test/
#             ├── clean/
#             └── noisy/
# ============================================================

ANC_DIR = ROOT / "datasets" / "anc"

RAW_DIR = ANC_DIR / "raw"

TRAIN_DIR = RAW_DIR / "train"
TEST_DIR = RAW_DIR / "test"

TRAIN_CLEAN_DIR = TRAIN_DIR / "clean"
TRAIN_NOISY_DIR = TRAIN_DIR / "noisy"

TEST_CLEAN_DIR = TEST_DIR / "clean"
TEST_NOISY_DIR = TEST_DIR / "noisy"


# ============================================================
# MUSAN
#
# Kept for optional future augmentation.
# It is NOT required for the prepared-dataset training path.
# ============================================================

MUSAN_DIR = ANC_DIR / "dataset" / "musan"

MUSAN_NOISE_DIR = MUSAN_DIR / "noise"


# ============================================================
# OPTIONAL DATASET OVERRIDE
# ============================================================

if os.environ.get("SHA_ANC_DIR"):
    ANC_DIR = Path(
        os.environ["SHA_ANC_DIR"]
    )

    RAW_DIR = ANC_DIR / "raw"

    TRAIN_DIR = RAW_DIR / "train"
    TEST_DIR = RAW_DIR / "test"

    TRAIN_CLEAN_DIR = TRAIN_DIR / "clean"
    TRAIN_NOISY_DIR = TRAIN_DIR / "noisy"

    TEST_CLEAN_DIR = TEST_DIR / "clean"
    TEST_NOISY_DIR = TEST_DIR / "noisy"

    MUSAN_DIR = ANC_DIR / "dataset" / "musan"
    MUSAN_NOISE_DIR = MUSAN_DIR / "noise"


# ============================================================
# FEDERATED PARTICIPANTS
# ============================================================

DEFAULT_WORKERS = 3

# HOST + Worker 1 + Worker 2
NUM_PARTICIPANTS = DEFAULT_WORKERS + 1


# ============================================================
# NETWORK
# ============================================================

HOST_IP = "170.20.10.2"

DEFAULT_PORT = 8080


# ============================================================
# NETWORK TIMEOUTS
# ============================================================

READY_TIMEOUT = 1800       # 30 minutes

ROUND_TIMEOUT = 3600       # 1 hour

WORKER_SOCKET_TIMEOUT = 0


# ============================================================
# TRAINING
# ============================================================

BATCH_SIZE = 32

GRADIENT_ACCUMULATION_STEPS = 2

EPOCHS = 15

LEARNING_RATE = 5e-4


# ============================================================
# DATASET
# ============================================================

# Kept for compatibility with code that still imports this.
#
# Since raw/train already contains prepared noisy examples,
# the multiplier is NOT used to regenerate audio.
DATASET_MULTIPLIER = 1


# ============================================================
# DATALOADER
# ============================================================

DATALOADER_WORKERS = 2


# ============================================================
# VALIDATION
# ============================================================

VALIDATION_RATIO = 0.05


# ============================================================
# EARLY STOPPING
# ============================================================

MIN_IMPROVEMENT = 0.001

PATIENCE = 3


# ============================================================
# GLOBAL EARLY STOPPING
# ============================================================

GLOBAL_MIN_IMPROVEMENT = 0.001

GLOBAL_PATIENCE = 3


# ============================================================
# CHECKPOINTS
# ============================================================

CHECKPOINT_DIR = ROOT / "checkpoints" / "anc"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GLOBAL_MODEL_PATH = (
    CHECKPOINT_DIR / "sha_anc_global.pt"
)


# ============================================================
# AUDIO CACHE
# ============================================================

AUDIO_CACHE_DIR = ROOT / "audio_cache"

AUDIO_CACHE_MAX_MEMORY_FILES = 1000


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42


# ============================================================
# FAST / 1-HOUR PRESET
# ============================================================

FAST_TRAINING = {
    "max_samples": 3000,
    "epochs": 15,
    "batch_size": 32,
    "grad_accum_steps": 2,
    "segment_seconds": 2,
    "timeout": 3600,
    "dataloader_workers": 2,
    "early_stop_patience": 3,
}