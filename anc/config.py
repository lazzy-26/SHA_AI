import os
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# AUDIO - OPTIMIZED FOR SPEED
# ============================================================

SAMPLE_RATE = 16000

# SHORTER segments = faster training
# 2 seconds is enough for speech enhancement to learn
SEGMENT_SECONDS = 2  # Reduced from 3

SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS


# ============================================================
# LOCAL DATA SOURCES
# ============================================================

SOURCES_DIR = ROOT / "datasets" / "anc" / "raw"

TRAIN_CLEAN_DIR = SOURCES_DIR / "train" / "clean"
TRAIN_NOISY_DIR = SOURCES_DIR / "train" / "noisy"

TEST_CLEAN_DIR = SOURCES_DIR / "test" / "clean"
TEST_NOISY_DIR = SOURCES_DIR / "test" / "noisy"

MUSAN_NOISE_DIR = ROOT / "datasets" / "anc" / "musan" / "noise"


# Optional override
if os.environ.get("SHA_SOURCES_DIR"):
    SOURCES_DIR = Path(os.environ["SHA_SOURCES_DIR"])
    TRAIN_CLEAN_DIR = SOURCES_DIR / "train" / "clean"
    TRAIN_NOISY_DIR = SOURCES_DIR / "train" / "noisy"
    TEST_CLEAN_DIR = SOURCES_DIR / "test" / "clean"
    TEST_NOISY_DIR = SOURCES_DIR / "test" / "noisy"

if os.environ.get("SHA_MUSAN_NOISE_DIR"):
    MUSAN_NOISE_DIR = Path(os.environ["SHA_MUSAN_NOISE_DIR"])


# ============================================================
# MUSAN AUGMENTATION (REDUCED FOR SPEED)
# ============================================================

# Lower ratio = faster because fewer mixed samples
MUSAN_MIX_RATIO = 0.1  # Reduced from 0.25

SNR_MIN_DB = -5
SNR_MAX_DB = 15


# ============================================================
# FEDERATED PARTICIPANTS
# ============================================================

DEFAULT_WORKERS = 2
NUM_PARTICIPANTS = DEFAULT_WORKERS + 1


# ============================================================
# NETWORK
# ============================================================

HOST_IP = "170.20.10.2"
DEFAULT_PORT = 8080


# ============================================================
# NETWORK TIMEOUTS
# ============================================================

WORKER_SOCKET_TIMEOUT = 7200  # 2 hours
READY_TIMEOUT = 600  # 10 minutes
ROUND_TIMEOUT = 7200  # 2 hours


# ============================================================
# TRAINING - OPTIMIZED FOR 2-HOUR RUN
# ============================================================

# Larger effective batch size via gradient accumulation
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 32

# Maximum epochs (early stopping will end earlier)
EPOCHS = 50  # Reduced from 150

# Higher learning rate for faster convergence
LEARNING_RATE = 5e-4  # Slightly higher than 3e-4

# DataLoader workers (more = faster loading)
DATALOADER_WORKERS = 4  # Increased from 2


# ============================================================
# VALIDATION
# ============================================================

# Smaller validation set = faster validation
VALIDATION_RATIO = 0.05  # Reduced from 0.10


# ============================================================
# EARLY STOPPING - MORE AGGRESSIVE
# ============================================================

# Less strict improvement threshold
MIN_IMPROVEMENT = 0.001  # Increased from 0.0005

# Fewer patience rounds = early stop sooner
PATIENCE = 5  # Reduced from 10


# ============================================================
# GLOBAL EARLY STOPPING
# ============================================================

GLOBAL_MIN_IMPROVEMENT = 0.001  # Increased
GLOBAL_PATIENCE = 5  # Reduced from 10


# ============================================================
# CHECKPOINTS
# ============================================================

CHECKPOINT_DIR = ROOT / "checkpoints" / "anc"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
GLOBAL_MODEL_PATH = CHECKPOINT_DIR / "sha_anc_global.pt"


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42


# ============================================================
# 2-HOUR TRAINING PRESET
# ============================================================

FAST_TRAINING = {
    'max_samples': 3000,  # Sweet spot: enough data, not too slow
    'epochs': 50,
    'batch_size': 8,
    'grad_accum_steps': 4,
    'segment_seconds': 2,
    'timeout': 7200,
    'dataloader_workers': 4,
    'early_stop_patience': 5,
}