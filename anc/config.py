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
# LOCAL DATA SOURCES
# ============================================================

SOURCES_DIR = ROOT / "datasets" / "anc" / "sources"

LIBRISPEECH_DIR = SOURCES_DIR / "librispeech"
MUSAN_DIR = SOURCES_DIR / "musan"


# Optional override
if os.environ.get("SHA_SOURCES_DIR"):
    SOURCES_DIR = Path(os.environ["SHA_SOURCES_DIR"])

    LIBRISPEECH_DIR = SOURCES_DIR / "librispeech"
    MUSAN_DIR = SOURCES_DIR / "musan"


# ============================================================
# FEDERATED PARTICIPANTS
# ============================================================

DEFAULT_WORKERS = 2

# HOST + Worker 1 + Worker 2 = 3
NUM_PARTICIPANTS = DEFAULT_WORKERS + 1


# ============================================================
# NETWORK
# ============================================================

HOST_IP = "170.20.10.2"

DEFAULT_PORT = 8080


# ============================================================
# NETWORK TIMEOUTS
# ============================================================

# Dataset preparation can take some time
READY_TIMEOUT = 1800       # 30 minutes

# Maximum time allowed for a federated round
ROUND_TIMEOUT = 3600       # 1 hour

# Persistent worker sockets stay open
WORKER_SOCKET_TIMEOUT = 0


# ============================================================
# TRAINING
# ============================================================

# Larger batch = fewer optimizer iterations
BATCH_SIZE = 32

# Effective batch size = 32 * 2 = 64
GRADIENT_ACCUMULATION_STEPS = 2

# Fewer epochs so training does not run unnecessarily long
EPOCHS = 15

LEARNING_RATE = 5e-4


# ============================================================
# DATA AUGMENTATION
# ============================================================

SNR_MIN_DB = -5

SNR_MAX_DB = 15

# Previously: 8
# Lower this significantly to reduce preprocessing/training time.
DATASET_MULTIPLIER = 3


# ============================================================
# DATALOADER
# ============================================================

# Increase this if your PC has multiple CPU cores.
# Start with 2; if CPU usage is low, try 4.
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

GLOBAL_MODEL_PATH = CHECKPOINT_DIR / "sha_anc_global.pt"


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
# 1-HOUR TRAINING PRESET
# ============================================================

FAST_TRAINING = {
    # Limit the number of training samples
    "max_samples": 5000,

    # Maximum number of epochs
    "epochs": 15,

    # Larger batch for faster training
    "batch_size": 32,

    # Effective batch = 32 * 2 = 64
    "grad_accum_steps": 2,

    # Keep 2-second audio segments
    "segment_seconds": 2,

    # Maximum training/round time
    "timeout": 3600,

    # Parallel audio loading
    "dataloader_workers": 2,

    # Stop earlier if model stops improving
    "early_stop_patience": 3,
}