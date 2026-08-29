
import os
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

SEGMENT_SECONDS = 2

SEGMENT_SAMPLES = (
    SAMPLE_RATE
    * SEGMENT_SECONDS
)


# ============================================================
# LOCAL DATASET
#
# EVERY PARTICIPANT HAS ITS OWN:
#
# sources/
# ├── librispeech/
# └── musan/
#
# No participant reads another participant's dataset.
# ============================================================

SOURCES_DIR = ROOT / "sources"

LIBRISPEECH_DIR = (
    SOURCES_DIR / "librispeech"
)

MUSAN_DIR = (
    SOURCES_DIR / "musan"
)


# ------------------------------------------------------------
# Optional environment overrides
#
# Useful if a particular PC stores sources elsewhere.
# ------------------------------------------------------------

if os.environ.get("SHA_SOURCES_DIR"):

    SOURCES_DIR = Path(
        os.environ["SHA_SOURCES_DIR"]
    )

    LIBRISPEECH_DIR = (
        SOURCES_DIR / "librispeech"
    )

    MUSAN_DIR = (
        SOURCES_DIR / "musan"
    )


# ============================================================
# FEDERATED PARTICIPANTS
# ============================================================

# There are TWO remote workers.
DEFAULT_WORKERS = 2

# Host + Worker 1 + Worker 2
NUM_PARTICIPANTS = (
    DEFAULT_WORKERS + 1
)


# ============================================================
# NETWORK
# ============================================================

HOST_IP = "10.187.224.242"

DEFAULT_PORT = 8080


# ============================================================
# TRAINING
# ============================================================

BATCH_SIZE = 8

EPOCHS = 30

LEARNING_RATE = 1e-3


# ============================================================
# DATA GENERATION
# ============================================================

SNR_MIN_DB = -10

SNR_MAX_DB = 15

DATASET_MULTIPLIER = 8


# ============================================================
# EARLY STOPPING
# ============================================================

# Minimum validation-loss improvement required to reset
# the participant patience counter.

MIN_IMPROVEMENT = 0.0005

# Number of consecutive rounds without sufficient
# improvement before participant convergence.

PATIENCE = 4


# ------------------------------------------------------------
# GLOBAL EARLY STOPPING
# ------------------------------------------------------------

GLOBAL_MIN_IMPROVEMENT = 0.0005

GLOBAL_PATIENCE = 4


# ============================================================
# CHECKPOINTS
# ============================================================

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "anc"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GLOBAL_MODEL_PATH = (
    CHECKPOINT_DIR
    / "sha_anc_global.pt"
)


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

