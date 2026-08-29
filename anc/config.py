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

SEGMENT_SAMPLES = (
    SAMPLE_RATE * SEGMENT_SECONDS
)


# ============================================================
# LOCAL DATA SOURCES
#
# EVERY PC MUST HAVE:
#
# SHA_AI/
# └── datasets/
#     └── anc/
#         └── sources/
#             ├── librispeech/
#             └── musan/
#
# The data remains local to each participant.
# ============================================================

SOURCES_DIR = (
    ROOT
    / "datasets"
    / "anc"
    / "sources"
)

LIBRISPEECH_DIR = (
    SOURCES_DIR
    / "librispeech"
)

MUSAN_DIR = (
    SOURCES_DIR
    / "musan"
)


# ------------------------------------------------------------
# Optional override
# ------------------------------------------------------------

if os.environ.get("SHA_SOURCES_DIR"):

    SOURCES_DIR = Path(
        os.environ["SHA_SOURCES_DIR"]
    )

    LIBRISPEECH_DIR = (
        SOURCES_DIR
        / "librispeech"
    )

    MUSAN_DIR = (
        SOURCES_DIR
        / "musan"
    )


# ============================================================
# FEDERATED PARTICIPANTS
# ============================================================

# TWO REMOTE WORKERS
DEFAULT_WORKERS = 2

# HOST + WORKER 1 + WORKER 2
NUM_PARTICIPANTS = (
    DEFAULT_WORKERS + 1
)


# ============================================================
# NETWORK
# ============================================================

HOST_IP = "10.178.216.99"

DEFAULT_PORT = 8080


# ============================================================
# TRAINING
# ============================================================

# Increased from 8 to reduce number of optimizer steps.
BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 1e-3


# ------------------------------------------------------------
# DataLoader workers
#
# Windows:
# Start conservatively with 2.
# ------------------------------------------------------------

DATALOADER_WORKERS = 2


# ============================================================
# DATA GENERATION
# ============================================================

# Reduced from 8 for the initial demonstration.
#
# Change to 4 or 8 later for a larger experiment.
DATASET_MULTIPLIER = 2

SNR_MIN_DB = -10

SNR_MAX_DB = 15


# ============================================================
# VALIDATION
# ============================================================

VALIDATION_RATIO = 0.20


# ============================================================
# PARTICIPANT EARLY STOPPING
# ============================================================

MIN_IMPROVEMENT = 0.0005

PATIENCE = 4


# ============================================================
# GLOBAL EARLY STOPPING
# ============================================================

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