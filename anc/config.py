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

# Longer segments give the model more temporal context to learn
# from, which tends to help quality — at the cost of more memory
# per sample. 3s is a common sweet spot for speech enhancement;
# drop back to 2 if you hit OOM.
SEGMENT_SECONDS = 3

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
#         ├── raw/
#         │   ├── train/
#         │   │     ├── clean/
#         │   │     └── noisy/
#         │   └── test/
#         │         ├── clean/
#         │         └── noisy/
#         └── musan/
#             └── noise/          <-- MUSAN noise-only clips (no
#                                     speech/music), used ONLY for
#                                     training-time augmentation.
#
# The data remains local to each participant.
# ============================================================

SOURCES_DIR = (
    ROOT
    / "datasets"
    / "anc"
    / "raw"
)

TRAIN_CLEAN_DIR = SOURCES_DIR / "train" / "clean"
TRAIN_NOISY_DIR = SOURCES_DIR / "train" / "noisy"

TEST_CLEAN_DIR = SOURCES_DIR / "test" / "clean"
TEST_NOISY_DIR = SOURCES_DIR / "test" / "noisy"

MUSAN_NOISE_DIR = (
    ROOT
    / "datasets"
    / "anc"
    / "musan"
    / "noise"
)


# ------------------------------------------------------------
# Optional override
# ------------------------------------------------------------

if os.environ.get("SHA_SOURCES_DIR"):

    SOURCES_DIR = Path(
        os.environ["SHA_SOURCES_DIR"]
    )

    TRAIN_CLEAN_DIR = SOURCES_DIR / "train" / "clean"
    TRAIN_NOISY_DIR = SOURCES_DIR / "train" / "noisy"

    TEST_CLEAN_DIR = SOURCES_DIR / "test" / "clean"
    TEST_NOISY_DIR = SOURCES_DIR / "test" / "noisy"

if os.environ.get("SHA_MUSAN_NOISE_DIR"):

    MUSAN_NOISE_DIR = Path(
        os.environ["SHA_MUSAN_NOISE_DIR"]
    )


# ============================================================
# MUSAN AUGMENTATION (TRAINING SET ONLY)
#
# A fraction of TRAINING samples are replaced with a synthetic
# mix: real clean speech + a random MUSAN noise clip, mixed at a
# random SNR. This widens the variety of noise the model sees
# beyond DEMAND's ~10 recorded environments (cafe, street, car,
# etc.) to include transient/tonal/non-stationary sounds (bells,
# alarms, technical noise).
#
# Kept deliberately as a MINORITY of training samples so real
# recorded noisy/clean pairs remain the dominant signal — this
# is augmentation, not a replacement dataset.
#
# Validation and test sets are NEVER augmented with MUSAN, so
# validation_loss / test-set metrics stay directly comparable to
# runs without this feature.
# ============================================================

# Fraction of TRAINING samples (per epoch) that use a synthetic
# MUSAN mix instead of the real paired noisy file. 0.0 disables
# MUSAN augmentation entirely.
MUSAN_MIX_RATIO = 0.25

SNR_MIN_DB = -5
SNR_MAX_DB = 15


# ============================================================
# FEDERATED PARTICIPANTS
# ============================================================

DEFAULT_WORKERS = 2

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

# Smaller batches add gradient noise, which often acts as a mild
# regularizer and generalizes better than very large batches —
# especially valuable with federated averaging across participants.
# Raise back to 16/32 only if training is unstable or too slow.
BATCH_SIZE = 8

# Ceiling only — GLOBAL_PATIENCE/PATIENCE below will stop training
# once validation loss plateaus, so a high ceiling is safe and lets
# the model use as many rounds as it actually needs.
EPOCHS = 150

# 1e-3 is aggressive for audio regression models and can cause
# noisy/unstable convergence, capping final quality. 3e-4 is a
# steadier default for this kind of task.
LEARNING_RATE = 3e-4


# ------------------------------------------------------------
# DataLoader workers
#
# Windows:
# Start conservatively with 2.
# ------------------------------------------------------------

DATALOADER_WORKERS = 2


# ============================================================
# VALIDATION
# ============================================================

# With ~11.5k train pairs, 10% (~1,150 samples) is already a
# stable validation estimate, and frees up more data for training.
VALIDATION_RATIO = 0.10


# ============================================================
# PARTICIPANT EARLY STOPPING
#
# MIN_IMPROVEMENT is a LOSS delta (how much validation loss must
# drop to count as "improved"). PATIENCE is a ROUND COUNT (how
# many rounds without sufficient improvement before a participant
# stops training). Keep these separate — do not use one where the
# other is expected.
# ============================================================

MIN_IMPROVEMENT = 0.0005

PATIENCE = 10


# ============================================================
# GLOBAL EARLY STOPPING
# ============================================================

GLOBAL_MIN_IMPROVEMENT = 0.0005

GLOBAL_PATIENCE = 10


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