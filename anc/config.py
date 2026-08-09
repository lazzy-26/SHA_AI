from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SAMPLE_RATE = 16000
SEGMENT_SECONDS = 2
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS

CLEAN_DIR = ROOT / "datasets" / "anc" / "clean"
NOISE_DIR = ROOT / "datasets" / "anc" / "noise"

CHECKPOINT_DIR = ROOT / "checkpoints" / "anc"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "anc_v1_best.pt"
LAST_MODEL_PATH = CHECKPOINT_DIR / "anc_v1_last.pt"

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

SNR_MIN_DB = -10
SNR_MAX_DB = 15

NUM_WORKERS = 0

SEED = 42