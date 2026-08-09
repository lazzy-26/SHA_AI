from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SAMPLE_RATE = 16000
SEGMENT_SECONDS = 2
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS

CLEAN_DIR = ROOT / "datasets" / "anc" / "clean"
NOISE_DIR = ROOT / "datasets" / "anc" / "noise"

CHECKPOINT_DIR = ROOT / "checkpoints" / "anc"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_MODEL_PATH = CHECKPOINT_DIR / "sha_anc_global.pt"

DEFAULT_PORT = 8080
DEFAULT_WORKERS = 3

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-3

SNR_MIN_DB = -10
SNR_MAX_DB = 15

SEED = 42