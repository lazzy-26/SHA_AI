"""
Evaluate a trained S.H.A ANC checkpoint on the held-out test set
(TEST_CLEAN_DIR / TEST_NOISY_DIR). Unlike training/validation, this
runs each pair at FULL LENGTH (no cropping to SEGMENT_SAMPLES) since
the model is fully convolutional and the goal here is a realistic,
representative score -- not a training-shaped one.
"""

import argparse
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from config import (
    GLOBAL_MODEL_PATH,
    SAMPLE_RATE,
    TEST_CLEAN_DIR,
    TEST_NOISY_DIR,
)

from dataset import load_audio, pair_clean_noisy
from model import create_model


# ============================================================
# OPTIONAL METRICS
# ============================================================

try:
    from pesq import pesq as pesq_fn
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi as stoi_fn
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False


def safe_pesq(clean, enhanced, sample_rate):
    """
    PESQ requires 8kHz ('nb') or 16kHz ('wb') and can raise on
    silent/degenerate segments -- treat failures as skippable,
    not fatal, since one bad file shouldn't kill the whole run.
    """
    if sample_rate not in (8000, 16000):
        return None

    mode = "wb" if sample_rate == 16000 else "nb"

    try:
        return float(pesq_fn(sample_rate, clean, enhanced, mode))
    except Exception:
        return None


def safe_stoi(clean, enhanced, sample_rate):
    try:
        return float(stoi_fn(clean, enhanced, sample_rate, extended=False))
    except Exception:
        return None


# ============================================================
# MODEL
# ============================================================

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    model = create_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("average_validation_loss")

    print(f"Model loaded from epoch {epoch}")

    if val_loss is not None:
        print(f"Checkpoint validation loss (held-out TRAIN split): {val_loss:.6f}")
    else:
        print("Checkpoint validation loss: unavailable")

    return model


@torch.no_grad()
def enhance(model, noisy, device):
    input_tensor = (
        torch.from_numpy(noisy)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )  # [1, 1, samples]

    output_tensor = model(input_tensor)

    return output_tensor.squeeze().cpu().numpy()


# ============================================================
# EVALUATION LOOP
# ============================================================

def evaluate(model_path, max_files=None, save_worst_n=0, output_dir=None):

    print("=" * 70)
    print("S.H.A ANC TEST-SET EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Test clean  : {TEST_CLEAN_DIR}")
    print(f"Test noisy  : {TEST_NOISY_DIR}")

    if not TEST_CLEAN_DIR.exists() or not TEST_NOISY_DIR.exists():
        raise RuntimeError(
            "Test directories do not exist. Expected:\n"
            f"  {TEST_CLEAN_DIR}\n"
            f"  {TEST_NOISY_DIR}"
        )

    pairs = pair_clean_noisy(TEST_CLEAN_DIR, TEST_NOISY_DIR)

    if not pairs:
        raise RuntimeError("No matching clean/noisy test pairs found.")

    if max_files is not None and max_files > 0:
        pairs = pairs[:max_files]

    print(f"Test pairs  : {len(pairs)}")

    if not PESQ_AVAILABLE:
        print(
            "\n[NOTE] PESQ not installed -- skipping PESQ scores.\n"
            "       Install with: pip install pesq"
        )

    if not STOI_AVAILABLE:
        print(
            "\n[NOTE] pystoi not installed -- skipping STOI scores.\n"
            "       Install with: pip install pystoi"
        )

    print()
    print(f"Loading model: {model_path}")
    model = load_model(model_path, device)

    criterion = nn.L1Loss()

    l1_losses = []
    pesq_scores = []
    stoi_scores = []

    per_file_results = []

    start_time = time.time()

    for index, (clean_path, noisy_path) in enumerate(pairs, start=1):

        clean = load_audio(clean_path)
        noisy = load_audio(noisy_path)

        # Align lengths WITHOUT cropping to a fixed segment size --
        # full-length evaluation, just trim to the shorter of the
        # (already near-identical) clean/noisy pair.
        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len]
        noisy = noisy[:min_len]

        if min_len == 0:
            print(f"[{index}/{len(pairs)}] SKIPPED (empty audio): {noisy_path.name}")
            continue

        enhanced = enhance(model, noisy, device)

        # Model output length should match input length (fully
        # convolutional, no pooling) -- but align defensively in
        # case of any off-by-one from padding/rounding.
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]

        l1 = float(
            criterion(
                torch.from_numpy(enhanced),
                torch.from_numpy(clean),
            )
        )
        l1_losses.append(l1)

        pesq_score = None
        stoi_score = None

        if PESQ_AVAILABLE:
            pesq_score = safe_pesq(clean, enhanced, SAMPLE_RATE)
            if pesq_score is not None:
                pesq_scores.append(pesq_score)

        if STOI_AVAILABLE:
            stoi_score = safe_stoi(clean, enhanced, SAMPLE_RATE)
            if stoi_score is not None:
                stoi_scores.append(stoi_score)

        per_file_results.append(
            {
                "noisy_path": noisy_path,
                "l1": l1,
                "pesq": pesq_score,
                "stoi": stoi_score,
            }
        )

        if index % 50 == 0 or index == len(pairs):
            print(f"[{index}/{len(pairs)}] processed...")

    elapsed = time.time() - start_time

    # ============================================================
    # SUMMARY
    # ============================================================

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Files evaluated : {len(l1_losses)}")
    print(f"Elapsed time    : {elapsed:.1f}s")

    if l1_losses:
        print(f"\nAverage L1 loss : {np.mean(l1_losses):.6f}")
        print(f"  Min / Max     : {np.min(l1_losses):.6f} / {np.max(l1_losses):.6f}")

    if pesq_scores:
        print(f"\nAverage PESQ    : {np.mean(pesq_scores):.3f}  (higher is better, range ~-0.5 to 4.5)")
        print(f"  Min / Max     : {np.min(pesq_scores):.3f} / {np.max(pesq_scores):.3f}")
        print(f"  Scored        : {len(pesq_scores)}/{len(l1_losses)} files")

    if stoi_scores:
        print(f"\nAverage STOI    : {np.mean(stoi_scores):.3f}  (higher is better, range 0 to 1)")
        print(f"  Min / Max     : {np.min(stoi_scores):.3f} / {np.max(stoi_scores):.3f}")
        print(f"  Scored        : {len(stoi_scores)}/{len(l1_losses)} files")

    print("=" * 70)

    # ============================================================
    # SAVE WORST N FILES FOR LISTENING
    # ============================================================

    if save_worst_n > 0 and per_file_results:

        if output_dir is None:
            output_dir = TEST_NOISY_DIR.parent / "worst_cases"

        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        worst = sorted(per_file_results, key=lambda r: r["l1"], reverse=True)[:save_worst_n]

        print(f"\nSaving {len(worst)} worst-performing enhanced files to:")
        print(f"  {output_dir}")

        for rank, result in enumerate(worst, start=1):
            noisy_path = result["noisy_path"]

            clean = load_audio(
                TEST_CLEAN_DIR / noisy_path.name
                if (TEST_CLEAN_DIR / noisy_path.name).exists()
                else noisy_path  # fallback, shouldn't normally hit
            )
            noisy = load_audio(noisy_path)

            min_len = min(len(clean), len(noisy))
            noisy = noisy[:min_len]

            enhanced = enhance(model, noisy, device)

            out_path = output_dir / f"{rank:02d}_l1_{result['l1']:.4f}_{noisy_path.stem}.wav"
            sf.write(str(out_path), enhanced, SAMPLE_RATE)

        print("Done.")

    return {
        "l1_losses": l1_losses,
        "pesq_scores": pesq_scores,
        "stoi_scores": stoi_scores,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate S.H.A ANC model on the held-out test set."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(GLOBAL_MODEL_PATH),
        help="Path to model checkpoint.",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit evaluation to the first N test pairs (for a quick check).",
    )

    parser.add_argument(
        "--save-worst",
        type=int,
        default=0,
        help="Save the N worst-scoring (highest L1 loss) enhanced files for listening.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save worst-case files (default: datasets/anc/worst_cases).",
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        max_files=args.max_files,
        save_worst_n=args.save_worst,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()