#!/usr/bin/env python
"""
Pre-generate the mixed dataset once to avoid on-the-fly mixing
during federated training.
"""

import argparse
from pathlib import Path

from config import (
    LIBRISPEECH_DIR,
    MUSAN_DIR,
    PREMIXED_DIR,
    SEED,
    VALIDATION_RATIO,
)
from dataset import (
    find_audio_files,
    generate_premixed_dataset,
    split_files,
)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate mixed audio dataset for faster training"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PREMIXED_DIR),
        help=f"Output directory for pre-mixed dataset (default: {PREMIXED_DIR})",
    )
    
    parser.add_argument(
        "--num-train",
        type=int,
        default=50000,
        help="Number of training samples to generate (default: 50000)",
    )
    
    parser.add_argument(
        "--num-val",
        type=int,
        default=10000,
        help="Number of validation samples to generate (default: 10000)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("=" * 70)
    print("PRE-GENERATING MIXED DATASET")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Training samples: {args.num_train}")
    print(f"Validation samples: {args.num_val}")
    print(f"Seed: {args.seed}")
    print("=" * 70)
    
    # Discover files
    print("\nDiscovering audio files...")
    clean_files = find_audio_files(LIBRISPEECH_DIR)
    noise_files = find_audio_files(MUSAN_DIR)
    
    print(f"Found {len(clean_files)} clean files (LibriSpeech)")
    print(f"Found {len(noise_files)} noise files (MUSAN)")
    
    if not clean_files:
        print("ERROR: No clean files found! Please check LIBRISPEECH_DIR path.")
        return
    
    if not noise_files:
        print("ERROR: No noise files found! Please check MUSAN_DIR path.")
        return
    
    # Split into train/val
    print("\nSplitting clean files into train/validation sets...")
    train_clean, val_clean = split_files(clean_files, validation_ratio=VALIDATION_RATIO, seed=args.seed)
    
    print(f"Train clean files: {len(train_clean)}")
    print(f"Val clean files: {len(val_clean)}")
    
    # Generate training set
    print("\n" + "=" * 70)
    print("GENERATING TRAINING SET")
    print("=" * 70)
    train_dir = output_dir / "train"
    generate_premixed_dataset(
        output_dir=train_dir,
        clean_files=train_clean,
        noise_files=noise_files,
        num_samples=args.num_train,
        seed=args.seed,
    )
    
    # Generate validation set
    print("\n" + "=" * 70)
    print("GENERATING VALIDATION SET")
    print("=" * 70)
    val_dir = output_dir / "val"
    generate_premixed_dataset(
        output_dir=val_dir,
        clean_files=val_clean,
        noise_files=noise_files,
        num_samples=args.num_val,
        seed=args.seed + 1000,
    )
    
    print()
    print("=" * 70)
    print("DATASET GENERATION COMPLETE! ✅")
    print("=" * 70)
    print(f"Training samples: {args.num_train}")
    print(f"Validation samples: {args.num_val}")
    print(f"Output directory: {output_dir}")
    print()
    print("You can now run the federated training with:")
    print(f"  python server.py --num-workers 4 --port 8080")
    print(f"  python worker.py --server 10.178.216.99 --id 1 --num-workers 4")
    print("=" * 70)


if __name__ == "__main__":
    main()