"""
Pre-process all audio files to 16kHz WAV format.
Run this once before training to eliminate resampling overhead.
"""

import os
from pathlib import Path
import soundfile as sf
import numpy as np
from tqdm import tqdm
from scipy.signal import resample_poly
from config import SAMPLE_RATE, TRAIN_CLEAN_DIR, TRAIN_NOISY_DIR

def preprocess_dataset(input_dir, output_dir, target_sr=16000):
    """Pre-process all WAV files in a directory to target sample rate"""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    files = list(input_dir.rglob("*.wav"))
    if not files:
        print(f"No files found in {input_dir}")
        return
    
    print(f"Processing {len(files)} files from {input_dir} -> {output_dir}")
    
    for file in tqdm(files, desc=f"Processing {input_dir.name}"):
        try:
            # Read audio
            audio, sr = sf.read(file)
            
            # Convert to mono if stereo
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != target_sr:
                audio = resample_poly(audio, target_sr, sr)
            
            # Normalize
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak
            
            # Save
            relative_path = file.relative_to(input_dir)
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as float32 WAV
            sf.write(output_path, audio.astype(np.float32), target_sr)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

def main():
    print("=" * 70)
    print("PRE-PROCESSING AUDIO FILES")
    print("=" * 70)
    print(f"Target sample rate: {SAMPLE_RATE} Hz")
    print()
    
    # Pre-process training data
    clean_output = TRAIN_CLEAN_DIR.parent / f"{TRAIN_CLEAN_DIR.name}_16k"
    noisy_output = TRAIN_NOISY_DIR.parent / f"{TRAIN_NOISY_DIR.name}_16k"
    
    print("Processing clean files...")
    preprocess_dataset(TRAIN_CLEAN_DIR, clean_output)
    
    print("\nProcessing noisy files...")
    preprocess_dataset(TRAIN_NOISY_DIR, noisy_output)
    
    print("\n" + "=" * 70)
    print("PRE-PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Clean output: {clean_output}")
    print(f"Noisy output: {noisy_output}")
    print("\nUpdate config.py to point to these directories:")
    print(f"TRAIN_CLEAN_DIR = Path('{clean_output}')")
    print(f"TRAIN_NOISY_DIR = Path('{noisy_output}')")

if __name__ == "__main__":
    main()