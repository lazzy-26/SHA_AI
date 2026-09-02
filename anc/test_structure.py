"""
Diagnostic script to find your dataset structure
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets" / "anc" / "raw"

print("=" * 70)
print("DATASET STRUCTURE DIAGNOSTIC")
print("=" * 70)

print(f"\nChecking: {DATASET_ROOT}")
print(f"Exists: {DATASET_ROOT.exists()}")

if DATASET_ROOT.exists():
    print("\nContents of raw/:")
    for item in sorted(DATASET_ROOT.iterdir()):
        if item.is_dir():
            print(f"  📁 {item.name}/")
            # Check one level deeper
            for subitem in sorted(item.iterdir()):
                if subitem.is_dir():
                    print(f"      📁 {subitem.name}/")
                    # Count files
                    wav_files = list(subitem.glob("*.wav"))
                    if wav_files:
                        print(f"         {len(wav_files)} .wav files")
        else:
            print(f"  📄 {item.name}")

print("\n" + "=" * 70)