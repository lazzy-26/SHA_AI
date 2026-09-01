"""
Quick script to verify VoiceBank-DEMAND dataset
"""

from config import TRAIN_CLEAN_DIR, TRAIN_NOISY_DIR, TEST_CLEAN_DIR, TEST_NOISY_DIR
from dataset import VoiceBankDataset

print("=" * 70)
print("VOICEBANK-DEMAND DATASET VERIFICATION")
print("=" * 70)

print(f"\nTrain clean: {TRAIN_CLEAN_DIR}")
print(f"Train noisy: {TRAIN_NOISY_DIR}")
print(f"Test clean : {TEST_CLEAN_DIR}")
print(f"Test noisy : {TEST_NOISY_DIR}")

# Check directories
print(f"\nTrain clean exists: {TRAIN_CLEAN_DIR.exists()}")
print(f"Train noisy exists: {TRAIN_NOISY_DIR.exists()}")
print(f"Test clean exists : {TEST_CLEAN_DIR.exists()}")
print(f"Test noisy exists : {TEST_NOISY_DIR.exists()}")

# Load training dataset
try:
    train_dataset = VoiceBankDataset(
        clean_dir=TRAIN_CLEAN_DIR,
        noisy_dir=TRAIN_NOISY_DIR,
        segment_samples=32000,  # 2 seconds at 16kHz
        deterministic=False,
    )
    print(f"\n✅ Training dataset: {len(train_dataset)} samples")
    
    # Sample one item
    noisy, clean = train_dataset[0]
    print(f"   Sample shape: noisy={noisy.shape}, clean={clean.shape}")
    
except Exception as e:
    print(f"\n❌ Error loading training dataset: {e}")

# Load test dataset
try:
    test_dataset = VoiceBankDataset(
        clean_dir=TEST_CLEAN_DIR,
        noisy_dir=TEST_NOISY_DIR,
        segment_samples=32000,
        deterministic=True,
    )
    print(f"\n✅ Test dataset: {len(test_dataset)} samples")
    
except Exception as e:
    print(f"\n⚠️ Test dataset not available: {e}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)