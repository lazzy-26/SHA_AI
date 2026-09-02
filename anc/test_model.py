import torch
import numpy as np
import soundfile as sf
from pathlib import Path

from model import create_model
from config import SAMPLE_RATE, GLOBAL_MODEL_PATH


def load_audio_for_test(audio_path):
    """
    Load and prepare audio for testing.

    Unlike training, inference does NOT truncate or pad to a fixed
    SEGMENT_SAMPLES length -- the model is fully convolutional
    (causal, no fixed-size layers), so it can process audio of any
    length. Forcing a fixed length here would silently discard the
    tail of longer files.
    """
    audio, sr = sf.read(audio_path)

    # Convert to mono if stereo
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, SAMPLE_RATE, sr)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio.astype(np.float32)


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
        print(f"Best validation loss: {val_loss:.6f}")
    else:
        print("Best validation loss: unavailable in checkpoint")

    return model


def enhance_audio(model, audio, device):
    with torch.no_grad():
        input_tensor = (
            torch.from_numpy(audio)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )  # [1, 1, samples]

        output_tensor = model(input_tensor)
        enhanced = output_tensor.squeeze().cpu().numpy()

    return enhanced


def test_model(model_path, audio_path, output_path=None):
    """Test the trained model on a single audio file"""

    print("=" * 70)
    print("S.H.A ANC MODEL TEST")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # 1. Load model
    print(f"\nLoading model from: {model_path}")
    model = load_model(model_path, device)

    # 2. Load audio
    print(f"\nLoading audio: {audio_path}")
    audio = load_audio_for_test(audio_path)

    # 3. Process through model
    print("\nProcessing audio through model...")
    enhanced = enhance_audio(model, audio, device)

    # 4. Save output
    audio_path = Path(audio_path)

    if output_path is None:
        output_path = audio_path.with_name(
            f"{audio_path.stem}_enhanced.wav"
        )
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(output_path), enhanced, SAMPLE_RATE)
    print(f"\nEnhanced audio saved to: {output_path}")

    # 5. Print stats
    print("\n" + "-" * 70)
    print("STATISTICS")
    print("-" * 70)
    print(f"Input audio length  : {len(audio)} samples ({len(audio)/SAMPLE_RATE:.2f}s)")
    print(f"Output audio length : {len(enhanced)} samples ({len(enhanced)/SAMPLE_RATE:.2f}s)")
    print(f"Input peak          : {np.max(np.abs(audio)):.4f}")
    print(f"Output peak         : {np.max(np.abs(enhanced)):.4f}")
    print(f"Input RMS           : {np.sqrt(np.mean(audio**2)):.4f}")
    print(f"Output RMS          : {np.sqrt(np.mean(enhanced**2)):.4f}")
    print("-" * 70)

    return enhanced


def test_batch(model_path, audio_dir, output_dir=None):
    """Test the model on all audio files in a directory"""

    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        print(f"Directory not found: {audio_dir}")
        return

    if output_dir is None:
        output_dir = audio_dir / "enhanced"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find audio files
    audio_files = sorted(
        list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.flac"))
    )
    print(f"Found {len(audio_files)} audio files")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(model_path, device)

    for i, audio_path in enumerate(audio_files):
        try:
            print(f"\n[{i+1}/{len(audio_files)}] Processing: {audio_path.name}")

            audio = load_audio_for_test(audio_path)
            enhanced = enhance_audio(model, audio, device)

            output_path = output_dir / f"{audio_path.stem}_enhanced.wav"
            sf.write(str(output_path), enhanced, SAMPLE_RATE)
            print(f"  Saved: {output_path.name}")

        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nDone! Enhanced files saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test S.H.A ANC model")
    parser.add_argument(
        "--model",
        type=str,
        default=str(GLOBAL_MODEL_PATH),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file or directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (for single file) or directory (for batch)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all audio files in the directory",
    )

    args = parser.parse_args()

    if args.batch:
        test_batch(args.model, args.audio, args.output)
    else:
        test_model(args.model, args.audio, args.output)