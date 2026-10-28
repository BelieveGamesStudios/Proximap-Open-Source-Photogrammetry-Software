"""Validate the bundled offline model before inference or release packaging."""

import hashlib
from pathlib import Path


SILUETA_SIZE = 44173029
SILUETA_SHA256 = "75da6c8d2f8096ec743d071951be73b4a8bc7b3e51d9a6625d63644f90ffeedb"


def validate_silueta(model_path):
    """Reject missing files, LFS pointers, and damaged model binaries."""
    path = Path(model_path)
    try:
        with path.open("rb") as model:
            header = model.read(256)
            if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(
                    f"{path} is a Git LFS pointer, not the Silueta ONNX model. "
                    "Install Git LFS and run 'git lfs pull' in the source checkout, "
                    "then rebuild/reinstall Proximap."
                )
            if path.stat().st_size != SILUETA_SIZE:
                raise ValueError(
                    f"{path} is incomplete or invalid (expected {SILUETA_SIZE} bytes). "
                    "Restore the model and rebuild/reinstall Proximap."
                )
            digest = hashlib.sha256(header)
            for chunk in iter(lambda: model.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(
            f"Cannot read the offline background-removal model at {path}. "
            "Restore the model and rebuild/reinstall Proximap."
        ) from exc
    if digest.hexdigest() != SILUETA_SHA256:
        raise ValueError(
            f"{path} failed its SHA-256 integrity check. "
            "Restore the model and rebuild/reinstall Proximap."
        )
    return path


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "models" / "silueta.onnx"
    try:
        validate_silueta(path)
    except ValueError as exc:
        sys.exit(f"[ERROR] {exc}")
    print(f"[INFO] Verified offline Silueta model: {path}")
