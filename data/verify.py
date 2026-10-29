"""Verify the immutable plotting archive; requires only Python's standard library."""

from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text())
for name, item in manifest["files"].items():
    path = root / name
    if not path.is_file():
        raise SystemExit(f"Missing data file: {name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise SystemExit(f"Checksum mismatch: {name}")
print(
    f"Verified {len(manifest['figures'])} figures and {len(manifest['files'])} data files."
)
