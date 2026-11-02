"""Verify example assets without changing the manuscript-data manifest."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "data/examples/manifest.json").read_text())
for name, record in manifest["files"].items():
    path = root / name
    if not path.is_file():
        raise SystemExit(f"Missing example asset: {name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise SystemExit(f"Example asset checksum mismatch: {name}")
print(f"Verified {len(manifest['files'])} example assets.")
