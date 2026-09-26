"""The vendored pgsg_1 snapshot must match its SHA-256 manifest byte for byte."""
import hashlib
import json
from pathlib import Path

import pytest

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "pgsg_1"


@pytest.mark.skipif(not (VENDOR / "MANIFEST.json").is_file(), reason="vendor/pgsg_1 not created")
def test_vendored_files_match_manifest():
    files = json.loads((VENDOR / "MANIFEST.json").read_text())["files"]
    assert "pgsg_v2.py" in files
    for rel, digest in files.items():
        assert hashlib.sha256((VENDOR / rel).read_bytes()).hexdigest() == digest, rel
