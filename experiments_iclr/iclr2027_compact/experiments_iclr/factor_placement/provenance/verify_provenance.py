#!/usr/bin/env python3
"""Check byte hashes of the additional factor-placement provenance files."""

import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(__file__).parent
    manifest = json.loads((root / "PROVENANCE_MANIFEST.json").read_text())
    assert manifest["schema"] == "factor-placement-supplemental-provenance-v1"
    seen = set()
    for record in manifest["files"]:
        rel = Path(record["path"])
        assert not rel.is_absolute() and ".." not in rel.parts
        assert str(rel) not in seen
        seen.add(str(rel))
        data = (root / rel).read_bytes()
        assert len(data) == record["size_bytes"], rel
        assert hashlib.sha256(data).hexdigest() == record["sha256"], rel
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    assert actual == seen | {"PROVENANCE_MANIFEST.json", "verify_provenance.py"}, actual ^ seen
    print(f"PASS: {len(seen)} supplemental provenance files match SHA-256")


if __name__ == "__main__":
    main()
