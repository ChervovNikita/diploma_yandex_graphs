#!/usr/bin/env python3
"""Companion to the staged planner: hash original split indices, never decode."""
import argparse
import hashlib
import json
from pathlib import Path

PHASE = "experiments_iclr/postsubmission_20260930"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-root", required=True, type=Path)
    parser.add_argument("--provider-request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    files = {}
    for rel in json.loads(args.provider_request.read_text())["repository_relative_directories"]:
        if not rel.startswith(PHASE + "/") or ".." in Path(rel).parts:
            raise ValueError("Noncanonical provider path")
        directory = args.phase_root / rel.removeprefix(PHASE + "/")
        candidates = list((directory / "split").rglob("*")) if (directory / "split").is_dir() else []
        candidates.append(directory / "split_dict.pt")
        for path in candidates:
            if path.is_file() and not path.is_symlink():
                h = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(chunk)
                target = PHASE + "/" + path.relative_to(args.phase_root).as_posix()
                files[target] = dict(source_relative=target, target_relative=target,
                                     bytes=path.stat().st_size, sha256=h.hexdigest())
    result = {"files": [files[k] for k in sorted(files)], "payloads_decoded": False,
              "TEST_forwards_or_scores": False, "file_count": len(files),
              "bytes": sum(v["bytes"] for v in files.values()),
              "scope": "Original provider split metadata including TEST indices, preferred PT shadows, and split_dict.pt; no result artifacts"}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, sort_keys=True))


if __name__ == "__main__":
    main()
