#!/usr/bin/env python3
"""Operator-run acquisition only. Do not execute on the Mac or import model code."""
import argparse
import ast
import datetime
import hashlib
import json
import math
import platform
from pathlib import Path, PurePosixPath
import shutil
import socket
import tarfile
import time
import urllib.request

RECORD_URL = "https://zenodo.org/api/records/22184581"
FILE_URL = "https://zenodo.org/records/22184581/files/HeaRT.tar.gz"
SIZE = 880240878
MD5 = "d5086c7456ee98892af00ce00816aba1"
NAMES = ("train_pos.txt", "valid_pos.txt", "test_pos.txt",
         "heart_valid_samples.npy", "heart_test_samples.npy", "gnn_feature")
WITHHELD = {"test_pos.txt", "heart_test_samples.npy"}
SELECTED_MAX = 512 * 1024 * 1024
PINNED_REPO = Path("/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git")
PINNED_PHASE = Path("/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
    return {"bytes": path.stat().st_size, "md5": md5.hexdigest(),
            "sha256": sha256.hexdigest()}


def npy_header(path):
    # Header only: no numpy, pickle, torch, array loading or model import.
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError("Not an NPY file: " + path.name)
        version = tuple(stream.read(2))
        width = 2 if version == (1, 0) else 4 if version in {(2, 0), (3, 0)} else 0
        if not width:
            raise ValueError("Unsupported NPY version")
        length = int.from_bytes(stream.read(width), "little")
        if length > 65536:
            raise ValueError("NPY header exceeds bound")
        header = ast.literal_eval(stream.read(length).decode("utf-8" if version == (3, 0) else "latin1").strip())
        shape, dtype = header["shape"], header["descr"]
        if not (isinstance(shape, tuple) and all(isinstance(n, int) and n >= 0 for n in shape)):
            raise ValueError("Invalid NPY shape")
        if not (isinstance(dtype, str) and len(dtype) >= 3 and dtype[0] in "<>=|" and dtype[1] in "iu" and dtype[2:] in {"1", "2", "4", "8"}):
            raise ValueError("Expected integer NPY pool without objects")
        expected_bytes = stream.tell() + math.prod(shape) * int(dtype[2:])
        if expected_bytes != path.stat().st_size:
            raise ValueError("NPY file size does not match header")
    return {"version": version, "shape": shape, "dtype": dtype,
            "fortran_order": header["fortran_order"], "header_payload_size_matches": True}


def split_counts(path):
    nodes, pairs, rows, self_loops, duplicate_rows = set(), set(), 0, 0, 0
    with path.open("rt", encoding="utf-8") as stream:
        for line in stream:
            fields = line.strip().split("\t")
            if len(fields) != 2:
                raise ValueError("Expected two tab-separated endpoints: " + path.name)
            u, v = map(int, fields)
            if min(u, v) < 0:
                raise ValueError("Negative endpoint ID")
            rows += 1
            nodes.update((u, v))
            self_loops += u == v
            pair = tuple(sorted((u, v)))
            duplicate_rows += pair in pairs
            pairs.add(pair)
    return {"raw_rows": rows, "native_nonself_rows": rows - self_loops,
            "self_loops": self_loops, "duplicate_undirected_rows": duplicate_rows,
            "unique_undirected_pairs": len(pairs), "distinct_endpoint_count": len(nodes)}, nodes, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PINNED_REPO)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--confirm-server-18-77", action="store_true", required=True)
    parser.add_argument("--output-relative", default="pubmed_heart_official_acquisition_server_20261004_v2")
    args = parser.parse_args()
    if platform.system() != "Linux" or socket.gethostname() != args.expected_hostname:
        raise SystemExit("Server guard failed; this utility must not run on the Mac.")
    repo = args.repo_root.resolve(strict=True)
    phase = PINNED_PHASE.resolve(strict=True)
    script = Path(__file__).resolve()
    if repo != PINNED_REPO.resolve(strict=True) or not phase.is_relative_to(repo) or not phase.is_dir() or not script.is_relative_to(phase):
        raise SystemExit("Script must be inside the pinned 18.77 repository phase.")
    rel = PurePosixPath(args.output_relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise SystemExit("Output must be a relative path inside the pinned phase.")
    output = phase / str(rel)
    if output.exists() or not output.resolve().is_relative_to(phase):
        raise SystemExit("Use a fresh acquisition destination inside the pinned phase.")
    if shutil.disk_usage(phase).free < SIZE + SELECTED_MAX + 128 * 1024 * 1024:
        raise SystemExit("Insufficient free disk for bounded acquisition.")
    output.mkdir(parents=True)
    started = time.monotonic()
    receipt = {"schema": "pubmed_official_acquisition_receipt_v1", "hostname": socket.gethostname(),
               "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "repo_root": str(repo), "phase_root": str(phase), "official_record_URL": RECORD_URL,
               "official_file_URL": FILE_URL, "scientific_execution": False}
    try:
        with urllib.request.urlopen(RECORD_URL, timeout=60) as response:
            metadata = json.loads(response.read(1024 * 1024))
        write_json(output / "official_record_metadata.json", metadata)
        entries = [f for f in metadata["files"] if f["key"] == "HeaRT.tar.gz"]
        if len(entries) != 1 or entries[0]["size"] != SIZE or entries[0]["checksum"] != "md5:" + MD5:
            raise ValueError("Official metadata changed from the reviewed descriptor")
        custody = output / "withheld_inputs"
        custody.mkdir()
        archive = custody / "HeaRT.tar.gz"
        partial = custody / "HeaRT.tar.gz.partial"
        received = 0
        with urllib.request.urlopen(FILE_URL, timeout=60) as response, partial.open("xb") as dest:
            receipt["file_final_URL"] = response.url
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > SIZE or time.monotonic() - started > 1800:
                    raise ValueError("Download exceeds reviewed byte/time bound")
                dest.write(chunk)
        archive_digest = digest(partial)
        receipt["archive"] = archive_digest
        receipt["official_checksum"] = entries[0]["checksum"]
        if archive_digest["bytes"] != SIZE or archive_digest["md5"] != MD5:
            raise ValueError("Archive size or official MD5 mismatch")
        partial.rename(archive)
        available = output / "available" / "pubmed"
        withheld = custody / "withheld_pubmed"
        available.mkdir(parents=True)
        withheld.mkdir()
        selected, prefixes, traversed, member_count = {}, set(), 0, 0
        with tarfile.open(archive, mode="r|gz") as bundle:
            for member in bundle:
                member_count += 1
                traversed += member.size
                if member_count > 50000 or traversed > 64 * 1024**3 or time.monotonic() - started > 3600:
                    raise ValueError("Archive traversal exceeds bound")
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts:
                    raise ValueError("Unsafe archive member path")
                if len(name.parts) < 3 or name.parts[-3:-1] != ("dataset", "pubmed") or name.name not in NAMES:
                    continue
                if name.name in selected or not member.isfile() or member.size > SELECTED_MAX:
                    raise ValueError("Duplicate, linked or oversized selected member")
                prefixes.add(name.parts[:-3])
                if len(prefixes) != 1 or sum(x["bytes"] for x in selected.values()) + member.size > SELECTED_MAX:
                    raise ValueError("Ambiguous prefix or selected-size bound exceeded")
                dest = (withheld if name.name in WITHHELD else available) / name.name
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError("Unreadable selected member")
                with source, dest.open("xb") as stream:
                    shutil.copyfileobj(source, stream, length=1024 * 1024)
                if dest.stat().st_size != member.size:
                    raise ValueError("Selected member length mismatch")
                selected[name.name] = {"tar_member": member.name, "relative_path": str(dest.relative_to(output)), **digest(dest)}
        if set(selected) != set(NAMES):
            raise ValueError("Required member identities unresolved: " + repr(sorted(set(NAMES) - set(selected))))
        node_union, pair_sets = set(), {}
        for split in ("train", "valid", "test"):
            item = selected[split + "_pos.txt"]
            counts, nodes, pairs = split_counts(output / item["relative_path"])
            item["counts"] = counts
            node_union.update(nodes)
            pair_sets[split] = pairs
        for split in ("valid", "test"):
            item = selected["heart_" + split + "_samples.npy"]
            item["npy_header"] = npy_header(output / item["relative_path"])
            shape = item["npy_header"]["shape"]
            expected = (selected[split + "_pos.txt"]["counts"]["native_nonself_rows"], 500, 2)
            if shape != expected or item["npy_header"]["fortran_order"]:
                raise ValueError("Fixed-negative pool schema/count mismatch: " + split)
        receipt.update({"selected_files": selected, "tar_member_count": member_count,
                        "transductive_endpoint_metadata": {"distinct_count": len(node_union), "min_ID": min(node_union), "max_ID": max(node_union), "dense_zero_based": min(node_union) == 0 and len(node_union) == max(node_union) + 1},
                        "split_pair_overlaps": {"train_valid": len(pair_sets["train"] & pair_sets["valid"]), "train_test": len(pair_sets["train"] & pair_sets["test"]), "valid_test": len(pair_sets["valid"] & pair_sets["test"])},
                        "status": "official_archive_authenticated_selected_pubmed_acquired_not_admitted",
                        "unresolved": ["Feature tensor shape, dtype, key and donor provenance", "Array endpoint ranges and per-query row alignment", "Scientific source/runtime competence"]})
        write_json(custody / "operator_receipt.json", receipt)
        write_json(output / "AVAILABLE_MANIFEST.json", {"files": {n: x for n, x in selected.items() if n not in WITHHELD}, "TEST_available_to_loader": False, "scientific_admission": False})
        print(json.dumps({"status": receipt["status"], "output": str(output), "archive_sha256": archive_digest["sha256"], "official_MD5_verified": True, "selected_files": len(selected), "scientific_admission": False}))
    except Exception as error:
        receipt.update({"status": "acquisition_failed_no_admission", "error": type(error).__name__ + ": " + str(error)})
        write_json(output / "FAILURE_RECEIPT.json", receipt)
        raise


if __name__ == "__main__":
    main()
