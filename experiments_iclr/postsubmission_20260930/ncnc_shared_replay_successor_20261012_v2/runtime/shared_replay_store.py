"""Stdlib immutable replay-store prototype. Not wired into any scientific packet.

The callbacks must be the authenticated source producer and reader. This module
does not generate samples, invoke Torch, mutate RNG or change optimizer work.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def stream_key(*, input_receipts, source_receipts, geometry, master_seed, runtime):
    """Fit seed, learning rate, arm and optimizer are intentionally absent.

    Runtime is compulsory: source hash equality alone does not admit a store.
    """
    if type(master_seed) is not int or not 0 <= master_seed < 2**63:
        raise ValueError("Invalid source replay master seed")
    if set(input_receipts) != {"graph", "train", "select"}:
        raise ValueError("Exact source input receipt schema required")
    if not source_receipts or not geometry or not runtime:
        raise ValueError("Whole source, TRAIN geometry and runtime key required")
    forbidden = {"heldout_targets", "test_targets", "evaluate_targets"}
    if forbidden.intersection(input_receipts):
        raise ValueError("No heldout target authority belongs in this store")
    return {
        "schema": "prospective-shared-source-replay-key-v1",
        "input_receipts": input_receipts,
        "source_receipts": source_receipts,
        "TRAIN_geometry": geometry,
        "source_master_seed": master_seed,
        "runtime": runtime,
        "dtype": "little-endian-int64",
        "epochs": 100,
    }


def file_pin(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def locked(path):
    with Path(path).open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class SharedReplayStore:
    def __init__(self, root, key):
        self.root = Path(root).resolve()
        self.key = key
        self.key_hash = digest(key)
        self.store = self.root / self.key_hash
        self.store.mkdir(parents=True, exist_ok=True)
        with locked(self.store / "STORE.lock"):
            manifest = self.store / "STORE.json"
            if manifest.exists():
                if json.loads(manifest.read_text()) != key:
                    raise ValueError("Replay store key changed")
            else:
                atomic_json(manifest, key)

    def ensure_epoch(self, epoch, producer, verifier):
        """Generate once, then authenticate every reader; failed attempts stay.

        producer(stage) writes exact source epoch files at a fresh directory.
        verifier(path) must run the source EpochReplay constructor and its
        source/data/RNG contract check; an integer receipt is not enough.
        """
        if type(epoch) is not int or not 1 <= epoch <= 100:
            raise ValueError("Source epoch outside 1..100")
        selected = self.store / ("epoch_" + str(epoch))
        publication = self.store / ("PUBLISHED_" + str(epoch) + ".json")
        with locked(self.store / ("epoch_" + str(epoch) + ".lock")):
            if not selected.exists():
                if publication.exists():
                    raise ValueError("A published replay authority is missing")
                stage = self.store / ("attempt_" + str(epoch) + "_" + uuid.uuid4().hex)
                try:
                    # The source write_epoch requires a nonexistent stage.
                    producer(stage)
                    verifier(stage)
                    files = sorted(p for p in stage.rglob("*") if p.is_file())
                    if any(p.is_symlink() for p in stage.rglob("*")):
                        raise ValueError("Replay authorities cannot be symlinks")
                    if {p.name for p in files} != {"EPOCH.json", "PERMUTATION.npy", "NEGATIVE_PAIRS.npy"}:
                        raise ValueError("Exact source epoch authority required")
                    manifest = {
                        "schema": "prospective-shared-replay-epoch-manifest-v1",
                        "key_hash": self.key_hash,
                        "epoch": epoch,
                        "files": {str(p.relative_to(stage)): file_pin(p) for p in files},
                        "source_verifier_passed": True,
                    }
                    atomic_json(stage / "SHARED_MANIFEST.json", manifest)
                    stage.rename(selected)
                    atomic_json(publication, {"key_hash": self.key_hash, "epoch": epoch,
                                              "manifest": file_pin(selected / "SHARED_MANIFEST.json")})
                except BaseException as error:
                    atomic_json(self.store / ("FAILED_" + stage.name + ".json"), {
                        "epoch": epoch, "attempt": stage.name,
                        "error_type": type(error).__name__,
                        "error": str(error), "failed_attempt_preserved": True,
                    })
                    raise
            if not publication.exists():
                raise ValueError("Uncommitted replay output requires explicit root recovery")
            authority = json.loads(publication.read_text())
            if authority != {"key_hash": self.key_hash, "epoch": epoch,
                             "manifest": file_pin(selected / "SHARED_MANIFEST.json")}:
                raise ValueError("Published replay manifest changed")
            manifest = json.loads((selected / "SHARED_MANIFEST.json").read_text())
            if manifest["key_hash"] != self.key_hash or manifest["epoch"] != epoch:
                raise ValueError("Shared epoch belongs to another source authority")
            for relative, pin in manifest["files"].items():
                path = selected / relative
                if path.is_symlink() or not path.resolve().is_relative_to(selected) or file_pin(path) != pin:
                    raise ValueError("Shared replay file corrupted")
            verifier(selected)
            return selected, file_pin(selected / "SHARED_MANIFEST.json")

    def bind_consumer(self, consumer, epoch, epoch_directory, manifest_pin):
        """Immutable per-fit pointer; no release/deletion is implemented here."""
        if not consumer or not consumer.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Safe explicit prospective consumer identity required")
        epoch_directory = Path(epoch_directory).resolve()
        if epoch_directory != self.store / ("epoch_" + str(epoch)):
            raise ValueError("Consumer cannot bind an unrelated epoch")
        if file_pin(epoch_directory / "SHARED_MANIFEST.json") != manifest_pin:
            raise ValueError("Consumer epoch receipt changed")
        root = self.store / "consumer_leases"
        root.mkdir(exist_ok=True)
        path = root / (consumer + "_epoch_" + str(epoch) + ".json")
        record = {"consumer": consumer, "epoch": epoch, "key_hash": self.key_hash,
                  "manifest": manifest_pin, "paid_consumer_or_retry_open": True,
                  "release_implemented": False}
        with locked(root / (consumer + ".lock")):
            if path.exists():
                if json.loads(path.read_text()) != record:
                    raise ValueError("Changed immutable consumer binding")
            else:
                atomic_json(path, record)
        return path
