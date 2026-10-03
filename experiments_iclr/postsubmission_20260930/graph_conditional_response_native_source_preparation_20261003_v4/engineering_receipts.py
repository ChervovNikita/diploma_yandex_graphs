"""Stdlib receipt persistence for separately authorized synthetic CPU runs."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import tempfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent


class EngineeringReceipt:
    """Fresh external output; atomic snapshots survive ordinary check exceptions.

    A process kill may leave status=running plus the last durable checkpoint.
    This is receipt persistence, not a full numerical dependency certificate or
    representative-data resource measurement. No source packet is ever written.
    """
    def __init__(self, path, authorization, names):
        requested = Path(path).expanduser()
        if not requested.is_absolute():
            raise ValueError("Explicit absolute local receipt path required")
        self.path = requested.resolve()
        self.path.relative_to(PROJECT)
        if not self.path.is_absolute() or self.path.exists():
            raise ValueError("Fresh absolute local receipt path required")
        for ancestor in (self.path.parent, *self.path.parents):
            if ancestor == PROJECT.parent:
                break
            if ancestor == ROOT or (ancestor / "MANIFEST.json").exists():
                raise ValueError("Execution receipt cannot be inside a sealed source/review packet")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {"schema": "durable-synthetic-native-engineering-receipt-v1",
                     "UTC_started": datetime.now(timezone.utc).isoformat(),
                     "packet": ROOT.name, "packet_manifest_sha256": sha256((ROOT / "MANIFEST.json").read_bytes()).hexdigest(),
                     "root_authorization_reference": authorization["root_authorization_reference"],
                     "declared_runtime_dependency_receipt": authorization.get("runtime_dependency_receipt"),
                     "status": "running", "phase": "runtime_gate", "declared_families": list(names),
                     "family_records": [], "Amazon_recipe_qualified": False, "DICE_FoRDE_qualified": False,
                     "PyG2_3_equivalence_qualified": False, "complete_dependency_closure_qualified": False,
                     "representative_resource_qualified": False, "data_or_fit_performed": False,
                     "pilot_launched": False}
        # Exclusive creation prevents silently replacing an existing run receipt.
        with self.path.open("x") as output:
            output.write(json.dumps(self.data, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())

    def persist(self):
        self.data["UTC_updated"] = datetime.now(timezone.utc).isoformat()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=self.path.name + ".",
                                             suffix=".tmp", dir=self.path.parent, delete=False) as output:
                temporary = Path(output.name)
                output.write(json.dumps(self.data, indent=2) + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def family_start(self, name):
        self.data["phase"] = name
        self.data["family_records"].append({"family": name, "status": "running"})
        self.persist()

    def family_progress(self, value):
        record = self.data["family_records"][-1]
        record["engineering_progress"] = value
        self.persist()

    def family_end(self, *, elapsed_seconds, result=None, error=None):
        record = self.data["family_records"][-1]
        record["elapsed_CPU_seconds"] = elapsed_seconds
        record["status"] = "failed" if error else "passed"
        if error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
        else:
            record["engineering_result"] = result
        self.persist()

    def finish(self, *, runtime=None, error=None):
        self.data["status"] = "failed" if error else "passed"
        self.data["UTC_finished"] = datetime.now(timezone.utc).isoformat()
        if runtime is not None:
            self.data["bound_synthetic_runtime"] = runtime
        if error:
            self.data["error"] = {"type": type(error).__name__, "message": str(error)}
        self.persist()
