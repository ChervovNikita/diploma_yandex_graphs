"""Source/receipt/output binding for the bounded all-layer amendment."""
import hashlib
import json
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent
PACKET = next(p for p in SOURCE_ROOT.parents if p.name == "all_layer_study_v1")
ALL_LAYER_ARM = "all_layer_be_sage_adaptation_prob"


def output_inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PACKET) or path == PACKET or path.is_relative_to(PACKET / "source") or path.is_relative_to(PACKET / "before_amendment"):
        raise ValueError("New outputs must stay in this amendment packet outside preserved/source trees")
    return path


def source_paths(root=SOURCE_ROOT):
    return {str(p.relative_to(root)): p for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and "runtime_classes" not in p.parts and p.suffix != ".pyc"}


def source_hashes(root=SOURCE_ROOT):
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_paths(root).items()}


def gate_receipts(args, sources):
    receipts = {}
    for kind, path in [("port", args.port_gate), ("integration", args.integration_gate)]:
        if path is None:
            raise ValueError("Freeze requires both completed source-port and integration CPU receipts")
        path = Path(path).resolve()
        value = json.loads(path.read_text())
        if value.get("status") not in ("PASS", "passed") or value.get("performance_evidence", False) or value.get("dataset_loads") != 0 or value.get("optimizer_steps") != 0:
            raise ValueError("CPU gate receipt is not an engineering PASS")
        if kind == "integration" and value.get("source_sha256") != sources:
            raise ValueError("Integration CPU gate did not bind this exact amended source")
        if kind == "port":
            runtime = json.loads((SOURCE_ROOT / "closest_port/EXPECTED_RUNTIME.json").read_text())["runtime"]
            provenance = json.loads((SOURCE_ROOT / "amendment_provenance.json").read_text())
            if value.get("runtime") != runtime or sources["closest_port/all_layer_sage_adapter.py"] != provenance["port_adapter_sha256"]:
                raise ValueError("Port gate/runtime/model-source binding differs")
        receipts[kind] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return receipts
