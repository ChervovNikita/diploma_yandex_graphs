"""Tiny finite source consumer; both actions require root-bound receipts."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import integration_contracts as contracts


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(release_path):
    release, cfg, paths, admission, external_owner = contracts.admit(release_path)
    output = Path(release["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    budget = contracts.Budget(release["limits"], external_owner)
    receipt = {"status": "RUNNING", "mode": release["mode"], "VALID_payload_deserializations": 0,
               "source_sha256": {k: v["sha256"] for k, v in release["sources"].items()},
               "safe_TRAIN_sha256": release["roles"]["train"]["sha256"],
               "runtime_fingerprint_sha256": admission["runtime_fingerprint_sha256"],
               "limits": release["limits"], "external_owner": release["external_owner"],
               "historical_cross_runtime_score_reuse": False, "TEST_access": False}
    try:
        budget.check()
        owner = load("pe_exact_F_owner", paths["F_owner"])
        packed = load("pe_sealed_packed", paths["PE_packed"])
        adapter = load("pe_sealed_adapter", paths["PE_adapter"])
        Parent = adapter.family_class(owner, packed)  # Check exact F pin before either action.
        if release["mode"] == "qualify":
            import numpy as np
            import torch
            import torch_geometric
            sys.path.insert(0, cfg["native_repo"])
            from models import Model
            from run_common import set_seed
            if str(torch.__version__) != admission["torch_version"] or str(torch_geometric.__version__) != admission["pyg_version"]:
                raise ValueError("Actual admitted runtime versions differ")
            routes = load("pe_qualification_routes", paths["common_routes"])
            device = torch.device(cfg["device"])
            if device.type != "cuda":
                raise ValueError("Actual authorized CUDA runtime required")
            torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
            with np.load(cfg["train_npz"], allow_pickle=False) as t:
                if set(t.files) != {"x", "edge_index", "ids", "y"}:
                    raise ValueError("Exact safe TRAIN keys required")
                arrays = [t[k].copy() for k in ("x", "edge_index", "ids", "y")]
            if arrays[0].dtype != np.float32 or any(a.dtype != np.int64 for a in arrays[1:]):
                raise ValueError("Original float32 X/int64 edges and compact roles required")
            x, edges, ids, y = [torch.from_numpy(a).to(device) for a in arrays]
            graph = SimpleNamespace(edge_index=edges)
            batch = {"x": x, "edge_index": edges, "graph": graph}
            qualifier = load("pe_full_TRAIN_qualifier", paths["qualifier"])
            receipt.update(qualifier.qualify(torch, routes, packed, Model, set_seed, batch, ids, y, output, budget))
            receipt["actual_graph"] = {"nodes": len(x), "features": x.shape[1], "edge_entries": edges.shape[1], "TRAIN_rows": len(ids)}
            write(output / "QUALIFICATION.json", receipt)
        else:
            class FiniteFamily(Parent):
                def progress(self, member, update):
                    budget.check(getattr(self, "torch", None), getattr(self, "device", None))
                    return super().progress(member, update)
            family = FiniteFamily(cfg, output, release["config"]["sha256"])
            import torch_geometric
            if str(family.torch.__version__) != admission["torch_version"] or str(torch_geometric.__version__) != admission["pyg_version"]:
                raise ValueError("Actual admitted runtime versions differ")
            family.source_hashes.update({str(p): release["sources"][k]["sha256"] for k, p in paths.items()})
            owner.write_json(output / "SOURCE_HASHES.json", {"config_sha256": release["config"]["sha256"], "files": family.source_hashes})
            budget.check(family.torch, family.device)
            family.run()  # Exact sealed PE Family; inherited F fitting unchanged.
            budget.check(family.torch, family.device)
            receipt.update({"status": "COMPLETE", "qualification": release["qualification"],
                            "endpoint_native_closure": release["endpoint_native_closure"],
                            "fresh_anchor_and_paired_reader_admission": "root-owned; no historical cross-runtime F scores",
                            "banks": 3, "optimizer_bundles": 3, "private_member_records": 12,
                            "comparison_interpreted": False})
            receipt.update(budget.costs(family.torch, family.device))
            write(output / "CONSUMER_COMPLETE.json", receipt)
        return receipt
    except BaseException as error:
        receipt.update({"status": "FAIL", "failure": {"type": type(error).__name__, "message": str(error)}})
        receipt["elapsed_seconds"] = time.perf_counter() - budget.started
        progress = output / "QUALIFIER_PROGRESS.json"
        if progress.is_file():
            receipt["qualification_progress"] = json.loads(progress.read_text())
        write(output / ("QUALIFICATION.json" if release["mode"] == "qualify" else "CONSUMER_FAILURE.json"), receipt)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True)
    args = parser.parse_args()
    run(args.release)


if __name__ == "__main__":
    main()
