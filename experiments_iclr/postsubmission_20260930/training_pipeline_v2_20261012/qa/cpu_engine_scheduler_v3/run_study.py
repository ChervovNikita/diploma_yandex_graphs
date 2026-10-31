"""Execute one approved model-fit unit. Dataset labels stay in its adapter."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

from registry import Registry
from spec import confined, load


def owner_identity():
    value = {"pid": os.getpid(), "hostname": socket.gethostname(), "started": time.time()}
    stat = Path("/proc") / str(os.getpid()) / "stat"
    if stat.exists():
        fields = stat.read_text().rsplit(")", 1)[1].split()
        value.update(start_ticks=int(fields[19]), pgid=int(fields[2]))
    return value


def require_approval(path, spec_hash):
    approval = json.loads(Path(path).read_text())
    approved = (approval.get("approved_spec_hash") == spec_hash or
                spec_hash in approval.get("approved_spec_hashes", []))
    if (not approved or approval.get("approval_source") != "explicit_user_go_ahead"
            or not approval.get("user_message_reference")):
        raise ValueError("This run list has not received the user's go-ahead")
    return approval


def load_factory(root, entry):
    relative, separator, function = entry.partition(":")
    if not separator or not function.isidentifier():
        raise ValueError("Use project-relative adapter_file.py:factory")
    path = confined(root, relative)
    name = "gnnm_adapter_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return getattr(module, function)


def terminal_owner(owner):
    """PID reuse is distinguished from the original process by Linux start ticks."""
    if owner.get("hostname") != socket.gethostname() or "start_ticks" not in owner:
        raise ValueError("The previous owner's termination cannot be verified on this host")
    path = Path("/proc") / str(owner["pid"]) / "stat"
    if path.exists():
        fields = path.read_text().rsplit(")", 1)[1].split()
        if int(fields[19]) == owner["start_ticks"] and fields[0] != "Z":
            raise ValueError("The previous exact owner is still alive")
    return {"previous_owner_terminal": True, "previous_owner": owner,
            "verified_on": socket.gethostname(), "verified_at": time.time()}


def execute(project_root, study_path, run_id, registry_path, output_root, approval_path, device,
            recover=False):
    project_root = Path(project_root).resolve()
    spec, digest = load(study_path, project_root)
    require_approval(approval_path, digest)
    rows = [row for row in spec["runs"] if row["run_id"] == run_id]
    if len(rows) != 1 or rows[0]["kind"] != "model_fit":
        raise ValueError("Run ID must identify one declared model fit")
    run = rows[0]
    if run["budget"]["policy"] != "full_budget":
        raise ValueError("The current engine implements full-budget trajectories")
    registry = Registry(registry_path)
    registry.register(spec)
    owner = owner_identity()
    admission = None
    if recover:
        prior = registry.run(spec["study_id"], run_id)
        admission = terminal_owner(prior["owner"] or {})
        retained = Path(output_root) / spec["study_id"] / run_id / "result.json"
        if not retained.is_file() or json.loads(retained.read_text()).get("status") != "failed":
            registry.close()
            raise ValueError("Recovery requires a retained interrupted engine result; inspect crashes manually")
        registry.recover(spec["study_id"], run_id, owner, admission)
    else:
        registry.claim(spec["study_id"], run_id, owner)
    tick = time.monotonic()
    try:
        import torch
        from engine import run_trajectory, resume_trajectory
        torch.set_num_threads(int(os.environ.get("GNNM_TORCH_THREADS", "2")))
        factory = load_factory(project_root, run["adapter"])
        adapter = factory(torch=torch, project_root=project_root, study=spec, run=run, device=device)
        engine_spec = {"fit_id": run_id, "budget_steps": run["budget"]["maximum"],
                       "budget_unit": run["budget"]["unit"],
                       "validation_interval": run["selection"]["evaluation_interval"],
                       "validate_initial": run["selection"]["checkpoint0"],
                       "prefixes": run["budget"].get("prefixes", []),
                       "recovery_interval": run["budget"].get("recovery_interval", 100),
                       "inherited_fit_ids": run["parents"],
                       "adapter_spec": {"study_hash": digest, "run": run,
                                        "data": spec["data"], "source": spec["source"]}}
        def guard(phase, step, result):
            if time.monotonic() - tick > spec["resources"]["wall_seconds"]:
                raise TimeoutError("Declared run wall budget exhausted")
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
                peak = int(torch.cuda.max_memory_allocated())
                ceiling = run["config"].get("maximum_cuda_bytes")
                if ceiling and peak > ceiling:
                    raise MemoryError("Declared run CUDA budget exhausted")
                return {"peak_cuda_allocated_bytes": peak}
            return {}
        destination = Path(output_root) / spec["study_id"] / run_id
        if admission:
            result = resume_trajectory(torch, adapter, engine_spec, destination,
                                       admission=admission, resource_check=guard)
        else:
            result = run_trajectory(torch, adapter, engine_spec, destination, resource_check=guard)
        registry.finish(spec["study_id"], run_id, result, success=True)
        return result
    except BaseException as error:
        retained = Path(output_root) / spec["study_id"] / run_id / "result.json"
        failure = json.loads(retained.read_text()) if retained.exists() else {}
        failure.update({"failure_type": type(error).__name__, "failure": str(error),
                        "wall_seconds": time.monotonic() - tick, "owner": owner})
        registry.finish(spec["study_id"], run_id,
                        failure, success=False)
        raise
    finally:
        registry.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recover-terminal", action="store_true",
                        help="Resume only a retained failed trajectory whose exact owner is terminal")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    for value in (args.study, args.registry, args.output, args.approval):
        if not Path(value).resolve().is_relative_to(root):
            raise ValueError("Study, registry, outputs and approval must stay inside the repository")
    execute(root, args.study, args.run_id, args.registry, args.output, args.approval, args.device,
            recover=args.recover_terminal)


if __name__ == "__main__":
    main()
