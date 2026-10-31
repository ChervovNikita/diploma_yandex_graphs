"""Closed fixed30 validation-only operations; no test exporter/evaluator exists."""
import sys
import os
# Discard script/cwd/PYTHONPATH before stdlib or phase-package imports.
sys.path[:] = [p for p in sys.path if p and os.path.isabs(p)
               and any(os.path.commonpath([p, root]) == root for root in (sys.prefix, sys.base_prefix))]
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def bootstrap(request_path):
    """Validate the request-bound guard before the first phase-package import."""
    request_path = Path(request_path)
    request_value = json.loads(request_path.read_text())
    manifest_path = HERE / "SOURCE_MANIFEST_v1.json"
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != request_value.get("source_manifest_sha256"):
        raise ValueError("Bootstrap source manifest differs from the exact root request")
    manifest = json.loads(raw)
    guards = [r for r in manifest["files"] if r["relative_path"] == "import_guard.py"]
    guard_path = HERE / "import_guard.py"
    if guard_path.is_symlink() or len(guards) != 1 or hashlib.sha256(guard_path.read_bytes()).hexdigest() != guards[0]["sha256"]:
        raise ValueError("Bootstrap import guard source changed")
    spec = importlib.util.spec_from_file_location("_shared_message_predictive_guard_v2", guard_path)
    guard = importlib.util.module_from_spec(spec)
    exec(compile(guard_path.read_bytes(), str(guard_path), "exec", dont_inherit=True), guard.__dict__)
    return guard, *guard.install(request_value["source_manifest_sha256"])


def main():
    # Parsing is stdlib only; no phase or scientific module is imported yet.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("fit", "replay", "lock-family"))
    p.add_argument("--binding", type=Path, required=True)
    p.add_argument("--validation-freeze", type=Path, required=True)
    p.add_argument("--supervisor", type=Path, required=True)
    p.add_argument("--task", choices=("Reddit2", "ogbn-products"))
    p.add_argument("--arm", choices=("exact_member", "shared_mean_message", "detached_mean_message", "single_base64", "single_parameters"))
    p.add_argument("--seed", type=int, choices=(17, 29, 43))
    args = p.parse_args()
    bootstrap(args.binding)
    from shared_message_predictive_v3.contract import binding, live, prescribed, runtime_before_import, write
    from shared_message_predictive_v3.core import fit, lock_family, replay, validation_freeze
    admitted, binding_ref = binding(args.binding)
    freeze, freeze_ref = validation_freeze(args.validation_freeze, admitted, binding_ref)
    runtime_before_import(admitted)
    context = live(args.supervisor, Path(__file__), args.binding, admitted)
    if context["gpu_uuid"] != admitted["gpu_uuid"]:
        raise ValueError("Wrong live authorized allocation")
    if args.operation in ("fit", "replay"):
        if any(v is None for v in (args.task, args.arm, args.seed)):
            p.error("One exact --task --arm --seed is required")
        cell = {"task": args.task, "arm": args.arm, "seed": args.seed}
        output = prescribed(admitted, "fits" if args.operation == "fit" else "replays", cell)
    else:
        if any(v is not None for v in (args.task, args.arm, args.seed)):
            p.error("Family lock accepts no selected cell or subset")
        output = prescribed(admitted, "family_lock")
    try:
        if args.operation == "fit":
            fit(admitted, binding_ref, freeze, freeze_ref, cell, context)
        elif args.operation == "replay":
            replay(admitted, binding_ref, freeze, freeze_ref, cell, context)
        else:
            lock_family(admitted, binding_ref, freeze_ref, context)
    except BaseException as error:
        if output.exists() and not (output / "FAILURE.json").exists():
            write(output / "FAILURE.json", {"type": type(error).__name__, "message": str(error), "operation": args.operation,
                "partial_outputs_preserved": True, "resume_retry_or_selective_promotion_authorized": False, "live": context})
        raise


if __name__ == "__main__":
    main()
