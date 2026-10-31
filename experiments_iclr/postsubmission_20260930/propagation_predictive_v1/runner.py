"""Closed prospective30 runner; every operation requires actual root admission."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
# Avoid the sealed engineering directory's profile.py shadowing stdlib profile.
# Imports are package-qualified, and no dependency is searched as a bare file.
entry_directory = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or Path.cwd()).resolve() != entry_directory]
sys.path.insert(0, str(entry_directory.parent))
from propagation_predictive_v1.contract import (ARMS, TASKS, HERE, PHASE, binding,
    live, reference, runtime_before_import, within, write)
from propagation_predictive_v1.core import (fit, lock_family, prescribed, replay,
                                           validation_freeze)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("fit", "replay", "lock-family", "final-test"))
    p.add_argument("--binding", type=Path, required=True)
    p.add_argument("--validation-freeze", type=Path, required=True)
    p.add_argument("--supervisor", type=Path, required=True)
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--seed", type=int, choices=(0, 1, 2))
    p.add_argument("--request", type=Path)
    p.add_argument("--family-lock", type=Path)
    p.add_argument("--test-freeze", type=Path)
    args = p.parse_args()
    admitted, binding_ref = binding(args.binding)
    freeze, freeze_ref = validation_freeze(args.validation_freeze, admitted, binding_ref)
    context = live(args.supervisor, Path(__file__))
    if context["gpu_uuid"] != admitted["gpu_uuid"]:
        raise ValueError("Scientific entrypoint received a different authorized GPU")
    runtime_before_import(admitted)
    output = None
    try:
        if args.operation in ("fit", "replay"):
            if args.task is None or args.arm is None or args.seed is None:
                p.error("fit/replay require one exact --task --arm --seed cell")
            if args.request or args.family_lock or args.test_freeze:
                p.error("cell fits/replays accept no test/finalization references")
            cell = {"task": args.task, "arm": args.arm, "seed": args.seed}
            if args.operation == "fit":
                output = prescribed(admitted, "fits", cell)
                fit(admitted, binding_ref, freeze, freeze_ref, cell, output, context)
            else:
                output = prescribed(admitted, "replays", cell)
                replay(admitted, binding_ref, freeze, freeze_ref, cell,
                       prescribed(admitted, "fits", cell), output, context)
        else:
            if any(v is not None for v in (args.task, args.arm, args.seed)):
                p.error("family operations accept no selectable cell")
            if args.operation == "lock-family":
                if args.request is None or args.family_lock or args.test_freeze:
                    p.error("lock-family requires only --request")
                request = json.loads(within(args.request, HERE).read_text())
                output = prescribed(admitted, "family_lock")
                lock_family(admitted, binding_ref, freeze_ref, request, output, context)
            else:
                if args.family_lock is None or args.test_freeze is None or args.request:
                    p.error("final-test requires --family-lock and --test-freeze")
                from propagation_predictive_v1.evaluate import final_test
                output = prescribed(admitted, "final_test")
                final_test(admitted, binding_ref, args.family_lock, args.test_freeze, output, context)
    except BaseException as error:
        if output is not None and output.exists() and not (output / "FAILURE.json").exists():
            write(output / "FAILURE.json", {"schema": "gnnm-propagation-scientific-entrypoint-failure-v1",
                "operation": args.operation, "type": type(error).__name__, "message": str(error),
                "partial_outputs_preserved": True, "retry_resume_or_partial_promotion_authorized": False,
                "live": context})
        raise


if __name__ == "__main__":
    main()
