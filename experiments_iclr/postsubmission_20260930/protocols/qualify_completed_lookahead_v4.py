"""Qualify a completed training-only lock; do not open predictive inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parents[1]
API = PHASE / "objective_lookahead_molhiv_admitted_v4"

def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PHASE) or output.exists():
        raise ValueError("Require a new output inside the phase")
    if os.environ.get("GNNM_PHASE_ROOT") != str(PHASE):
        raise ValueError("Require the repository environment wrapper")
    sys.path.insert(0, str(API))
    import admission
    if Path(admission.__file__).resolve() != API / "admission.py":
        raise ValueError("Conflicting method API")
    admitted = admission.qualify_completed_lock(args.lock)
    if any(n == "torch" or n.startswith("torch.") or n == "ogb" or n.startswith("ogb.") for n in sys.modules):
        raise ValueError("Completed-method qualification must remain tensor/provider free")
    value = {"schema": "gnnm-completed-training-only-screen-consumer-audit-v1",
        "status": "qualified_complete_training_only_mechanism_lock",
        "mechanism_lock": admitted.evidence["completed_mechanism_lock"],
        "source_sha256": admitted.evidence["source_sha256"],
        "qualified_gpu_uuid": admitted.qualified_gpu_uuid,
        "loss_families": admitted.evidence["loss_families"],
        "all_six_sources_qualified": len(admitted.descriptors) == 6,
        "all_48_gains_and_selection_replayed": admitted.evidence["all_48_gains_and_selection_replayed_without_tensor_loading"],
        "evidence_sha256": hashlib.sha256(admission.canonical(admitted.evidence)).hexdigest(),
        "tensor_provider_or_validation_imported": False,
        "predictive_result_or_promotion_claimed": False}
    with output.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": value["status"], "loss_families": value["loss_families"]}), flush=True)

if __name__ == "__main__":
    main()
