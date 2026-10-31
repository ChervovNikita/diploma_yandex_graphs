"""Execute one approved non-training endpoint with explicit parent custody."""
import argparse
import json
from pathlib import Path
import time

from registry import Registry
from run_study import load_factory, owner_identity, require_approval, require_phase_approval
from spec import canonical, confined, load


def execute(root, study_path, run_id, registry_path, output_root, approval_path, device):
    spec, digest = load(study_path, root)
    approval = require_approval(approval_path, digest)
    runs = [r for r in spec["runs"] if r["run_id"] == run_id]
    if len(runs) != 1 or runs[0]["kind"] == "model_fit":
        raise ValueError("Expected a non-training endpoint")
    run = runs[0]
    require_phase_approval(root, spec, run, approval)
    registry = Registry(registry_path)
    registry.register(spec)
    ready = (registry.required_terminal if run["kind"] == "selection" else registry.required_complete)
    if run["parents"] and not ready(spec["study_id"], run["parents"]):
        registry.close()
        raise ValueError("Endpoint parent custody is not ready")
    owner = owner_identity()
    registry.claim(spec["study_id"], run_id, owner)
    out = Path(output_root) / spec["study_id"] / run_id
    tick = time.monotonic()
    try:
        if out.exists() and any(out.iterdir()):
            raise ValueError("An endpoint output already exists; retain it for inspection")
        out.mkdir(parents=True, exist_ok=True)
        factory = load_factory(root, run["adapter"])
        result = factory(project_root=root, study=spec, run=run, output_root=Path(output_root),
                         output_dir=out, registry=registry, device=device)
        if not isinstance(result, dict) or result.get("status") != "complete":
            raise ValueError("Endpoint did not return a complete custody receipt")
        result.update(wall_seconds=time.monotonic()-tick, owner=owner)
        (out / "result.json").write_text(canonical(result)+"\n")
        registry.finish(spec["study_id"], run_id, result, success=True)
        return result
    except BaseException as error:
        result = {"status": "failed", "failure_type": type(error).__name__, "failure": str(error),
                  "owner": owner, "wall_seconds": time.monotonic()-tick}
        out.mkdir(parents=True, exist_ok=True)
        (out / "failure.json").write_text(canonical(result)+"\n")
        registry.finish(spec["study_id"], run_id, result, success=False)
        raise
    finally:
        registry.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("project-root", "study", "run-id", "registry", "output", "approval"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    for path in (args.study, args.registry, args.output, args.approval):
        if not Path(path).resolve().is_relative_to(root):
            raise ValueError("Endpoint files must stay inside the project")
    execute(root, args.study, args.run_id, args.registry, args.output, args.approval, args.device)


if __name__ == "__main__":
    main()
