"""Local source/identity/parity/diagnostic checks, without model/data loading."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_selected_replay_contract", HERE / "replay.py")
backend = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = backend
spec.loader.exec_module(backend)
torch.set_num_threads(1)


def rejects(call):
    try:
        call()
    except (backend.FidelityRefusal, AssertionError, RuntimeError):
        return
    raise AssertionError("Unsupported identity/source/parity input accepted")


def main():
    registry = json.loads((HERE / "REGISTRY.json").read_text())
    outcomes = []
    for profile in registry["runtimes"]:
        binding = {"runtime_sha256": copy.deepcopy(profile["runtime_sha256"]), "callback_sha256": "c" * 64,
                   "extension_sha256": None, "dependency_sha256": {}}
        if profile["id"] == "runtime_v2":
            binding["policy_sha256"] = None
        assert backend.match_runtime(binding, registry["runtimes"]) == profile
        changed = copy.deepcopy(binding)
        changed["runtime_sha256"]["adapters.py"] = "unknown"
        rejects(lambda: backend.match_runtime(changed, registry["runtimes"]))
        changed = {**binding, "unknown_fallback": True}
        rejects(lambda: backend.match_runtime(changed, registry["runtimes"]))
        sentinel = types.ModuleType("common")
        previous = sys.modules.get("common")
        sys.modules["common"] = sentinel
        with backend.runtime_namespace(profile, HERE.parent.parent.parent):
            assert sys.modules["common"] is not sentinel
            assert Path(sys.modules["adapters"].__file__).parent.name == ("runtime" if profile["id"] == "runtime_v1" else "runtime_v2")
        assert sys.modules["common"] is sentinel
        if previous is None:
            sys.modules.pop("common")
        else:
            sys.modules["common"] = previous
    outcomes.append("two exact runtime profiles; unknown hash/key refusal; isolated module imports/restoration")

    job = {"id": "own-job", "cohort_id": "own-cohort", "family": "node", "procedure": "native_single",
           "seed": 17, "policy": {"track": "fixed"}, "recipe": {"dataset": "citeseer", "id": "polyformer-citeseer-cheb"}}
    pin = {"path": str(HERE / "selected.pt"), "bytes": 123, "sha256": "a" * 64}
    checkpoint = {"schema": "native-full-epoch-recovery-v1", "job_hash": backend.digest(job), "TEST_accessed": False,
                  "selector": {"selected_epoch": 2}, "adapter": {"completed_epochs": 2}}
    exported = {"schema": "own-selected-VALID-logits-v1", "role": "VALID", "TEST_accessed": False,
        "job_hash": backend.digest(job), "job_id": job["id"], "cohort_id": job["cohort_id"], "family": "node",
        "procedure": "native_single", "seed": 17, "body": None, "split": 0, "policy": job["policy"],
        "dataset": "citeseer", "recipe_id": job["recipe"]["id"], "selected_checkpoint": pin, "selected_epoch": 2}
    assert backend.selected_identity(job, checkpoint, exported, pin)[1] == 2
    for field, value in (("seed", 18), ("selected_epoch", 3), ("role", "TEST")):
        wrong = copy.deepcopy(exported)
        wrong[field] = value
        rejects(lambda: backend.selected_identity(job, checkpoint, wrong, pin))
    wrong = copy.deepcopy(checkpoint)
    wrong["adapter"]["completed_epochs"] = 3
    rejects(lambda: backend.selected_identity(job, wrong, exported, pin))
    outcomes.append("own JOB/cohort/seed/epoch/export checkpoint receipt identity; no recovery/final/TEST substitution")

    logits = torch.tensor([[0.1, 0.8], [-0.2, 0.5]], dtype=torch.float32)
    actual = {"ids": torch.tensor([3, 7]), "labels": torch.tensor([1, 0]), "outputs": logits,
              "loss": 0.9, "metrics": {"accuracy": 0.5}}
    saved = {"ids": actual["ids"], "labels": actual["labels"], "raw_logits": logits + 1e-5,
             "validation": {"role": "VALID", "complete_source_VALID": True, "TEST_accessed": False,
                            "records": 2, "loss": 0.90001, "metrics": {"accuracy": 0.501}}}
    assert backend.compare_export(torch, actual, saved)["passed"]
    for key, value in (("ids", torch.tensor([7, 3])), ("labels", torch.tensor([0, 1])),
                       ("outputs", logits + 0.1), ("outputs", logits.double())):
        changed = {**actual, key: value}
        rejects(lambda: backend.compare_export(torch, changed, saved))
    changed = {**actual, "metrics": {"accuracy": 0.51}}
    rejects(lambda: backend.compare_export(torch, changed, saved))
    outcomes.append("fixed practical logit/loss/metric tolerances; exact IDs/labels/shapes/dtypes; meaningful mismatch refusal")

    class Session:
        def __init__(self):
            self.calls = 0
        def validate(self):
            self.calls += 1
            return {**actual, "role": "VALID", "complete_source_VALID": True, "TEST_accessed": False,
                    "same_bank_mean_outputs": logits * 0.8, "same_bank_mean_loss": 1.,
                    "same_bank_mean_metrics": {"accuracy": 0.5}}
    session = Session()
    adapter = types.SimpleNamespace(session=session)
    def validate():
        raw = adapter.session.validate()
        return {k: raw[k] for k in (*actual, "role", "complete_source_VALID", "TEST_accessed")}
    adapter.validate = validate
    result, mean = backend.actual_validation(torch, adapter, True)
    assert session.calls == 1 and torch.equal(mean["same_bank_mean_raw_logits"], logits * 0.8)
    assert mean["used_for_selection"] is False and torch.equal(result["outputs"], logits)
    assert session.validate.__func__ is Session.validate
    outcomes.append("C6 actual+mean diagnostics captured from one native VALID call; original method restored; no selector change")

    with tempfile.TemporaryDirectory(dir=HERE) as folder:
        root = Path(folder)
        data = root / "cache/ogbg_molbace"
        paths = {"processed": data / "processed/geometric_data_processed.pt",
                 "train_split": data / "split/scaffold/train.pt", "valid_split": data / "split/scaffold/valid.pt"}
        for key, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((key + "-synthetic-descriptor").encode())
        declared = {key: backend.receipt(path) for key, path in paths.items()}
        data_job = {"family": "ogb", "data_root": str(root / "cache"), "recipe": {"dataset": "ogbg-molbace"}}
        assert set(backend.data_preflight(data_job, declared, root)) == set(paths)
        rejects(lambda: backend.data_preflight(data_job, {**declared, "test_split": declared["valid_split"]}, root))
        paths["processed"].write_bytes(b"changed")
        rejects(lambda: backend.data_preflight(data_job, declared, root))
        rejects(lambda: backend.inside(root.parent / "outside", root))
    outcomes.append("explicit source data receipts; TEST-role/changed bytes/outside-repository refusal")
    result = {"schema": "selected-VALID-local-contract-checks-v1", "passed": True, "torch": str(torch.__version__),
        "checks": outcomes, "native_model_or_data_loads": 0, "optimizer_steps": 0, "scientific_fits": 0, "TEST_accessed": False}
    (HERE / "CHECKS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
