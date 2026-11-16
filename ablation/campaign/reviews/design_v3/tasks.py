"""Execute one immutable campaign task with readable Python entry points."""
import argparse
from dataclasses import fields
import hashlib
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .schema import load_plan, canonical_hash

REPO = Path(__file__).resolve().parents[2]

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(value, path):
    from ablation.common.resumable import atomic_write_json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(value, Path(path))

def output(root, task_id):
    return root / "tasks" / task_id / "output"

def result(root, task_id):
    return read_json(output(root, task_id) / "done.json")

def resolve_spec(metadata, root):
    spec = dict(metadata["spec"])
    resolver = metadata.get("resolve")
    if resolver:
        if "selection" in resolver:
            decision = result(root, resolver["selection"])
            setting = resolver["setting"]
            ready = decision.get("ready_" + setting, decision.get("ready", True))
            if not ready or setting not in decision["selections"]:
                raise ValueError("Confirmation requires a feasible, ready validation selection")
            selected = decision["selections"][resolver["setting"]]["spec"]
            for key in ("num_layers", "lr", "weight_decay", "dropout", "attention_dropout", "num_heads"):
                spec[key] = selected[key]
        else:
            gate = result(root, resolver["gate"])
            hp = gate["chosen_configs"][resolver["grid_slot"]]
            spec.update(hp)
            spec["weight_decay"] = hp.get("weight_decay", 0.0)
    if spec.get("variant") is None:
        spec.pop("variant", None)
    return spec

def source_binding():
    import hashlib
    paths = sorted((REPO / "ablation").rglob("*.py")) + [REPO / "models.py", REPO / "datasets.py", REPO / "run_common.py"]
    return canonical_hash({str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in paths})

def node_fit(meta, out, root, device, smoke_updates=None):
    import torch
    from datasets import load_dataset
    from ablation.common.resumable import JobStore
    from ablation.common.worker import StopFlag
    from ablation.common.tabm_trainer import TrainSettings, train_resumable
    from ablation.p05_hparam_grid.selected import build_selected
    spec = resolve_spec(meta, root)
    spec["source_sha256"] = source_binding()
    data_path = REPO / "data" / (spec["dataset"].replace("-", "_") + ".npz")
    spec["dataset_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    spec["planned_caps"] = [spec["num_steps"], spec["continuation_steps"]]
    if smoke_updates:
        spec.update(num_steps=smoke_updates, min_steps=smoke_updates,
                    patience=smoke_updates, continuation_steps=smoke_updates,
                    planned_caps=[smoke_updates], role="engineering_smoke")
    else:
        spec["role"] = "scientific"
    store = JobStore(out)
    store.ensure_config(spec)
    device = torch.device(device)
    data, train, val, test, _, targets, binary = load_dataset(
        spec["dataset"], add_self_loops=True, device=device, data_dir=str(REPO / "data"))
    bundle = (data, train, val, test, binary)
    stop = StopFlag()
    settings_fields = {field.name for field in fields(TrainSettings)}
    settings = {key: value for key, value in spec.items() if key in settings_fields}
    receipts_path = out / "continuations.json"
    receipts = read_json(receipts_path) if receipts_path.exists() else []
    saved = store.load_state()
    already_trained = saved["progress"]["step"] if saved else 0
    for cap in sorted(set(spec["planned_caps"])):
        if cap < already_trained:
            continue
        settings["num_steps"] = cap
        try:
            trained = train_resumable(TrainSettings(**settings), store, bundle, device,
                                     lambda: build_selected(spec, data, targets, device, calibrate=True)[0], stop=stop)
        except FloatingPointError as error:
            failure = dict(spec, scientific_status="numerical_failure", error=str(error),
                           needs_convergence_extension=False, effective_cap=cap)
            write_json(failure, out / "numerical_failure.json")
            return failure
        receipts.append(dict(cap=cap, stopped=trained["stopped_step"],
                             best_step=trained["best_step"], stop_reason=trained["stop_reason"]))
        write_json(receipts, out / "continuations.json")
        if not trained["needs_convergence_extension"]:
            break
    trained.update(spec)
    trained["selected_checkpoint_sha256"] = hashlib.sha256((out / "model.pt").read_bytes()).hexdigest()
    trained["selected_predictions_sha256"] = hashlib.sha256((out / "predictions.pt").read_bytes()).hexdigest()
    trained["effective_cap"] = cap
    trained["continuations"] = receipts
    write_json(trained, out / "done.json")
    if not trained["needs_convergence_extension"]:
        store.clear_state()
    return trained

def node_cell(meta, root, task_id):
    import torch
    from datasets import compute_metrics
    fits = [result(root, name) for name in meta["fits"]]
    failed = {index: fit for index, fit in enumerate(fits)
              if fit.get("scientific_status") == "numerical_failure"}
    predictions = [None if index in failed else
                   torch.load(output(root, name) / "predictions.pt", map_location="cpu", weights_only=False)
                   for index, name in enumerate(meta["fits"])]
    surviving = [artifact for artifact in predictions if artifact is not None]
    if surviving:
        reference = surviving[0]["roles"]["val"]
        for artifact in surviving:
            other = artifact["roles"]["val"]
            if not torch.equal(reference["indices"], other["indices"]) or not torch.equal(reference["labels"], other["labels"]):
                raise ValueError("Cell members have conflicting validation ordering")
            if artifact["is_binary"] != surviving[0]["is_binary"]:
                raise ValueError("Cell members have conflicting target types")
            if "test" in artifact["roles"]:
                raise ValueError("Pilot cell contains held-out predictions")
    if failed:
        scores = {}
        indices = {"BASE": [0], "ENS": [0, 1, 2, 3], "GNNM": [4]}
        for setting, members in indices.items():
            scores[setting] = None
            if any(index in failed for index in members):
                continue
            role = predictions[members[0]]["roles"]["val"]
            logits = torch.cat([predictions[index]["roles"]["val"]["logits"] for index in members]).mean(0)
            mask = torch.ones(len(role["labels"]), dtype=torch.bool)
            scores[setting] = compute_metrics(logits, role["labels"], mask,
                 predictions[members[0]]["is_binary"], dataset_name=meta["dataset"])["metric"]
        return dict(task_id=task_id, role="val", dataset=meta["dataset"], model=meta["model"],
                    config_name=meta["config_name"], depth=meta["depth"], split=meta["split"],
                    scores=scores, specs={"BASE": fits[0], "ENS": fits[0], "GNNM": fits[4]},
                    fit_ids=meta["fits"], scientific_failures={meta["fits"][index]: fit["error"] for index, fit in failed.items()},
                    unresolved_convergence=[name for name, fit in zip(meta["fits"], fits)
                                            if fit.get("needs_convergence_extension")])
    labels = reference["labels"]
    mask = torch.ones(len(labels), dtype=torch.bool)
    def metric(logits):
        return compute_metrics(logits, labels, mask, predictions[0]["is_binary"],
                               dataset_name=meta["dataset"])["metric"]
    logits = [artifact["roles"]["val"]["logits"].mean(0) for artifact in predictions]
    return dict(task_id=task_id, role="val", dataset=meta["dataset"], model=meta["model"],
                config_name=meta["config_name"], depth=meta["depth"], split=meta["split"],
                scores={"BASE": metric(logits[0]), "ENS": metric(torch.stack(logits[:4]).mean(0)),
                        "GNNM": metric(logits[4])},
                specs={"BASE": fits[0], "ENS": fits[0], "GNNM": fits[4]},
                fit_ids=meta["fits"], elapsed_seconds={
                    "BASE": fits[0]["fit_seconds"], "ENS": sum(item["fit_seconds"] for item in fits[:4]),
                    "GNNM": fits[4]["fit_seconds"]},
                unresolved_convergence=[name for name, fit in zip(meta["fits"], fits)
                                        if fit["needs_convergence_extension"]])

def node_score(meta, root, out, device):
    from ablation.p05_hparam_grid.selected import score_selected
    dirs = [output(root, name) for name in meta["fits"]]
    failed = [name for name in meta["fits"] if result(root, name).get("scientific_status") == "numerical_failure"]
    if failed:
        return dict(meta, scientific_status="numerical_failure", failed_fit_ids=failed,
                    num_layers=result(root, meta["fits"][0])["num_layers"],
                    val_metric=None, test_metric=None, fit_ids=meta["fits"])
    scored = score_selected(dirs, out, roles=meta.get("roles", ("val", "test")), data_dir=str(REPO / "data"), device=device)
    scored.update({key: meta[key] for key in ("setting", "dataset", "model", "split", "stage") if key in meta})
    scored["fit_ids"] = meta["fits"]
    scored["config"] = result(root, meta["fits"][0])
    scored["unresolved_convergence"] = [
        name for name in meta["fits"] if result(root, name)["needs_convergence_extension"]]
    return scored

def selected_fit_dirs(meta, root):
    if "score" in meta:
        scored = result(root, meta["score"])
    else:
        candidates = [result(root, name) for name in meta["candidates"]]
        candidates = [row for row in candidates if row.get("val_metric") is not None]
        if not candidates:
            return []
        scored = sorted(candidates, key=lambda item: (-item["val_metric"], item["num_layers"]))[0]
    if scored.get("scientific_status") == "numerical_failure":
        return []
    return [output(root, name) for name in scored["fit_ids"]]

def transfer_arguments(meta, out, device):
    module, pair, method = meta["module"], meta["pair"], meta["method"]
    argv = [sys.executable, "-m", f"ablation.{module}.run", "--root", str(out / "training"),
            "--data_dir", str(REPO / "data"), "--device", device,
            "--cohorts", str(meta["cohort"]), "--no_wait"]
    if module == "p09_molecular":
        dataset, model = pair.split("_")
        method = "ens_m" + method[-1] if method.startswith("native") else "gnnm_private_bn" if method == "private_bn" else method
        argv += ["--datasets", "ogbg-" + dataset, "--models", model, "--methods", method,
                 "--convergence", "--defer-test"]
    elif module == "p11_collab_buddy":
        method = {"private_bn": "gnnm_private_bn", "link_same_query_global": "gnnm_private_bn_contrast"}.get(method, method)
        if method.startswith("native"):
            argv += ["--only", "native", "--member", method[-1]]
        else:
            argv += ["--only", method]
        argv += ["--convergence", "--defer-test", "--require-cache"]
    else:
        argv += ["--methods", method, "--epochs", "2000", "--min-epochs", "100",
                 "--patience", "50", "--initial-cap", "1000", "--defer-test"]
    return argv

def inherited_task_locks(out):
    inherited = tuple(int(value) for value in os.environ.get("GNNM_CAMPAIGN_LOCK_FDS", "").split(",") if value)
    if not inherited:
        return ()
    expected = (out.parent / ".ownership.lock").stat()
    matched = []
    for descriptor in inherited:
        actual = os.fstat(descriptor)
        if (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino):
            matched.append(descriptor)
    if not matched:
        raise ValueError("Inherited descriptors do not own this campaign task")
    # This nonblocking operation preserves the inherited open-file-description
    # lock and rejects a matching descriptor that is owned by another process.
    fcntl.flock(matched[0], fcntl.LOCK_EX | fcntl.LOCK_NB)
    return inherited


def transfer_fit(meta, out, device, smoke_updates=None):
    argv = transfer_arguments(meta, out, device)
    inherited = inherited_task_locks(out)
    if inherited:
        # Campaign ownership persists in live descendants. Once acquired here,
        # no earlier trainer can still hold this task. Legacy claim files can
        # survive SIGKILL even after their process has gone.
        for claim in (out / "training").rglob("claim"):
            claim.unlink()
    if smoke_updates:
        if meta["module"] == "p09_molecular":
            argv += ["--smoke_updates", str(smoke_updates)]
        elif meta["module"] == "p11_collab_buddy":
            argv += ["--smoke", "--smoke_epochs", "1", "--smoke_max_batches", str(smoke_updates)]
        else:
            argv += ["--smoke-updates", str(smoke_updates)]
    write_json({"argv": argv}, out / "command.json")
    subprocess.run(argv, cwd=REPO, check=True, pass_fds=inherited)
    completed = list((out / "training").rglob("done.json"))
    if len(completed) != 1:
        raise RuntimeError(f"Expected one completed transfer fit, found {len(completed)}")
    return dict(fit_dir=str(completed[0].parent), argv=argv, fit=read_json(completed[0]),
                pair=meta["pair"], method=meta["method"], cohort=meta["cohort"],
                role="engineering_smoke" if smoke_updates else "scientific")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke-updates", type=int)
    args = parser.parse_args()
    import setproctitle
    setproctitle.setproctitle("AI Auto Research Pipeline")
    if not args.root.resolve().is_relative_to(REPO):
        raise ValueError("Campaign outputs must remain inside the repository")
    plan = load_plan(args.plan)
    task = next(item for item in plan["tasks"] if item["id"] == args.task)
    meta = task["metadata"]
    out = output(args.root, task["id"])
    out.mkdir(parents=True, exist_ok=True)
    action = meta["action"]
    if action == "node_fit":
        completed = node_fit(meta, out, args.root, args.device, args.smoke_updates)
    elif action == "node_cell":
        completed = node_cell(meta, args.root, task["id"])
    elif action == "select":
        from .decisions import select
        completed = select([result(args.root, name) for name in meta["cells"]])
        completed.update(dataset=meta["dataset"], model=meta["model"])
    elif action == "expand":
        from .decisions import expansion
        completed = expansion([result(args.root, name) for name in meta["decisions"]])
    elif action == "node_score":
        completed = node_score(meta, args.root, out, args.device)
    elif action in ("diversity", "profile"):
        dirs = selected_fit_dirs(meta, args.root)
        if not dirs:
            write_json(dict(scientific_status="infeasible", reason="All required candidate fits failed numerically"), out / "done.json")
            return
        if action == "diversity":
            from ablation.p06_member_diversity.selected import analyse_selected
            completed = analyse_selected(dirs, out, role="val", data_dir=str(REPO / "data"), device=args.device)
            from .structure import analyse_structure
            completed["structure"] = analyse_structure(dirs, out, data_dir=str(REPO / "data"))
        else:
            from ablation.p07_cost_small_setup.selected import profile_selected
            completed = profile_selected(dirs, out, data_dir=str(REPO / "data"), device=args.device)
    elif action == "initialization":
        from ablation.p06_member_diversity.initialization import initialization_diagnostics
        completed = initialization_diagnostics(meta["spec"], out, data_dir=str(REPO / "data"), device=args.device)
    elif action == "transfer_fit":
        completed = transfer_fit(meta, out, args.device, args.smoke_updates)
    else:
        from .analyses import execute_analysis
        completed = execute_analysis(action, meta, args.root, out, args.device, plan)
    write_json(completed, out / "done.json")
    print(json.dumps({"task": task["id"], "action": action, "complete": True}))

if __name__ == "__main__":
    main()
