"""Exercise archived replay with a clearly labeled synthetic engineering fixture.

No dataset or GPU is accessed. No training result is constructed or claimed:
300 identical synthetic prediction records exercise the complete trajectory
reader and earliest-epoch tie rule. Fresh subprocesses check source drift and
archived model rebuilding. Real checkpoint replay remains required after fits.
"""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True


def load(path):
    spec = importlib.util.spec_from_file_location("ppi_archived_replay_subject", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def child(args):
    import torch
    study = load(args.runner)
    fixture = args.fixture.resolve()
    graphs = torch.load(fixture / "synthetic_graphs.pt", map_location="cpu")
    # Dataset-provider verification is intentionally excluded from this
    # synthetic reader/model/source test. It is gated separately on real data.
    study.PPI = lambda root, split: graphs if split == "val" else []
    study.data_freeze = lambda path: json.loads(Path(path).read_text())
    args.run = fixture
    args.data_manifest = fixture / "data_manifest.json"
    args.allow_engineering_fixture = args.allow_fixture
    try:
        study.replay(args)
    except ValueError as exc:
        print(json.dumps({"status": "expected_or_unexpected_rejection", "error": str(exc)}))
        return 3
    return 0


def parent(args):
    import torch
    from torch_geometric.data import Data
    study = load(Path(__file__).with_name("ppi_study.py"))
    root = study.inside(args.output)
    if root.exists():
        raise ValueError("Refuse to overwrite a verification")
    root.mkdir(parents=True)
    fixture = root / "ENGINEERING_FIXTURE"
    fixture.mkdir()
    torch.set_num_threads(1)
    torch.manual_seed(991)
    graphs = []
    for nodes in (3, 4):
        pairs = [(i, j) for i in range(nodes) for j in range(nodes) if i != j]
        graphs.append(Data(x=torch.randn(nodes, 50), y=torch.randint(2, (nodes, 121)).float(),
                           edge_index=torch.tensor(pairs).T.contiguous()))
    torch.save(graphs, fixture / "synthetic_graphs.pt")
    for name, path in study.source_paths().items():
        target = fixture / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    study.write_json(fixture / "data_manifest.json", {"scope": "synthetic engineering fixture, no public dataset"})
    contract = {"arm": "gnnm", "seed": 0, "source_sha256": study.sources(),
                "training_fit": False, "engineering_fixture": True,
                "data_manifest_sha256": study.sha(fixture / "data_manifest.json")}
    study.write_json(fixture / "contract.json", contract)
    torch.manual_seed(0)
    model, mode, _ = study.impl.make_model("gnnm", 3)
    ops = [study.impl.MeanOperator.from_edges(g.edge_index, g.num_nodes, dtype=g.x.dtype) for g in graphs]
    metrics, logits = study.impl.evaluate(model, mode, graphs, ops)
    target_list = [g.y for g in graphs]
    torch.save(target_list, fixture / "validation_targets.pt")
    epochs = fixture / "validation_epochs"
    epochs.mkdir()
    rows = [{"epoch": epoch, **metrics} for epoch in range(1, 301)]
    with (fixture / "validation_trace.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for epoch in range(1, 301):
        torch.save(logits, epochs / f"epoch{epoch:03d}.pt")
    torch.save(logits, fixture / "selected_validation_member_logits.pt")
    study.write_json(fixture / "selected.json", rows[0])
    torch.save({"state_dict": model.state_dict(), "epoch": 1, "validation": metrics,
                "contract_sha256": study.sha(fixture / "contract.json")}, fixture / "selected.pt")
    study.write_json(fixture / "result.json", {"status": "completed_engineering_fixture", "selected": rows[0],
                                              "no_training_or_research_performance_result": True})
    study.artifact_manifest(fixture)
    clone = root / "LIVE_CODE_DRIFT_FIXTURE"
    shutil.copytree(fixture / "source", clone)
    live = clone / "protocols/ppi_study.py"
    with live.open("a") as stream:
        stream.write("\n# Deliberate engineering source drift; never a production modification.\n")
    results = []
    env = dict(os.environ)
    env.pop("GNNM_PHASE_ROOT", None)  # exercise archived phase inference
    for label, runner, allow, expected in (
        ("live_source_drift_rejected", live, True, 3),
        ("fixture_requires_explicit_flag", fixture / "source/protocols/ppi_study.py", False, 3),
        ("archived_source_replays_without_root_override", fixture / "source/protocols/ppi_study.py", True, 0),
    ):
        output = root / f"{label}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--child", "--runner", str(runner),
                   "--fixture", str(fixture), "--output", str(output)]
        if allow: command.append("--allow-fixture")
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
        (root / f"{label}.log").write_text(result.stdout + result.stderr)
        if result.returncode != expected:
            raise AssertionError(f"{label}: exit{result.returncode}; see preserved log")
        if expected == 3:
            required = "archived fitted source" if label.startswith("live") else "explicit engineering flag"
            if required not in result.stdout:
                raise AssertionError("Rejection was for an unrelated reason")
        results.append({"check": label, "exit_code": result.returncode, "status": "PASS"})
    study.write_json(root / "REPORT.json", {"status": "PASS", "checks": results,
            "scope": "synthetic source/root/trajectory-reader/selected-model replay engineering check",
            "dataset_accessed": False, "optimizer_steps": 0, "GPU_context": False,
            "no_research_performance_evidence": True,
            "source_sha256": study.sources(), "verifier_sha256": study.sha(Path(__file__))})
    print(json.dumps({"status": "PASS", "scope": "synthetic archived replay engineering fixture", "checks": results}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--child", action="store_true")
    p.add_argument("--runner", type=Path)
    p.add_argument("--fixture", type=Path)
    p.add_argument("--allow-fixture", action="store_true")
    args = p.parse_args()
    if args.child: sys.exit(child(args))
    parent(args)
