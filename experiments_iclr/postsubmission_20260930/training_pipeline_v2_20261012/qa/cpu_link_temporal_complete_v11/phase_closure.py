"""Close distributed phases from complete per-host registry evidence.

Host receipts are copied through the project's normal synchronization route.
The collector cannot invent missing host results or waive failed comparisons.
"""
import argparse
import hashlib
import json
from pathlib import Path

from spec import canonical, confined, fingerprint


def expected_keys(plan, runs, placement, gate_name, host):
    stages = set(plan["global_gates"][gate_name]["source_stages"])
    return sorted(key for key, record in runs.items()
                  if placement[key]["worker"].split("/")[0] == host
                  and record["run"]["config"].get("stage") in stages)


def write_host_receipts(root, plan, digest, runs, placement, registry, host, already_closed=None):
    """Called by supervisor; emits each immutable proof once at actual closure."""
    root = Path(root).resolve()
    states = registry.states()
    already_closed = already_closed if already_closed is not None else set()
    for name, gate in plan.get("global_gates", {}).items():
        if name in already_closed:
            continue
        if host not in gate["required_hosts"]:
            continue
        keys = expected_keys(plan, runs, placement, name, host)
        if not keys:
            raise ValueError("A required host has no declared phase jobs")
        # Failed search fits remain part of closure, provided every selection
        # endpoint succeeds. A failed comparative/calibration bank blocks it.
        def acceptable(key):
            row = runs[key]["run"]
            return states[key] == "complete" or (row["kind"] == "model_fit"
                and row["config"].get("stage") == "search" and states[key] == "failed")
        if not all(acceptable(key) for key in keys):
            continue
        path = confined(root, str(Path(gate["path"]).parent / (host + "_" + name + ".json")))
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("plan_hash") != digest or old.get("run_keys") != keys:
                raise ValueError("A phase proof from another plan already exists")
            already_closed.add(name)
            continue
        results = {}
        for key in keys:
            record = runs[key]
            row = registry.run(record["study"], record["run"]["run_id"])
            results[key] = {"state": row["state"], "result_hash": fingerprint(row["result"]),
                            "contract_hash": fingerprint(row["contract"])}
        value = {"status": "complete", "host": host, "closed_phase": name,
                 "plan_hash": digest, "run_keys": keys, "results": results,
                 "closure_authority": "whole_assigned_host_registry"}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(canonical(value)+"\n")
        already_closed.add(name)


def collect(root, plan, digest, runs, placement, gate_name):
    root = Path(root).resolve()
    gate = plan["global_gates"][gate_name]
    target = confined(root, gate["path"])
    if target.exists():
        raise ValueError("A global phase closure already exists; retain it unchanged")
    proofs = {}
    for host in gate["required_hosts"]:
        path = target.parent / (host + "_" + gate_name + ".json")
        raw = path.read_bytes()
        value = json.loads(raw)
        keys = expected_keys(plan, runs, placement, gate_name, host)
        if (value.get("status") != "complete" or value.get("host") != host
                or value.get("closed_phase") != gate_name or value.get("plan_hash") != digest
                or value.get("run_keys") != keys or set(value.get("results", {})) != set(keys)):
            raise ValueError("A copied host receipt omits or changes the assigned cohort")
        for key in keys:
            row = runs[key]["run"]
            result = value["results"][key]
            failed_search = row["kind"] == "model_fit" and row["config"].get("stage") == "search"
            if result["state"] != "complete" and not (failed_search and result["state"] == "failed"):
                raise ValueError("Phase receipt contains an incomplete required comparison")
            if result["contract_hash"] != fingerprint(row) or not result["result_hash"]:
                raise ValueError("Host proof has different run contracts")
        proofs[host] = {"path": str(path.relative_to(Path(root))), "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()}
    value = {"status": "complete", "plan_hash": digest, "closed_phase": gate_name,
             "host_receipts": proofs, "closure_authority": "all_declared_hosts_with_exact_contracts"}
    with target.open("x") as stream:
        stream.write(canonical(value)+"\n")
    return value


def main():
    from launcher import load_plan
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--gate", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    plan, digest, _, runs, placement, _ = load_plan(root, args.plan)
    print(canonical(collect(root, plan, digest, runs, placement, args.gate)))


if __name__ == "__main__":
    main()
