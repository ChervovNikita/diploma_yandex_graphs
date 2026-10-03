#!/usr/bin/env python3
"""Read-only integrity/source-binding audit. Never import or run author code."""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
CONTROL = {"MANIFEST.json", "SEAL.json"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(name):
    return json.loads((ROOT / name).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_path(name):
    p = ROOT / name
    require(p.resolve().is_relative_to(ROOT), f"Outside packet: {name}")
    require(p.is_file() and not p.is_symlink(), f"Invalid payload: {name}")
    return p


def validate_sources():
    retrieval = read_json("SOURCE_RETRIEVAL.json")
    repos = read_json("REPOSITORY_RETRIEVAL.json")
    trees = {r["repository"]: read_json(r["tree_file"]) for r in repos}
    require(len(retrieval) == 24, "Source unit count")
    seen = set()
    raw_checked = 0
    for r in retrieval:
        key = (r["repository"], r["commit"], r["path"])
        require(key not in seen, f"Duplicate receipt {key}")
        seen.add(key)
        require(r["http_status"] == 200, f"HTTP receipt {key}")
        t = trees[r["repository"]]
        require(not t["truncated"], "Truncated tree")
        require(t["commit"] == r["commit"], "Pin mismatch")
        entry = {x["path"]: x for x in t["entries"]}[r["path"]]
        require(entry["type"] == "blob", "Nonblob source")
        require(entry["sha"] == r["blob_sha1"], "Tree/receipt blob mismatch")
        require(entry["size"] == r["bytes"], "Tree/receipt length mismatch")
        data = local_path(r["saved_path"]).read_bytes()
        require(digest(data) == r["saved_sha256"], "Saved source changed")
        if r["path"] != "README.md":
            require(digest(data) == r["raw_sha256"], "Raw/saved hash mismatch")
            require(len(data) == r["bytes"], "Raw/saved size mismatch")
            blob = b"blob " + str(len(data)).encode() + b"\0" + data
            require(hashlib.sha1(blob).hexdigest() == entry["sha"], "Raw blob mismatch")
            raw_checked += 1
        else:
            require(all(not line.split(": ", 1)[-1].lstrip().startswith("|")
                        for line in data.decode().splitlines()), "README pipe table retained")
    ncn_paths = [x["path"].lower() for x in trees["GraphPKU/NeuralCommonNeighbor"]["entries"]]
    require(not any("license" in x or "copying" in x for x in ncn_paths), "NCN license metadata changed")
    license_text = local_path("source/pencil__LICENSE").read_text()
    require("Apache License" in license_text and "Version 2.0, January 2004" in license_text,
            "PENCIL license title")
    require(local_path("source/pencil__args__official__ogbl_collab_bert.yaml").read_bytes()
            == local_path("source/pencil__args__official__heart_ogbl_collab_bert.yaml").read_bytes(),
            "Config duplicate assertion")
    return retrieval, raw_checked


def validate_scope_and_sites(retrieval):
    scope = read_json("READ_SCOPE.json")
    require(scope["retrieved_source_units"] == 24, "Read scope retrieval count")
    require(scope["distinct_inspected_source_units"] == 22, "Read scope inspected count")
    require(scope["semantic_inspected_source_units"] == 21, "Read scope semantic count")
    require(scope["semantic_author_code_units"] == 15, "Read scope code count")
    require(scope["new_semantic_primary_paper_reads"] == 0, "Primary read credit")
    require(all(v == 0 for v in scope["prohibited_access_counts"].values()), "Prohibited count assertion")
    receipts = {(r["repository"], r["path"]): r for r in retrieval}
    for unit in scope["units"]:
        r = receipts[(unit["repository"], unit["path"])]
        require(unit["commit"] == r["commit"] and unit["saved_path"] == r["saved_path"],
                "Read scope pin/path mismatch")
        ranges = unit["ranges"]
        if isinstance(ranges, list):
            lines = len(local_path(unit["saved_path"]).read_text().splitlines())
            require(all(1 <= a <= b <= lines for a, b in ranges), "Read range out of source")
    with local_path("SOURCE_SITES.csv").open(newline="") as f:
        sites = list(csv.DictReader(f))
    require(len(sites) == 46, "Source site count")
    for row in sites:
        r = receipts[(row["repository"], row["author_path"])]
        require(row["commit"] == r["commit"] and row["saved_path"] == r["saved_path"],
                "Source site pin/path mismatch")
        lines = len(local_path(row["saved_path"]).read_text().splitlines())
        require(1 <= int(row["start_line"]) <= int(row["end_line"]) <= lines,
                "Source site line range invalid")
    boundaries = read_json("PROTOCOL_BOUNDARIES.json")
    require(not boundaries["adopted"] and not boundaries["new_seed_freeze"]
            and not boundaries["new_arm"] and boundaries["no_running_buddy_changes"],
            "Adoption boundary assertion")
    cost = read_json("RESOURCE_UNKNOWNS.json")
    require(not cost["execution_performed"] and not cost["compute_demand"]
            and cost["resource_request"] is None
            and not cost["author_numeric_accuracy_or_cost_claims_used"], "Cost/claim assertion")
    decision = read_json("DECISION.json")
    require(not decision["ncn_ncnc"]["runtime_verified"]
            and not decision["recent_competitor"]["runtime_verified"]
            and not decision["recent_competitor"]["superiority_established"], "Readiness limit assertion")
    recipes = read_json("NATIVE_RECIPES.json")
    require(recipes["not_executed"] and recipes["not_adopted"], "Native recipe boundary")
    require(recipes["pencil"]["native_year_minimum"] == 2007
            and recipes["pencil"]["training_query_sampling"]["percent"] == 50
            and recipes["pencil"]["edge_weights_in_tokens"] is False
            and recipes["pencil"]["training"]["gradient_accumulation_steps_total"] == 8
            and recipes["pencil"]["evaluation"]["eval_test_during_training"] is True,
            "Native recipe binding assertion")
    return len(sites)


def validate_retained_inputs():
    reused = read_json("REUSED_CONCLUSIONS.json")
    require(reused["new_primary_semantic_paper_reads"] == 0, "Retained primary credit")
    for x in reused["inputs"]:
        data = Path(x["path"]).read_bytes()
        require(digest(data) == x["sha256"] and len(data) == x["bytes"],
                f"Retained input changed: {x['path']}")
    # Read saved href receipts, not saved HTML bodies or any model/data/result file.
    require(len(reused["saved_primary_href_only"]) == 2, "Href locator count")
    return len(reused["inputs"])


def validate_seal():
    manifest_bytes = local_path("MANIFEST.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    seal = read_json("SEAL.json")
    require(digest(manifest_bytes) == seal["manifest_sha256"], "Manifest hash mismatch")
    entries = manifest["files"]
    expected = {x["path"] for x in entries}
    actual = {str(x.relative_to(ROOT)) for x in ROOT.rglob("*")
              if x.is_file() and str(x.relative_to(ROOT)) not in CONTROL}
    require(expected == actual, "Packet file set mismatch")
    require(len(entries) == seal["payload_file_count"] == manifest["payload_file_count"],
            "Manifest count mismatch")
    require(len(expected) == len(entries), "Manifest duplicate path")
    for x in entries:
        data = local_path(x["path"]).read_bytes()
        require(len(data) == x["bytes"] and digest(data) == x["sha256"],
                f"Payload mismatch: {x['path']}")
    return len(entries), seal["manifest_sha256"]


def main():
    preseal = sys.argv[1:] == ["--preseal"]
    require(preseal or not sys.argv[1:], "Use no args, or --preseal before sealing")
    retrieval, raw_checked = validate_sources()
    sites = validate_scope_and_sites(retrieval)
    inputs = validate_retained_inputs()
    result = {"status": "PASS", "scope": "read-only integrity/static source binding, not runtime/scientific verification",
              "source_units": len(retrieval), "raw_nonreadme_blob_checks": raw_checked,
              "readme_excerpt_checks": 2, "source_sites": sites, "retained_input_hash_checks": inputs,
              "model_imports_or_execution": 0, "data_checkpoint_result_log_access": 0}
    if not preseal:
        count, manifest_hash = validate_seal()
        result.update(payload_file_count=count, manifest_sha256=manifest_hash, sealed=True)
    else:
        result["sealed"] = False
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
