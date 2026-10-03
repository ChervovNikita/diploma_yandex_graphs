"""Read-only stdlib custody/scope verification; no model or study imports."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    seal = read("SEAL.json")
    manifest_bytes = (ROOT / "MANIFEST.json").read_bytes()
    check(sha(manifest_bytes) == seal["manifest_sha256"], "Manifest custody")
    manifest = json.loads(manifest_bytes)
    entries = manifest["files"]
    check(len(entries) == seal["payload_files"], "Payload count")
    actual = {str(x.relative_to(ROOT)) for x in ROOT.rglob("*") if x.is_file()
              and x.name not in {"MANIFEST.json", "SEAL.json"}}
    check(actual == {x["path"] for x in entries}, "File set")
    for x in entries:
        p = ROOT / x["path"]
        check(p.resolve().is_relative_to(ROOT) and not p.is_symlink(), "Payload path")
        b = p.read_bytes()
        check(sha(b) == x["sha256"] and len(b) == x["bytes"], "Payload custody")
    records = []
    for receipt in ("PRIMARY_RETRIEVAL.json", "PRIMARY_RECENT_RETRIEVAL.json"):
        records += read(receipt)["papers"]
    check(len(records) == 6, "Primary ID count")
    source_requests = 0
    for record in records:
        check(len(record["requests"]) == 2, "Abs/HTML request count")
        for request in record["requests"]:
            check(request["status"] == 200, "Primary retrieval failed")
            b = (ROOT / request["saved_path"]).read_bytes()
            check(sha(b) == request["sha256"] and len(b) == request["bytes"], "Primary custody")
            check(record["id"] in request["url"], "Unversioned request")
            source_requests += 1
    scope = read("READ_SCOPES.json")
    check(scope["new_unique_primary_method_papers"] == 6 and scope["new_full_paper_reads"] == 0,
          "Primary credit")
    pp = hh = mm = 0
    for record in scope["paper_records"]:
        ident = record["id"]
        blocks = read(f"primary/{ident}_blocks.json")
        selected = []
        for a, b in record["ranges_zero_based_inclusive"]:
            check(0 <= a <= b < len(blocks), "Block range")
            selected += blocks[a:b + 1]
        paragraphs = sum(x["tag"] == "p" for x in selected)
        headings = len(selected) - paragraphs
        check(paragraphs == record["paragraphs"] and headings == record["headings"], "Block counts")
        expected = "\n\n".join(f"[{x['index']}] {x['text']}" for x in selected) + "\n"
        check(expected == (ROOT / f"excerpts/{ident}_method.txt").read_text(), "Method excerpt")
        maths = read(f"primary/{ident}_math.json")
        math_selected = [maths[i] for i in record["math_indices_zero_based"]]
        check(math_selected == read(f"excerpts/{ident}_math.json"), "Formula excerpt")
        pp += paragraphs
        hh += headings
        mm += len(math_selected)
    check((pp, hh, mm) == (162, 34, 42), "Scope totals")
    check(all(v == 0 for v in scope["execution_counts"].values()), "Scope assertion")
    reused = read("REUSED_CONCLUSIONS.json")
    check(sha(Path(reused["index_path"]).read_bytes()) == reused["index_sha256"], "Index snapshot")
    index = json.loads(Path(reused["index_path"]).read_text())
    known = {re.sub(r"v\d+$", "", x["canonical_id"].lower().split(":")[-1])
             for x in index["paper_records"]}
    check(all(re.sub(r"v\d+$", "", x["id"]) not in known for x in records), "New ID claim")
    operator = read("OPERATOR_AND_GAP.json")
    check(operator["root_owns_execution_and_adoption"], "Owner assertion")
    variant = read("ONE_OPTIONAL_VARIANT.json")
    check(variant["count"] <= 1 and not variant["changes_to_running_comparison"]
          and not variant["new_arm_seed_protocol_or_resource_demand"], "Variant scope")
    print(json.dumps({"status": "PASS", "payload_files": len(entries), "primary_papers": 6,
                      "versioned_abs_html_receipts": source_requests, "paragraphs": pp,
                      "headings": hh, "math_nodes_or_fragments": mm,
                      "manifest_sha256": seal["manifest_sha256"],
                      "scope": "read-only custody/scope assertions, not scientific/runtime verification"}, indent=2))


if __name__ == "__main__":
    main()
