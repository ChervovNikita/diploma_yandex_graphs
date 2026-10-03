"""Verify local source packet custody with stdlib only; no data/model access."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

PACKET = Path(__file__).resolve().parent
RESEARCH = PACKET.parent


def read(name):
    return json.loads((PACKET / name).read_text())


def digest(path):
    b = path.read_bytes()
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


failures = []
counts = {}


def require(condition, message):
    if not condition:
        failures.append(message)


manifest = read("MANIFEST.json")
for record in manifest["files"]:
    p = PACKET / record["path"]
    require(p.is_file(), f"missing packet file: {record['path']}")
    if p.is_file():
        require(digest(p) == {k: record[k] for k in ["bytes", "sha256"]}, f"packet digest mismatch: {record['path']}")
counts["manifest_files"] = len(manifest["files"])
manifest_paths = {v["path"] for v in manifest["files"]}
observed_paths = {str(v.relative_to(PACKET)) for v in PACKET.rglob("*") if v.is_file() and v.name not in manifest["manifest_excludes"]}
require(manifest_paths == observed_paths, "packet inventory differs from manifest")

inputs = read("INPUT_BINDINGS.json")
for record in inputs["inputs"]:
    p = RESEARCH / record["path"]
    require(p.is_file() and digest(p) == {k: record[k] for k in ["bytes", "sha256"]}, f"input custody mismatch: {record['path']}")
counts["bound_inputs"] = len(inputs["inputs"])

successful = failed = 0
for receipt_file in sorted(PACKET.glob("RETRIEVAL*.json")):
    for receipt in json.loads(receipt_file.read_text()):
        if receipt["status"] == 200:
            successful += 1
            require(digest(PACKET / receipt["saved_name"]) == {k: receipt[k] for k in ["bytes", "sha256"]}, f"retrieval digest mismatch: {receipt['saved_name']}")
        else:
            failed += 1
            require(not (PACKET / receipt["saved_name"]).exists(), f"failed source incorrectly retained: {receipt['saved_name']}")
counts["successful_retained_retrievals"] = successful
counts["failed_access_receipts"] = failed

for passage in read("PRIMARY_EXCERPTS.json")["passages"]:
    file = PACKET / passage["page_text"]
    require(digest(file) == passage["page_text_binding"], f"page text digest mismatch: {passage['page_text']}")
    lines = file.read_text().splitlines()
    extracted = "\n".join(lines[passage["line_start"] - 1:passage["line_end"]])
    require(extracted == passage["extracted_text"], f"excerpt coordinate mismatch: {passage['page_text']}")
    require(hashlib.sha256(extracted.encode()).hexdigest() == passage["excerpt_utf8_sha256"], f"excerpt digest mismatch: {passage['page_text']}")
counts["primary_passages"] = len(read("PRIMARY_EXCERPTS.json")["passages"])

scopes = read("READ_SCOPES.json")
sources = read("CONCLUSIONS.json")["sources"]
require(scopes["new_primary_method_scopes"] == len(sources) == 3, "new primary method scope count mismatch")
require(scopes["new_primary_method_scopes"] <= scopes["maximum_authorized_new_primary_method_scopes"], "new scope limit exceeded")
require(scopes["full_primary_papers_read"] == 0 and scopes["author_code_scopes_read"] == 0, "bounded read accounting mismatch")
require(scopes["saved_dice_forde_adp_primary_reopens"] == 0, "retained primary reread accounting mismatch")
require([s["canonical_id"] for s in sources] == scopes["new_primary_method_identities"], "method identity mismatch")
index_text = (RESEARCH / "literature_memory/index_v31/LITERATURE_INDEX.json").read_text().casefold()
for source in sources:
    arxiv_id = source["canonical_id"].removeprefix("arxiv:")
    require(arxiv_id not in index_text and source["title"].casefold() not in index_text, f"source was already indexed: {arxiv_id}")
    require(digest(PACKET / source["pdf"]) == {k: source[k] for k in ["bytes", "sha256"]}, f"primary PDF binding mismatch: {source['pdf']}")
    for visual in source["visual_checks"]:
        require(visual in manifest_paths, f"visual check missing from packet: {visual}")
counts["new_primary_method_scopes"] = len(sources)

before = read("PRIOR_REPORT_CUSTODY_BEFORE.json")
after = read("PRIOR_REPORT_CUSTODY_AFTER.json")
require(len(before) == after["targets"] == len(after["files"]) == 270, "prior report custody count mismatch")
require(after["all_unchanged"], "prior report preservation failed at packet build")
for expected, saved in zip(before, after["files"]):
    require(expected["path"] == saved["path"], "prior custody path/order mismatch")
    target = RESEARCH / expected["path"]
    require(target.is_file() and digest(target) == {k: expected[k] for k in ["bytes", "sha256"]}, f"prior report custody mismatch: {expected['path']}")
counts["preserved_prior_reports_reviews"] = len(before)

result = {"schema": "source-packet-stdlib-verification-v1", "UTC": datetime.now(timezone.utc).isoformat(),
          "status": "PASS" if not failures else "FAIL", "counts": counts, "failures": failures,
          "scope": "File hashes, retrieval receipts, excerpt coordinates, reading counts and input/prior-report custody only; no algebraic numerical test, model/data operation or utility verdict."}
(PACKET / "VERIFICATION.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False))
raise SystemExit(bool(failures))
