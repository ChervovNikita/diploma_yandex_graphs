"""Verify additive index custody only; no retrieval/model/data operations."""
from pathlib import Path
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
prior_file = ROOT / "literature_memory/index_v31/LITERATURE_INDEX.json"
old = json.loads(prior_file.read_text())
new = json.loads((OUT / "LITERATURE_INDEX.json").read_text())
receipt = json.loads((OUT / "APPEND_RECEIPT.json").read_text())
failures = []


def check(ok, message):
    if not ok:
        failures.append(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


check(sha(prior_file) == "109cb9b981db5320d509df9d5b4e1bd44336690a954f2c0621c7ca8673365c0c", "immutable v31 hash changed")
check(new["paper_records"][:130] == old["paper_records"], "older conclusion records changed")
check(new["existing_packets"][:165] == old["existing_packets"], "older catalog entries changed")
check(new["canonical_identifier_normalization"]["groups"][:87] == old["canonical_identifier_normalization"]["groups"], "older normalization groups changed")
check(len(new["paper_records"]) == 133, "conclusion count mismatch")
groups = new["canonical_identifier_normalization"]["groups"]
check(sum(g["kind"] == "paper" for g in groups) == 88, "paper identity count mismatch")
check(sum(g["kind"] != "paper" for g in groups) == 2, "software identity count mismatch")
check(len({g["normalized_identifier"] for g in groups}) == len(groups), "duplicate normalization group")
check(sha(OUT / "LITERATURE_INDEX.json") == receipt["new_index"]["sha256"], "new index receipt hash mismatch")
for record in new["paper_records"][130:]:
    check(sha(ROOT / record["conclusion_file"]) == record["conclusion_file_sha256"], "new conclusion binding mismatch")
    check(sha(ROOT / record["read_scope_reference"]["path"]) == record["read_scope_reference"]["sha256"], "new scope binding mismatch")
for item in new["existing_packets"][165:]:
    check(sha(ROOT / item["path"]) == item["sha256"], "new catalog binding mismatch: " + item["path"])
result = {"status": "PASS" if not failures else "FAIL", "failures": failures, "older_conclusion_records_preserved_exactly": 130, "older_catalog_entries_preserved_exactly": 165, "older_normalization_groups_preserved_exactly": 87, "new_conclusion_records": 3, "new_paper_identities": 3, "conclusion_records": 133, "paper_identities": 88, "software_identities": 2, "integration_primary_retrievals_or_reads": 0, "full_paper_read_total_certified": False}
(OUT / "VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
raise SystemExit(bool(failures))
