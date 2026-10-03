"""Verify the prospective packet and unchanged saved evidence."""
import hashlib
import json
from pathlib import Path

packet = Path(__file__).resolve().parent
research = packet.parent
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads((packet / "MANIFEST.json").read_text())
listed = {f["path"] for f in manifest["files"]}
actual = {str(path.relative_to(packet)) for path in packet.rglob("*") if path.is_file() and path.name != "MANIFEST.json"}
assert listed == actual and len(listed) == manifest["payload_file_count"]
for item in manifest["files"]:
    path = packet / item["path"]
    assert sha(path) == item["sha256"] and path.stat().st_size == item["size"]
    if path.suffix == ".json":
        json.loads(path.read_text())
provenance = json.loads((packet / "PROVENANCE.json").read_text())
for item in provenance["bindings"]:
    path = research / item["path"]
    assert sha(path) == item["sha256"] and path.stat().st_size == item["size"]
pilot = json.loads((packet / "PILOT_SPEC.json").read_text())
assert pilot["pilots_proposed"] == 1 and pilot["pilots_executed"] == 0
assert len(pilot["fixed_continuation_arms"]) == 8
assert not pilot["launch_authorized_by_this_packet"]
scopes = json.loads((packet / "READ_SCOPES.json").read_text())
assert scopes["new_primary_method_scopes"] == scopes["full_paper_certifications"] == scopes["author_source_reads_or_audits"] == 0
assert json.loads((packet / "CANDIDATE.json").read_text())["candidate_count"] == 1
print(json.dumps({"integrity": "PASS", "payload_files": len(listed), "external_bindings_unchanged": len(provenance["bindings"]), "manifest_sha256": sha(packet / "MANIFEST.json"), "report_sha256": sha(packet / "REPORT.md"), "new_primary_method_scopes": 0, "full_paper_certifications": 0, "candidate_count": 1, "prospective_pilots": 1, "pilots_executed": 0}, indent=2))
