"""Check packet and reused-summary custody, without importing author code."""
import hashlib
import json
from pathlib import Path

packet = Path(__file__).resolve().parent
research = packet.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((packet / "MANIFEST.json").read_text())
assert len(manifest["files"]) == manifest["payload_file_count"]
listed = {item["path"] for item in manifest["files"]}
actual = {str(path.relative_to(packet)) for path in packet.rglob("*") if path.is_file() and path.name != "MANIFEST.json"}
assert listed == actual, (listed - actual, actual - listed)
for item in manifest["files"]:
    path = packet / item["path"]
    assert path.stat().st_size == item["size"], item["path"]
    assert sha(path) == item["sha256"], item["path"]
    if path.suffix == ".json":
        json.loads(path.read_text())
provenance = json.loads((packet / "PROVENANCE.json").read_text())
for item in provenance["external_unchanged_bindings"]:
    path = research / item["path"]
    assert path.stat().st_size == item["size"], item["path"]
    assert sha(path) == item["sha256"], item["path"]
read_scopes = json.loads((packet / "READ_SCOPES.json").read_text())
assert len(read_scopes["papers"]) == read_scopes["new_primary_method_scopes"] == 3
assert read_scopes["new_full_paper_certifications"] == 0
for paper in read_scopes["papers"]:
    assert not paper["full_paper_read"]
    assert sha(packet / paper["pdf"]["path"]) == paper["pdf"]["sha256"]
assert json.loads((packet / "PILOT_DECISION.json").read_text())["pilots_proposed"] == 0
assert json.loads((packet / "HGEN_SOURCE_CHECK.json").read_text())["static_parse"]["exception"] == "TabError"
print(json.dumps({"integrity": "PASS", "payload_files": len(listed), "external_unchanged_bindings": len(provenance["external_unchanged_bindings"]), "report_sha256": sha(packet / "REPORT.md"), "manifest_sha256": sha(packet / "MANIFEST.json"), "new_primary_method_scopes": 3, "full_paper_certifications": 0, "pilots_proposed": 0}, indent=2))
