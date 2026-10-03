"""Verify the bounded access packet and unchanged source bindings."""
import hashlib
import json
from pathlib import Path

packet = Path(__file__).resolve().parent
research = packet.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((packet / "MANIFEST.json").read_text())
listed = {f["path"] for f in manifest["files"]}
actual = {str(p.relative_to(packet)) for p in packet.rglob("*") if p.is_file() and p.name != "MANIFEST.json"}
assert listed == actual
assert len(listed) == manifest["payload_file_count"]
for f in manifest["files"]:
    path = packet / f["path"]
    assert sha(path) == f["sha256"] and path.stat().st_size == f["size"]
    if path.suffix == ".json":
        json.loads(path.read_text())
for binding in json.loads((packet / "INPUT_BINDINGS.json").read_text())["bindings"]:
    path = research / binding["path"]
    assert sha(path) == binding["sha256"] and path.stat().st_size == binding["size"]
routes = json.loads((packet / "ROUTE_RECEIPTS.json").read_text())
assert len(routes) == len({r["url"] for r in routes}) == 6
assert all(not r["previously_blocked_sciencedirect_url_retried"] for r in routes)
scopes = json.loads((packet / "READ_SCOPES.json").read_text())
assert scopes["new_primary_method_scopes"] == scopes["new_full_paper_reads_or_certifications"] == scopes["author_source_files_read"] == scopes["retained_primary_method_revisits"] == 0
assert routes[-1]["status"] == 403
print(json.dumps({"integrity": "PASS", "payload_files": len(listed), "manifest_sha256": sha(packet / "MANIFEST.json"), "report_sha256": sha(packet / "REPORT.md"), "routes": 6, "new_primary_method_scopes": 0, "full_paper_certifications": 0, "author_source_reads": 0}, indent=2))
