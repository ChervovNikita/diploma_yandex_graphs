"""Verify retained text, quotations and source integrity; not scientific validity."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    seal = json.loads((HERE / "SEAL.json").read_text())
    assert digest(HERE / "MANIFEST.json") == seal["manifest_sha256"]
    for row in manifest["files"]:
        path = HERE / row["path"]
        assert path.is_file() and path.stat().st_size == row["bytes"], row
        assert digest(path) == row["sha256"], row
    bindings = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    for row in bindings["external_read_only_bindings"]:
        path = PHASE / row["path"]
        assert path.stat().st_size == row["bytes"], row
        assert digest(path) == row["sha256"], row
    spec = importlib.util.spec_from_file_location("scoped_prior_extract", HERE / "EXTRACT_BLOCKS.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = module.extract(HERE / "primary/2602_15747v1_html.html")
    retained = json.loads((HERE / "primary/2602_15747v1_blocks.json").read_text())
    assert actual == retained
    passages = json.loads((HERE / "PRIMARY_PASSAGES.json").read_text())
    for group in passages["new_method_scope"]:
        for passage in group["passages"]:
            assert passage == actual[passage["index"]]
    for row in passages["reused_lora_ga"]["scope"]:
        text = Path(row["source"]["path"]).read_text()
        for passage in row["passages"]:
            assert text[passage["start"]:passage["end_exclusive"]] == passage["text"]
    incidental = passages["incidental_unadopted_gradient_starvation"]
    original = json.loads((PHASE / incidental["binding"]["path"]).read_text())
    for passage in incidental["passages"]:
        assert original[passage["index"]] == passage
    reads = json.loads((HERE / "READ_SCOPES.json").read_text())
    assert reads["accounting"]["new_primary_scope_papers_budget_used"] == 2
    assert reads["accounting"]["new_full_paper_reads"] == 0
    return {
        "status": "PASS", "local_manifest_files": len(manifest["files"]),
        "external_bindings": len(bindings["external_read_only_bindings"]),
        "new_quotation_groups": len(passages["new_method_scope"]),
        "scope": "Exact retained bytes/quotation custody and read-accounting checks only",
        "novelty_or_predictive_validity_verified": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
