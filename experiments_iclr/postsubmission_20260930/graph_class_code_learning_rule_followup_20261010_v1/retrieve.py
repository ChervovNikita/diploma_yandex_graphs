"""Bounded literature retrieval only. Does not access project data or outcomes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
from urllib.parse import quote
from datetime import datetime, timezone
import json, hashlib, time

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"
SOURCES.mkdir(exist_ok=True)
REQUESTS = [
    ("anytime_hierarchical_ensemble_ar5iv.html", "https://ar5iv.labs.arxiv.org/html/2003.01474"),
    ("bcnn_ar5iv.html", "https://ar5iv.labs.arxiv.org/html/1709.09890"),
    ("spectral_ecoc_openalex.json", "https://api.openalex.org/works/https://doi.org/10.1109/ICCV.2009.5459355"),
    ("learn_hierarchy_openalex.json", "https://api.openalex.org/works/https://doi.org/10.1007/s10489-020-02103-6"),
]
for name, query in [
    ("multi_hierarchy_discovery.json", "ensemble multiple label hierarchies neural classification"),
    ("class_graph_ecoc_discovery.json", "graph class similarity error correcting output codes"),
    ("hierarchy_loss_ensemble_discovery.json", "diverse hierarchy classification ensembles loss"),
    ("shared_ecoc_discovery.json", "neural error correcting output codes shared backbone"),
]:
    REQUESTS.append((name, "https://api.openalex.org/works?search=" + quote(query) + "&per-page=5"))

def retrieve(item):
    name, url = item
    start = time.monotonic()
    receipt = {"url": url, "UTC": datetime.now(timezone.utc).isoformat(), "intended_scope": "mechanical retrieval; semantic scope declared separately"}
    try:
        with urlopen(Request(url, headers={"User-Agent": "Research source verification/1.0"}), timeout=25) as response:
            body = response.read()
            receipt.update({"final_url": response.url, "http_status": response.status, "content_type": response.headers.get("content-type")})
        p = SOURCES / name
        p.write_bytes(body)
        receipt.update({"status": "retrieved", "path": str(p.relative_to(ROOT)), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
    except Exception as exc:
        receipt.update({"status": "failed", "error": repr(exc)})
    receipt["seconds"] = time.monotonic() - start
    return receipt

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(retrieve, REQUESTS))
    (ROOT / "RETRIEVALS.json").write_text(json.dumps(receipts, indent=2) + "\n")
    for r in receipts:
        print(json.dumps(r))
