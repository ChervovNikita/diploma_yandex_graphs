"""Scoped public arXiv metadata retrieval, no method or numerical execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
QUERIES = {
    "graphlora": 'all:"GraphLoRA"',
    "context_lowrank": 'all:"graph" AND (all:"low-rank" OR all:"LoRA") AND (all:"mixture" OR all:"context" OR all:"adaptive" OR all:"condition")',
    "structural_experts": '(all:"GraphMETRO" OR all:"graph" AND all:"structural" AND all:"experts")',
}
key = sys.argv[1]
url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": QUERIES[key], "start": 0, "max_results": 15, "sortBy": "relevance", "sortOrder": "descending"})
receipt = {"UTC": datetime.now(timezone.utc).isoformat(), "key": key, "query": QUERIES[key], "URL": url, "scope": "discovery metadata only; no method read"}
directory = ROOT / "discovery"
directory.mkdir(exist_ok=True)
try:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Scoped-research-scout/1.0"}), timeout=30) as response:
        data = response.read()
        receipt.update(status=response.status, final_URL=response.url)
    path = directory / (key + ".xml")
    path.write_bytes(data)
    receipt["output"] = {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
    tree = ET.fromstring(data)
    receipt["total_results"] = int(tree.findtext("o:totalResults", default="0", namespaces=ns))
    receipt["entries"] = [{"id": row.findtext("a:id", namespaces=ns), "title": " ".join(row.findtext("a:title", default="", namespaces=ns).split()), "published": row.findtext("a:published", namespaces=ns), "summary": " ".join(row.findtext("a:summary", default="", namespaces=ns).split())} for row in tree.findall("a:entry", ns)]
except Exception as error:
    receipt["error"] = type(error).__name__ + ": " + str(error)
(directory / (key + "_RECEIPT.json")).write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"key": key, "total": receipt.get("total_results"), "entries": [{k: row[k] for k in ("id", "title")} for row in receipt.get("entries", [])], "error": receipt.get("error")}, indent=2))
