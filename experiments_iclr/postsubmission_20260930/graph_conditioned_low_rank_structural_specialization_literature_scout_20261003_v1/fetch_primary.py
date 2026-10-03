"""Retrieve primary article HTML; no downloaded content is executed."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
key, version = sys.argv[1:]
directory = ROOT / "primary"
directory.mkdir(exist_ok=True)
url = "https://arxiv.org/html/" + version
receipt = {"UTC": datetime.now(timezone.utc).isoformat(), "key": key, "version": version,
           "URL": url, "retrieval_is_not_method_or_whole_paper_read": True}
try:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Scoped-research-scout/1.0"}), timeout=30) as response:
        data = response.read()
        receipt.update(status=response.status, final_URL=response.url)
    path = directory / (key + ".html")
    path.write_bytes(data)
    receipt["output"] = {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
except Exception as error:
    receipt["error"] = type(error).__name__ + ": " + str(error)
(directory / (key + "_RECEIPT.json")).write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
