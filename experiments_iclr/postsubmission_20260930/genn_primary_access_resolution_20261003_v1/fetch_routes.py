"""Bounded public source requests. No models, datasets or experiment artifacts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

PACKET = Path(__file__).resolve().parent
RECEIPT = PACKET / "ROUTE_RECEIPTS.json"
BLOCKED_URL = "https://www.sciencedirect.com/science/article/abs/pii/S1566253524002392"


def fetch(spec):
    assert spec["url"] != BLOCKED_URL
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "Mozilla/5.0 (bounded academic source retrieval)", "Accept": spec.get("accept", "application/json, application/xml;q=0.9, text/html;q=0.8")})
    result = {**spec, "requested_UTC": datetime.now(timezone.utc).isoformat(), "previously_blocked_sciencedirect_url_retried": False}
    body = None
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read(16_000_001)
            assert len(body) <= 16_000_000
            result.update(status=response.status, actual_response_url=response.geturl(), response_content_type=response.headers.get("Content-Type"))
    except urllib.error.HTTPError as error:
        body = error.read(1_000_000)
        result.update(status=error.code, error=str(error), actual_response_url=error.geturl(), response_content_type=error.headers.get("Content-Type"))
    except Exception as error:
        result.update(status="network_error", error=repr(error))
    if body is not None:
        content_type = result.get("response_content_type", "") or ""
        extension = ".json" if "json" in content_type else ".xml" if "xml" in content_type or "atom" in content_type else ".pdf" if body.startswith(b"%PDF") else ".txt"
        path = PACKET / "routes" / (spec["key"] + extension)
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(body)
        result.update(saved_path=str(path.relative_to(PACKET)), sha256=hashlib.sha256(body).hexdigest(), size=len(body))
    return result


def main():
    existing = json.loads(RECEIPT.read_text()) if RECEIPT.exists() else []
    requested = json.loads(sys.argv[1])
    assert len(existing) + len(requested) <= 6, "At most six targeted request routes total"
    assert len({r["url"] for r in existing + requested}) == len(existing) + len(requested), "No route retries"
    with ThreadPoolExecutor(max_workers=len(requested)) as executor:
        results = list(executor.map(fetch, requested))
    RECEIPT.write_text(json.dumps(existing + results, ensure_ascii=False, indent=2) + "\n")
    for result in results:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
