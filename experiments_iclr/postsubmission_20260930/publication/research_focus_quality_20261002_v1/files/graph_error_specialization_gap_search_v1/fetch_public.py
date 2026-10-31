"""Public text retrieval only; no models, datasets, scientific runtime or remote hosts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent


def retrieve(item):
    start = datetime.now(timezone.utc).isoformat()
    req = urllib.request.Request(item['url'], headers={'User-Agent': 'Mozilla/5.0 (bounded academic literature review)'})
    path = HERE / item['file']
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {'key': item['key'], 'requested_url': item['url'], 'accessed_UTC': start,
              'file': item['file'], 'purpose': item['purpose']}
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            body = response.read()
            record.update(status=response.status, final_url=response.url,
                          content_type=response.headers.get('Content-Type'))
        path.write_bytes(body)
        record.update(success=True, bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    except urllib.error.HTTPError as error:
        body = error.read()
        path.write_bytes(body)
        record.update(success=False, status=error.code, final_url=error.url,
                      bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), error=str(error))
    except Exception as error:
        record.update(success=False, error_type=type(error).__name__, error=str(error))
    return record


def main():
    request = json.loads((HERE / sys.argv[1]).read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(retrieve, request))
    target = HERE / sys.argv[2]
    if target.exists():
        raise RuntimeError('Never replace an earlier retrieval log')
    target.write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps([{k:r.get(k) for k in ['key','success','status','bytes','error']} for r in records]))


if __name__ == '__main__':
    main()
