from pathlib import Path
import concurrent.futures
import datetime
import hashlib
import json
import urllib.request
import urllib.error
import urllib.parse
import sys

ROOT = Path(__file__).resolve().parent
(ROOT / 'sources').mkdir(exist_ok=True)
(ROOT / 'discovery').mkdir(exist_ok=True)

def fetch(task):
    key, url, relative, role = task
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt = {'key': key, 'url': url, 'requested_UTC': started, 'role': role}
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Public scholarly literature scope retrieval/1.0'})
        with urllib.request.urlopen(req, timeout=35) as r:
            body = r.read()
            receipt.update(http_status=r.status, final_url=r.url,
                           content_type=r.headers.get('Content-Type'))
        path = ROOT / relative
        path.write_bytes(body)
        receipt.update(path=relative, bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
                       read_status='retrieved_only_semantic_scope_pending')
    except urllib.error.HTTPError as e:
        body = e.read()
        path = ROOT / ('discovery/' + key + '_failure_body.bin')
        path.write_bytes(body)
        receipt.update(http_status=e.code, error=str(e), failure_body_path=str(path.relative_to(ROOT)),
                       failure_bytes=len(body), failure_sha256=hashlib.sha256(body).hexdigest(),
                       read_status='failed_route_not_method_evidence')
    except Exception as e:
        receipt.update(error=type(e).__name__ + ': ' + str(e), read_status='failed_route_not_method_evidence')
    receipt['received_UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return receipt

if __name__ == '__main__':
    tasks=json.loads(sys.stdin.read())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts=list(pool.map(fetch,tasks))
    ledger=ROOT/'RETRIEVAL_LEDGER.json'
    old=json.loads(ledger.read_text()) if ledger.exists() else []
    ledger.write_text(json.dumps(old+receipts,indent=2)+'\n')
    for r in receipts:
        print(json.dumps(r))
