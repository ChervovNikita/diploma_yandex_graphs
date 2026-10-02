"""Bounded public source retrieval; no numerical/scientific execution."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
for sub in ['sources', 'discovery', 'evidence']:
    (ROOT / sub).mkdir(exist_ok=True)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def fetch(item):
    name, url, relative, purpose = item
    record = {'id': name, 'requested_url': url, 'purpose': purpose,
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'timeout_seconds': 18, 'automatic_retries': 0}
    request = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (compatible; source-literature-review/1.0)',
        'Accept': 'application/pdf,text/html,application/json,*/*'})
    try:
        with urllib.request.urlopen(request, timeout=18) as response:
            body = response.read(20 * 1024 * 1024 + 1)
            if len(body) > 20 * 1024 * 1024:
                raise ValueError('Response exceeds 20 MiB retrieval limit')
            record.update(status='retrieved', http_status=response.status,
                          final_url=response.url, headers=dict(response.headers),
                          sha256=sha(body), bytes=len(body), saved_path=relative,
                          pdf_magic=body.startswith(b'%PDF-'))
            (ROOT / relative).write_bytes(body)
    except urllib.error.HTTPError as exc:
        body = exc.read(1024 * 1024)
        failure_path = 'discovery/' + name + '_http_error.bin'
        (ROOT / failure_path).write_bytes(body)
        record.update(status='http_error', http_status=exc.code, final_url=exc.url,
                      headers=dict(exc.headers), error=str(exc),
                      failure_body_path=failure_path,
                      failure_body_sha256=sha(body), failure_body_bytes=len(body))
    except Exception as exc:
        record.update(status='transport_error', error_type=type(exc).__name__, error=str(exc))
    record['completed_utc'] = datetime.now(timezone.utc).isoformat()
    return record

def run_batch(batch_name, items):
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, items))
    out = ROOT / 'discovery' / (batch_name + '.json')
    out.write_text(json.dumps(records, indent=2) + '\n')
    for r in records:
        print(r['id'], r['status'], r.get('http_status'), r.get('bytes'), r.get('pdf_magic'), r.get('error', ''))
    return records

if __name__ == '__main__':
    run_batch('retrieval_batch1', [
        ('morgan_pdf_fresh', 'https://ojs.aaai.org/index.php/AAAI/article/download/39553/43514', 'sources/morgan_publisher_response.bin', 'Fresh canonical version-of-record PDF attempt; prior RemoteDisconnected retained'),
        ('fagel_pdf_fresh', 'https://link.springer.com/content/pdf/10.1007/978-3-032-37657-2_35.pdf', 'sources/fagel_publisher_response.bin', 'Publisher chapter PDF attempt'),
        ('morgan_crossref', 'https://api.crossref.org/works/10.1609/aaai.v40i28.39553', 'discovery/morgan_crossref.json', 'Canonical identity and registered full-text links'),
        ('fagel_crossref', 'https://api.crossref.org/works/10.1007/978-3-032-37657-2_35', 'discovery/fagel_crossref.json', 'Canonical identity and registered full-text links'),
        ('morgan_openalex', 'https://api.openalex.org/works/https://doi.org/10.1609/aaai.v40i28.39553', 'discovery/morgan_openalex.json', 'Public OA-location metadata discovery, not a primary read'),
        ('fagel_openalex', 'https://api.openalex.org/works/https://doi.org/10.1007/978-3-032-37657-2_35', 'discovery/fagel_openalex.json', 'Public OA-location metadata discovery, not a primary read'),
        ('morgan_releases', 'https://api.github.com/repos/lihuiliullh/Morgan/releases', 'discovery/morgan_releases.json', 'Official author-repository public paper releases discovery'),
        ('fagel_releases', 'https://api.github.com/repos/Chrisshen12/FAGEL/releases', 'discovery/fagel_releases.json', 'Official author-repository public paper releases discovery'),
    ])
