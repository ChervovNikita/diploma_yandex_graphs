"""Retrieve public primary locators once; preserve inaccessible responses."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent / 'primary'
HERE.mkdir(exist_ok=False)
urls = [('omoe_v1', 'https://arxiv.org/html/2501.10062v1'),
        ('htkge_publisher', 'https://doi.org/10.3390/sym16091166'),
        ('house_metadata', 'https://api.openalex.org/works?' + urllib.parse.urlencode(dict(search='HousE Knowledge Graph Embedding with Householder Parameterization', per_page=5)))]
records = []
for name, url in urls:
    started = datetime.now(timezone.utc).isoformat()
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'GNNM-research-literature/1.0'})
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read(3_000_001)
            if len(raw) > 3_000_000:
                raise RuntimeError('Payload exceeded bounded source retrieval')
            result = dict(status=response.status, resolved_url=response.url,
                          content_type=response.headers.get('Content-Type'))
        with (HERE / (name + '.source')).open('xb') as handle:
            handle.write(raw)
        result.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    except Exception as error:
        result = dict(status='FAILURE', error_type=type(error).__name__, error=str(error), automatic_retries=False)
    records.append(dict(name=name, url=url, start_UTC=started, terminal_UTC=datetime.now(timezone.utc).isoformat(),
                        primary_read=False, **result))
with (HERE / 'RETRIEVAL.json').open('x') as handle:
    json.dump(records, handle, indent=2)
    handle.write('\n')
print(json.dumps(records))
