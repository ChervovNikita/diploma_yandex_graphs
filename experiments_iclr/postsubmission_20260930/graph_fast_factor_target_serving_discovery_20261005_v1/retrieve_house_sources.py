from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.request

HERE = Path(__file__).resolve().parent / 'primary'
records = []
for name, paper in [('house_v1', '2202.07919v1'), ('uop_v1', '2405.08540v1')]:
    url = 'https://arxiv.org/html/' + paper
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'GNNM-research-literature/1.0'}), timeout=20) as response:
            raw = response.read(3_000_001)
            if len(raw) > 3_000_000:
                raise RuntimeError('Primary payload exceeds bound')
            value = dict(status=response.status, resolved_url=response.url, content_type=response.headers.get('Content-Type'))
        with (HERE / (name + '.source')).open('xb') as handle:
            handle.write(raw)
        value.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    except Exception as error:
        value = dict(status='FAILURE', error_type=type(error).__name__, error=str(error), automatic_retries=False)
    records.append(dict(name=name, url=url, UTC=datetime.now(timezone.utc).isoformat(), primary_read=False, **value))
with (HERE / 'HOUSE_SOURCE_RETRIEVAL.json').open('x') as handle:
    json.dump(records, handle, indent=2)
    handle.write('\n')
print(json.dumps(records))
