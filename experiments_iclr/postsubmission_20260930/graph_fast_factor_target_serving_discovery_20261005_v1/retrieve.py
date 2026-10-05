"""Bounded public metadata discovery; results are locators, not paper reads."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
QUERIES = (
    'graph neural network ensemble shared backbone fast weights',
    'graph ensemble low rank parameter sharing predictive diversity',
    'graph neural network ensemble relation uncertainty weight sharing',
    'graph neural network ensemble propagation diversity accuracy',
)
records = []
for number, query in enumerate(QUERIES, 1):
    url = 'https://api.openalex.org/works?' + urllib.parse.urlencode(dict(search=query, per_page=10))
    request = urllib.request.Request(url, headers={'User-Agent': 'GNNM-research-literature/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise RuntimeError('Metadata response exceeded bounded payload')
            status = response.status
        data = json.loads(raw)
        with (HERE / ('query%02d.json' % number)).open('xb') as handle:
            handle.write(raw)
        rows = []
        for work in data.get('results', []):
            rows.append(dict(id=work.get('id'), title=work.get('title'), doi=work.get('doi'),
                publication_date=work.get('publication_date'),
                primary_location=work.get('primary_location'), open_access=work.get('open_access'),
                locations=work.get('locations'), abstract_inverted_index=work.get('abstract_inverted_index')))
        records.append(dict(query=query, url=url, UTC=datetime.now(timezone.utc).isoformat(),
            status=status, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), results=rows,
            primary_read=False))
    except Exception as error:
        records.append(dict(query=query, url=url, UTC=datetime.now(timezone.utc).isoformat(),
            status='FAILURE', error_type=type(error).__name__, error=str(error),
            primary_read=False, automatic_retries=False))
with (HERE / 'LOCATORS.json').open('x') as handle:
    json.dump(records, handle, indent=2)
    handle.write('\n')
print(json.dumps([dict(query=row['query'], status=row['status'], titles=[work['title'] for work in row.get('results', [])]) for row in records]))
