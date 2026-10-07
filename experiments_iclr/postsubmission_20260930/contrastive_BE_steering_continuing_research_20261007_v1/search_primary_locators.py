"""Bounded public literature locators; discovery is not a paper reading."""
import datetime
import json
from pathlib import Path
import urllib.parse
import urllib.request

OUT = Path(__file__).resolve().parent / 'literature'
QUERIES = [
    'graph supervised contrastive within class multimodal positives',
    'ensemble supervised contrastive graph context diversity shared backbone',
    'supervised contrastive learning subclass collapse',
]

def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for i, query in enumerate(QUERIES):
        url = 'https://api.openalex.org/works?' + urllib.parse.urlencode({
            'search': query, 'per-page': 8,
            'select': 'id,title,doi,publication_year,primary_location,open_access',
        })
        row = {'query': query, 'url': url,
               'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'GNNM research scoped literature review'})
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
                row['status'] = response.status
            (OUT / f'discovery_{i}.json').write_bytes(payload)
            data = json.loads(payload)
            row['works'] = data.get('results', [])
        except Exception as exc:
            row['error'] = str(exc)
        rows.append(row)
    (OUT / 'DISCOVERY_LOG.json').write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(row['query'], row.get('status'), row.get('error', ''))
        for work in row.get('works', []):
            print(work['title'], work.get('doi'), work.get('primary_location'))

if __name__ == '__main__':
    main()
