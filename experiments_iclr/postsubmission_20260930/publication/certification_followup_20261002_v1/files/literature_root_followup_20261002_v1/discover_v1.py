"""Bounded new literature discovery; metadata is not a paper-read claim."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
QUERIES = {
    'ensemble_initialization': 'all:ensemble AND (all:initialization OR all:tangent OR all:diversification) AND submittedDate:[202401010000 TO 202610022359]',
    'graph_diversity': 'all:graph AND (all:ensemble OR all:experts) AND (all:diversity OR all:initialization) AND submittedDate:[202601010000 TO 202610022359]',
}

def fetch(item):
    key, query = item
    url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode(
        dict(search_query=query, start=0, max_results=10, sortBy='submittedDate', sortOrder='descending'))
    start = datetime.now(timezone.utc).isoformat()
    receipt = dict(query=query, url=url, start_UTC=start, scope='Discovery metadata only')
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'GNNM-literature-research/1.0'})
        with urllib.request.urlopen(request, timeout=25) as response:
            raw = response.read(2 * 1024 * 1024)
            receipt['status_code'] = response.status
        with (ROOT/(key+'.xml')).open('xb') as stream:
            stream.write(raw)
        receipt.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        tree = ET.fromstring(raw)
        ns = {'a': 'http://www.w3.org/2005/Atom'}
        receipt['entries'] = [dict(id=e.findtext('a:id', namespaces=ns),
            title=' '.join(e.findtext('a:title', '', ns).split()),
            summary=' '.join(e.findtext('a:summary', '', ns).split()),
            published=e.findtext('a:published', namespaces=ns),
            updated=e.findtext('a:updated', namespaces=ns)) for e in tree.findall('a:entry', ns)]
    except Exception as error:
        receipt.update(failure_type=type(error).__name__, failure=str(error))
    receipt['terminal_UTC'] = datetime.now(timezone.utc).isoformat()
    with (ROOT/(key+'_RETRIEVAL.json')).open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    return receipt

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fetch, QUERIES.items()))
    for r in results:
        print(json.dumps(dict(query=r['query'], failure=r.get('failure'),
            entries=[dict(id=e['id'], title=e['title']) for e in r.get('entries', [])])))
