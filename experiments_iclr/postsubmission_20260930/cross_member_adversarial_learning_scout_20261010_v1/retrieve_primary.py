"""Bounded public-paper discovery/retrieval only; no framework/model work."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from urllib.parse import urlencode

HERE = Path(__file__).resolve().parent
SOURCES = HERE/'sources'
REQUESTS = (
    ('dct_arxiv_html', 'https://arxiv.org/html/1803.05984v2', 'primary_candidate_method'),
    ('dct_ar5iv', 'https://ar5iv.labs.arxiv.org/html/1803.05984', 'primary_candidate_method_fallback'),
    ('dct_arxiv_abs', 'https://arxiv.org/abs/1803.05984', 'bibliographic_metadata_only'),
    ('vat_arxiv_html', 'https://arxiv.org/html/1704.03976v9', 'closest_self_adversarial_method'),
    ('recent_graph_adversarial_openalex', 'https://api.openalex.org/works?'+urlencode({'search':'adversarial co-training graph neural networks ensemble','per-page':8}), 'discovery_metadata_only'),
    ('recent_dct_openalex', 'https://api.openalex.org/works?'+urlencode({'search':'deep co-training shared weights adversarial ensemble','per-page':8}), 'discovery_metadata_only'))


def fetch(entry):
    key,url,scope = entry; started = time.monotonic()
    record = dict(key=key, requested_url=url, intended_scope=scope)
    try:
        with urlopen(Request(url, headers={'User-Agent':'GNNM-public-literature-scout/1.0'}), timeout=25) as response:
            payload = response.read(7_000_001)
            if len(payload) > 7_000_000: raise ValueError('Bounded public retrieval too large')
            path = SOURCES/(key+('.json' if 'openalex' in key else '.html'))
            path.write_bytes(payload)
            record.update(status='retrieved_unread', final_url=response.geturl(), HTTP_status=response.status,
                          path=str(path.relative_to(HERE)), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    except Exception as error:
        record.update(status='retrieval_failed', error_type=type(error).__name__, error=str(error))
    record['seconds'] = time.monotonic()-started
    return record


if __name__ == '__main__':
    SOURCES.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=3) as pool: records = list(pool.map(fetch, REQUESTS))
    (HERE/'PRIMARY_RETRIEVALS.json').write_text(json.dumps(records, indent=2, sort_keys=True)+'\n')
    print(json.dumps(records, indent=2, sort_keys=True))
