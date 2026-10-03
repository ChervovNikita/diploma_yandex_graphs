"""Bounded graph-filter/member-neighborhood metadata search."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parent
QUERIES=[
    '"graph ensemble" "filter"',
    '"graph" "ensemble" "shared" "neighborhood"',
    '"graph" "learnable filter bank"',
    'Is Homophily a Necessity for Graph Neural Networks',
    'Specformer: Spectral Graph Neural Networks Meet Transformers',
    'HopGNN: Simplifying Multi-hop Graph Neural Networks',
    '"graph" "ensemble" "heterophily"',
]


def query(q):
    url='https://api.openalex.org/works?'+urllib.parse.urlencode({'search':q,'per-page':12,
        'select':'id,title,publication_date,doi,primary_location,best_oa_location,abstract_inverted_index'})
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'bounded-graph-literature-audit/1.0'}),timeout=30) as f:
            d=json.load(f)
        return {'query':q,'url':url,'retrieved_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'total':d['meta']['count'],'results':d['results']}
    except Exception as e:return {'query':q,'url':url,'error':repr(e)}


if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:r=list(pool.map(query,QUERIES))
    (ROOT/'discovery').mkdir(exist_ok=True)
    (ROOT/'discovery/OPENALEX.json').write_text(json.dumps(r,indent=2)+'\n')
    for x in r:
        print(json.dumps({k:x.get(k) for k in ['query','total','error']}))
        for v in x.get('results',[]): print(json.dumps({'title':v['title'],'date':v['publication_date'],'doi':v['doi'],'id':v['id']}))
