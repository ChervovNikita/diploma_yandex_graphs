from pathlib import Path
import urllib.request,urllib.parse,json,datetime,hashlib
from concurrent.futures import ThreadPoolExecutor
W=Path(__file__).resolve().parent
queries=['graph mamba Euler tour traversal','graph state space model spanning tree walk','graph mamba random walk node sequence ensemble']
def query(a):
 i,q=a;u='https://api.openalex.org/works?'+urllib.parse.urlencode({'search':q,'per-page':8});row={'query':q,'url':u,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  with urllib.request.urlopen(u,timeout=45) as f:d=f.read();row.update(status=f.status,final_url=f.url)
  p=W/'queries'/f'openalex_{i}.json';p.write_bytes(d);row.update(path=str(p.relative_to(W)),sha256=hashlib.sha256(d).hexdigest(),bytes=len(d),disposition='retrieved')
  row['titles']=[{'title':x['display_name'],'doi':x.get('doi'),'openalex_id':x['id']} for x in json.loads(d)['results']]
 except Exception as e:row.update(disposition='failed',error=str(e))
 return row
rows=list(ThreadPoolExecutor(max_workers=3).map(query,enumerate(queries,1)));(W/'SEARCH_RETRIEVAL.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(r['query']);print(json.dumps(r.get('titles',r),indent=2))
