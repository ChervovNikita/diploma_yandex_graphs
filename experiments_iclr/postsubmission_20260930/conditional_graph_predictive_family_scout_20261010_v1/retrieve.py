"""Bounded literature discovery for teacher-free conditional predictive families."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
from urllib.parse import quote
from datetime import datetime,timezone
import json,hashlib,time

ROOT=Path(__file__).resolve().parent
SOURCES=ROOT/'sources';SOURCES.mkdir(exist_ok=True)
REQUESTS=[]
for name,query in [
 ('graph_neural_process_discovery.json','graph neural processes node classification Bayesian'),
 ('message_passing_neural_process_discovery.json','message passing neural processes graph'),
 ('rank_one_bayesian_discovery.json','rank 1 Bayesian neural networks'),
 ('correlated_shared_private_ensemble_discovery.json','Bayesian neural network shared weights correlated ensemble posterior'),
]:
 REQUESTS.append((name,'https://api.openalex.org/works?search='+quote(query)+'&per-page=5'))
def retrieve(item):
 name,url=item;start=time.monotonic();r={'url':url,'UTC':datetime.now(timezone.utc).isoformat(),'intended_scope':'mechanical retrieval; read scope separately declared'}
 try:
  with urlopen(Request(url,headers={'User-Agent':'Research source verification/1.0'}),timeout=25) as response:
   b=response.read();r.update({'final_url':response.url,'http_status':response.status,'content_type':response.headers.get('content-type')})
  p=SOURCES/name;p.write_bytes(b);r.update({'status':'retrieved','path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
 except Exception as exc:r.update({'status':'failed','error':repr(exc)})
 r['seconds']=time.monotonic()-start;return r
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as pool:receipts=list(pool.map(retrieve,REQUESTS))
 (ROOT/'DISCOVERY_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\n')
 for r in receipts:print(json.dumps(r))
