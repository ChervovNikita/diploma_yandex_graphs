from pathlib import Path
import urllib.request, urllib.parse, hashlib, json, datetime
from concurrent.futures import ThreadPoolExecutor
W=Path(__file__).resolve().parent
requests=[('cgnn_abs','https://arxiv.org/abs/2307.11981'),('cgnn_api','https://export.arxiv.org/api/query?id_list=2307.11981'),('textgcn_exact_query','https://api.openalex.org/works?search='+urllib.parse.quote('Graph Convolutional Networks for Text Classification')+'&per-page=5')]
def get(item):
 key,url=item
 row={'key':key,'url':url,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  with urllib.request.urlopen(url,timeout=45) as r:
   data=r.read(); row.update(status=r.status,final_url=r.url)
  p=W/'primary'/f'{key}.raw';p.write_bytes(data);row.update(path=str(p.relative_to(W)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),disposition='retrieved')
 except Exception as e:row.update(disposition='failed',error=str(e))
 return row
rows=list(ThreadPoolExecutor(max_workers=3).map(get,requests))
(W/'PRIMARY_RETRIEVAL.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(r)
