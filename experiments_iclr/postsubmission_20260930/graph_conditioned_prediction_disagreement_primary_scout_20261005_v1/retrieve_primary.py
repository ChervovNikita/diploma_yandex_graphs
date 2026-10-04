from pathlib import Path
import urllib.request,hashlib,json,datetime,concurrent.futures
P=Path(__file__).resolve().parent
routes=[('emrgnn','https://arxiv.org/html/2205.12076v1'),('gre','https://arxiv.org/html/1909.02811v1')]
def get(item):
 key,url=item;rec={'key':key,'url':url,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'retrieval only, unread until display'}
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'literature-scout/1.0'}),timeout=45) as r:
   b=r.read();rec.update(status=r.status,final_url=r.url,content_type=r.headers.get('Content-Type'),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
  path=P/'primary'/f'{key}.html';path.write_bytes(b); rec['path']=str(path.relative_to(P))
 except Exception as e:rec['error']=repr(e)
 return rec
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:out=list(ex.map(get,routes))
(P/'primary'/'RETRIEVAL.json').write_text(json.dumps(out,indent=2)+'\n')
for r in out: print(r)
