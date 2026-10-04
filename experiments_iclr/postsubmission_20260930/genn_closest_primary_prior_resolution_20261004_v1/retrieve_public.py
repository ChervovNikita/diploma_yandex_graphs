"""Public discovery/retrieval with exact receipts; never execute downloaded content."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,sys,urllib.request,urllib.error

HERE=Path(__file__).resolve().parent
DEST=HERE/'discovery'
DEST.mkdir(exist_ok=True)

def request(item):
 name,url,kind=item
 record={'name':name,'requested_url':url,'kind':kind,
         'UTC':datetime.now(timezone.utc).isoformat(),'request_method':'GET',
         'request_headers':{'User-Agent':'Mozilla/5.0 (research public document discovery)'},
         'credentials_used':False}
 req=urllib.request.Request(url,headers=record['request_headers'])
 try:
  with urllib.request.urlopen(req,timeout=30) as r:
   body=r.read(5_000_001)
   record.update(status=r.status,final_url=r.url,content_type=r.headers.get('Content-Type'))
 except urllib.error.HTTPError as e:
  body=e.read(5_000_001)
  record.update(status=e.code,final_url=e.url,content_type=e.headers.get('Content-Type'),error=str(e))
 except Exception as e:
  body=b'';record.update(status=None,error=repr(e))
 record['response_truncated']=len(body)>5_000_000
 body=body[:5_000_000]
 path=DEST/name;path.write_bytes(body)
 record.update(path=str(path.relative_to(HERE)),bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
 (DEST/(name+'.receipt.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 return record

if __name__=='__main__':
 items=json.loads(Path(sys.argv[1]).read_text())
 with ThreadPoolExecutor(max_workers=6) as pool:
  results=list(pool.map(request,items))
 for r in results:print(json.dumps({k:r.get(k) for k in ['name','status','final_url','bytes','error','response_truncated']},ensure_ascii=False))
