import json, urllib.request, urllib.error, hashlib, datetime, concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def fetch(key,url):
 target=ROOT/'sources'/key
 receipt={'key':key,'requested_url':url,'retrieved_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'GNNM-Literature-Review/1.0 (bounded public scholarly retrieval)'})
  with urllib.request.urlopen(req,timeout=35) as r:
   data=r.read(); receipt.update(status=r.status,final_url=r.url,content_type=r.headers.get('Content-Type'),headers=dict(r.headers))
 except urllib.error.HTTPError as e:
  data=e.read(); receipt.update(status=e.code,final_url=e.url,error=str(e),headers=dict(e.headers))
 except Exception as e:
  data=b''; receipt.update(error=repr(e))
 target.write_bytes(data);receipt.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),path=str(target.relative_to(ROOT)))
 return receipt
if __name__=='__main__':
 import sys
 jobs=json.loads((ROOT/sys.argv[1]).read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex: results=list(ex.map(lambda x:fetch(*x), jobs))
 out=ROOT/(Path(sys.argv[1]).stem+'_receipts.json');out.write_text(json.dumps(results,indent=2))
 for r in results: print(r['key'],r.get('status'),r.get('bytes'),r.get('error',''))
