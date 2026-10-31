"""Pinned CF-GNN author source as text only. Never import/execute."""
from pathlib import Path
import urllib.request,datetime,json,hashlib,concurrent.futures
ROOT=Path(__file__).resolve().parent
assert not (ROOT/'SEAL.json').exists()
(ROOT/'code_primary').mkdir(exist_ok=True)
commit='3564a3d6d9bd4ff69e36c57672f8272c5fb0ff39'
def get(name):
 url=f'https://raw.githubusercontent.com/snap-stanford/conformalized-gnn/{commit}/{name}';rec=dict(path_requested=name,commit=commit,url=url,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 try:
  with urllib.request.urlopen(url,timeout=35) as r:data=r.read();rec.update(status=r.status,final_url=r.url)
  p=ROOT/'code_primary'/name.replace('/','_');p.write_bytes(data);rec.update(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest())
 except Exception as e:rec['error']=repr(e)
 return rec
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(get,['conformalized_gnn/model.py','README.md']))
(ROOT/'code_primary/RETRIEVAL.json').write_text(json.dumps(records,indent=2)+'\n')
for r in records:print(r['path_requested'],r.get('sha256',r.get('error')))
