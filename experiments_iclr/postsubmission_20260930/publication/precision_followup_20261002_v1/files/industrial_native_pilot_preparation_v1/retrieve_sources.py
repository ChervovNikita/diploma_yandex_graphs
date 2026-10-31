import json,hashlib,urllib.request,urllib.error,datetime,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'industrial_graph_followup_v1'
COMMIT='3b9b115490249cc777227c846babfb55f35bd8c4'
TREE=json.loads((BASE/'discovery/graphpfn_tree.json').read_text())
paths=['paper/lib/graph/deep.py','paper/lib/graph/data.py','paper/lib/deep.py','paper/lib/data.py']
def one(remote):
 url=f'https://raw.githubusercontent.com/yandex-research/graphpfn/{COMMIT}/{remote}'
 out=ROOT/'sources'/remote.replace('/','__')
 rec={'url':url,'remote_path':remote,'commit':COMMIT,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  with urllib.request.urlopen(url,timeout=25) as r:blob=r.read();rec.update(status=r.status,final_url=r.url)
 except Exception as e:blob=b'';rec['error']=repr(e)
 out.write_bytes(blob);expected=next(x['sha'] for x in TREE['tree'] if x['path']==remote)
 actual=hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
 rec.update(path=str(out.relative_to(ROOT)),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),tree_git_blob_sha1=expected,actual_git_blob_sha1=actual,matches_pinned_tree=actual==expected)
 return rec
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:rs=list(ex.map(one,paths))
(ROOT/'SOURCE_RETRIEVALS.json').write_text(json.dumps(rs,indent=2))
for x in rs:print(x['remote_path'],x.get('status'),x['bytes'],x['matches_pinned_tree'])
