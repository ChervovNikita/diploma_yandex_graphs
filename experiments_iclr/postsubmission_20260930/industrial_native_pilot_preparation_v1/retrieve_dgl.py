import json,urllib.request,urllib.error,hashlib,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
records=[]
def get(name,url):
 r={'url':url,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'path':'sources/'+name}
 try:
  with urllib.request.urlopen(url,timeout=25) as response:data=response.read();r.update(status=response.status,final_url=response.url)
 except urllib.error.HTTPError as e:data=e.read();r.update(status=e.code,error=str(e))
 except Exception as e:data=b'';r['error']=repr(e)
 p=ROOT/r['path'];p.write_bytes(data);r.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());records.append(r);return data,r
# Preserve the failed exact-tag spelling; request the release tag with its actual prefix.
_,failed=get('dgl_tag_2_4_0_failed.json','https://api.github.com/repos/dmlc/dgl/git/ref/tags/2.4.0')
data,rec=get('dgl_tag_v2_4_0.json','https://api.github.com/repos/dmlc/dgl/git/ref/tags/v2.4.0')
if rec.get('status')==200:
 x=json.loads(data);ref=x['object']['sha']
 if x['object']['type']=='tag':
  b,_=get('dgl_annotated_tag.json',x['object']['url']);ref=json.loads(b)['object']['sha']
 for path in ['python/dgl/backend/pytorch/sparse.py','python/dgl/ops/edge_softmax.py']:
  _,r=get('dgl__'+path.replace('/','__'),'https://raw.githubusercontent.com/dmlc/dgl/'+ref+'/'+path);r['release']='v2.4.0';r['commit']=ref
(ROOT/'DGL_SOURCE_RECEIPTS.json').write_text(json.dumps(records,indent=2))
for r in records:print(r['path'],r.get('status'),r['bytes'],r.get('commit',''))
