import json,hashlib,urllib.request
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from discover_primary import Parser,clean
OUT=Path(__file__).resolve().parent

def fetch(z):
 key,ident=z;url='https://arxiv.org/html/'+ident
 req=urllib.request.Request(url,headers={'User-Agent':'Scoped academic method research'})
 now=datetime.now(timezone.utc).isoformat()
 try:
  with urllib.request.urlopen(req,timeout=40) as f:raw=f.read();final=f.url;status=f.status
  (OUT/('_'+key+'.html')).write_bytes(raw)
  parser=Parser();parser.feed(raw.decode('utf-8'))
  headings=[{'tag':n.tag,'id':n.attrs.get('id'),'class':n.attrs.get('class'),'text':clean(n.text())} for tag in ['h1','h2','h3','h4','h5','h6'] for n in parser.root.find(tag)]
  return {'key':key,'arxiv_id':ident,'url':url,'retrieved_utc':now,'status':status,'final_url':final,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'headings':headings,'read_accounting':'Headings only displayed at retrieval; semantic method scope is selected separately. Full-page network retrieval is not full-paper reading.'}
 except Exception as e:return {'key':key,'arxiv_id':ident,'url':url,'retrieved_utc':now,'error':str(e)}
with ThreadPoolExecutor(max_workers=2) as e:rs=list(e.map(fetch,[('hopper','2608.09031v1'),('nba','2310.07430v1')]))
(OUT/'PRIMARY_RETRIEVAL.json').write_text(json.dumps(rs,ensure_ascii=False,indent=2)+'\n')
for r in rs:print(json.dumps(r,ensure_ascii=False))
