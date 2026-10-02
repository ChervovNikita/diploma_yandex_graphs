"""Resolve GPN exact source via title metadata; source/document request only."""
from pathlib import Path
import urllib.request,urllib.parse,datetime,json,hashlib
ROOT=Path(__file__).resolve().parent
url='https://api.openalex.org/works?'+urllib.parse.urlencode(dict(search='Graph posterior network Bayesian predictive uncertainty for node classification',per_page=5))
rec=dict(url=url,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
try:
 with urllib.request.urlopen(url,timeout=40) as r:data=r.read();rec.update(status=r.status,final_url=r.url)
 p=ROOT/'discovery/gpn_openalex.json';p.write_bytes(data);rec.update(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest());v=json.loads(data)
 for e in v.get('results',[]):print(e.get('title'),e.get('id'),e.get('locations'))
except Exception as e:rec['error']=repr(e)
(ROOT/'discovery/GPN_RESOLUTION_RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n')
print(rec)
