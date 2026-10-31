"""Focused closest-complete-method discovery. Source/document retrieval only."""
from pathlib import Path
import urllib.request,urllib.parse,datetime,json,hashlib,concurrent.futures,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
assert not (ROOT/'SEAL.json').exists()
(ROOT/'discovery').mkdir(exist_ok=True)
queries=[('ensemble_graph','(ti:"graph" OR ti:"GNN") AND (ti:"uncertainty" OR ti:"Bayesian") AND (all:"ensemble" OR all:"covariance")'),('conformal_graph','(ti:"conformal") AND (ti:"graph" OR ti:"GNN")'),('uncertain_edges','(ti:"uncertainty" OR ti:"uncertainty-aware") AND (all:"edge" OR all:"structure learning") AND (ti:"graph")'),('graph_calibration','(ti:"calibration" OR ti:"temperature scaling") AND (ti:"graph" OR ti:"GNN")')]
def get(spec):
 name,query=spec;url='https://export.arxiv.org/api/query?'+urllib.parse.urlencode(dict(search_query=query,start=0,max_results=100,sortBy='relevance',sortOrder='descending'));rec=dict(name=name,query=query,url=url,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 try:
  with urllib.request.urlopen(url,timeout=50) as r:data=r.read();rec.update(status=r.status,final_url=r.url)
  p=ROOT/'discovery'/f'{name}.xml';p.write_bytes(data);ns={'a':'http://www.w3.org/2005/Atom','o':'http://a9.com/-/spec/opensearch/1.1/'};feed=ET.fromstring(data)
  rec.update(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest(),total_results=feed.findtext('o:totalResults',namespaces=ns),entries=[dict(id=e.findtext('a:id',namespaces=ns),title=' '.join(e.findtext('a:title',namespaces=ns).split()),summary=' '.join(e.findtext('a:summary',namespaces=ns).split()),updated=e.findtext('a:updated',namespaces=ns)) for e in feed.findall('a:entry',ns)])
 except Exception as e:rec['error']=repr(e)
 return rec
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(get,queries))
(ROOT/'discovery/RECEIPTS.json').write_text(json.dumps(records,indent=2)+'\n')
for r in records:
 print(r['name'],r.get('total_results',r.get('error')),len(r.get('entries',[])))
 for e in r.get('entries',[]):print(e['id'],e['title'])
