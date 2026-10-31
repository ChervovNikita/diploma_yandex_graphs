"""Retrieve bounded exact-version primary PDFs; source extraction only, no method execution."""
from pathlib import Path
import urllib.request,datetime,json,hashlib,concurrent.futures
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent
assert not (ROOT/'SEAL.json').exists()
(ROOT/'primary').mkdir(exist_ok=True)
specs=[('bayesian_cp','2310.11479v3'),('snaps','2405.14303v2'),('sparsification','2410.21618v1')]
def get(spec):
 key,identifier=spec;rec=dict(key=key,identifier=identifier,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),attempts=[])
 for host in ['arxiv.org','export.arxiv.org']:
  url=f'https://{host}/pdf/{identifier}';a=dict(url=url)
  try:
   with urllib.request.urlopen(url,timeout=50) as r:data=r.read();a.update(status=r.status,final_url=r.url)
   assert data.startswith(b'%PDF'), 'Response not PDF'
   p=ROOT/'primary'/f'{key}.pdf';p.write_bytes(data);reader=PdfReader(p);t=ROOT/'primary'/f'{key}.txt';t.write_text('\n'.join(f'[PAGE {i+1}]\n{page.extract_text()}' for i,page in enumerate(reader.pages)))
   a['success']=True;rec['attempts'].append(a);rec.update(pdf=str(p.relative_to(ROOT)),pdf_sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),text=str(t.relative_to(ROOT)),text_sha256=hashlib.sha256(t.read_bytes()).hexdigest(),pages=len(reader.pages));break
  except Exception as e:a['error']=repr(e);rec['attempts'].append(a)
 return rec
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(get,specs))
(ROOT/'primary/RETRIEVAL.json').write_text(json.dumps(records,indent=2)+'\n')
for r in records:print(r['key'],r.get('pages'),r['attempts'])
