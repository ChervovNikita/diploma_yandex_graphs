"""Retrieve title-resolved exact GPN primary, preserving mistaken-ID receipt separately."""
from pathlib import Path
import urllib.request,json,datetime,hashlib
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent
assert not (ROOT/'SEAL.json').exists()
rec=dict(key='gpn_resolved',identifier='2110.14012v1',title_expected='Graph Posterior Network',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),attempts=[])
for host in ['arxiv.org','export.arxiv.org']:
 url=f'https://{host}/pdf/2110.14012v1';a=dict(url=url)
 try:
  with urllib.request.urlopen(url,timeout=45) as r:data=r.read();a.update(status=r.status,final_url=r.url)
  assert data.startswith(b'%PDF')
  p=ROOT/'primary/gpn_resolved.pdf';p.write_bytes(data);reader=PdfReader(p);t=ROOT/'primary/gpn_resolved.txt';t.write_text('\n'.join(f'[PAGE {i+1}]\n{page.extract_text()}' for i,page in enumerate(reader.pages)))
  assert 'Graph Posterior Network' in reader.pages[0].extract_text()
  a['success']=True;rec['attempts'].append(a);rec.update(pdf=str(p.relative_to(ROOT)),pdf_sha256=hashlib.sha256(data).hexdigest(),text=str(t.relative_to(ROOT)),text_sha256=hashlib.sha256(t.read_bytes()).hexdigest(),pages=len(reader.pages));break
 except Exception as e:a['error']=repr(e);rec['attempts'].append(a)
(ROOT/'primary/GPN_RETRIEVAL.json').write_text(json.dumps(rec,indent=2)+'\n');print(rec)
