"""Retrieve exactly three selected unread primary method papers."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
SPECS=[('bankgcn','2106.09910'),('specformer','2303.01028'),('hgen',None)]


def get(spec):
    name,ident=spec
    folder=ROOT/'primary';folder.mkdir(exist_ok=True)
    receipt={'key':name,'retrieved_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':[]}
    urls=[('metadata.html',f'https://arxiv.org/abs/{ident}'),('pdf',f'https://arxiv.org/pdf/{ident}')] if ident else [('metadata.html','https://www.ijcai.org/proceedings/2025/685'),('pdf','https://www.ijcai.org/proceedings/2025/0685.pdf')]
    for ext,url in urls:
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'bounded-graph-literature-audit/1.0'}),timeout=40) as f:blob=f.read();endurl=f.url
            path=folder/(name+'.'+ext);path.write_bytes(blob)
            receipt['sources'].append({'url':url,'actual_response_url':endurl,'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(blob).hexdigest(),'size':len(blob)})
        except Exception as e:receipt['sources'].append({'url':url,'error':repr(e)})
    pdf=folder/(name+'.pdf')
    if pdf.exists():
        pages=[p.extract_text() for p in PdfReader(pdf).pages]
        (folder/(name+'.pages.json')).write_text(json.dumps(pages,indent=2)+'\n')
        (folder/(name+'.txt')).write_text('\n\n'.join('=== PDF PAGE '+str(i+1)+' ===\n'+p for i,p in enumerate(pages)))
        receipt['page_count']=len(pages);receipt['first_page_id']=[s for s in pages[0].splitlines() if 'arXiv' in s or 'Proceedings' in s];receipt['first_page_header']=pages[0][:450]
    return receipt


if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:r=list(pool.map(get,SPECS))
    (ROOT/'primary/RETRIEVAL.json').write_text(json.dumps(r,indent=2)+'\n')
    for x in r:print(json.dumps(x,indent=2))
