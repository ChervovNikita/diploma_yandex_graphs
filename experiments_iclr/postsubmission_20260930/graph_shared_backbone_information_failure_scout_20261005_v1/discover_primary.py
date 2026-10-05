import json,hashlib,re,urllib.request,urllib.parse
from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor

OUT=Path(__file__).resolve().parent
class Node:
 def __init__(self,tag='root',attrs=None): self.tag=tag;self.attrs=dict(attrs or []);self.children=[]
 def text(self): return ' '.join((c.text() if isinstance(c,Node) else c) for c in self.children)
 def find(self,tag=None,cls=None):
  out=[]
  for c in self.children:
   if isinstance(c,Node):
    if (tag is None or c.tag==tag) and (cls is None or cls in c.attrs.get('class','').split()): out.append(c)
    out.extend(c.find(tag,cls))
  return out
class Parser(HTMLParser):
 def __init__(self): super().__init__();self.root=Node();self.stack=[self.root]
 def handle_starttag(self,tag,attrs):
  n=Node(tag,attrs);self.stack[-1].children.append(n)
  if tag not in ['area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr']:self.stack.append(n)
 def handle_endtag(self,tag):
  for i in range(len(self.stack)-1,0,-1):
   if self.stack[i].tag==tag:self.stack=self.stack[:i];break
 def handle_data(self,d):self.stack[-1].children.append(d)
def clean(s):return re.sub(r'\s+',' ',s).strip()
def get(q):
 url='https://arxiv.org/search/?'+urllib.parse.urlencode({'query':q,'searchtype':'all','abstracts':'show','order':'-announced_date_first','size':'50'})
 now=datetime.now(timezone.utc).isoformat(); req=urllib.request.Request(url,headers={'User-Agent':'Scoped academic method research; contact via arXiv public web interface'})
 try:
  with urllib.request.urlopen(req,timeout=40) as r: raw=r.read();status=r.status;final=r.url
  p=Parser();p.feed(raw.decode('utf-8'));rs=[]
  for n in p.root.find('li','arxiv-result'):
   links=n.find('p','list-title');titles=n.find('p','title');authors=n.find('p','authors');abstracts=n.find('span','abstract-full')
   linktext=clean(links[0].text()) if links else ''
   m=re.search(r'arXiv:(\d{4}\.\d{4,5})',linktext)
   rs.append({'arxiv_id':m.group(1) if m else None,'title':clean(titles[0].text()) if titles else '', 'authors':clean(authors[0].text()) if authors else '', 'abstract':clean(abstracts[0].text()) if abstracts else ''})
  page=clean(p.root.text());mm=re.search(r'Showing .*? results for all:',page)
  return {'query':q,'url':url,'retrieved_utc':now,'provider':'primary arXiv search; public metadata only','status':status,'final_url':final,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'result_summary':mm.group(0) if mm else ('No results' if 'no results' in page.lower() else None),'results':rs}
 except Exception as e:return {'query':q,'url':url,'retrieved_utc':now,'error':str(e),'results':[]}
if __name__=='__main__':
 queries=['"graph neural" AND "non-backtracking"','"graph neural" AND "information loss" AND "link prediction"','"graph neural" AND "oversquashing" AND "ensemble"']
 with ThreadPoolExecutor(max_workers=3) as e: rows=list(e.map(get,queries))
 (OUT/'DISCOVERY.json').write_text(json.dumps({'queries':rows,'read_accounting':'Metadata discovery only; no primary method reading credit'},ensure_ascii=False,indent=2)+'\n')
 for r in rows:
  print(json.dumps({'query':r['query'],'result_summary':r.get('result_summary'),'error':r.get('error'),'results':r['results']},ensure_ascii=False))
