from pathlib import Path
import urllib.request,urllib.parse,json,hashlib,datetime,concurrent.futures
P=Path(__file__).resolve().parent
routes=[('geeni_crossref','https://api.crossref.org/works/10.1145/3489517.3530416'),('geeni_publisher','https://dl.acm.org/doi/pdf/10.1145/3489517.3530416'),('geeni_github','https://api.github.com/search/repositories?'+urllib.parse.urlencode({'q':'GEENI'}))]
def get(item):
 k,u=item;r={'key':k,'url':u,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'retrieval only'}
 try:
  with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'literature-scout/1.0','Accept':'*/*'}),timeout=45) as res:
   b=res.read();r.update(status=res.status,final_url=res.url,bytes=len(b),content_type=res.headers.get('Content-Type'),sha256=hashlib.sha256(b).hexdigest())
  ext='pdf' if b.startswith(b'%PDF') else 'json' if b[:1]in(b'{',b'[')else 'html';path=P/('primary' if ext=='pdf'else'discovery')/(k+'.'+ext);path.write_bytes(b);r['path']=str(path.relative_to(P))
 except Exception as e:r['error']=repr(e)
 return r
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:out=list(ex.map(get,routes))
(P/'discovery'/'GEENI_RETRIEVAL.json').write_text(json.dumps(out,indent=2)+'\n')
for r in out:print(r)
f=P/'discovery/geeni_crossref.json'
if f.exists():
 d=json.loads(f.read_text())['message'];print('AUTHORS',d.get('author'));print('LINKS',d.get('link'));print('RESOURCE',d.get('resource'))
f=P/'discovery/geeni_github.json'
if f.exists():
 d=json.loads(f.read_text());print('GITHUB',d.get('total_count'));print([(x['full_name'],x.get('description'),x['html_url'])for x in d.get('items',[])[:10]])
