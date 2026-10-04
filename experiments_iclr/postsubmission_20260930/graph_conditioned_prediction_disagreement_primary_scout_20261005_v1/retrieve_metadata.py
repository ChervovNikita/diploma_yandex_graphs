import urllib.request, urllib.parse, json, hashlib, datetime, concurrent.futures
from pathlib import Path
P=Path(__file__).resolve().parent
queries=[('negative_correlation','graph neural negative correlation ensemble'),('conditional_diversity','graph neural conditional diversity ensemble'),('representation_ensemble','Graph representation ensemble learning'),('prediction_disagreement','ensemble graph neural prediction disagreement')]
def run(q):
 key,query=q
 url='https://api.openalex.org/works?'+urllib.parse.urlencode({'search':query,'per-page':15})
 start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 rec={'key':key,'query':query,'url':url,'UTC':start,'method':'GET'}
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'literature-scout/1.0'}),timeout=45) as r:
   b=r.read();rec.update({'status':r.status,'final_url':r.url,'content_type':r.headers.get('Content-Type'),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
  (P/'discovery'/f'{key}.json').write_bytes(b); d=json.loads(b)
  rec['total']=d.get('meta',{}).get('count')
  rec['rows']=[{'title':x.get('title'),'doi':x.get('doi'),'year':x.get('publication_year'),'id':x.get('id'),'primary_location':x.get('primary_location'),'best_oa_location':x.get('best_oa_location')} for x in d.get('results',[])]
 except Exception as e: rec['error']=repr(e)
 return rec
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex: out=list(ex.map(run,queries))
(P/'discovery'/'RETRIEVAL.json').write_text(json.dumps(out,indent=2)+'\n')
for r in out:
 print('QUERY',r['key'],'total',r.get('total'),'error',r.get('error'))
 for x in r.get('rows',[]): print(x['year'],x['title'],x['doi'], (x.get('best_oa_location')or{}).get('pdf_url'),(x.get('primary_location')or{}).get('landing_page_url'))
