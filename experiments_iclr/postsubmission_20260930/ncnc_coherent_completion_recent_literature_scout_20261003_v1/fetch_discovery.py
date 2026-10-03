"""Bounded primary arXiv discovery metadata retrieval; no method read."""
from pathlib import Path
import urllib.request,urllib.parse,json,datetime,hashlib,xml.etree.ElementTree as ET
import sys
p=Path(__file__).resolve().parent
queries=json.loads((p/'DISCOVERY_QUERIES.json').read_text())
key=sys.argv[1]; query=queries[key]
url='https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'search_query':query,'start':0,'max_results':25,'sortBy':'submittedDate','sortOrder':'descending'})
r={'key':key,'query':query,'requested_URL':url,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'discovery metadata only'}
try:
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Research-literature-scout/1.0'}),timeout=40) as response:
  data=response.read();r['status']=response.status;r['final_URL']=response.url
 out=p/'discovery'/f'{key}.xml';out.write_bytes(data)
 r['output']={'path':out.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 ns={'a':'http://www.w3.org/2005/Atom'};j=ET.fromstring(data)
 entries=[{'id':e.findtext('a:id',namespaces=ns),'title':' '.join(e.findtext('a:title',namespaces=ns).split()),'published':e.findtext('a:published',namespaces=ns),'summary':' '.join(e.findtext('a:summary',namespaces=ns).split())} for e in j.findall('a:entry',ns)]
 r['entries']=entries
 print(json.dumps([{'id':e['id'],'title':e['title'],'summary_locator':e['summary'][:230]} for e in entries],indent=2))
except Exception as e:
 r['error']=type(e).__name__+': '+str(e);print(r['error'])
(p/'discovery'/f'{key}_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n')
