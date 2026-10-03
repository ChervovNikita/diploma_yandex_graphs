"""Primary article retrieval only; no execution of downloaded content."""
from pathlib import Path
import urllib.request,json,datetime,hashlib,sys
p=Path(__file__).resolve().parent
key,version=sys.argv[1:]
url='https://arxiv.org/html/'+version
r={'key':key,'versioned_id':version,'requested_URL':url,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'retrieval_is_not_full_paper_read':True}
try:
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Research-literature-scout/1.0'}),timeout=45) as response:
  data=response.read();r['status']=response.status;r['final_URL']=response.url
 out=p/'primary'/f'{key}.html';out.write_bytes(data)
 r['output']={'path':out.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
except Exception as e:r['error']=type(e).__name__+': '+str(e)
(p/'primary'/f'{key}_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
