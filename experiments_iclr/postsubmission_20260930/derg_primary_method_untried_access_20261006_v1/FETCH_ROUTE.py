"""Bounded public literature retrieval only; no model/data/SSH execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys, urllib.request, urllib.error

key, url, purpose = sys.argv[1:4]
out = Path(__file__).resolve().parent / 'sources'
out.mkdir(exist_ok=True)
log = out / (key + '.route.json')
if log.exists():
    raise SystemExit('Route already attempted; automatic repeats refused')
row = {'key': key, 'url': url, 'purpose': purpose,
       'UTC': datetime.now(timezone.utc).isoformat(), 'new_targeted_route': True,
       'credentials_or_access_challenge_bypass': False}
log.write_text(json.dumps({**row, 'status': 'STARTED'}, indent=2) + '\n')
request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (public literature method review)',
                                               'Accept': 'application/json,text/html,application/xml;q=0.9,*/*;q=0.5'})
body = b''
try:
    with urllib.request.urlopen(request, timeout=25) as response:
        body = response.read(5*1024*1024+1)
        row.update(status=response.status, final_url=response.geturl(),
                   content_type=response.headers.get('Content-Type'),
                   content_length=response.headers.get('Content-Length'))
except urllib.error.HTTPError as error:
    body = error.read(5*1024*1024+1)
    row.update(status=error.code, final_url=error.geturl(), error=str(error),
               content_type=error.headers.get('Content-Type'))
except Exception as error:
    row.update(status='TRANSPORT_ERROR', error_type=type(error).__name__, error=str(error))
if body:
    path = out / (key + '.body')
    path.write_bytes(body)
    row.update(file='sources/'+path.name, bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
               capped_at_5MiB=len(body)>5*1024*1024)
row['retrieval_is_not_method_read'] = True
log.write_text(json.dumps(row, indent=2) + '\n')
print(json.dumps(row))
