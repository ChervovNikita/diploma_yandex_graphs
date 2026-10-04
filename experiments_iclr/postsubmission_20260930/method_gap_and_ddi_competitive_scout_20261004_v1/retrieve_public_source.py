import sys, json, hashlib, urllib.request, datetime, pathlib
url, dest, ledger = sys.argv[1:]
p=pathlib.Path(dest); p.parent.mkdir(parents=True,exist_ok=True)
r={"requested_url":url,"retrieved_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"status":"pending"}
try:
 req=urllib.request.Request(url,headers={"User-Agent":"Research-source-scout/1.0 (primary-source documentation)"})
 with urllib.request.urlopen(req,timeout=60) as res:
  data=res.read(20*1024*1024+1)
  if len(data)>20*1024*1024: raise RuntimeError("source exceeded 20 MiB bound")
  r.update(final_url=res.geturl(),http_status=res.status,content_type=res.headers.get("Content-Type"),bytes=len(data))
 p.write_bytes(data); r.update(status="retrieved",saved_path=str(p),sha256=hashlib.sha256(data).hexdigest())
except Exception as e:
 r.update(status="failed",error_type=type(e).__name__,error=str(e))
pathlib.Path(ledger).write_text(json.dumps(r,indent=2)+"\n")
print(json.dumps(r))
