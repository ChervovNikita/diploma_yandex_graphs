"""Read-only public scholarly document retrieval; no model/data execution."""
import concurrent.futures, datetime, hashlib, json, pathlib, sys, urllib.error, urllib.parse, urllib.request
ROOT = pathlib.Path(__file__).resolve().parent
SOURCES = ROOT / 'sources'
SOURCES.mkdir(exist_ok=True)
class ScopedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).hostname in {'dl.acm.org', 'www.dl.acm.org'}:
            raise urllib.error.URLError('Known blocked publisher excluded')
        return super().redirect_request(req, fp, code, msg, headers, newurl)
def retrieve(item):
    key, url = item
    receipt={'key':key,'requested_url':url,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'discovery_or_primary_access_until_read_scope_recorded'}
    try:
        if urllib.parse.urlparse(url).hostname in {'dl.acm.org','www.dl.acm.org'}:raise ValueError('Known blocked publisher excluded')
        req=urllib.request.Request(url,headers={'User-Agent':'Public scholarly literature retrieval','Accept':'*/*'})
        with urllib.request.build_opener(ScopedRedirect()).open(req,timeout=20) as r:
            data=r.read()
            receipt.update(status=r.status,final_url=r.url,content_type=r.headers.get('Content-Type'),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            (SOURCES/key).write_bytes(data)
    except Exception as e:receipt.update(error=str(e),status=getattr(e,'code',None))
    (ROOT/(key+'_RECEIPT.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
if __name__=='__main__':
    items=json.loads(pathlib.Path(sys.argv[1]).read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for receipt in executor.map(retrieve,items):print(json.dumps(receipt))
