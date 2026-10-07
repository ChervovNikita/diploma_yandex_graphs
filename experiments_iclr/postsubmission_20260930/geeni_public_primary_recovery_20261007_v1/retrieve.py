"""Public document retrieval with exact receipts; blocked publisher is excluded."""
import concurrent.futures, datetime, hashlib, json, pathlib, sys, urllib.error, urllib.parse, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
SOURCES = ROOT / 'sources'
SOURCES.mkdir(exist_ok=True)

class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).hostname in {'dl.acm.org', 'www.dl.acm.org'}:
            raise urllib.error.URLError('Blocked ACM publisher request excluded by task scope')
        return super().redirect_request(req, fp, code, msg, headers, newurl)

def retrieve(item):
    key, url = item
    receipt = {'key': key, 'requested_url': url, 'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'read_scope': 'discovery_or_locator_only_until_inspected'}
    if urllib.parse.urlparse(url).hostname in {'dl.acm.org', 'www.dl.acm.org'}:
        raise ValueError('Publisher request excluded')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Public scholarly literature retrieval', 'Accept': '*/*'})
        with urllib.request.build_opener(PublicRedirect()).open(req, timeout=20) as response:
            data = response.read()
            receipt.update(status=response.status, final_url=response.url, content_type=response.headers.get('Content-Type'), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            (SOURCES / key).write_bytes(data)
    except Exception as error:
        receipt.update(error=str(error), status=getattr(error, 'code', None))
    (ROOT / (key + '_RECEIPT.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt

if __name__ == '__main__':
    items = json.loads(pathlib.Path(sys.argv[1]).read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for receipt in executor.map(retrieve, items):
            print(json.dumps(receipt))
