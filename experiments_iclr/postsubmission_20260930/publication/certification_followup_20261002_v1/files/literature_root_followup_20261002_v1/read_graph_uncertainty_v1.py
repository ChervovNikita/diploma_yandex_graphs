"""Retrieve one newly identified primary HTML without executing its content."""
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent / 'graph_uncertainty_collapse_v1'
URL = 'https://arxiv.org/html/2605.22593v1'

class Extractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skip += 1
        if tag in ('p', 'div', 'h1', 'h2', 'h3', 'h4', 'li', 'table', 'tr', 'section'):
            self.parts.append('\n')
        if tag == 'math':
            a = dict(attrs)
            if a.get('alttext'):
                self.parts.append(' [MATH: '+a['alttext']+'] ')
    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skip -= 1
        if tag in ('p', 'div', 'h1', 'h2', 'h3', 'h4', 'li', 'table', 'tr', 'section'):
            self.parts.append('\n')
    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

def main():
    ROOT.mkdir(exist_ok=False)
    receipt = dict(url=URL, canonical_id='arXiv:2605.22593v1', start_UTC=datetime.now(timezone.utc).isoformat())
    try:
        with urllib.request.urlopen(urllib.request.Request(URL,
                headers={'User-Agent': 'GNNM-literature-research/1.0'}), timeout=25) as response:
            raw = response.read(6*1024*1024)
            receipt['status_code'] = response.status
            receipt['final_url'] = response.url
        (ROOT/'primary.html').write_bytes(raw)
        parser = Extractor()
        parser.feed(raw.decode('utf-8'))
        lines = [' '.join(s.split()) for s in ''.join(parser.parts).splitlines()]
        text = '\n'.join(s for s in lines if s)+'\n'
        (ROOT/'primary.txt').write_text(text)
        receipt.update(primary_sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                       extracted_sha256=hashlib.sha256(text.encode()).hexdigest(),
                       read_claim='Retrieval only; read scope recorded separately')
    except Exception as error:
        receipt.update(failure_type=type(error).__name__, failure=str(error))
    receipt['terminal_UTC'] = datetime.now(timezone.utc).isoformat()
    (ROOT/'RETRIEVAL.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
