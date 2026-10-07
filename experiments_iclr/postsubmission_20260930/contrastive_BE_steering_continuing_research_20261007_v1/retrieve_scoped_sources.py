"""Retrieve a bounded set of public primary bodies, preserving request failures."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

OUT = Path(__file__).resolve().parent / 'literature'
SOURCES = {
    'simplicity_class_collapse_v1': 'https://arxiv.org/pdf/2305.16536v1',
    'supcon_balance_v1': 'https://arxiv.org/pdf/2204.07596v1',
}

def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for key, url in SOURCES.items():
        row = {'key': key, 'url': url,
               'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'GNNM primary literature scoped review'})
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
                row['status'] = response.status
                row['content_type'] = response.headers.get('Content-Type')
            if not payload.startswith(b'%PDF'):
                raise ValueError('Not a PDF payload')
            destination = OUT / f'{key}.pdf'
            destination.write_bytes(payload)
            row.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            result = subprocess.run(['pdftotext', '-layout', str(destination), str(OUT / f'{key}.txt')], capture_output=True, text=True)
            row['extract_exit'] = result.returncode
            row['extract_error'] = result.stderr
        except Exception as exc:
            row['error'] = str(exc)
        rows.append(row)
    (OUT / 'PRIMARY_RETRIEVAL_LOG.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))

if __name__ == '__main__':
    main()
