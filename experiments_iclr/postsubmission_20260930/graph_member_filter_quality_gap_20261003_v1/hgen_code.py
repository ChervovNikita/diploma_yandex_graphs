"""Inspect author source metadata without cloning data/checkpoints."""
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parent
D=ROOT/'author_source';D.mkdir(exist_ok=True)


def fetch(name,url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'bounded-graph-literature-audit/1.0'}),timeout=30) as f:b=f.read()
    (D/name).write_bytes(b)
    return json.loads(b),{'url':url,'path':'author_source/'+name,'sha256':hashlib.sha256(b).hexdigest(),'retrieved_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}


if __name__=='__main__':
    head,r1=fetch('HGEN_HEAD.json','https://api.github.com/repos/Chrisshen12/HGEN/commits/HEAD')
    commit=head['sha']
    tree,r2=fetch('HGEN_TREE.json',f'https://api.github.com/repos/Chrisshen12/HGEN/git/trees/{commit}?recursive=1')
    (D/'METADATA_RETRIEVAL.json').write_text(json.dumps([r1,r2],indent=2)+'\n')
    print('COMMIT',commit,head['commit']['committer']['date'])
    for x in tree['tree']:
        if x['type']=='blob' and (x['path'].endswith('.py') or x['path'].lower().endswith('readme.md')):print(x['path'])
