"""Fetch the pinned public WikiCS and Actor files into this study directory."""
from __future__ import annotations
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
URLS={
    'data/wiki_cs/data.json':'https://raw.githubusercontent.com/pmernyei/wiki-cs-dataset/master/dataset/data.json',
    'data/actor/out1_node_feature_label.txt':'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data/film/out1_node_feature_label.txt',
    'data/actor/out1_graph_edges.txt':'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data/film/out1_graph_edges.txt',
    'data/actor/film_split_0.6_0.2_0.npz':'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/splits/film_split_0.6_0.2_0.npz',
}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    frozen=json.loads((ROOT/'PRETRAIN_FREEZE.json').read_text())['data_and_source_sha256']
    for name,url in URLS.items():
        expected=frozen[name]
        dest=ROOT/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and sha(dest)==expected:
            print('verified existing',name)
            continue
        temporary=dest.with_suffix(dest.suffix+'.download')
        with urllib.request.urlopen(url,timeout=60) as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response,stream)
        if sha(temporary)!=expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f'Public source bytes differ from the frozen hash: {name}')
        temporary.replace(dest)
        print('verified download',name)
if __name__=='__main__': main()
