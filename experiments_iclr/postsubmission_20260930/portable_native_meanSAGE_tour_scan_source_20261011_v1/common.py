"""Portable public source/receipt utilities; standard-library imports only."""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys

HERE=Path(__file__).resolve().parent
BLOCKS=(10103,10427,10949)
ARMS=('P1','F1','I4','S4','PV1','WI4','V1','SW4','C4','R4')

def require(value,message):
    if not value:raise ValueError(message)

def receipt(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as stream:
        for value in iter(lambda:stream.read(1048576),b''):h.update(value)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())

def checked(item):
    require(receipt(item['path'])==item,'Local file receipt differs: '+item['path'])
    return Path(item['path'])

def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def read(path):return json.loads(Path(path).read_text())

def load(name):
    spec=importlib.util.spec_from_file_location('portable_tour_'+name,HERE/'deps'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

def bundle():
    manifest=read(HERE/'MANIFEST.json')
    for name,row in manifest['files'].items():
        value=receipt(HERE/name)
        require(value['bytes']==row['bytes'] and value['sha256']==row['sha256'],'Portable source changed: '+name)
    return receipt(HERE/'MANIFEST.json')

def runtime_versions():
    versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','torch','torch-geometric')}
    require(sys.version_info[:2]==(3,11) and versions['numpy']=='1.26.4' and versions['torch'].split('+')[0]=='2.1.2' and versions['torch-geometric']=='2.7.0','Fixed Python3.11/NumPy1.26.4/Torch2.1.2/PyG2.7.0 source recipe required')
    return dict(python=sys.version,platform=platform.platform(),packages=versions,portable_runtime_verified=False)
