"""Check the actual lexical venv/prefix/binary before any native runtime import."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import socket
import subprocess
import sys

from host_binding import guard, load_config
ROOT=Path(load_config()[0]["project_root"])
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def require(ok,message):
    if not ok:raise ValueError(message)


def validate_runtime_environment(spec):
    require(spec.get('schema')=='qualified-lexical-venv-python-v1','Explicit lexical venv environment required')
    require(sys.executable==spec['lexical_executable'],'Actual lexical Python executable differs')
    require(sys.prefix==spec['prefix']and sys.base_prefix==spec['base_prefix']and sys.prefix!=sys.base_prefix,
            'Actual child virtual-environment/base prefix differs')
    binary=Path(sys.executable).resolve()
    require(str(binary)==spec['resolved_executable']and hashlib.sha256(binary.read_bytes()).hexdigest()==spec['binary_sha256'],
            'Actual resolved Python binary differs')
    cfg=spec['pyvenv_config'];path=Path(cfg['path'])
    require(path==Path(sys.prefix)/'pyvenv.cfg','Actual venv configuration path differs')
    data=path.read_bytes();require(len(data)==cfg['bytes']and hashlib.sha256(data).hexdigest()==cfg['sha256'],
                               'Actual venv configuration changed')
    return {'lexical_executable':sys.executable,'resolved_executable':str(binary),'prefix':sys.prefix,
            'base_prefix':sys.base_prefix,'binary_sha256':spec['binary_sha256'],'Torch_imported':'torch'in sys.modules}


def main():
    guard()  # Explicit enabled host, hostname, GPU UUID and environment before runtime import.
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment',required=True);parser.add_argument('--environment-sha256',required=True)
    parser.add_argument('--entrypoint');parser.add_argument('--entrypoint-sha256');parser.add_argument('--probe-only',action='store_true')
    parser.add_argument('runtime_args',nargs=argparse.REMAINDER);args=parser.parse_args()
    path=Path(args.environment).resolve();require(path.is_relative_to(ROOT),'Environment path escapes approved root')
    data=path.read_bytes();require(hashlib.sha256(data).hexdigest()==args.environment_sha256,'Environment receipt changed')
    result=validate_runtime_environment(json.loads(data))
    if args.probe_only:
        require(args.entrypoint is None and not args.runtime_args,'Probe cannot import a runtime')
        print(json.dumps({'schema':'qualified-python-no-model-probe-v1','status':'PASS','environment':result,
            'models':0,'forwards':0,'optimizer_updates':0,'TEST_accessed':False}));return
    entry=Path(args.entrypoint).resolve();require(entry.is_relative_to(ROOT),'Runtime entrypoint escapes approved root')
    require(entry.name in ('run_fit.py','aggregate_selected.py'),'Fixed reviewed native/export handler required')
    require(hashlib.sha256(entry.read_bytes()).hexdigest()==args.entrypoint_sha256,'Runtime entrypoint changed')
    argv=args.runtime_args
    require(bool(argv)and argv[0]=='--','Explicit fixed runtime arguments required')
    sys.argv=[str(entry),*argv[1:]]
    sys.path.insert(0,str(entry.parent))
    runpy.run_path(str(entry),run_name='__main__')


if __name__=='__main__':main()
