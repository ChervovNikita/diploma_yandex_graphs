"""One disposable actual full-TRAIN call under the separate finite owner."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=Path(__file__).resolve().parent
S=P/'native_SAGE_endpoint_value_transport_source_20261011_v1'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
cfgpath=H/'QUALIFICATION_CONFIG_V1.json'
spec=importlib.util.spec_from_file_location('actual_endpoint_value_qualifier',S/'qualify_full_train.py')
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
mod.qualify(json.loads(cfgpath.read_text()),H/'qualify_v1',hashlib.sha256(cfgpath.read_bytes()).hexdigest(),
            wall_seconds=300,cuda_bytes=32*1024**3,rss_bytes=16*1024**3)
