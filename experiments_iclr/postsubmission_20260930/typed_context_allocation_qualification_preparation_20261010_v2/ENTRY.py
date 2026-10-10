"""Run only the separately reviewed complete-input qualification release."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import socket
import subprocess
import sys

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[GPU]
parser=argparse.ArgumentParser()
parser.add_argument('--release-sha256',required=True)
args=parser.parse_args()
release=HERE/'RELEASE.json'
assert hashlib.sha256(release.read_bytes()).hexdigest()==args.release_sha256
spec=json.loads(release.read_text())
assert spec['action']=='qualify_complete_native_context_pair' and not spec['scientific_execution_approved']
sys.path.insert(0,str(PHASE))
package='typed_label_context_factor_source_prototype_20261010_v3'
caps_module=importlib.import_module(package+'.caps')
driver=importlib.import_module(package+'.driver')
caps=caps_module.Caps(source_bound=True,model=True,data=True,runtime=True,scientific=False,root_review_sha256=spec['root_review']['sha256'])
result=driver.execute(release,caps=caps)
print(json.dumps(dict(complete=result['complete'],qualification_passed=result['qualification_passed'],conditions=result['conditions'],scientific_result=False)),flush=True)
