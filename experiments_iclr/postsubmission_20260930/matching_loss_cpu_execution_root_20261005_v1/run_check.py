"""Run the one bounded algebra/gradient check on the authorized 18.77 host."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / 'degree_preserving_matching_loss_source_20261005_v1/module.py'
spec = importlib.util.spec_from_file_location('existing_transport', PHASE / 'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
transport.HERE = HERE
raw = SOURCE.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
plan = dict(UTC=datetime.now(timezone.utc).isoformat(), source_sha256=digest,
            operation='One cpu_correctness_check() call; constructed inputs only',
            graph_or_checkpoint_reads=False, fits_or_updates=0,
            CUDA_visible_devices='', threads=2, child_timeout_seconds=45,
            automatic_retries=False)
with (HERE / 'PLAN.json').open('x') as handle:
    json.dump(plan, handle, indent=2)
    handle.write('\n')
child_code = '''from pathlib import Path
import hashlib,importlib.util,json,time,torch
p=Path(SOURCE_PATH)
assert hashlib.sha256(p.read_bytes()).hexdigest()==SOURCE_SHA
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
assert not torch.cuda.is_available()
spec=importlib.util.spec_from_file_location('matching_loss',p)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
t=time.monotonic();result=module.cpu_correctness_check()
print(json.dumps(dict(status='PASS',result=result,check_seconds=time.monotonic()-t,torch_version=torch.__version__,source_sha256=SOURCE_SHA)))
'''
remote = transport.REMOTE_PHASE / 'matching_loss_cpu_execution_root_20261005_v1'
remote_source = remote / 'module.py'
child_code = 'SOURCE_PATH=' + repr(str(remote_source)) + '\nSOURCE_SHA=' + repr(digest) + '\n' + child_code
code = '''from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess,time
repo=Path(REPO);root=Path(ROOT);source=root/'module.py'
assert Path.cwd()==repo and os.uname().nodename=='peptide'
g=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15)
assert set(g.stdout.splitlines())=={'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}
assert root.resolve().is_relative_to(repo) and not root.exists()
root.mkdir();raw=base64.b64decode(RAW,validate=True)
assert hashlib.sha256(raw).hexdigest()==SHA
with source.open('xb') as h:h.write(raw)
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1')
t=time.monotonic()
try:
 r=subprocess.run([PYTHON,'-B','-c',CHILD],cwd=repo,env=environment,capture_output=True,text=True,timeout=45)
 receipt=dict(status='PASS' if r.returncode==0 else 'FAIL',exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
except subprocess.TimeoutExpired as e:
 receipt=dict(status='TIMEOUT',timeout_seconds=45,stdout=(e.stdout or b'').decode() if isinstance(e.stdout,bytes) else e.stdout,stderr=(e.stderr or b'').decode() if isinstance(e.stderr,bytes) else e.stderr)
receipt.update(UTC=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-t,route=dict(repository=str(repo),hostname=os.uname().nodename,GPU_UUIDs=g.stdout.splitlines()),source_sha256=SHA,graph_or_checkpoint_reads=False,fits_or_updates=0,automatic_retries=False)
with (root/'RECEIPT.json').open('x') as h:json.dump(receipt,h,indent=2);h.write('\\n')
print(json.dumps(receipt))
'''
code = ('REPO=' + repr(str(transport.REPO)) + '\nROOT=' + repr(str(remote))
        + '\nRAW=' + repr(base64.b64encode(raw).decode()) + '\nSHA=' + repr(digest)
        + '\nPYTHON=' + repr(transport.PYTHON) + '\nCHILD=' + repr(child_code) + '\n' + code)
with (HERE / 'REMOTE_CODE.py.txt').open('x') as handle:
    handle.write(code)
receipt = transport.run('matching_loss_cpu_check_20261005_v1', code)
with (HERE / 'RECEIPT.json').open('x') as handle:
    json.dump(receipt, handle, indent=2)
    handle.write('\n')
print(json.dumps(receipt))
