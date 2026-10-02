"""Root CPU-only qualification of immutable ports; synthetic inputs only."""
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
PACKET = PHASE / 'efficient_graph_ensemble_ports_preparation_v1'
REMOTE = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SSH = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes', LOGIN]

CHILD = r'''
import hashlib,json,os,pathlib,platform,resource,sys,time,unittest
packet=pathlib.Path(sys.argv[1]); output=pathlib.Path(sys.argv[2])
manifest=json.loads((packet/'MANIFEST.json').read_text())
for row in manifest['files']:
 p=packet/row['path']
 if not p.resolve().is_relative_to(packet) or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
  raise ValueError('Immutable source changed')
sys.path.insert(0,str(packet))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('CPU-only visibility required')
started=time.monotonic()
import torch,torch_geometric
torch.set_num_threads(1); torch.set_num_interop_threads(1)
if torch.__version__!='2.1.2+cu118' or torch_geometric.__version__!='2.7.0':
 raise ValueError('Retained runtime versions differ')
from torch_geometric.nn import GATConv
import inspect
gat_source=pathlib.Path(inspect.getsourcefile(GATConv))
if hashlib.sha256(gat_source.read_bytes()).hexdigest() != next(r['sha256'] for r in manifest['files'] if r['path']=='sources/gat_conv_2_7_pinned.py'):
 raise ValueError('Imported GAT source differs from preparation pin')
events=[]
class Result(unittest.TextTestResult):
 def startTest(self,test):
  self.start=time.monotonic(); super().startTest(test)
 def note(self,test,state,detail=None):
  events.append({'test':test.id(),'state':state,'seconds':time.monotonic()-self.start,'detail':detail})
 def addSuccess(self,test):
  self.note(test,'passed'); super().addSuccess(test)
 def addSkip(self,test,reason):
  self.note(test,'unsupported_backend_skip',reason); super().addSkip(test,reason)
 def addFailure(self,test,err):
  self.note(test,'failed',self._exc_info_to_string(err,test)); super().addFailure(test,err)
 def addError(self,test,err):
  self.note(test,'error',self._exc_info_to_string(err,test)); super().addError(test,err)
suite=unittest.defaultTestLoader.loadTestsFromName('test_ports')
with (output/'unittest.txt').open('x') as stream:
 result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=Result).run(suite)
receipt={'schema':'efficient-graph-ports-cpu-qualification-v1','tests_run':result.testsRun,
 'passed':sum(r['state']=='passed' for r in events),'skips':len(result.skipped),
 'failures':len(result.failures),'errors':len(result.errors),'events':events,
 'seconds':time.monotonic()-started,'torch':torch.__version__,'pyg':torch_geometric.__version__,
 'python':platform.python_version(),'GAT_source_sha256':hashlib.sha256(gat_source.read_bytes()).hexdigest(),
 'torch_source_path':torch.__file__,'threads':torch.get_num_threads(),
 'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 'cuda_initialized':torch.cuda.is_initialized(),'GPU_computation':False,
 'inputs':'six-node synthetic engineering fixtures only; no dataset or role files',
 'predictive_or_efficiency_result':False,'skips_never_qualify_a_backend':True,
 'all_tests_passed_without_skips':result.wasSuccessful() and not result.skipped}
with (output/'QUALIFICATION.json').open('x') as stream:
 json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(receipt,allow_nan=False))
raise SystemExit(0 if result.wasSuccessful() else 1)
'''

OUTER = r'''
import base64,datetime,hashlib,json,os,pathlib,signal,subprocess,sys,time
repo=pathlib.Path(sys.argv[1]); uuid=sys.argv[2]; child_code=sys.argv[3]
if repo.resolve()!=repo: raise ValueError('Actual fixed repository required')
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10)
if gpu.stdout.strip().splitlines()!=[uuid]: raise ValueError('Authorized one-GPU route differs')
phase=repo/'experiments_iclr/postsubmission_20260930'
packet=phase/'efficient_graph_ensemble_ports_preparation_v1'
output=phase/'efficient_graph_ensemble_ports_cpu_qualification_v1/root01'
payload=json.load(sys.stdin)
for root in (packet,output):
 cursor=repo
 for part in root.relative_to(repo).parts:
  cursor/=part
  if cursor.is_symlink(): raise ValueError('Symlink destination refused')
if output.exists(): raise ValueError('Single-use qualification identity already used')
for row in payload['files']:
 rel=pathlib.PurePosixPath(row['path'])
 if rel.is_absolute() or '..' in rel.parts: raise ValueError('Invalid source path')
 data=base64.b64decode(row['data'])
 if hashlib.sha256(data).hexdigest()!=row['sha256']: raise ValueError('Transfer hash differs')
 target=packet.joinpath(*rel.parts)
 if not target.resolve().is_relative_to(packet): raise ValueError('Source escapes packet')
 if target.exists():
  if target.read_bytes()!=data: raise ValueError('Existing source differs')
 else:
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as stream: stream.write(data)
output.mkdir(parents=True,exist_ok=False)
(output/'executed_child.py').write_text(child_code)
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HOME=str(output),TMPDIR=str(output))
started=time.monotonic()
child=subprocess.Popen([str(repo/'.venv/bin/python'),'-I','-B','-c',child_code,str(packet),str(output)],
 stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env,cwd=output,start_new_session=True)
try:
 stdout,stderr=child.communicate(timeout=180); timed_out=False
except subprocess.TimeoutExpired:
 os.killpg(child.pid,signal.SIGKILL);stdout,stderr=child.communicate();timed_out=True
(output/'stdout.txt').write_text(stdout);(output/'stderr.txt').write_text(stderr)
record={'schema':'root-graph-ports-cpu-whole-process-v1','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'UUID_inventory':uuid,'exit_code':child.returncode,'timed_out':timed_out,'cap_seconds':180,
 'whole_seconds':time.monotonic()-started,'source_manifest_sha256':hashlib.sha256((packet/'MANIFEST.json').read_bytes()).hexdigest(),
 'executed_child_sha256':hashlib.sha256(child_code.encode()).hexdigest(),'CPU_only':True,'no_real_dataset_or_labels':True}
with (output/'TERMINAL.json').open('x') as stream: json.dump(record,stream,indent=2);stream.write('\n')
files={p.name:base64.b64encode(p.read_bytes()).decode() for p in output.iterdir() if p.is_file()}
print(json.dumps({'terminal':record,'files':files}))
raise SystemExit(0 if child.returncode==0 and not timed_out else 1)
'''


def main():
    output = HERE / 'root01'
    output.mkdir(exist_ok=False)
    manifest_path = PACKET / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    rows = list(manifest['files']) + [{'path': 'MANIFEST.json', 'sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}]
    files = []
    for row in rows:
        path = PACKET / row['path']
        if not path.resolve().is_relative_to(PACKET) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Reviewed local source differs')
        files.append({'path': row['path'], 'sha256': row['sha256'], 'data': base64.b64encode(path.read_bytes()).decode()})
    command = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', OUTER, REMOTE, UUID, CHILD])
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(SSH + [command], input=json.dumps({'files': files}), capture_output=True, text=True, timeout=225)
    (output / 'ssh_stderr.txt').write_text(result.stderr)
    with (output / 'LAUNCH.json').open('x') as stream:
        json.dump({'UTC': started, 'terminal_UTC': datetime.now(timezone.utc).isoformat(), 'exit_code': result.returncode,
                   'destination': LOGIN, 'CPU_only': True, 'source_manifest_sha256': rows[-1]['sha256'],
                   'automatic_retry': False}, stream, indent=2)
        stream.write('\n')
    try:
        received = json.loads(result.stdout)
    except ValueError:
        (output / 'transport_stdout.txt').write_text(result.stdout)
        raise RuntimeError('Qualification failed before receipt transfer')
    for name, encoded in received['files'].items():
        if Path(name).name != name or name in ('LAUNCH.json', 'ssh_stderr.txt'):
            raise ValueError('Invalid evidence transfer name')
        with (output / name).open('xb') as stream:
            stream.write(base64.b64decode(encoded))
    qualification = json.loads((output / 'QUALIFICATION.json').read_text()) if (output / 'QUALIFICATION.json').exists() else None
    print(json.dumps({'exit_code': result.returncode, 'terminal': received['terminal'],
                      'qualification': qualification, 'output': str(output)}))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
