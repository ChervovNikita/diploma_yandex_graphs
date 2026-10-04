"""Prepare one CPU-only available-data inspection inside the authorized repository."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import base64
import hashlib
import json
import lzma
import shlex

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
SOURCE = PHASE / 'pubmed_heart_available_inspector_native_adapter_20261004_v1'
SOURCE_SHA = '8d05f0bbf5bbca50cd3abc98f63bf796244510528d471329428ed5edbc9f3f5b'
ENTRY_SHA = 'bae1dcc325bce6432ce3dfa6cf8a9bfc618abf07aeb3f73960bce17351ae77ce'
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE = REPO / 'experiments_iclr/postsubmission_20260930'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(SOURCE / 'MANIFEST.json') == SOURCE_SHA
assert sha(SOURCE / 'inspect_available.py') == ENTRY_SHA
manifest = json.loads((SOURCE / 'MANIFEST.json').read_text())
for row in manifest['files']:
    path = SOURCE / row['path']
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
ast.parse((SOURCE / 'inspect_available.py').read_text())
admission = dict(schema='root-Pubmed-available-inspection-admission-v1',
                 UTC=datetime.now(timezone.utc).isoformat(), source_manifest_sha256=SOURCE_SHA,
                 inspector_sha256=ENTRY_SHA, root_read_complete_inspector=True,
                 authorized_files=['gnn_feature', 'train_pos.txt', 'valid_pos.txt', 'heart_valid_samples.npy'],
                 weights_only_CPU_feature_load=True, NumPy_allow_pickle=False,
                 feature_and_negative_pool_geometry_only=True,
                 complete_available_pool_collisions_reported_without_altering_benchmark=True,
                 literal_feature_exporter_authority_unresolved=True,
                 scientific_training_or_TEST_authorized=False, GPU_execution=False,
                 wall_cap_seconds=90, owned_child_RSS_cap_bytes=4*1024**3,
                 automatic_retry=False, server_configuration_changes=False)
with (ROOT / 'ROOT_INSPECTION_ADMISSION.json').open('x') as handle:
    json.dump(admission, handle, indent=2)
    handle.write('\n')
paths = [SOURCE / row['path'] for row in manifest['files']] + [SOURCE / 'MANIFEST.json', SOURCE / 'SEAL.json', ROOT / 'ROOT_INSPECTION_ADMISSION.json']
rows = [dict(path=str(path.relative_to(PHASE)), bytes=path.stat().st_size, sha256=sha(path)) for path in paths]
metadata = json.dumps(rows).encode()
data = len(metadata).to_bytes(8,'little') + metadata + b''.join(path.read_bytes() for path in paths)
packed = base64.b64encode(lzma.compress(data, preset=9)).decode()
code = '''from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,lzma,os,signal,socket,subprocess,time
repo=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git');phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='peptide'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
 with path.open('x') as handle:json.dump(value,handle,indent=2,allow_nan=False);handle.write('\\n')
def physical(pid):
 p=Path('/proc')/str(pid);raw=(p/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
 return dict(PID=pid,start_time_ticks=int(v[19]),process_group=int(v[2]),session=int(v[3]))
'''
code += 'root=phase/'+repr(ROOT.name)+'\nsource=phase/'+repr(SOURCE.name)+'\npacked='+repr(packed)+'\n'
code += '''assert not root.exists(),'Inspection identity already exists; no retry'
python=Path('/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12')
assert sha(python)=='14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'
data=lzma.decompress(base64.b64decode(packed,validate=True));assert len(data)<2*1024**2
length=int.from_bytes(data[:8],'little');assert length<1024**2
rows=json.loads(data[8:8+length]);offset=8+length;assert len(data)==offset+sum(r['bytes'] for r in rows)
for row in rows:
 path=(phase/row['path']).resolve();assert path.is_relative_to(source) or path.is_relative_to(root)
 raw=data[offset:offset+row['bytes']];offset+=row['bytes']
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if path.exists():assert path.is_file() and not path.is_symlink() and path.read_bytes()==raw
 else:
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as handle:handle.write(raw)
assert sha(source/'MANIFEST.json')=='8d05f0bbf5bbca50cd3abc98f63bf796244510528d471329428ed5edbc9f3f5b'
assert sha(source/'inspect_available.py')=='bae1dcc325bce6432ce3dfa6cf8a9bfc618abf07aeb3f73960bce17351ae77ce'
for row in json.loads((source/'MANIFEST.json').read_text())['files']:
 path=source/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
argv=[str(python),'-B',str(source/'inspect_available.py'),'--receipt',str(root/'AVAILABLE_INSPECTION.json')]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
began=time.monotonic();UTC=datetime.now(timezone.utc).isoformat();violation=None;peak=0;signal_sent=False
with (root/'CHILD.stdout.log').open('x') as stdout,(root/'CHILD.stderr.log').open('x') as stderr:
 child=subprocess.Popen(argv,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
 identity=physical(child.pid);assert identity['process_group']==identity['session']==child.pid
 save(root/'CHILD_STARTED.json',dict(UTC=UTC,physical_identity=identity,argv=argv,CPU_only=True))
 while True:
  waited,status,usage=os.wait4(child.pid,os.WNOHANG)
  if waited:exit_code=os.waitstatus_to_exitcode(status);child.returncode=exit_code;break
  try:rss=next((int(line.split()[1])*1024 for line in (Path('/proc')/str(child.pid)/'status').read_text().splitlines() if line.startswith('VmRSS:')),0)
  except FileNotFoundError:rss=0
  peak=max(peak,rss);elapsed=time.monotonic()-began
  if violation is None and (elapsed>=90 or rss>4*1024**3):
   violation=dict(kind='wall' if elapsed>=90 else 'owned_child_RSS',elapsed_seconds=elapsed,RSS_bytes=rss)
   try:
    assert physical(child.pid)['start_time_ticks']==identity['start_time_ticks'];os.killpg(child.pid,signal.SIGKILL);signal_sent=True
   except ProcessLookupError:pass
  time.sleep(.25)
terminal=dict(schema='pubmed-available-inspection-physical-terminal-v1',UTC=datetime.now(timezone.utc).isoformat(),status='COMPLETE' if exit_code==0 and violation is None else 'FAILED',physical_identity=identity,child_exit_code=exit_code,physical_inclusive_wall_seconds=time.monotonic()-began,kernel_wait4_peak_RSS_bytes=usage.ru_maxrss*1024,sampled_peak_owned_child_RSS_bytes=peak,cap_violation=violation,owned_child_signal_sent=signal_sent,other_jobs_signaled=False,GPU_execution=False,TEST_opened=False,scientific_training=False,automatic_retry=False)
receipt=root/'AVAILABLE_INSPECTION.json'
if receipt.is_file():terminal['inspection_receipt']=dict(path=str(receipt),bytes=receipt.stat().st_size,sha256=sha(receipt))
save(root/'PHYSICAL_TERMINAL.json',terminal)
result=dict(physical_terminal=terminal,scalar_inspection=json.loads(receipt.read_text()) if receipt.is_file() else None,stderr=(root/'CHILD.stderr.log').read_text())
print(json.dumps(result))
assert terminal['status']=='COMPLETE','Preserve failed inspection; no retry'
'''
ast.parse(code)
command = 'cd '+shlex.quote(str(REPO))+'\n'+shlex.join(['/usr/bin/python3','-I','-S','-B','-c',code])+'\n'
assert len(command.encode()) < 100000
path = PHASE/'gpu77_connection_recovery_v1/pubmed_available_inspection_20261004_v1_command.txt'
with path.open('x') as handle:
    handle.write(command)
with (ROOT/'COMMAND_PREPARATION.json').open('x') as handle:
    json.dump(dict(command=str(path.relative_to(PHASE)),bytes=len(command.encode()),sha256=sha(path),staged_payloads=rows,source_and_command_AST_parsed=True,executed=False),handle,indent=2)
    handle.write('\n')
print(json.dumps(dict(command=str(path),bytes=len(command.encode()),payloads=len(rows))))
