"""Deploy exact FiLM source and own one disposable full-graph runtime check."""
import ast,base64,hashlib,importlib.util,json,shlex,zlib
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent
S=P/'native_PyG_FiLM_WikiCS_comparator_source_20261011_v1'
REMOTE=r"""
import base64,datetime,hashlib,json,os,socket,subprocess,sys,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';H=P/'native_PyG_FiLM_WikiCS_comparator_execution_root_20261011_v1'
for row in payload:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 path=P/rel;assert path.resolve().is_relative_to(P)
 b=base64.b64decode(row['data']);assert hashlib.sha256(b).hexdigest()==row['sha256'];path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():assert path.read_bytes()==b
 else:path.write_bytes(b)
assert not (H/'READINESS_LAUNCH_V1.json').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-44039938-fd82-41d2-fefd-de71514e2fac',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',DGLBACKEND='pytorch')
env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
bound=subprocess.run([str(R/'.venv/bin/python'),'-B',str(H/'bind_readiness.py')],cwd=R,env=env,capture_output=True,text=True,timeout=60)
(H/'binding_stdout.log').write_text(bound.stdout);(H/'binding_stderr.log').write_text(bound.stderr)
assert bound.returncode==0,bound.stderr[-2500:]
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=32768,'Insufficient currently free CUDA memory; no child spawned'
with (H/'owner_outer_stdout.log').open('x') as out,(H/'owner_outer_stderr.log').open('x') as err:
 child=subprocess.Popen([str(R/'.venv/bin/python'),'-B',str(H/'readiness_owner.py')],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
value=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner_pid=child.pid,source_manifest_sha256='f66f2448cd1df8b2a77efa5f8b71b5fe5d9a50d54ed75da01d2c28a308031a0d',GPU_free_MiB_before_spawn=free,scientific_fit_or_selection=False,one_disposable_update=True,TEST_access=False,science_launch_admitted=False,binding=json.loads((H/'READINESS_BINDING_V1.json').read_text()))
(H/'READINESS_LAUNCH_V1.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
"""
def main():
 manifest=json.loads((S/'MANIFEST.json').read_text());paths=[]
 for row in manifest['files']:
  path=S/row['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'];paths.append(path)
 paths.extend([S/'MANIFEST.json',S/'SEAL.json',H/'bind_readiness.py',H/'readiness_owner.py'])
 rows=[]
 for path in paths:
  b=path.read_bytes()
  if path.suffix=='.py':ast.parse(b)
  rows.append(dict(path=str(path.relative_to(P)),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
 spec=importlib.util.spec_from_file_location('payload_transport',P/'publication/publish_exact_inventory_v19.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'FiLM-readiness','literal-route','one-disposable-update'])]
 result=mod.terminal_transport(argv,input=base64.b64encode(zlib.compress(json.dumps(rows).encode(),9)).decode(),capture_output=True,text=True,timeout=90)
 output=H/'READINESS_LAUNCH_TRANSPORT_V1.json';assert not output.exists();output.write_text(json.dumps(dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
 assert result.returncode==0,result.stderr[-2000:]+result.stdout[-2000:]
 value=json.loads(result.stdout);(H/'READINESS_LAUNCH_V1.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(dict(owner_pid=value['owner_pid'],GPU_free_MiB=value['GPU_free_MiB_before_spawn'],scientific_fit_or_selection=False,science_launch_admitted=False)))
if __name__=='__main__':main()
