"""Dispatch the committed root-admitted FiLM queue; no immediate scientific fit."""
import base64,hashlib,importlib.util,json,shlex,zlib
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent
REMOTE=r"""
import base64,datetime,hashlib,json,os,socket,subprocess,sys,time,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';H=P/'native_PyG_FiLM_WikiCS_comparator_execution_root_20261011_v1';S=P/'native_PyG_FiLM_WikiCS_comparator_source_20261011_v1'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==payload['commit']
assert subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip()=='codex/postsubmission-research-20260930'
admission=H/'ROOT_PIPELINE_ADMISSION_V1.json';assert hashlib.sha256(admission.read_bytes()).hexdigest()==payload['admission_sha256']
root=json.loads(admission.read_text());assert root['status']=='ROOT_FIXED_FiLM_PIPELINE_ADMITTED'
for name,row in root['execution_sources'].items():
 q=H/name;assert hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256']
 assert subprocess.check_output(['git','show','HEAD:'+str(q.relative_to(R))],cwd=R)==q.read_bytes()
for row in json.loads((S/'MANIFEST.json').read_text())['files']:
 q=S/row['path'];assert hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256']
 assert subprocess.check_output(['git','show','HEAD:'+str(q.relative_to(R))],cwd=R)==q.read_bytes()
assert not (H/'QUEUE_LAUNCH_V1.json').exists() and not (H/'PIPELINE_START_V1.json').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-44039938-fd82-41d2-fefd-de71514e2fac',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',DGLBACKEND='pytorch');env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
command=[str(R/'.venv/bin/python'),'-B',str(H/'queue_full_pipeline_v1.py'),'--admission',str(admission)]
with (H/'queue_outer_stdout_v1.log').open('x') as out,(H/'queue_outer_stderr_v1.log').open('x') as err:
 child=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
value=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue_PID=child.pid,commit=payload['commit'],source_manifest_sha256=root['source_manifest']['sha256'],root_admission_sha256=payload['admission_sha256'],banks=9,optimizer_bundles=18,calibration_attempts=45,queue_wait_cap_seconds=86400,fit_wall_cap_seconds=21600,readout_wall_cap_seconds=900,wait_for_physical_acquisition_closure_any_outcome=True,prior_predictive_outcomes_read=False,TEST_access=False,retry=False,paper_review=False)
(H/'QUEUE_LAUNCH_V1.json').write_text(json.dumps(value,indent=2)+'\n')
for _ in range(50):
 if (H/'PIPELINE_START_V1.json').exists():value['pipeline_start_present']=True;break
 if child.poll() is not None:value['early_queue_exit']=child.returncode;value['stderr']=(H/'queue_outer_stderr_v1.log').read_text()[-2500:];break
 time.sleep(.1)
print(json.dumps(value))
"""
def main():
 commit=json.loads((P/'publication/Native_FiLM9_18_45_ready_fixed_queue_and_updated_closest_prior_20261011_v1/COMMIT_RECEIPT.json').read_text())['commit']
 payload=dict(commit=commit,admission_sha256=hashlib.sha256((H/'ROOT_PIPELINE_ADMISSION_V1.json').read_bytes()).hexdigest())
 spec=importlib.util.spec_from_file_location('payload_transport',P/'publication/publish_exact_inventory_v19.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'FiLM-fixed-queue','literal-route','committed-source'])]
 r=mod.terminal_transport(argv,input=base64.b64encode(zlib.compress(json.dumps(payload).encode(),9)).decode(),capture_output=True,text=True,timeout=60)
 f=H/'QUEUE_LAUNCH_TRANSPORT_V1.json';assert not f.exists();f.write_text(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n');assert r.returncode==0,r.stderr[-2000:]+r.stdout[-2000:]
 value=json.loads(r.stdout);(H/'QUEUE_LAUNCH_V1.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
