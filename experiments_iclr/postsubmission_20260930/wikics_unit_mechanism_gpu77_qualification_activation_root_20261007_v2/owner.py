from pathlib import Path
from types import SimpleNamespace
import os,socket,subprocess,importlib.util,json,hashlib,datetime,time
R=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git');P=R/'experiments_iclr/postsubmission_20260930';A=P/'wikics_unit_mechanism_gpu77_qualification_activation_root_20261007_v2';S=P/'internal_BE_WikiCS_unit_mechanism_ablation_source_20261007_v2'
assert Path.cwd()==R and socket.gethostname()=='peptide'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def write(f,a):Path(f).write_text(json.dumps(a,indent=2)+'\n')
def phase_file(s):
 f=(P/s).resolve();assert f.is_relative_to(P) and f.is_file();return f
def module(name,row):
 f=phase_file(row['path']);assert sha(f)==row['sha256'];spec=importlib.util.spec_from_file_location(name,f);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
pins=json.loads((S/'SOURCE_BINDINGS.json').read_text());cfg=json.loads((A/'RELEASE.json').read_text());assert sha(A/'RELEASE.json')=='9f9ce7752694183539344e808a65858692b8954241c4229c79f943d0e9105308'
helper=module('qual_Wiki_mechanism_helper',pins['reviewed_ownership_helper']);helper.GPU=cfg['physical_gpu_uuid'];supervisor=module('qual_Wiki_mechanism_existing_runfit',pins['reviewed_owned_fit_helper'])
def physical():
 assert socket.gethostname()=='peptide' and Path.cwd()==R
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==pins['physical_gpu_inventory']
context=SimpleNamespace(REPO=R,SOURCE=S,SOURCE_SHA=cfg['source_manifest_sha256'],GPU_UUID=cfg['physical_gpu_uuid'],GPU_UUIDS=tuple(pins['physical_gpu_inventory']),phase_file=phase_file,physical_host=physical,sha=sha,write=write)
limits=dict(owned_tree_GPU_memory_cap_bytes=cfg['owned_GPU_memory_cap_bytes'],owned_tree_RSS_cap_bytes=34359738368,combined_child_log_cap_bytes=8388608,own_fit_output_cap_bytes=4294967296,minimum_fresh_GPU_free_bytes=pins['minimum_fresh_free_GPU_bytes'],resource_wait_seconds=1800,poll_interval_seconds=5,telemetry_timeout_seconds=10)
(A/'logs').mkdir();write(A/'PARENT_OWNER.json',helper.identity(os.getpid()))
entry=dict(cell_id='qualifier',job_relative=str((A/'RELEASE.json').relative_to(P)),job_sha256=sha(A/'RELEASE.json'),hard_seconds=cfg['external_hard_seconds'],argv=[pins['python'],'-B',str(S/'qualify.py'),'--release',str(A/'RELEASE.json'),'--release-sha256',sha(A/'RELEASE.json')])
env=dict(os.environ,CUDA_VISIBLE_DEVICES=cfg['physical_gpu_uuid'],PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2');env.pop('PYTHONHOME',None)
try:
 receipt=supervisor.run_fit(helper,A,entry,{'resource_limits':limits},env,P/cfg['output'],context);child=receipt['raw_identity_observation'];absent=child is not None and helper.identity(child['PID']) is None
 cuda=helper.query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],10);no_cuda=child is not None and not any(len(parts)>1 and parts[1].strip()==str(child['PID']) for parts in (x.split(',') for x in cuda))
 passed=receipt['exit_code']==0 and receipt['reason'] is None and not receipt['signals_sent'] and receipt['terminal_wait_observed'] and absent and no_cuda and receipt['elapsed_seconds']<=cfg['external_hard_seconds']
 endpoint=P/cfg['output']/'QUALIFIED.json'
 if passed:
  a=json.loads(endpoint.read_text());assert a['complete'] is True and len(a['rows'])==8 and a['source_manifest_sha256']==cfg['source_manifest_sha256'] and a['release_sha256']==sha(A/'RELEASE.json')
 result=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete=bool(passed),receipt=receipt,child_absent=absent,child_no_CUDA_rows=no_cuda,qualified=dict(path=str(endpoint.relative_to(P)),sha256=sha(endpoint)) if passed else None,scientific_fits=0,discarded_updates=8 if passed else None)
 write(A/('COMPLETE.json' if passed else 'FAILURE.json'),result)
except Exception as e:write(A/'FAILURE.json',dict(complete=False,error=type(e).__name__+': '+str(e),scientific_fits=0));raise
