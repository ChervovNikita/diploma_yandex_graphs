"""One owned allocation CPU extraction/census; compact receipts stay local."""
from pathlib import Path
import ast
import base64
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / 'private_transfer_pubmed_expansion_train_geometry_preparation_20261005_v1'
MANIFEST_SHA = '4924b9308031ecb3b9187c518ead4a26140fd701aa68e8da01b56b1ca4646c5f'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
REMOTE = r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,signal,socket,subprocess,sys,time
payload=json.load(sys.stdin)
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'private_transfer_pubmed_train_geometry_execution_root_20261005_v1'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=20).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert not root.exists() and root.resolve().is_relative_to(phase)
def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):d.update(b)
 return d.hexdigest()
def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
prepared=[]
for r in payload['files']:
 rel=Path(r['path']);assert not rel.is_absolute() and '..' not in rel.parts
 assert rel.parts[0]=='source' or rel.name in ('ROOT_REVIEW.md','ROOT_RELEASE.json','INDEPENDENT_SOURCE_REVIEW.md')
 b=base64.b64decode(r['base64'],validate=True)
 assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
 prepared.append((rel,b))
root.mkdir()
for rel,b in prepared:
 f=root/rel;f.parent.mkdir(parents=True,exist_ok=True)
 with f.open('xb') as out:out.write(b)
assert sha(root/'source/SOURCE_MANIFEST.json')==payload['source_manifest_sha256']
manifest=json.loads((root/'source/SOURCE_MANIFEST.json').read_text())
for r in manifest['files']:
 f=root/'source'/r['path'];assert f.resolve().is_relative_to(root/'source') and not f.is_symlink()
 assert sha(f)==r['sha256'] and f.stat().st_size==r['bytes']
job=json.loads((root/'ROOT_RELEASE.json').read_text())
assert sha(root/'ROOT_RELEASE.json')==payload['root_release_sha256']
assert job['source_manifest_sha256']==payload['source_manifest_sha256']
assert sha(root/'INDEPENDENT_SOURCE_REVIEW.md')==job['independent_source_review_sha256']
assert sha(root/'ROOT_REVIEW.md')==job['root_review_sha256']
assert all(job[k] is True for k in ('source_review_approved','TRAIN_member_extraction_authorized','TRAIN_geometry_measurement_authorized','external_900_second_hard_bound_confirmed','external_2100_second_hard_bound_confirmed'))
assert all(job[k] is False for k in ('network_access','VALID_TEST_member_access','feature_payload_access','model_access','fits_authorized','retry','training_admission'))
assert job['root_review_reference']==str(root/'ROOT_REVIEW.md')
assert job['extraction_output_directory']==str(root/'train') and job['geometry_output_directory']==str(root/'census')
assert job['train_only_input']['path']==str(root/'train/train_pos.txt')
caps=job['prospective_external_memory_output_caps']
assert caps['peak_RSS_bytes']==4294967296 and caps['output_bytes']==268435456
interpreter=phase/'native_ncn_runtime_20261005_v1/.venv/bin/python';assert interpreter.is_file()
env=dict(os.environ)
env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONPATH=str(phase/'native_ncn_dependency_overlay_20261005_v1')+':'+str(repo/'.venv/lib/python3.11/site-packages'))
env.pop('PYTHONHOME',None)
write(root/'STAGING_RECEIPT.json',dict(UTC=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),source_manifest_sha256=sha(root/'source/SOURCE_MANIFEST.json'),job_sha256=sha(root/'ROOT_RELEASE.json'),root_review_sha256=sha(root/'ROOT_REVIEW.md'),files=[{k:r[k] for k in ('path','bytes','sha256')} for r in payload['files']],fits=0))
def physical(pid):
 p=Path('/proc')/str(pid)
 try:
  raw=(p/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
  argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
  status=(p/'status').read_text().splitlines()
  rss=next((int(x.split()[1])*1024 for x in status if x.startswith('VmRSS:')),0)
  hwm=next((int(x.split()[1])*1024 for x in status if x.startswith('VmHWM:')),0)
  return dict(PID=pid,PPID=int(v[1]),start_ticks=int(v[19]),state=v[0],argv=argv,cwd=str((p/'cwd').resolve()) if v[0]!='Z' else None,RSS_bytes=rss,peak_RSS_bytes=hwm)
 except FileNotFoundError:return None
def output_bytes():
 total=0
 for f in root.rglob('*'):
  assert not f.is_symlink()
  if f.is_file():total+=f.stat().st_size
 return total
def run(stage,script,output,hard):
 command=[str(interpreter),'-B',str(root/'source'/script),'--recipe',str(root/'ROOT_RELEASE.json'),'--output',str(output)]
 start=time.monotonic();peak=0;failure=None;signals=[];usage=None;code=None
 with (root/(stage+'.stdout.log')).open('xb') as out,(root/(stage+'.stderr.log')).open('xb') as err:
  child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
  # Popen creates this direct child and no poll()/wait() reaps it. Its PID
  # cannot be reused until wait4 returns the authoritative terminal status.
  owner=physical(child.pid)
  def reap(block=False):
   nonlocal code,peak,usage
   if code is not None:return True
   pid,status,rusage=os.wait4(child.pid,0 if block else os.WNOHANG)
   if not pid:return False
   code=os.waitstatus_to_exitcode(status);child.returncode=code
   peak=max(peak,int(rusage.ru_maxrss)*1024)
   usage=dict(peak_RSS_bytes=int(rusage.ru_maxrss)*1024,user_seconds=rusage.ru_utime,system_seconds=rusage.ru_stime)
   return True
  ended=reap()
  ownership_admitted=True  # kernel-created, direct, unreaped Popen child
  scope_admitted=ended or (owner is not None and owner['argv']==command and owner['cwd']==str(repo))
  write(root/(stage+'.START.json'),dict(UTC=datetime.now(timezone.utc).isoformat(),identity=owner,command=command,ownership_admitted=ownership_admitted,command_scope_admitted=scope_admitted,ownership_basis='direct unreaped Popen child plus parent/start identity; terminal via wait4',hard_seconds=hard,one_attempt=True))
  if not ended and (owner is None or owner['PPID']!=os.getpid()):failure='physical_ownership_observation'
  if not scope_admitted:failure='command_scope_admission'
  while code is None:
   if reap():break
   now=physical(child.pid)
   if now is None:
    if reap():break
    failure='physical_ownership_observation'
   elif owner is None or now['PPID']!=os.getpid() or now['start_ticks']!=owner['start_ticks']:failure='physical_identity_changed'
   elif now['state']!='Z' and (now['argv']!=command or now['cwd']!=str(repo)):failure='command_scope_changed'
   if now is not None:peak=max(peak,now['RSS_bytes'],now['peak_RSS_bytes'])
   size=output_bytes()
   write_value=dict(UTC=datetime.now(timezone.utc).isoformat(),stage=stage,identity=now,elapsed_seconds=time.monotonic()-start,peak_observed_RSS_bytes=peak,own_output_bytes=size,quality_access=False)
   (root/'CURRENT_METADATA.json').write_text(json.dumps(write_value,indent=2)+'\n')
   if peak>caps['peak_RSS_bytes']:failure='owned_RSS_cap'
   if size>caps['output_bytes']:failure='owned_output_cap'
   if time.monotonic()-start>hard:failure='owned_hard_time_cap'
   if failure:
    fresh=physical(child.pid)
    if not reap():
     # This remains the same unreaped direct child even if /proc observation
     # failed. No unrelated/reused PID can occupy its kernel wait handle.
     child.terminate();signals.append('SIGTERM')
     deadline=time.monotonic()+5
     while not reap() and time.monotonic()<deadline:time.sleep(.05)
     if code is None:
      child.kill();signals.append('SIGKILL');reap(block=True)
    break
   time.sleep(1)
  if code is None:reap(block=True)
 elapsed=time.monotonic()-start;size=output_bytes()
 if peak>caps['peak_RSS_bytes']:failure='terminal_RSS_cap'
 if size>caps['output_bytes']:failure='terminal_output_cap'
 if elapsed>hard:failure='terminal_hard_time_cap'
 receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),stage=stage,identity=owner,exit_code=code,elapsed_seconds=elapsed,peak_observed_RSS_bytes=peak,authoritative_wait4_usage=usage,own_output_bytes=size,signals_sent=signals,failure=failure,one_attempt=True,fits=0,VALID_TEST_feature_model_access=False)
 write(root/(stage+'.TERMINAL.json'),receipt)
 print(json.dumps({'stage':stage,'terminal':receipt}),flush=True)
 if code or failure or signals:raise RuntimeError('Owned CPU stage failed; artifacts retained and no retry')
 return receipt
stages=[]
try:
 stages.append(run('extraction','extract_pubmed_train_only.py',root/'train',900))
 extraction=json.loads((root/'train/TRAIN_EXTRACTION_RECEIPT.json').read_text())
 assert extraction['status']=='TRAIN_EXTRACTED_NOT_TRAINING_ADMITTED'
 assert extraction['archive_sha256']==job['archive_sha256'] and extraction['train_sha256']==job['train_only_input']['sha256']
 assert extraction['rows']==37676 and extraction['bytes']==409212 and extraction['fits']==0
 assert extraction['source_manifest_sha256']==payload['source_manifest_sha256'] and extraction['job_sha256']==sha(root/'ROOT_RELEASE.json')
 assert sha(root/'train/train_pos.txt')==job['train_only_input']['sha256']
 assert (root/'train/train_pos.txt').stat().st_size==409212
 stages.append(run('census','inspect_train_episode_geometry.py',root/'census',2100))
 result_path=root/'census/RESULT.json';assert result_path.is_file()
 v=json.loads(result_path.read_text());assert v['status']=='PASS_TRAIN_GEOMETRY_MEASUREMENT_ONLY' and v['fits']==0 and v['VALID_TEST_access'] is False
 assert v['source_manifest_sha256']==payload['source_manifest_sha256'] and v['job_sha256']==sha(root/'ROOT_RELEASE.json')
 assert v['runtime']==job['expected_runtime_versions'] and v['candidate_geometry']==job['candidate_geometry']
 assert len(v['episodes'])==589 and [r['episode'] for r in v['episodes']]==list(range(589))
 assert v['schedule']['outer_episodes']==589 and v['schedule']['outer_positive_rows_dropped']==0
 assert v['full_TRAIN']['sha256']==job['train_only_input']['sha256']
 assert v['peak_RSS_bytes']<=caps['peak_RSS_bytes'] and output_bytes()<=caps['output_bytes']
 feasible=v['schedule']['paired_feasible_episodes']==589 and v['schedule']['endpoint_feasible_episodes']==589 and v['schedule']['rejected_episode_indices']==[] and all(r['endpoint_feasible'] and r['control_feasible'] and 'paired_endpoint' in r for r in v['episodes'])
 def distribution(values):
  return {'count':len(values),'minimum':min(values),'maximum':max(values),'mean':sum(values)/len(values)} if values else {'count':0}
 paired=[r['paired_endpoint']['masked_positive_facts'] for r in v['episodes'] if 'paired_endpoint' in r]
 own=[r['endpoint_own_mask']['masked_positive_facts'] for r in v['episodes']]
 compact=dict(UTC=datetime.now(timezone.utc).isoformat(),status=v['status'],scope=v['scope'],source_manifest_sha256=v['source_manifest_sha256'],job_sha256=v['job_sha256'],result_sha256=sha(result_path),result_bytes=result_path.stat().st_size,runtime=v['runtime'],elapsed_seconds=v['elapsed_seconds'],peak_RSS_bytes=v['peak_RSS_bytes'],full_TRAIN=v['full_TRAIN'],schedule=v['schedule'],endpoint_exposure=v['endpoint_exposure'],matched_random_exposure=v['matched_random_exposure'],control_contract=v['control_contract'],paired_masked_facts=distribution(paired),own_masked_facts=distribution(own),paired_removed_fractions=distribution([x/37676 for x in paired]),all_feasible_endpoint_rows_have_zero_outer_touches=all(r['paired_endpoint']['inner_queries_touching_outer_endpoints']==0 for r in v['episodes'] if 'paired_endpoint' in r),all_feasible_rows_match_pre_mask_strata=all(r.get('pre_mask_strata_match_exact_per_route',False) for r in v['episodes'] if 'paired_endpoint' in r),training_admitted=False,fits=0,VALID_TEST_feature_model_access=False)
 compact['all_589_geometry_feasible']=feasible
 compact['candidate_geometry_decision']='FEASIBLE_SEED0_CYCLE0_ONLY' if feasible else 'REJECT_CANDIDATE_GEOMETRY_NO_FALLBACK'
 write(root/'COMPACT_RESULT.json',compact)
 write(root/'EXECUTION_RECEIPT.json',dict(complete=True,stages=stages,source_bytes_unchanged=all(sha(root/r['path'])==r['sha256'] for r in payload['files']),authenticated_extraction=extraction,train_sha256=sha(root/'train/train_pos.txt'),fits=0,result_sha256=sha(result_path)))
 assert output_bytes()<=caps['output_bytes']
 print(json.dumps({'completed':True,'compact_result':compact,'execution_receipt':json.loads((root/'EXECUTION_RECEIPT.json').read_text())}),flush=True)
except Exception as error:
 write(root/'EXECUTION_FAILURE.json',dict(UTC=datetime.now(timezone.utc).isoformat(),error=type(error).__name__+': '+str(error),completed_stages=stages,partial_files_preserved=True,no_retry=True,fits=0))
 raise
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ast.parse(REMOTE)
    assert sha(SOURCE/'SOURCE_MANIFEST.json') == MANIFEST_SHA
    assert not (HERE/'LOCAL_INTENT.json').exists()
    files = []
    for f in sorted(SOURCE.iterdir()):
        assert not f.is_symlink()
        if f.is_file():
            b = f.read_bytes()
            files.append(dict(path='source/'+f.name, bytes=len(b), sha256=hashlib.sha256(b).hexdigest(), base64=base64.b64encode(b).decode()))
    for name in ('ROOT_REVIEW.md', 'ROOT_RELEASE.json','INDEPENDENT_SOURCE_REVIEW.md'):
        f = HERE/name
        b = f.read_bytes()
        files.append(dict(path=name, bytes=len(b), sha256=hashlib.sha256(b).hexdigest(), base64=base64.b64encode(b).decode()))
    payload = dict(files=files, source_manifest_sha256=MANIFEST_SHA,root_release_sha256=sha(HERE/'ROOT_RELEASE.json'))
    ssh = ['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20',LOGIN]
    command = 'cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs && exec python3 -I -B -c '+shlex.quote(REMOTE)
    with (HERE/'LOCAL_INTENT.json').open('x') as out:
        json.dump(dict(single_attempt=True,singleton_only=True,client_sha256=sha(Path(__file__)),remote_source_sha256=hashlib.sha256(REMOTE.encode()).hexdigest(),root_release_sha256=payload['root_release_sha256'],relaunch_after_disconnect=False),out,indent=2);out.write('\n')
    try:
        result = subprocess.run([*ssh, command], input=json.dumps(payload), capture_output=True, text=True, timeout=3100)
    except subprocess.TimeoutExpired as error:
        def decoded(value):return value.decode(errors='replace') if isinstance(value,bytes) else (value or '')
        (HERE/'CONSOLE.jsonl').write_text(decoded(error.stdout))
        (HERE/'TRANSPORT.json').write_text(json.dumps(dict(transport_timeout=True,remote_terminal_status='UNKNOWN_REPOLL_EXACT_ROOT_DO_NOT_RELAUNCH',stderr=decoded(error.stderr),singleton_only=True),indent=2)+'\n')
        raise
    (HERE/'CONSOLE.jsonl').write_text(result.stdout)
    (HERE/'TRANSPORT.json').write_text(json.dumps(dict(exit_code=result.returncode,stderr=result.stderr,remote_source_sha256=hashlib.sha256(REMOTE.encode()).hexdigest(),client_sha256=sha(Path(__file__)),singleton_only=True),indent=2)+'\n')
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    value = json.loads(result.stdout.splitlines()[-1])
    assert value['completed'] is True
    for key,name in [('compact_result','COMPACT_RESULT.json'),('execution_receipt','EXECUTION_RECEIPT.json')]:
        with (HERE/name).open('x') as out:
            json.dump(value[key],out,indent=2);out.write('\n')
    print(json.dumps({'status':value['compact_result']['status'],'schedule':value['compact_result']['schedule'],'geometry_only':True,'training_admitted':False,'raw_result_fetched':False}))


if __name__ == '__main__':
    main()
