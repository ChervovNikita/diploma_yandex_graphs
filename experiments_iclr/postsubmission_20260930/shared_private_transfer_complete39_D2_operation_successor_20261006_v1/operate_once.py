"""Create the explicitly approved actual-binding D2 release and run unchanged source once."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,shlex,subprocess,time,zlib
HERE=Path(__file__).resolve().parent;P=HERE.parent
SOURCE=P/'shared_private_transfer_amended39_d2_activation_preparation_20261006_v1'
HISTORY_REQUEST=P/'shared_private_transfer_complete39_history_inventory_root_request_preparation_20261006_v1'
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
PHASE=REPO+'/experiments_iclr/postsubmission_20260930'
OUT='shared_private_transfer_complete39_D2_analysis_execution_root_20261006_v1'
NATIVE=PHASE+'/native_ncn_runtime_20261005_v1/.venv/bin/python'
def sha(b):return hashlib.sha256(b).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(n,v):
 with (HERE/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 (HERE/n).chmod(0o444)
def bind(p):
 b=p.read_bytes();return {'path':str(p.relative_to(P)),'bytes':len(b),'sha256':sha(b)}
write('ROOT_AUTHORIZATION_RECEIPT.json',{'schema':'trusted_parent_exact_D2_creation_and_execution_authorization_receipt_v1','UTC':now(),'sender':'/root',
 'text':'Root explicitly approves creating/enabling and invoking the D2 release from ROOT_D2_ANALYSIS_RELEASE_PENDING.json with only actual history binding2b2b562d3a74b98b6edb3819b9240f4588b1cbc2767b6b5e03bd56027134ba2f, root_analysis_approved=true, collector30_plus9_custody_reviewed=true, and descriptive status updated. Registry/source/reviews/selector/gates remain exact. Bind actual approved collection/history metadata; execute unchanged reviewed accounting source0fdd061b… once into fresh complete39 D2 output, preserve logs/terminal/failures/cost and full output. No TEST,fit,model,new selection. No further root roundtrip required. Allocation only; no77 access.',
 'saved_as_trusted_instruction_receipt_not_root_generated_authority_file':True})
pending=json.loads((HISTORY_REQUEST/'ROOT_D2_ANALYSIS_RELEASE_PENDING.json').read_text());release=json.loads(json.dumps(pending))
release['VALID_history_inventory']={'path':'shared_private_transfer_complete39_history_inventory_execution_root_20261006_v1/VALID_HISTORY_INVENTORY.json','sha256':'2b2b562d3a74b98b6edb3819b9240f4588b1cbc2767b6b5e03bd56027134ba2f'}
release['root_analysis_approved']=True;release['collector30_plus9_custody_reviewed']=True;release['status']='ROOT_APPROVED_ACTUAL_COMPLETE39_HISTORY_D2_EXECUTION'
assert [k for k in pending if pending[k]!=release[k]]==['VALID_history_inventory','collector30_plus9_custody_reviewed','root_analysis_approved','status']
assert release['complete39_registry']['sha256']=='b3b318ed1a64bf73a5e453c4665183e3540fc7f8ef114fa41345922cf4860d25'
assert release['selection']=='first_maximum_complete_VALID_MRR_rounded4'
for k in ('TEST_access','fits_authorized','original_scores_recalculation','checkpoint_reselection','scientific_model_execution','promotion_policy_change'):assert release[k] is False
write('ROOT_D2_ANALYSIS_RELEASE.json',release);release_ref=bind(HERE/'ROOT_D2_ANALYSIS_RELEASE.json')
assert sha((SOURCE/'analyze_d2_accounting_proposal.py').read_bytes())=='0fdd061baaa71bb525c7122ff8bdf98c8dfe450074c41819cc3ab659e4ab648a'
assert sha((SOURCE/'MANIFEST.json').read_bytes())==release['analysis_source_manifest_sha256']=='3fc153c6f690af54080150725dc5e575e79ede5747d5b7cf989ad949ff0cf3f4'
files={}
def add(p):
 b=p.read_bytes();rel=str(p.relative_to(P));files[rel]={'path':rel,'bytes':len(b),'sha256':sha(b),'base64':base64.b64encode(b).decode()}
for r in json.loads((SOURCE/'MANIFEST.json').read_text())['files']:
 p=SOURCE/r['path'];b=p.read_bytes();assert sha(b)==r['sha256'] and len(b)==r['bytes'];add(p)
add(SOURCE/'MANIFEST.json');add(SOURCE/'SEAL.json')
for r in release['custody_review_evidence']:
 p=P/r['path'];assert sha(p.read_bytes())==r['sha256'];add(p)
accounting=json.loads((SOURCE/'OLD77_RETROSPECTIVE_ACCOUNTING_ADDENDUM.json').read_text())
for r in accounting['evidence_bindings']:
 p=P/r['path'];assert sha(p.read_bytes())==r['sha256'];add(p)
add(HERE/'ROOT_D2_ANALYSIS_RELEASE.json');assert len(files)==23
argv=[NATIVE,'-B',PHASE+'/shared_private_transfer_amended39_d2_activation_preparation_20261006_v1/analyze_d2_accounting_proposal.py','--release',PHASE+'/'+release_ref['path'],'--output',PHASE+'/'+OUT]
pins=[release['complete39_registry'],release['VALID_history_inventory'],
 {'path':'shared_private_transfer_complete39_collection_execution_root_20261006_v1/COLLECTION_RELEASE.json','sha256':'b9375929fcc458d21a3309757b4ff515d26eaf37d6edc3ff32a70159a03b065e'},
 {'path':'shared_private_transfer_complete39_history_inventory_root_request_preparation_20261006_v1/ROOT_HISTORY_INVENTORY_RELEASE.json','sha256':'d84424c9a4746c24d26a1c96f9bc686fe9ed6038b3d575e2339b244bd6905c78'}]
remote=r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,socket,subprocess,sys,zlib
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split();assert uuids==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
value=json.loads(zlib.decompress(base64.b64decode(sys.stdin.buffer.read())));rows=value['files'];pins=value['actual_pins'];argv=value['D2_argv'];output=phase/value['output']
assert len(rows)==23 and not output.exists() and Path(argv[0]).is_file()
for r in pins:
 p=phase/r['path'];assert p.resolve(strict=True).is_relative_to(phase) and not p.is_symlink() and p.stat().st_mode&0o222==0 and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
for r in rows:
 p=phase/r['path'];assert not Path(r['path']).is_absolute() and '..' not in Path(r['path']).parts and p.resolve().is_relative_to(phase)
 for q in [p,*p.parents]:
  if q==phase.parent:break
  assert not q.is_symlink()
 b=base64.b64decode(r['base64']);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
 if p.exists():assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'] and p.stat().st_size==r['bytes']
result=[]
for r in rows:
 p=phase/r['path'];existed=p.exists()
 if not existed:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(base64.b64decode(r['base64']))
  p.chmod(0o444)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
 result.append({k:r[k] for k in ('path','bytes','sha256')}|{'mode':oct(p.stat().st_mode&0o777),'existing_exact_bytes_reused_without_chmod':existed})
assert not output.exists()
stat=Path('/proc/self/stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
print(json.dumps({'schema':'exact_D2_staging_and_once_only_invocation_started_v1','UTC':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),'GPU_UUIDs':uuids,'files':result,
 'D2_process_identity':{'PID':os.getpid(),'start_ticks':ticks},'argv':argv,'fresh_output_absent':True,'staging_read_scientific_payload_semantics':False,'no77_connection':True}),flush=True)
env=dict(os.environ);env.update({'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
os.execve(argv[0],argv,env)
'''
data={'files':[files[k] for k in sorted(files)],'actual_pins':pins,'D2_argv':argv,'output':OUT}
remote=remote.replace("value=json.loads(zlib.decompress(base64.b64decode(sys.stdin.buffer.read())));rows=value['files'];","value="+repr(data)+";rows=value['files'];")
with (HERE/'STAGE_AND_INVOKE_REMOTE_SOURCE.py').open('x') as f:f.write(remote)
(HERE/'STAGE_AND_INVOKE_REMOTE_SOURCE.py').chmod(0o444)
payload=base64.b64encode(zlib.compress(remote.encode(),9)).decode()+'\n'
request=json.loads((HISTORY_REQUEST/'ROOT_APPROVAL_REQUEST.json').read_text());prefix=request['remote_invocation_argv'][:-1]
bootstrap="import sys,base64,zlib;exec(compile(zlib.decompress(base64.b64decode(sys.stdin.buffer.read())), '<exact-D2-staging-invocation>', 'exec'))"
command=prefix+['cd '+shlex.quote(REPO)+' && exec /usr/bin/python3 -I -S -B -c '+shlex.quote(bootstrap)]
write('LAUNCH_INTENT.json',{'UTC':now(),'release':release_ref,'source_sha256':'0fdd061baaa71bb525c7122ff8bdf98c8dfe450074c41819cc3ab659e4ab648a','argv':command,'D2_argv':argv,
 'remote_source_sha256':sha(remote.encode()),'stdin_sha256':sha(payload.encode()),'staged_exact_files':[{'path':r['path'],'bytes':r['bytes'],'sha256':r['sha256']} for r in data['files']],
 'actual_pins':pins,'once_only':True,'retry_or_resume':False,'device':'cpu','source_and_numerical_selectors_unchanged':True,'no_TEST_fit_model_or_new_selection':True})
print(json.dumps({'approved_D2_release':release_ref,'D2_argv':argv,'staged_source_and_metadata_files':23}),flush=True)
t=time.monotonic();started=now();proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
write('LOCAL_SSH_PROCESS_STARTED.json',{'UTC':started,'PID':proc.pid,'argv':command,'monotonic_started_seconds':t})
stdout,stderr=proc.communicate(input=payload)
exit_record={'UTC':now(),'started_UTC':started,'local_SSH_PID':proc.pid,'exit_code':proc.returncode,'terminal_wait_observed':True,'elapsed_seconds':time.monotonic()-t,
 'stdout':stdout,'stderr':stderr,'one_D2_invocation_requested':True,'external_timeout_signals_retry_or_resume':False,'no77_access':True}
lines=stdout.splitlines()
if lines:
 try:exit_record['staging_and_remote_D2_identity']=json.loads(lines[0])
 except json.JSONDecodeError:pass
write('D2_EXIT.json',exit_record);print(json.dumps({'D2_exit':proc.returncode,'elapsed_seconds':exit_record['elapsed_seconds']}),flush=True)
assert proc.returncode==0,'D2 failed/uncertain. Preserve same output and handle; do not rerun.'
gather=r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,socket,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
root=phase/OUTPUT_TEXT;p=root/'D2_RESULTS.json';assert {q.name for q in root.iterdir()}=={'D2_RESULTS.json'} and p.is_file() and not p.is_symlink() and p.stat().st_size<64_000_000
b=p.read_bytes();value=json.loads(b);assert len(value['records'])==39 and value['diagnostics']==['D2'] and value['numeric_runtime']['device']=='cpu'
for k in ('TEST_access','fits_authorized','checkpoint_reselection','scientific_model_execution','original_scores_recalculation'):assert value[k] is False
before=hashlib.sha256(b).hexdigest();p.chmod(0o444);assert hashlib.sha256(p.read_bytes()).hexdigest()==before
print(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'path':str(p.relative_to(phase)),'bytes':len(b),'sha256':before,'mode':'0444','utf8':b.decode()}))
'''.replace('OUTPUT_TEXT',repr(OUT))
compressed=base64.b64encode(zlib.compress(gather.encode(),9)).decode()+'\n'
gcommand=prefix+['cd '+shlex.quote(REPO)+' && exec /usr/bin/python3 -I -S -B -c '+shlex.quote(bootstrap)]
t=time.monotonic();g=subprocess.run(gcommand,input=compressed,capture_output=True,text=True,timeout=55)
receipt={'UTC':now(),'exit_code':g.returncode,'elapsed_seconds':time.monotonic()-t,'stderr':g.stderr,'remote_source_sha256':sha(gather.encode())}
if g.returncode==0:receipt['result']=json.loads(g.stdout)
else:receipt['stdout']=g.stdout
write('RESULT_GATHER_RECEIPT.json',receipt);assert g.returncode==0,'D2 succeeded but output gather failed; do not rerun.'
row=receipt['result'];b=row['utf8'].encode();assert len(b)==row['bytes'] and sha(b)==row['sha256']
with (HERE/'D2_RESULTS.json').open('xb') as f:f.write(b)
(HERE/'D2_RESULTS.json').chmod(0o444)
write('COMPLETION.json',{'UTC':now(),'status':'SUCCESS_FULL39_D2_RESULTS_PRESERVED','result':{k:row[k] for k in ('path','bytes','sha256','mode')},'local_result':bind(HERE/'D2_RESULTS.json'),
 'release':release_ref,'D2_invocations':1,'exit_code':0,'terminal_wait_observed':True,'cost_seconds':{'staging_and_D2':exit_record['elapsed_seconds'],'result_gather':receipt['elapsed_seconds']},
 'no_TEST_fit_model_new_selection_retry_or77_access':True})
print(json.dumps({'status':'SUCCESS_FULL39_D2_RESULTS_PRESERVED','result':{k:row[k] for k in ('path','bytes','sha256')},'D2_release':release_ref}),flush=True)
