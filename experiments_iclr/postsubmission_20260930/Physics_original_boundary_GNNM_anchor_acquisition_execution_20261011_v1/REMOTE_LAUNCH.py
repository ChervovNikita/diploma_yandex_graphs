import pathlib,json,hashlib,socket,subprocess,os,base64,zlib,time
R=pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==[UUID]
os.chdir(R)
payload=json.loads(zlib.decompress(base64.b64decode('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')))
phase=R/'experiments_iclr/postsubmission_20260930'
source=phase/'Physics_original_boundary_GNNM_anchor_source_20261011_v1'
readyH=phase/'Physics_original_boundary_GNNM_anchor_readiness_execution_20261011_v1'
H=phase/'Physics_original_boundary_GNNM_anchor_acquisition_execution_20261011_v1'
def receipt(p):
 p=pathlib.Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(r):
 assert receipt(r['path'])==r,r['path']
 return pathlib.Path(r['path'])
def write(p,v):
 with p.open('x') as f:f.write(json.dumps(v,indent=2)+'\n')
def identity(pid):
 p=pathlib.Path('/proc')/str(pid)/'stat'
 if not p.exists():return None
 a=p.read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,start_ticks=int(a[19]),pgid=int(a[2]),state=a[0])
assert receipt(source/'MANIFEST.json')['sha256']==payload['source_manifest_sha256']
manifest=json.loads((source/'MANIFEST.json').read_text())
for row in manifest['files']:
 r=receipt(source/row['path']);assert r['bytes']==row['bytes'] and r['sha256']==row['sha256']
rr=json.loads(checked(payload['readiness']['result']).read_text())
ro=json.loads(checked(payload['readiness']['owner_end']).read_text())
assert rr['complete'] and rr['training_updates']==1 and rr['restoration_exact'] and not rr['SELECT_array_read'] and not rr['CAL_TEST_access']
assert ro['status']=='COMPLETE_EXIT_0' and ro['exit_code']==0 and ro['direct_child_wait'] and ro['child_pid_absent'] and ro['owned_cuda_pid_absent'] and not ro['timed_out']
prior=json.loads((readyH/'SPEC.json').read_text())
bindings=json.loads((source/'SOURCE_BINDINGS.json').read_text())
assert prior['roles']==bindings['Physics_roles']
for block in prior['roles'].values():
 for item in block.values():checked(item)
for item in prior['source_receipts'].values():checked(item)
metadata=receipt(phase/'native_SAGE_Physics_confirmation_root_20261011_v1/FROZEN_BINDINGS_V1.json')
readydecision=json.loads((readyH/'ROOT_DECISION.json').read_text())
assert metadata==readydecision['existing_Physics_metadata_binding']
source_sha={str(source/name):receipt(source/name)['sha256'] for name in ['acquire.py','PLAN.json','INTERFACE.json','SOURCE_BINDINGS.json']}
assert rr['source_sha256']==source_sha
H.mkdir(exist_ok=False)
for name in ['worker.py','owner.py']:
 with (H/name).open('x') as f:f.write(payload[name[:-3]])
limits=dict(wall_seconds=43200,cuda_bytes=48*1024**3,rss_bytes=64*1024**3)
decision=dict(status='ROOT_COMPLETE_ORIGINAL_PHYSICS_ANCHOR_ADMITTED',
 authority='Explicit root approval after reading exact readiness/result/owner/decision receipts: all ten frozen blocks and ten fresh original native bodies',
 action='Physics_original_boundary_acquisition',limits=limits,external_owner_wall_seconds=43320,
 source_directory=str(source),source_sha256=source_sha,source_manifest=receipt(source/'MANIFEST.json'),
 readiness=payload['readiness'],readiness_body_discarded_not_promoted=True,
 existing_Physics_metadata_binding=metadata,verified_runtime_sources=prior['source_receipts'],
 verified_all_TRAIN_SELECT_receipts=prior['roles'],runtime_executable=str(R/'.venv/bin/python'),
 owner_source=receipt(H/'owner.py'),worker_source=receipt(H/'worker.py'),
 banks=10,actual_fits=10,selected_native_routes=40,blocks=list(map(int,bindings['Physics_roles'])),
 native_recipe='Original multiclass ownCE/4;5000cap;300patience;strict per10 save;global sequential dropout;raw-logit mean',
 CAL_TEST_access=False,new_candidate=False,source_or_recipe_change=False,
 full_scientific_acquisition_admitted=True,no_retry=True,no_partial_quality_interpretation=True,
 calibration_or_serving_followup_admitted=False,encountered_graph_exploration=True,unused_confirmation_claim=False)
write(H/'ROOT_DECISION.json',decision)
spec=dict(output=str(H/'acquisition_v1'),limits=limits,decision=receipt(H/'ROOT_DECISION.json'),
 source_receipts=prior['source_receipts'],roles=prior['roles'],readiness=payload['readiness'])
write(H/'SPEC.json',spec)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
argv=[str(R/'.venv/bin/python'),'-B','-u',str(H/'owner.py')]
with (H/'owner.log').open('xb') as log:
 owner=subprocess.Popen(argv,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 birth=identity(owner.pid);assert birth and birth['pgid']==owner.pid
launch=dict(status='OWNER_DISPATCHED',owner=birth,command=argv,source_manifest=receipt(source/'MANIFEST.json'),
 spec=receipt(H/'SPEC.json'),decision=receipt(H/'ROOT_DECISION.json'),readiness=payload['readiness'],
 limits=limits,external_owner_seconds=43320,full_panel={'banks':10,'actual_fits':10,'selected_native_routes':40},
 detached_direct_owner=True,retry=False,CAL_TEST_access=False,partial_quality_opened=False)
write(H/'LAUNCH.json',launch)
deadline=time.monotonic()+15
while not (H/'OWNER_START.json').exists():
 if owner.poll() is not None:raise RuntimeError('Owner exited before direct-child start: '+(H/'owner.log').read_text()[-3000:])
 if time.monotonic()>deadline:raise RuntimeError('Launch metadata not ready; retain actual dispatch, never redispatch')
 time.sleep(.1)
start=json.loads((H/'OWNER_START.json').read_text())
child=identity(start['child']['pid'])
assert child and child['start_ticks']==start['child']['start_ticks']
progress=dict(status='RUNNING',hostname=socket.gethostname(),GPU_UUID=UUID,owner=identity(owner.pid),child=child,
 owner_start=receipt(H/'OWNER_START.json'),launch=receipt(H/'LAUNCH.json'),decision=receipt(H/'ROOT_DECISION.json'),
 spec=receipt(H/'SPEC.json'),source_manifest=receipt(source/'MANIFEST.json'),full_panel=launch['full_panel'],
 limits=limits,external_owner_seconds=43320,quality_scores_read=False,CAL_TEST_access=False)
write(H/'INITIAL_PROGRESS.json',progress)
names=['ROOT_DECISION.json','SPEC.json','LAUNCH.json','OWNER_START.json','INITIAL_PROGRESS.json']
files=[]
for name in names:
 p=H/name;b=p.read_bytes()
 files.append(dict(name=name,**receipt(p),base64=base64.b64encode(b).decode()))
print(json.dumps(dict(status='RUNNING',progress=progress,files=files,execution_directory=str(H))),flush=True)
