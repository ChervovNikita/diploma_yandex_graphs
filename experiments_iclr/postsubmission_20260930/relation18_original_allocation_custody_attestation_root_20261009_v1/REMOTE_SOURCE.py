
import socket,subprocess,json,hashlib,datetime,importlib.util,os
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[GPU]
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
os.chdir(R)
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(p.relative_to(P)),bytes=p.stat().st_size,sha256=sha(p))
def identity(h):
 pid=h.get('PID',h.get('pid'));assert pid is not None
 try:s=Path('/proc',str(pid),'stat').read_text()
 except FileNotFoundError:return dict(PID=pid,expected_start_ticks=h.get('start_ticks'),present=False)
 f=s[s.rfind(')')+2:].split();ticks=int(f[19]);assert h.get('start_ticks') is None or ticks==h['start_ticks'],'PID reused'
 return dict(PID=pid,start_ticks=ticks,present=True,state=f[0],pgid=int(f[2]),sid=int(f[3]))
E=P/'Wiki24_selected_analysis_after_closure_execution_root_20261007_v1'
CPU=P/'Wiki24_analysis_after_closure_execution_root_20261007_v1'
export=read(CPU/'extraction/METADATA_EXPORT.json');saved=export['gate'];act=P/saved['activation']['path'];cfg=read(act)
source=P/'internal_BE_Wiki24_closed_family_reader_source_20261007_v3/gate.py'
assert sha(source)=='4e08d843f670802d757e4e7ed3256bb5237ae1f7fed88a551733d5bf2b4de901'
spec=importlib.util.spec_from_file_location('_original_custody_gate',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
_,gate=m.preflight(act,saved['activation']['sha256'],cfg['stage'])
assert all(gate[k]==saved[k] for k in ('closure','owner','source_manifest_sha256','reader_manifest_sha256'))
assert [(r['cell'],r['status'],r.get('selected_checkpoint')) for r in gate['cells']]==[(r['cell'],r['status'],r.get('selected_checkpoint')) for r in saved['cells']]
prior_terminal=read(E/'TERMINAL_EVIDENCE.json');observations=[]
for row in prior_terminal['current_observed_handles']:
 h={k:row[k] for k in ('PID','start_ticks') if k in row};observations.append(identity(h))
child_owner=read(E/'CHILD_OWNER.json');child_terminal=read(E/'CHILD_TERMINAL.json')
assert child_terminal['reaped'] is True and child_terminal['exit_code']==0 and not child_terminal['timed_out'] and not child_terminal['cap_exceeded'] and child_terminal['error'] is None and child_terminal['cleanup_error'] is None
assert child_owner['identity']==child_terminal['identity']
child_observation=identity(child_terminal['identity']);observations.append(child_observation)
assert all(not r['present'] for r in observations),'Historical owner/child still present'
pids={r['PID'] for r in observations}
cu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True).splitlines()
owned=[line for line in cu if int(line.split(',')[1].strip()) in pids];assert not owned
checks=[]
for state in gate['cells']:
 if state.get('selected_checkpoint'):
  row=state['selected_checkpoint'];p=P/row['path'];assert bind(p)==row;checks.append(dict(cell=state['cell'],selected_checkpoint=row))
refs=read(P/'graph_relation18_selected_readout_source_20261009_v1/REFERENCE_POINTERS.json')['records']
archives=[]
for row in refs:
 old=row['raw_prediction_archive_inherited_binding'];assert bind(P/old['path'])==old;archives.append(dict(cell_id=row['cell_id'],original_archive=old))
relative='relation18_original_allocation_custody_attestation_root_20261009_v1';target=P/relative;target.mkdir(mode=0o700,exist_ok=False)
(target/'RECHECKED_GATE.json').write_text(json.dumps(gate,indent=2,allow_nan=False)+'\n')
result=dict(schema='relation18-original-route-historical-custody-attestation-v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_observed=True,hostname=socket.gethostname(),repository=str(R),physical_GPU_uuid=GPU,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),original_reader_gate_passed=True,original_gate=bind(target/'RECHECKED_GATE.json'),historical_metadata_export=bind(CPU/'extraction/METADATA_EXPORT.json'),historical_extraction_cost=bind(CPU/'extraction/EXTRACTION_COST.json'),historical_collector_release=bind(E/'COLLECT_RELEASE.json'),historical_terminal_evidence=bind(E/'TERMINAL_EVIDENCE.json'),historical_child_owner=bind(E/'CHILD_OWNER.json'),historical_child_terminal=bind(E/'CHILD_TERMINAL.json'),historical_collection=bind(E/'predictions/compact/COLLECTION.json'),historical_cost=bind(E/'predictions/compact/COST.json'),checkpoint_hashes_reverified=checks,historical_archives=archives,actual_observations=observations,own_CUDA_rows=owned,collector_child_direct_wait_reap=True,watcher_direct_OS_exit_or_wait_observed=False,original_scores_changed=False,raw_payloads_decoded=False,TEST_access=False,new_scientific_fits=0)
(target/'ATTESTATION.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(attestation=bind(target/'ATTESTATION.json'),rechecked_gate=bind(target/'RECHECKED_GATE.json'),collector_owner=child_owner,collector_terminal=child_terminal,archives=archives,observations=len(observations),payloads_decoded=False)))
