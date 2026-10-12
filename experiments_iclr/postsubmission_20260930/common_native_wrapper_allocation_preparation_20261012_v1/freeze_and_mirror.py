"""Guarded own-source-only final receipts; no graph/target/checkpoint access."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zlib

from stage_and_cpu_qa import DESTINATION, GUARD, HERE, PHASE_LOCAL, PREFIX, PRIOR

REMOTE = GUARD + r'''
import gzip
packet=phase/sys.argv[1];lineage=packet/'lineage/sealed_local_v1'
inventory=json.loads((lineage/'INVENTORY.json').read_text())
assert inventory['file_count']==48
def receipt(path):
 raw=path.read_bytes();return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
for row in inventory['files']:
 path=lineage/row['path'];observed=receipt(path)
 assert observed['bytes']==row['bytes'] and observed['sha256']==row['sha256']
code_checks=[]
for row in inventory['files']:
 if row['path'].endswith('.py'):
  observed=receipt(packet/row['path'])
  assert observed['bytes']==row['bytes'] and observed['sha256']==row['sha256']
  code_checks.append({'path':observed['path'],'sha256':observed['sha256'],'unchanged_from_sealed_v1':True})
custody=[]
for dataset in ('wikics','amazon-photo'):
 copied=packet/'metadata'/dataset/'CUSTODY.json'
 declared=json.loads((packet/'metadata'/dataset/'RECEIPT.json').read_text())
 actual=phase/declared['path']
 # Metadata only: the original current CUSTODY bytes, never its tensor files.
 original=receipt(actual)
 assert original==declared and copied.read_bytes()==actual.read_bytes()
 value=json.loads(copied.read_text())
 custody.append({'dataset':dataset,'allocation_custody':original,'successor_copy':receipt(copied),
  'current_original_metadata_byte_identical':True,'split_ids':[r['split_id'] for r in value['splits']],
  'provider_graph_metadata':value['model_graph'],'tensor_payloads_read':False})
qa=json.loads((packet/'VALIDATION_SYNTHETIC.json').read_text())
assert qa['status']=='passed' and len(qa['native_cases'])==36 and qa['synthetic_train_primitive_calls']==72
ingest=json.loads((packet/'ACTUAL_CONTROLLER_V5_INGEST.json').read_text())
plan=json.loads((packet/'matrix/PLAN.json').read_text())
assert plan['scientific_launch_enabled'] is False and plan['scientific_roster_approved'] is False
assert plan['resource_admission_status']=='pending_root_qualification' and plan['source_commit_status']=='pending_root_owned'
for row in plan['studies']:
 study=json.loads(gzip.decompress((phase/row['path']).read_bytes()))
 assert study['source']['source_commit_status']=='pending_root_owned' and study['source']['git_commit']=='PENDING_ROOT_SOURCE_COMMIT_BINDING'
 assert study['source']['payload_bindings_status'].startswith('original custody declarations; payloads unread/unverified locally;')
handoff=json.loads((packet/'PREPARATION_HANDOFF.json').read_text())
handoff['schema']='common-native-wrapper-allocation-handoff-v2'
handoff['numerical_QA']={'status':'PASSED_DEFINED_ALLOCATION_CPU_SYNTHETIC_QA','script':'runtime/tests/check_new_port_synthetic.py',
 'scope':'fresh nine-node CPU fixtures only; no benchmark fitting/scoring, real provider payloads or old audit rerun',
 'native_cases':36,'I4_independence_cases':12,'toy_optimizer_steps':72,'benchmark_optimizer_steps':0,'scientific_fit_count':0,
 'actual_runtime_versions':{name:qa[name] for name in ('torch_version','pyg_version','python')},'runtime_pin_claimed':False}
handoff['actual_controller_load']={'loader':'frozen_controller_v5','status':'passed','plan_hash':ingest['plan_hash'],
 'counts':ingest['counts']['totals'],'receipt':'ACTUAL_CONTROLLER_V5_INGEST.json'}
handoff['root_owned_remaining']=[item for item in handoff['root_owned_remaining'] if not item.startswith('allocation staging') and not item.startswith('run focused new-port synthetic CPU QA')]
handoff['root_owned_remaining'].insert(0,'verify scientific execution environment and runtime/source compatibility; this CPU toy receipt is not a runtime pin or CUDA/resource qualification')
handoff['successor_plan']='matrix/PLAN.json'
handoff['predecessor_plan']='lineage/sealed_local_v1/prepared/PLAN.json; retained lineage, not the successor entry'
handoff['no_execution']['remote_operations']=True
handoff['no_execution']['Git_operations']=0
handoff['no_execution']['synthetic_toy_optimizer_steps']=72
(packet/'PREPARATION_HANDOFF.json').write_text(json.dumps(handoff,indent=2,sort_keys=True)+'\n')
final={'schema':'common-native-wrapper-allocation-final-source-validation-v2','status':'CPU_AND_ACTUAL_CONTROLLER_SOURCE_METADATA_QA_PASSED',
 'predecessor_exact_assets_preserved':48,'code_checks':code_checks,'current_custody_metadata':custody,
 'actual_controller_file':ingest['actual_controller_file'],'actual_loader':'frozen_controller_v5','plan_hash':ingest['plan_hash'],
 'actual_controller_counts':ingest['counts']['totals'],'admission_rejections':ingest['source_guard_rejections'],
 'source_learning_code_changed':False,'metadata_path_binding_only':True,'source_operator_and_first_factor_initialization_unchanged':True,
 'source_commit_status':'pending_root_owned','scientific_roster_approved':False,'resource_admission_status':'pending_root_qualification',
 'runtime_pin_claimed':False,'CUDA_resource_qualification':False,'actual_numeric_runtime_observed':{name:qa[name] for name in ('torch_version','pyg_version','python')},
 'engineering_counts':{'toy_optimizer_steps':72,'toy_source_backward_calls':108,'toy_native_TRAIN_forward_calls':216,
  'standalone_route_block_gradient_checks':2,'native_cases':36,'I4_independence_cases':12},
 'benchmark_counts':{'fit_starts':0,'optimizer_steps':0,'predictive_scores':0},'real_tensor_payload_reads':0,'checkpoint_payload_reads':0,
 'GO_created':False,'scientific_launch_enabled':False,'remote_v1_written':False,'Git_operations':0}
(packet/'FINAL_VALIDATION.json').write_text(json.dumps(final,indent=2,sort_keys=True,allow_nan=False)+'\n')
text="""Common native wrapper allocation successor v2

Exact 48 sealed local v1 assets are retained in lineage/sealed_local_v1.
Remote v1 was absent and remains untouched. No v1 upstream was invented.
All 26 source/runtime/test Python files retain their sealed v1 bytes.
Provider CUSTODY metadata matches both current original allocation receipts.
No real provider tensors, labels or checkpoints were opened or rehashed.

Defined new-port CPU synthetic QA passed in allocation R/.venv/bin/python,
CUDA_VISIBLE_DEVICES empty, with36 native cases and12 I4 independence cases.
Engineering-only counts:72 toy optimizer steps,108 toy source backward calls,
216 toy native TRAIN forward calls,2 standalone route-block gradient checks.
Benchmark fit starts/optimizer steps/scores:0/0/0. CUDA compute:0.

Actual frozen_controller_v5 load_packet passed on matrix/PLAN.json,
with6 studies,4428 optimizer/body/fit records,22140000 nominal model updates,
2160 served banks/CAL500/evaluation jobs,540 genuine I4,zero member0 reuse.
All required reference cohorts match. Disabled/pending roster-resource/source
admission rejection cases pass. Master entry is MASTER_INPUT_ENTRY.json.

Observed CPU-test runtime: Torch2.1.2+cu118 and PyG2.7.0. This is an observation,
not a new runtime pin, CUDA qualification, resource admission or pushed source.
Source commit, runtime/scientific resource admission, master integration,
scientific roster approval and explicit user GO remain pending/root-owned.
All source/operator/first-factor seeds and literal train primitives unchanged.

matrix contains the new successor plan. The inherited prepared directory and
its exact lineage copy are predecessor metadata, not a successor master entry.
No scientific training, GO creation or Git operation occurred.
"""
(packet/'REPORT.txt').write_text(text+'\nSuccessor plan fingerprint: '+ingest['plan_hash']+'\n')
rows=[]
for path in sorted(packet.rglob('*')):
 assert not path.is_symlink()
 if path.is_file() and path!=packet/'INVENTORY.json':
  observed=receipt(path);rows.append({**observed,'path':str(path.relative_to(packet))})
final_inventory={'schema':'common-native-wrapper-allocation-successor-inventory-v2','status':'FROZEN_DISABLED_CPU_QA_AND_ACTUAL_CONTROLLER_LOADED',
 'files':rows,'file_count':len(rows),'bytes':sum(row['bytes'] for row in rows),'predecessor_exact_assets':48,
 'plan_hash':ingest['plan_hash'],'source_code_changed':False,'scientific_launch_enabled':False,'source_commit_status':'pending_root_owned',
 'scientific_roster_approved':False,'resource_admission_status':'pending_root_qualification','benchmark_fit_starts':0,'toy_optimizer_steps':72,
 'real_tensor_payload_reads':0,'GO_created':False,'Git_operations':0}
(packet/'INVENTORY.json').write_text(json.dumps(final_inventory,indent=2,sort_keys=True)+'\n')
for path in packet.rglob('*'):
 if path.is_file():path.chmod(0o444)
# Fetch only this own package's source/metadata assets; there are no tensors.
files=[]
for path in sorted(packet.rglob('*')):
 if path.is_file():
  assert path.suffix not in ('.pt','.pth','.npy','.npz','.pkl')
  raw=path.read_bytes();files.append({'path':str(path.relative_to(packet)),'bytes':len(raw),
   'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
snapshot={'destination':packet.name,'files':files,'inventory':receipt(packet/'INVENTORY.json'),'final_validation':receipt(packet/'FINAL_VALIDATION.json'),
 'plan':receipt(packet/'matrix/PLAN.json'),'summary':receipt(packet/'matrix/SUMMARY.json'),'master_entry':receipt(packet/'MASTER_INPUT_ENTRY.json')}
print(base64.b64encode(zlib.compress(json.dumps(snapshot,separators=(',',':')).encode())).decode())
'''


def main():
    command = PREFIX + [shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE, DESTINATION])]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=90)
    lines = [line for line in completed.stdout.splitlines() if line and not line.startswith('{')]
    record = {'operation': 'verified own-source final metadata/current-CUSTODY check, freeze and source-only mirror',
        'exit_code': completed.returncode, 'stdout_bytes': len(completed.stdout.encode()),
        'stdout_sha256': hashlib.sha256(completed.stdout.encode()).hexdigest(), 'stderr': completed.stderr,
        'route_guard_before_project_operations': True, 'benchmark_fits': 0, 'real_tensor_payload_reads': 0, 'GO': False}
    (HERE / 'FREEZE_AND_MIRROR_TRANSPORT.json').write_text(json.dumps(record, indent=2) + '\n')
    if completed.returncode:
        (HERE / 'FREEZE_AND_MIRROR_FAILURE.txt').write_text(completed.stdout + '\n' + completed.stderr)
        raise RuntimeError('Final own-source mirror failed; exact failure retained')
    snapshot = json.loads(zlib.decompress(base64.b64decode(lines[-1])))
    destination = PHASE_LOCAL / snapshot['destination']
    destination.mkdir(exist_ok=False)
    for row in snapshot['files']:
        rel = Path(row['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        raw = base64.b64decode(row['data'])
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    old = json.loads((PRIOR / 'INVENTORY.json').read_text())
    for row in old['files']:
        raw = (PRIOR / row['path']).read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        assert raw == (destination / 'lineage/sealed_local_v1' / row['path']).read_bytes()
    inventory = json.loads((destination / 'INVENTORY.json').read_text())
    for row in inventory['files']:
        raw = (destination / row['path']).read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    summary = {key: snapshot[key] for key in ('destination', 'inventory', 'final_validation', 'plan', 'summary', 'master_entry')}
    summary.update(local_v1_unchanged=True, mirrored_files=len(snapshot['files']), all_receipts_verified=True,
        benchmark_fits=0, toy_optimizer_steps=72, GO=False)
    (HERE / 'FINAL_MIRROR_RECEIPT.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
