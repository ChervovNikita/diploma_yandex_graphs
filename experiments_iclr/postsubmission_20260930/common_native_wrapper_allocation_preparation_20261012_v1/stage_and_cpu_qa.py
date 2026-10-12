"""Own guarded allocation source staging/new-port CPU QA, never science/GO/Git."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import zlib

HERE = Path(__file__).resolve().parent
PHASE_LOCAL = HERE.parent
PRIOR = PHASE_LOCAL / 'common_native_wrapper_packet_20261012_v1'
DESTINATION = 'common_native_wrapper_packet_20261012_v2'
TRANSPORT = 'common_native_wrapper_allocation_source_transport_20261012_v1'
PREFIX = ['ssh', '-tt', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
    '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
    '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
    '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no',
    'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
GUARD = r'''
import base64,hashlib,json,os,socket,subprocess,sys,time,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()
assert uuids==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
'''
APPEND = GUARD + r'''
transport=phase/sys.argv[1]
target=transport/'payload.b64';offset=int(sys.argv[2])
if offset==0:
 assert not transport.exists();transport.mkdir()
else:assert target.stat().st_size==offset
with target.open('ab') as stream:stream.write(sys.argv[3].encode())
print(json.dumps({'transport_bytes':target.stat().st_size,'hostname':socket.gethostname(),'gpu_uuids':uuids,
 'scientific_fits':0,'real_tensor_reads':0,'CUDA_compute':0,'GO':False}))
'''
STAGE = GUARD + r'''
encoded=(phase/sys.argv[1]/'payload.b64').read_text()
assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))
packet=phase/payload['destination'];assert not packet.exists();packet.mkdir()
lineage=packet/'lineage/sealed_local_v1';lineage.mkdir(parents=True)
def atomic(path,value):
 temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)
def receipt(path):
 raw=path.read_bytes();return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
assert len(payload['assets'])==48
for row in payload['assets']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 frozen=lineage/rel;frozen.parent.mkdir(parents=True,exist_ok=True);frozen.write_bytes(raw);frozen.chmod(0o444)
 # Copy source/metadata into the fresh successor. Numerical source/code
 # stays byte-identical; successor path receipts are separately rebound.
 target=packet/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
 if rel.suffix=='.py':target.chmod(0o444)
raw=base64.b64decode(payload['inventory_proof']['data'])
assert hashlib.sha256(raw).hexdigest()==payload['inventory_proof']['sha256']
(lineage/'INVENTORY.json').write_bytes(raw);(lineage/'INVENTORY.json').chmod(0o444)
prior_bindings=json.loads((packet/'SOURCE_BINDINGS.json').read_text())
bindings={key:{**row,'path':row['path'].replace(payload['prior_name']+'/',payload['destination']+'/',1)} for key,row in prior_bindings.items()}
for key,row in bindings.items():assert receipt(phase/row['path'])==row
atomic(packet/'SOURCE_BINDINGS.json',bindings)
# The original v1 plan/source paths are absent on this allocation. Preserve
# all exact v1 bytes in lineage, then compile a fresh disabled successor in
# a distinct directory instead of claiming a nonexistent v1 upstream.
binding_changes={'schema':'common-wrapper-successor-path-binding-v1','prior_local_packet':payload['prior_name'],
 'local_inventory_sha256':payload['inventory_proof']['sha256'],'staged_exact_assets':48,
 'source_code_changed':False,'learned_operator_changed':False,'provider_role_changes':False,
 'identified_integration_defect':'literal v1 paths are absent on allocation; bind successor source/runtime/spec paths to fresh v2',
 'source_binding_changes':{key:{'prior':prior_bindings[key],'successor':bindings[key]} for key in bindings},
 'immutable_lineage':str(lineage.relative_to(phase)),'remote_v1_preexisted':(phase/payload['prior_name']).exists(),
 'remote_v1_written':False,'benchmark_fits':0,'real_tensor_reads':0,'GO':False}
atomic(packet/'SUCCESSOR_LINEAGE.json',binding_changes)
host={'hostname':socket.gethostname(),'gpu_uuids':uuids,'verified_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 'literal_destination':'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','port':2222,'CUDA_compute':0,'scientific_fits':0}
atomic(packet/'HOST_BINDING.json',host)
print(json.dumps({'destination':payload['destination'],'exact_assets_staged':48,'source_code_changed':False,
 'lineage_inventory':receipt(lineage/'INVENTORY.json'),'source_bindings':receipt(packet/'SOURCE_BINDINGS.json'),
 'successor_lineage':receipt(packet/'SUCCESSOR_LINEAGE.json'),'host':host,'real_tensor_reads':0,'scientific_fits':0,'GO':False}))
'''
CPU_QA = GUARD + r'''
packet=phase/sys.argv[1];runtime=packet/'runtime';python=repo/'.venv/bin/python'
assert packet.is_dir() and python.is_file()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
tick=time.monotonic()
q=subprocess.run([str(python),'-B',str(runtime/'tests/check_new_port_synthetic.py')],cwd=runtime,env=env,capture_output=True,text=True,timeout=180)
record={'schema':'common-wrapper-allocation-CPU-QA-transport-v1','exit_code':q.returncode,'stdout':q.stdout,'stderr':q.stderr,
 'wall_seconds':time.monotonic()-tick,'python':str(python),'CUDA_VISIBLE_DEVICES':'','defined_test_only':True,
 'synthetic_toy_optimizer_steps':72 if q.returncode==0 else None,'toy_steps_on_failure':'unknown partial count; original stdout/stderr retained' if q.returncode else None,
 'benchmark_optimizer_steps':0,'benchmark_fits':0,'real_tensor_reads':0,'CUDA_compute':0,'GO':False}
(packet/'ALLOCATION_CPU_QA.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps(record))
if q.returncode:raise SystemExit(q.returncode)
result=json.loads((packet/'VALIDATION_SYNTHETIC.json').read_text())
assert result['status']=='passed' and result['scientific_fits']==0 and result['real_tensor_reads']==0 and result['CUDA_operations']==0
'''
COMPILE_AND_LOAD = GUARD + r'''
import importlib,importlib.util
packet=phase/sys.argv[1];runtime=packet/'runtime';python=repo/'.venv/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
q=subprocess.run([str(python),'-B',str(runtime/'compile_common_native_wrapper_matrix.py'),'--project-root',str(phase),
 '--packet',str(packet),'--output',str(packet/'matrix')],cwd=runtime,env=env,capture_output=True,text=True,timeout=60)
record={'exit_code':q.returncode,'stdout':q.stdout,'stderr':q.stderr,'numerical_imports':0,'real_tensor_reads':0,'scientific_fits':0,'GO':False}
(packet/'ALLOCATION_COMPILATION.json').write_text(json.dumps(record,indent=2)+'\n')
assert q.returncode==0,record
controller=phase/'master_controller_source_20261012_v5/runtime'
sys.path.insert(0,str(controller))
master=importlib.import_module('compile_master_plan')
assert Path(master.__file__).resolve()==(controller/'compile_master_plan.py').resolve()
def receipt(path):
 raw=path.read_bytes();return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
entry={'id':'common_native_wrapper_source_v2','plan':receipt(packet/'matrix/PLAN.json'),'summary':receipt(packet/'matrix/SUMMARY.json'),
 'plan_hash':json.loads((packet/'matrix/SUMMARY.json').read_text())['plan_hash'],'coverage':[receipt(packet/'SUCCESSOR_LINEAGE.json')]}
loaded=master.load_packet(phase,entry)
counts=loaded['counts']['totals']
assert len(loaded['studies'])==6
assert counts['atomic_fit_run_groups']==counts['paid_optimizer_starts']==counts['physical_slow_body_instances']==4428
assert counts['known_registered_optimizer_step_calls']==22140000
assert counts['served_banks']==2160 and counts['genuine_separately_selected_I4_banks']==540 and counts['reused_exact_member0_fit_references']==0
assert loaded['declared_source_data_bindings_rehashed'] is False
assert all(row['matched'] for rows in loaded['references'].values() for row in rows)
guard_spec=importlib.util.spec_from_file_location('common_wrapper_admission_guard',runtime/'preparation_guard.py')
guard=importlib.util.module_from_spec(guard_spec);guard_spec.loader.exec_module(guard)
rejections=[]
for name,plan,studies in [('disabled',loaded['plan'],{}),('roster_pending',{**loaded['plan'],'scientific_launch_enabled':True},{}),
 ('source_commit_pending',{**loaded['plan'],'scientific_launch_enabled':True,'scientific_roster_approved':True,'resource_admission_status':'qualified'},
 {study['study_id']:study for study in loaded['studies']})]:
 try:guard.require_admitted_plan(plan,studies)
 except ValueError as error:rejections.append({'case':name,'reason':str(error)})
 else:raise AssertionError('Invalid scientific admission accepted')
numeric=json.loads((packet/'VALIDATION_SYNTHETIC.json').read_text())
report={'schema':'common-wrapper-actual-controller-v5-ingest-v1','actual_loader':'frozen_controller_v5',
 'actual_controller_file':receipt(controller/'compile_master_plan.py'),'plan_hash':loaded['plan_hash'],'entry':entry,
 'studies':len(loaded['studies']),'counts':loaded['counts'],'references':loaded['references'],
 'source_guard_rejections':rejections,'optimizer_source_CPU_QA_status':numeric['status'],'actual_runtime_versions':
 {name:numeric[name] for name in ('torch_version','pyg_version','python')},'runtime_pin_claimed':False,
 'controller_metadata_only':True,'declared_source_data_bindings_rehashed':False,'benchmark_fits':0,
 'benchmark_optimizer_steps':0,'synthetic_toy_optimizer_steps':72,'real_tensor_reads':0,'CUDA_compute':0,'GO':False}
(packet/'MASTER_INPUT_ENTRY.json').write_text(json.dumps(entry,indent=2)+'\n')
(packet/'ACTUAL_CONTROLLER_V5_INGEST.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report))
'''


def invoke(code, args, name, timeout=60):
    command = PREFIX + [shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', code, *args])]
    tick = time.monotonic()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        value = {'exit_code': completed.returncode, 'stdout': completed.stdout, 'stderr': completed.stderr}
    except subprocess.TimeoutExpired as error:
        value = {'exit_code': None, 'stdout': (error.stdout or b'').decode() if isinstance(error.stdout, bytes) else error.stdout or '',
            'stderr': (error.stderr or b'').decode() if isinstance(error.stderr, bytes) else error.stderr or '', 'transport_timeout': True}
    value.update(operation=name, wall_seconds=time.monotonic()-tick, destination=PREFIX[-1], port=2222,
        route_guard_before_project_operations=True, benchmark_fits=0, real_tensor_reads=0, GO=False)
    (HERE / (name + '_TRANSPORT.json')).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    if value['exit_code'] != 0:
        raise RuntimeError('Guarded operation failed; receipt preserved: ' + name)
    lines = [line for line in value['stdout'].splitlines() if line.startswith('{')]
    return json.loads(lines[-1]) if lines else {}


def main():
    inventory = json.loads((PRIOR / 'INVENTORY.json').read_text())
    assert inventory['file_count'] == 48
    assets = []
    for row in inventory['files']:
        raw = (PRIOR / row['path']).read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        assets.append({**row, 'data': base64.b64encode(raw).decode()})
    inventory_raw = (PRIOR / 'INVENTORY.json').read_bytes()
    payload = {'destination': DESTINATION, 'prior_name': PRIOR.name, 'assets': assets,
        'inventory_proof': {'sha256': hashlib.sha256(inventory_raw).hexdigest(), 'data': base64.b64encode(inventory_raw).decode()}}
    encoded = base64.b64encode(zlib.compress(json.dumps(payload, separators=(',', ':')).encode())).decode()
    transport_record = {'encoded_bytes': len(encoded), 'encoded_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
        'asset_count': len(assets), 'asset_bytes': sum(row['bytes'] for row in assets), 'destination': DESTINATION,
        'local_v1_inventory_sha256': payload['inventory_proof']['sha256'], 'local_v1_modified': False}
    (HERE / 'PAYLOAD_RECEIPT.json').write_text(json.dumps(transport_record, indent=2) + '\n')
    for offset in range(0, len(encoded), 50000):
        result = invoke(APPEND, [TRANSPORT, str(offset), encoded[offset:offset+50000]], 'TRANSFER_' + str(offset // 50000))
        print(json.dumps({'transfer_chunk': offset // 50000, 'transport_bytes': result['transport_bytes']}), flush=True)
    result = invoke(STAGE, [TRANSPORT, transport_record['encoded_sha256']], 'STAGE')
    (HERE / 'ALLOCATION_STAGE.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'staged_exact_assets': result['exact_assets_staged'], 'destination': DESTINATION}), flush=True)
    result = invoke(CPU_QA, [DESTINATION], 'CPU_QA', 210)
    (HERE / 'ALLOCATION_CPU_QA.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'CPU_QA_exit_code': result['exit_code'], 'synthetic_toy_optimizer_steps': result['synthetic_toy_optimizer_steps'], 'benchmark_fits': 0}), flush=True)
    result = invoke(COMPILE_AND_LOAD, [DESTINATION], 'COMPILE_CONTROLLER_LOAD', 90)
    (HERE / 'ACTUAL_CONTROLLER_V5_INGEST.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (HERE / 'MASTER_INPUT_ENTRY.json').write_text(json.dumps(result['entry'], indent=2) + '\n')
    print(json.dumps({'actual_loader': result['actual_loader'], 'studies': result['studies'], 'counts': result['counts']['totals'],
        'plan_hash': result['plan_hash'], 'synthetic_toy_optimizer_steps': 72, 'benchmark_fits': 0, 'GO': False}), flush=True)


if __name__ == '__main__':
    main()
