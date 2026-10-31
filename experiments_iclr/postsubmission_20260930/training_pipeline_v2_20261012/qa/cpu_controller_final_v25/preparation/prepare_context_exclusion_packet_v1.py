"""Freeze/qualify context port, prepare TRAIN targets, compile disabled packet.

Every allocation request checks hostname and sole GPU before project access.
Numerical qualification uses synthetic fixtures only. Target custody opens
GRAPH/TRAIN only. No scientific fit, score, GO, remote Git or77 access occurs.
"""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import zlib

PACKAGE = Path(__file__).resolve().parents[1]
PHASE = PACKAGE.parent
NAME = "massive_context_exclusion_packet_20261012_v1"
LOCAL = PHASE / NAME
sys.path.insert(0, str(PACKAGE))
from adapters.context_exclusion import SOURCE_ARMS, SOURCE_FILES
from compile_native_matrix import RUNTIME_FILES

PREFIX = ["ssh", "-tt", "-p", "2222", "-i", "/Users/alex/.ssh/mlspace__private_key_anogena.txt",
          "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
          "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
          "-o", "StrictHostKeyChecking=yes", "-o", "UpdateHostKeys=no",
          "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"]
GUARD = r'''
import base64,hashlib,json,os,socket,subprocess,sys,time,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0', 'Excluded route; project unopened'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac'], 'Excluded GPU route; project unopened'
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'massive_context_exclusion_packet_20261012_v1'
runtime=root/'runtime'
python=str(repo/'.venv/bin/python')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(), 'Preserve existing receipt: '+str(path)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
def bound(path):
    raw=path.read_bytes()
    return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def checked(label,command,timeout,cwd=None,environment=None):
    tick=time.monotonic()
    result=subprocess.run(command,cwd=cwd or phase,env=environment or env,capture_output=True,text=True,timeout=timeout)
    value={'schema':'context-preparation-operation-v1','UTC':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
           'operation':label,'exit_code':result.returncode,'wall_seconds':time.monotonic()-tick,
           'stdout':result.stdout,'stderr':result.stderr,'scientific_fits':0,'predictive_scores':0,'GO_created':False}
    write_json(root/'receipts'/(label+'.json'),value)
    assert result.returncode==0, json.dumps(value)
    return value
'''


def request(label, body, arguments=(), *, timeout=60):
    command = PREFIX + [shlex.join(["/usr/bin/python3", "-I", "-S", "-B", "-c", GUARD + body, *arguments])]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    value = {"UTC": datetime.now(timezone.utc).isoformat(), "exit_code": result.returncode,
             "stdout": result.stdout, "stderr": result.stderr}
    destination = LOCAL / "transport" / (label + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    destination.write_text(json.dumps(value, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError("Allocation operation failed; preserved " + str(destination))
    print(json.dumps({"operation": label, "exit_code": 0}), flush=True)
    return result.stdout


def main():
    assert not LOCAL.exists(), "Fresh local/remote packet required; preserve previous attempts"
    LOCAL.mkdir()
    files = []
    def add(path, relative):
        raw = path.read_bytes()
        files.append({"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                      "data": base64.b64encode(raw).decode()})
        target = LOCAL / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    runtime = list(RUNTIME_FILES) + ["adapters/context_exclusion.py", "compile_context_exclusion_matrix.py",
                                    "compile_native_matrix.py", "prepare_node_data.py"]
    for name in runtime:
        add(PACKAGE / name, "runtime/" + name)
    add(PACKAGE / "tests/test_context_exclusion.py", "runtime/tests/test_context_exclusion.py")
    add(PACKAGE / "compile_master_plan.py", "verification/compile_master_plan.py")
    bindings = {}
    for key, (relative, expected) in SOURCE_FILES.items():
        source = PHASE / relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
        add(source, "sources/" + relative)
        row = files[-1]
        bindings[key] = {"path": NAME + "/" + row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
    for version in (19, 20):
        for name in ("TRANSPORT.json", "SOURCE_INVENTORY.json"):
            add(PACKAGE / ("qa/cpu_context_exclusion_v" + str(version)) / name,
                "receipts/historical_qa" + str(version) + "_" + name)
    resources = {"schema": "context-exclusion-provisional-resources-v1", "resource_scope": "provisional_unmeasured",
                 "CUDA_qualified": False, "worker_capacities_cuda_bytes": {"allocation/gpu0": 78 * 1024 ** 3},
                 "capacity_basis": "78GiB allocation scheduling allowance; guard confirms hostname and sole UUID, not measured admission",
                 "scientific_launch_enabled": False, "human_GO_required": True, "profiles": {}}
    for arm, (architecture, members, mode) in SOURCE_ARMS.items():
        resources["profiles"]["wikics/context-exclusion-" + arm] = {
            "reservation_mib": 32768, "wall_seconds": 250000000,
            "estimated_seconds_per_update": 5.0 if members == 4 else 2.0,
            "estimated_export_seconds": 120, "estimated_scalar_cal_seconds": 60, "estimated_evaluation_seconds": 60,
            "evidence": "Explicit unmeasured scheduling placeholder. CPU semantic QA does not qualify CUDA SAGE/context targets, checkpoint serialization, endpoints or concurrent admission; root resource QA required.",
            "resource_scope": "provisional_unmeasured", "CUDA_qualified": False}
    for name, value in (("SOURCE_BINDINGS.json", bindings), ("RESOURCES_PROVISIONAL.json", resources)):
        path = LOCAL / name
        path.write_text(json.dumps(value, indent=2) + "\n")
        raw = path.read_bytes()
        files.append({"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                      "data": base64.b64encode(raw).decode()})
    inventory = {"schema": "context-exclusion-final-freeze-v1", "project_relative_root": NAME,
                 "files": [{k: v for k, v in row.items() if k != "data"} for row in files],
                 "scientific_fits": 0, "GO_created": False}
    (LOCAL / "SOURCE_INVENTORY.json").write_text(json.dumps(inventory, indent=2) + "\n")
    payload = base64.b64encode(zlib.compress(json.dumps({"files": files, "inventory": inventory}).encode())).decode()
    transfer = r'''
staging=phase/'training_pipeline_v2_20261012/preparation/context_exclusion_packet_v1_transport'
offset=int(sys.argv[1])
if offset==0:
    assert not root.exists() and not staging.exists()
    staging.mkdir(parents=True)
else:assert (staging/'payload.b64').stat().st_size==offset
with (staging/'payload.b64').open('ab') as stream:stream.write(sys.argv[2].encode())
print((staging/'payload.b64').stat().st_size)
'''
    for offset in range(0, len(payload), 24000):
        request("TRANSFER_" + str(offset), transfer, [str(offset), payload[offset:offset + 24000]])
    request("FREEZE", r'''
payload=(phase/'training_pipeline_v2_20261012/preparation/context_exclusion_packet_v1_transport/payload.b64').read_bytes()
assert hashlib.sha256(payload).hexdigest()==sys.argv[1]
value=json.loads(zlib.decompress(base64.b64decode(payload)))
assert not root.exists();root.mkdir()
for row in value['files']:
    rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
    raw=base64.b64decode(row['data'])
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
write_json(root/'SOURCE_INVENTORY.json',value['inventory'])
print(json.dumps({'frozen_files':len(value['files']),'scientific_fits':0,'GO':False}))
''', [hashlib.sha256(payload.encode()).hexdigest()])
    request("FINAL_CPU_QA", r'''
qaenv=dict(env,GNNM_REFERENCE_PROJECT=str(root/'sources'))
result=checked('FINAL_CPU_QA',[python,'-B','-m','unittest','discover','-s','tests','-p','test_context_exclusion.py','-v'],120,cwd=runtime,environment=qaenv)
assert 'Ran 7 tests' in result['stderr'] and '\nOK\n' in result['stderr'] and 'skipped=' not in result['stderr']
receipt={'schema':'context-exclusion-final-cpu-fixture-v1','passed':True,'tests':7,'skips':0,
         'engineering_only':True,'synthetic_graphs_only':True,'scientific_fits':0,'predictive_scores':0,'GO_created':False,
         'runtime_inventory':bound(root/'SOURCE_INVENTORY.json'),'operation':bound(root/'receipts/FINAL_CPU_QA.json'),
         'source_bindings':bound(root/'SOURCE_BINDINGS.json'),'wall_seconds':result['wall_seconds']}
write_json(root/'receipts/FINAL_CPU_QA_RESULT.json',receipt)
print(json.dumps(receipt))
''', timeout=180)
    request("PREPARE_GRAPH_TRAIN_TARGETS", r'''
checked('PREPARE_GRAPH_TRAIN_TARGETS',[python,'-B',str(runtime/'compile_context_exclusion_matrix.py'),
    '--project-root',str(phase),'--custody','training_pipeline_v2_20261012/prepared_data/wikics/CUSTODY.json',
    '--source-bindings',str(root/'SOURCE_BINDINGS.json'),'--prepare-targets','--output',str(root/'targets')],900)
print(json.dumps({'target_splits':len(json.loads((root/'targets/CUSTODY.json').read_text())['splits']),
                  'scientific_fits':0,'GO':False}))
''', timeout=960)
    request("COMPILE_DISABLED_PACKET", r'''
checked('COMPILE_DISABLED_PACKET',[python,'-B',str(runtime/'compile_context_exclusion_matrix.py'),
    '--project-root',str(phase),'--custody','training_pipeline_v2_20261012/prepared_data/wikics/CUSTODY.json',
    '--source-bindings',str(root/'SOURCE_BINDINGS.json'),'--targets-custody',str(root/'targets/CUSTODY.json'),
    '--runtime-package',str(runtime),'--resources',str(root/'RESOURCES_PROVISIONAL.json'),
    '--fixture-receipts',str(root/'receipts/FINAL_CPU_QA_RESULT.json'),str(root/'receipts/FINAL_CPU_QA.json'),
    str(root/'receipts/historical_qa19_TRANSPORT.json'),str(root/'receipts/historical_qa20_TRANSPORT.json'),
    '--output',str(root/'matrix')],180)
print(json.dumps(json.loads((root/'matrix/SUMMARY.json').read_text())['counts']))
''', timeout=240)
    stdout = request("VERIFY_AND_HANDOFF", r'''
import gzip,importlib.util
sys.path.insert(0,str(runtime))
spec=importlib.util.spec_from_file_location('context_master_ingest',root/'verification/compile_master_plan.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
entry={'id':'context_positive_exclusion_wikics','plan':bound(root/'matrix/PLAN.json'),'summary':bound(root/'matrix/SUMMARY.json')}
write_json(root/'MASTER_INPUT_ENTRY.json',entry)
packet=module.load_packet(phase,entry)
study=packet['studies'][0]
plan=packet['plan'];summary=packet['summary'];counts=summary['counts']
assert counts['atomic_fit_groups']==1242 and counts['all_run_records']==3411
assert counts['optimizer_starts_by_stage']=={'search':162,'comparative':1080}
assert counts['full_budget_optimizer_updates']==6210000 and counts['comparative_banks']==720
assert counts['genuine_I4_banks']==180 and counts['reused_I4_body0']==180
assert plan['scientific_launch_enabled'] is False and plan['human_GO_required'] is True
assert plan['worker_capacities_cuda_bytes']=={'allocation/gpu0':78*1024**3}
assert {g['gpu_assignment'] for g in plan['groups']}=={'allocation/gpu0'}
assert all(row['allowed_workers']==['allocation/gpu0'] for row in plan['groups'])
assert len(json.loads((root/'SOURCE_BINDINGS.json').read_text()))==7
assert all(not f.name.startswith('GO') for f in root.rglob('*'))
targets=json.loads((root/'targets/CUSTODY.json').read_text())
assert len(targets['splits'])==20 and targets['SELECT_CAL_TEST_tensors_opened'] is False
for row in targets['splits']:
    assert bound(phase/row['target']['path'])==row['target']
    assert all(n>0 for n in row['preparation']['changed_incidence_entries_per_context'])
verification={'schema':'context-exclusion-master-ingest-verification-v1','verified':True,
    'plan_hash':packet['plan_hash'],'counts':packet['counts'],'reference_matches':packet['references'],
    'model_updates_declared':6210000,'scalar_CAL_updates_declared':360000,
    'source_method_scopes':7,'exact_source_files':7,'target_preparations':20,
    'actual_scientific_fits':0,'actual_predictive_scores':0,'GO_created':False,
    'SELECT_CAL_TEST_target_tensors_decoded':False,'target_pack_bytes_rehashed':True,
    'worker':'allocation/gpu0','resources':'provisional_unmeasured',
    'remaining_gates':['CUDA_resource_and_serializer_endpoint_concurrent_qualification','combined_root_master_and_global_closures','explicit_human_GO']}
write_json(root/'MASTER_INGEST_VERIFICATION.json',verification)
handoff={'schema':'context-exclusion-packet-handoff-v1','status':'PREPARED_DISABLED_RESOURCE_QA_MASTER_AND_GO_PENDING',
    'entry':entry,'plan_hash':packet['plan_hash'],'counts':counts,'targets':bound(root/'targets/CUSTODY.json'),
    'final_cpu_qa':bound(root/'receipts/FINAL_CPU_QA_RESULT.json'),'ingest_verification':bound(root/'MASTER_INGEST_VERIFICATION.json'),
    'scientific_launch_enabled':False,'human_GO_required':True,'scientific_fits':0,'GO_created':False,
    'resource_scope':'provisional_unmeasured','full_249_coverage':False,'source_canonical_methods':7}
write_json(root/'SUMMARY.json',handoff)
mirrors=['SOURCE_INVENTORY.json','MASTER_INPUT_ENTRY.json','MASTER_INGEST_VERIFICATION.json','SUMMARY.json',
    'matrix/PLAN.json','matrix/SUMMARY.json','targets/CUSTODY.json','receipts/FINAL_CPU_QA.json',
    'receipts/FINAL_CPU_QA_RESULT.json','receipts/PREPARE_GRAPH_TRAIN_TARGETS.json','receipts/COMPILE_DISABLED_PACKET.json']
mirrors += [str((phase/row['path']).relative_to(root)) for row in plan['studies']]
value={relative:base64.b64encode((root/relative).read_bytes()).decode() for relative in mirrors}
print(base64.b64encode(zlib.compress(json.dumps(value).encode())).decode())
''', timeout=180)
    mirrored = json.loads(zlib.decompress(base64.b64decode(stdout.strip())))
    for relative, encoded in mirrored.items():
        target = LOCAL / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = base64.b64decode(encoded)
        if target.exists():
            assert json.loads(target.read_bytes()) == json.loads(raw)
        else:
            target.write_bytes(raw)
    print(json.dumps({"packet": NAME, "status": "PREPARED_DISABLED", "scientific_fits": 0,
                      "GO": False, "local_metadata_files": len(mirrored)}), flush=True)


if __name__ == "__main__":
    main()
