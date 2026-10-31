"""Complete old failure and diagnostic text custody; no scientific artifacts."""
from pathlib import Path
from admission_support import HERE, PHASE, REMOTE_PHASE, REMOTE_REPO, TEXT_EXTENSIONS, read, bound, descriptor, require, write

def preflight_remote_lineage():
    """Read-only complete local/remote text inventory before transport or claim."""
    import hashlib
    import json
    import shlex
    import subprocess
    from finite_coordinator import SSH, utc
    require(PHASE != REMOTE_PHASE, 'Inventory preflight belongs to local bootstrap')
    expected = read(HERE/'LINEAGE_BINDINGS.json')['all_bound_text_files']
    code = '''import hashlib,json,pathlib,sys
base=pathlib.Path(sys.argv[1]); roots=json.loads(sys.argv[2]); suffixes=set(json.loads(sys.argv[3])); rows=[]
for root in roots:
 for p in sorted((base/root).rglob('*')):
  if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts:
   if p.is_symlink(): raise ValueError('symlink forbidden')
   rows.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
print(json.dumps(rows))'''
    command = shlex.join([str(REMOTE_REPO/'.venv/bin/python'), '-c', code, str(REMOTE_PHASE),
                          json.dumps(EXPECTED_ROOTS), json.dumps(sorted(TEXT_EXTENSIONS))])
    result = subprocess.run([*SSH, command], capture_output=True, text=True, check=True, timeout=60)
    actual_remote = json.loads(result.stdout)
    actual_local = [{'path':str(REMOTE_PHASE/p.relative_to(PHASE)),
                     'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
        for root in EXPECTED_ROOTS for p in sorted((PHASE/root).rglob('*'))
        if p.is_file() and p.suffix in TEXT_EXTENSIONS and '__pycache__' not in p.parts]
    def differences(rows):
        want = {r['path']:r for r in expected}; got = {r['path']:r for r in rows}
        return {'count':len(rows),'missing':sorted(set(want)-set(got)),
                'extra':[got[k] for k in sorted(set(got)-set(want))],
                'changed':[{'expected':want[k],'actual':got[k]} for k in sorted(set(want)&set(got)) if want[k]!=got[k]]}
    local = differences(actual_local); remote = differences(actual_remote)
    equal = all(not d[k] for d in (local,remote) for k in ('missing','extra','changed'))
    value = {'schema':'graph-init-pre-registration-read-only-lineage-preflight-v2','UTC':utc(),
        'expected_count':len(expected),'local':local,'remote':remote,'passed':equal,
        'transport_performed':False,'registration_claimed':False,'scientific_execution':False,
        'discrepancy_is_recoverable_metadata_before_claim':True,'new_study_required_by_preflight':False}
    name = 'lineage_preflight_v2_'+hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()+'.json'
    write(HERE/name, value)
    require(equal, 'Read-only lineage discrepancy; recover metadata and repeat preflight before claiming this same registration identity: '+name)
    verify_lineage()
    return {'receipt':descriptor(HERE/name),'complete_text_count':len(expected),'passed':True}

EXPECTED_ROOTS = ['graph_init_execution_root_v1','graph_init_execution_continuation_v1',
    'photo_fd_arithmetic_diagnostic_v1','photo_fd_arithmetic_diagnostic_root_v1',
    'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v2']

def verify_lineage():
    value=read(HERE/'LINEAGE_BINDINGS.json')
    require(value['schema']=='graph-init-precision-lineage-bindings-v1' and value['preserved_roots']==EXPECTED_ROOTS and
        value['precision_amendment_designed_after_known_failure'] is True and value['old_source_and_registry_preserved'] is True and
        value['old_checkpoint_or_phase_output_inherited'] is False and value['old_cold_qualification_inherited'] is False and
        value['old_diagnostic_as_new_cold_qualification'] is False and
        value['all_six_cold_and_every_actual_warm_requalification_required'] is True, 'Complete disclosed precision lineage required')
    records=value['all_bound_text_files']
    require(records and len({r['path'] for r in records})==len(records), 'Unique old text identities required')
    for record in records:
        bound(record)
    expected={str(REMOTE_PHASE/p.relative_to(PHASE)) for root in EXPECTED_ROOTS for p in (PHASE/root).rglob('*')
              if p.is_file() and p.suffix in TEXT_EXTENSIONS and '__pycache__' not in p.parts}
    require({r['path'] for r in records}==expected, 'Old failure/diagnostic text inventory must remain complete and immutable')
    registries=[r for r in records if Path(r['path']).name=='GRAPH_INIT_ATTEMPT_REGISTRY.json']
    require(registries==value['prior_attempt_registries'] and len(registries)==1 and
            registries[0]['path']==str(REMOTE_PHASE/'graph_init_execution_root_v1/study_v1/GRAPH_INIT_ATTEMPT_REGISTRY.json'),
            'Do not omit or substitute any failed v2 registry')
    status=value['old_v2_status']
    require(status['registered_attempts']==72 and status['claimed_attempts']==4 and status['completed_successfully']==3 and
            status['failed_attempts']==1 and status['unexecuted_attempts']==68 and status['warm_initialize_fit_attempts_completed']==0,
            'Original v2 numerical history must remain failed')
    bound(value['diagnostic_interpretation']);bound(value['local_receipt_parent_failure'])
    return value

def prior_registries():
    return verify_lineage()['prior_attempt_registries']

def lineage_descriptors():
    repair=read(HERE/'REPAIR_BINDINGS.json')
    for row in repair['preserved_text_records']:
        bound(row)
    return [descriptor(HERE/'LINEAGE_BINDINGS.json'),descriptor(HERE/'REPAIR_BINDINGS.json')]+verify_lineage()['all_bound_text_files']+repair['preserved_text_records']
