"""Complete old failure and diagnostic text custody; no scientific artifacts."""
from pathlib import Path
from admission_support import HERE, PHASE, REMOTE_PHASE, TEXT_EXTENSIONS, read, bound, descriptor, require

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
    return [descriptor(HERE/'LINEAGE_BINDINGS.json')]+verify_lineage()['all_bound_text_files']
