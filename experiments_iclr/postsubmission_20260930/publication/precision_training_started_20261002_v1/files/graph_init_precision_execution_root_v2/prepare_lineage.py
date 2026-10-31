"""Inventory all old v2/diagnostic text custody; never opens arrays/models."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REMOTE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
TEXT = {'.json','.py','.md','.csv','.txt','.sh','.toml','.yaml','.yml','.diff','.patch','.log','.jsonl'}
ROOTS = ['graph_init_execution_root_v1','graph_init_execution_continuation_v1',
         'photo_fd_arithmetic_diagnostic_v1','photo_fd_arithmetic_diagnostic_root_v1',
         'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v2']

def descriptor(p):
    assert p.is_file() and not p.is_symlink() and p.suffix in TEXT
    return {'path':str(REMOTE/p.relative_to(PHASE)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes':p.stat().st_size}

rows=[]
for root in ROOTS:
    for p in sorted((PHASE/root).rglob('*')):
        if p.is_file() and p.suffix in TEXT and '__pycache__' not in p.parts:
            rows.append(descriptor(p))
registries=[x for x in rows if Path(x['path']).name=='GRAPH_INIT_ATTEMPT_REGISTRY.json']
assert len(registries)==1
old=PHASE/'graph_init_execution_root_v1/study_v1'
claims=[json.loads(p.read_text()) for p in (old/'claims').glob('*.json')]
terminals=[json.loads(p.read_text()) for p in (old/'terminals').glob('*.json')]
assert len(claims)==len(terminals)==4
summary={'registered_attempts':72, 'claimed_attempts':4, 'completed_successfully':sum(x.get('completed') is True for x in terminals),
         'failed_attempts':sum(x.get('completed') is False for x in terminals), 'unexecuted_attempts':68,
         'failed_context':'Photo/seed17/cold qualify','warm_initialize_fit_attempts_completed':0}
assert summary['completed_successfully']==3 and summary['failed_attempts']==1
value={'schema':'graph-init-precision-lineage-bindings-v1','preserved_roots':ROOTS,
       'all_bound_text_files':rows,'prior_attempt_registries':registries,'old_v2_status':summary,
       'precision_amendment_designed_after_known_failure':True,
       'diagnostic_interpretation':descriptor(PHASE/'photo_fd_arithmetic_diagnostic_root_v1/ROOT_DIAGNOSTIC_ANALYSIS.json'),
       'local_receipt_parent_failure':descriptor(PHASE/'photo_fd_arithmetic_diagnostic_root_v1/LOCAL_LAUNCH_RECEIPT_WRITE_FAILURE.json'),
       'old_source_and_registry_preserved':True,'old_checkpoint_or_phase_output_inherited':False,
       'old_cold_qualification_inherited':False,'old_diagnostic_as_new_cold_qualification':False,
       'all_six_cold_and_every_actual_warm_requalification_required':True,
       'scientific_execution':False,'arrays_models_or_checkpoints_opened':False}
with (HERE/'LINEAGE_BINDINGS.json').open('x') as stream:
    json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps({'protected_text_files':len(rows),'prior_registries':len(registries),'status':summary}))
