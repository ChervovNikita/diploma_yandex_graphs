"""Read the 32 existing engineering receipts once; never construct a model."""
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD, invoke

HERE = Path(__file__).resolve().parent / 'new_source_resource_closure_root_20261012_v1'
REMOTE = GUARD + r'''
from collections import Counter
proposal=json.loads((phase/'new_source_resource_stage_20261012_v2/source/PROPOSAL.json').read_text())
cases={r['case_id']:r for r in proposal['requested_cases']};assert len(cases)==32
prefix=json.loads((phase/'new_source_engineering_prefix_root_20261012_v1/CLOSURE.json').read_text())
recovery=json.loads((phase/'new_source_engineering_remaining_root_20261012_v1/ROOT_EXISTING_CASE_RECOVERY.json').read_text())
tail=json.loads((phase/'new_source_engineering_native_remaining7_root_20261012_v1/CLOSURE.json').read_text())
rows=prefix['attempts']+recovery['completed_cases']+tail['attempts']
assert len(rows)==32 and {r['case_id'] for r in rows}==set(cases)
assert not recovery['failed_cases'] and all(r.get('exit_code',0)==0 and r['child_absent'] for r in rows)
def receipt(p):return {'path':str(p.relative_to(phase)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
gpu_pids=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10).splitlines())
closed=[]
for row in rows:
 pid=row['child']['pid'];assert not (Path('/proc')/str(pid)).exists() and str(pid) not in gpu_pids
 p=phase/row['result']['path'];assert receipt(p)==row['result']
 value=json.loads(p.read_text());case=cases[row['case_id']]
 assert value['actual_optimizer_updates']==value['original_optimizer_kernels']==value['scores_emitted']==value['heldout_target_files_opened']==0
 item={'case_id':row['case_id'],'family':case['family'],'dataset':case['dataset'],
  'study_id':case.get('study_id'),'source_run_id':case.get('source_run_id'),
  'arm':case.get('arm'),'backbone':case.get('backbone'),'source_run_hash':case.get('source_run_hash'),
  'source_case':case,'result':row['result'],'child':row['child'],'child_and_CUDA_PID_absent':True}
 keys=('CUDA_peak_allocated_bytes','CUDA_peak_reserved_bytes','parameter_bytes','gradient_bytes',
  'manual_moment_bytes','extra_workspace_bytes','process_peak_RSS_KiB','elapsed_wall_seconds',
  'warm_median_seconds','full_graph_serving_shape','full_graph_serving_seconds',
  'full_graph_logits_CPU_bytes','random_engine_restore','CPU_bank_export_proxy','packs',
  'optimizer_attempts','optimizer_functional_attempts','cycle_action_counts','current_child_allocator_cap')
 item['observations']={k:value[k] for k in keys if k in value};closed.append(item)
family_counts=dict(Counter(r['family'] for r in closed))
aggregate={'schema':'new-source-zero-update-resource-closure-v1','cases':closed,
 'total_cases':32,'family_counts':family_counts,'all_children_and_CUDA_PIDs_absent':True,
 'actual_optimizer_updates':0,'optimizer_kernels':0,'scientific_fits':0,'scores_emitted':0,
 'heldout_target_files_opened':0,'GO':False,'two_process_pairs_qualified':False,
 'scope':'Random-state source callbacks and actual checkpoint round trips; no training, selected checkpoints or whole-study ETA',
 'preserved_collector_failures':['new_source_engineering_remaining_root_20261012_v1',
 'new_source_queue_metadata_recovery_root_20261012_v1'],
 'evidence':[receipt(phase/'new_source_engineering_prefix_root_20261012_v1/CLOSURE.json'),
 receipt(phase/'new_source_engineering_remaining_root_20261012_v1/ROOT_EXISTING_CASE_RECOVERY.json'),
 receipt(phase/'new_source_engineering_native_remaining7_root_20261012_v1/CLOSURE.json')]}
out=phase/'new_source_resource_closure_root_20261012_v1';assert not out.exists();out.mkdir()
(out/'CLOSURE.json').write_text(json.dumps(aggregate,indent=2,allow_nan=False)+'\n')
aggregate['closure_receipt']=receipt(out/'CLOSURE.json')
print(json.dumps(aggregate,allow_nan=False))
'''

def main():
    assert not HERE.exists()
    HERE.mkdir()
    (HERE/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes())
    transport=invoke(REMOTE,[],HERE,'READ_EXISTING_CLOSURE',120)
    assert transport['exit_code']==0, 'Read-only aggregation failure preserved'
    value=json.loads(transport['stdout'])
    (HERE/'CLOSURE.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:value[k] for k in ('total_cases','family_counts','all_children_and_CUDA_PIDs_absent','actual_optimizer_updates','scores_emitted','GO')}))

if __name__=='__main__':main()
