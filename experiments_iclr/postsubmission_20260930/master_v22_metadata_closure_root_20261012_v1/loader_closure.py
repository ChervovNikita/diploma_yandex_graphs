
import hashlib,importlib,json,socket,subprocess,sys,time
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930';out=Path(__file__).resolve().parent
stage=phase/'massive_practical_master_20261012_v22';plan=json.loads((stage/'PLAN.json').read_text());summary=json.loads((stage/'SUMMARY.json').read_text())
blocked={'torch','numpy','scipy','sklearn','torch_geometric','torch_sparse','torch_scatter','ogb','pickle','_pickle','pandas'}
class NoScientific:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in blocked:raise AssertionError('Scientific imports denied in metadata closure: '+fullname)
sys.meta_path.insert(0,NoScientific());sys.path.insert(0,str(phase/plan['runtime_package']))
actual,digest,studies,runs,placements,deps=importlib.import_module('launcher').load_plan(phase,stage/'PLAN.json')
assert actual==plan and digest==summary['plan_hash']
assert set(runs)==set(placements)==set(deps)
assert all(placements[p]['worker'].split('/')[0]==placements[k]['worker'].split('/')[0] for k,ps in deps.items() for p in ps)
for value in studies.values():
 s=value['spec'];dataset=''.join(c for c in s['dataset'].lower() if c.isalnum());assert dataset not in {'romanempire','amazonratings','minesweeper','questions','tolokers'}
 assert len({placements[s['study_id']+'/'+r['run_id']]['worker'] for r in s['runs']})==1
count=importlib.import_module('compile_master_plan').typed_counts([s['spec'] for s in studies.values()]);assert count==summary['counts']
verification={'plan_hash':digest,'loaded_studies':len(studies),'assigned_records':len(runs),'model_fit_groups':count['totals']['atomic_fit_run_groups'],'all_records_exactly_once':True,'same_host_dependencies':True,'comparative_methods_within_study_same_GPU':True,'source_bytes_rehashed':True,'scientific_libraries_denied':True,'data_arrays_deserialized':False,'scientific_fits':0,'human_GO_created':False,'assembly_transport_interruption_preserved':True}
(stage/'ACTUAL_CONTROLLER_LOAD_VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
compact={'plan_hash':digest,'status':summary['status'],'studies':len(studies),'registered_records':len(runs),'model_fit_groups':count['totals']['atomic_fit_run_groups'],'counts':{k:v for k,v in count.items() if k not in {'native_stopping_run_keys','bounded_trial_run_keys','AMP_bounded_commit_run_keys'}},'original_paper_experiment_reruns':False,'all_method_inventory_migration_completed':False,'resource_gate':'UNQUALIFIED_FAMILIES_AND_ENDPOINT_CONCURRENCY_STORAGE_PENDING','scientific_fits':0,'GO_created':False,'source_families_added':['NCNC private/pooled-after-clamp','BUDDY','HGT typed/mixed','FiLM','E Rank1 Bayesian factors'],'controller_load_verification':verification}
(stage/'COMPACT_SUMMARY.json').write_text(json.dumps(compact,indent=2)+'\n')
result={'plan_hash':digest,'studies':len(studies),'model_fits':count['totals']['atomic_fit_run_groups'],'optimizer_call_bounds':[count['model_optimizer_step_call_minimum'],count['model_optimizer_step_call_maximum']],'scientific_fits':0,'GO':False}
(out/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
