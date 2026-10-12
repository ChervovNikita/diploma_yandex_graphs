"""Isolated Q21 source/CPU/TRAIN-table/disabled-spec freeze on allocation."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zlib
from prepare_molecular_packets_allocation import GUARD,TRANSFER,invoke

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from adapters.private_bootstrap_source import SOURCES,CONTRACT
from compile_native_matrix import RUNTIME_FILES
NAME='massive_private_bootstrap_packet_20261012_v1'
LABEL='private_bootstrap_allocation_20261012_v1'
OWNED=list(dict.fromkeys([*RUNTIME_FILES,'adapters/context_exclusion.py','adapters/private_bootstrap_source.py',
 'compile_private_bootstrap_matrix.py','prepare_private_bootstrap_weights.py','compile_native_matrix.py','prepare_node_data.py','tests/test_private_bootstrap_source.py']))

REMOTE=GUARD+r'''
transport=phase/(sys.argv[1]+'_transport');encoded=(transport/'payload.b64').read_text();assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)));stage=phase/sys.argv[1];assert not stage.exists();stage.mkdir();runtime=stage/'runtime';runtime.mkdir()
def receipt(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for part in iter(lambda:stream.read(1048576),b''):h.update(part)
 return {'path':str(path.relative_to(phase)),'bytes':path.stat().st_size,'sha256':h.hexdigest()}
def save(path,value):
 assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');path.chmod(0o444)
for row in payload['files']:
 relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
 raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 path=runtime/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o444)
bindings={}
for key,(relative,sha) in payload['sources'].items():
 path=phase/relative;assert not path.is_symlink() and path.resolve().is_relative_to(phase.resolve())
 row=receipt(path);assert row['sha256']==sha;bindings[key]=row
save(stage/'SOURCE_BINDINGS.json',bindings)
save(stage/'SOURCE_INVENTORY.json',{'owned_runtime':[receipt(runtime/r['path']) for r in payload['files']],'pinned_original_sources':bindings,
 'common_controller_modified':False,'other_frozen_packets_modified':False,'source_contract':payload['contract'],'science_fits':0,'CUDA_executions':0})
save(stage/'HISTORY_SOURCE_CITATIONS.json',payload['history'])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',GNNM_REFERENCE_PROJECT=str(phase),PYTHONPATH=str(runtime))
tmp=stage/'engineering_tmp';tmp.mkdir();env['TMPDIR']=str(tmp)
def run(argv,label,seconds):
 tick=time.monotonic();child=subprocess.Popen(argv,cwd=runtime,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 f=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split();save(stage/'receipts'/(label+'_START.json'),{'pid':child.pid,'start_ticks':int(f[19]),'command':argv,'CUDA_executions':0,'scientific_fits':0})
 try:stdout,stderr=child.communicate(timeout=seconds)
 except subprocess.TimeoutExpired:child.kill();stdout,stderr=child.communicate();stderr+='\nBounded synthetic CPU/metadata deadline retained.\n'
 value={'exit_code':child.returncode,'wall_seconds':time.monotonic()-tick,'stdout':stdout,'stderr':stderr,'CUDA_executions':0,'scientific_fits':0,'scores_emitted':0,'GO_created':False}
 save(stage/'receipts'/(label+'_RESULT.json'),value);print(json.dumps({'label':label,**value}),flush=True);return value
python=str(repo/'.venv/bin/python')
qa=run([python,'-B','-m','unittest','discover','-s','tests','-p','test_private_bootstrap_source.py','-v'],'SYNTHETIC_CPU_SOURCE_QA',180)
if qa['exit_code']!=0:sys.exit(1)
assert 'Ran 5 tests' in qa['stderr'] and '\nOK\n' in qa['stderr'] and 'skipped' not in qa['stderr']
save(stage/'CPU_SOURCE_PROOF.json',{'tests':5,'skips':0,'passed':True,'synthetic_inputs_only':True,'actual_source_GAT_SAGE_models_and_maps':True,
 'source_block_gradients_verified':True,'fixed_class_mass_weights_and_local_generator_verified':True,'actual_moment_and_stream_restoration_verified':True,
 'original_real_R_S_B_verified':True,'all_masks_reference_metadata_and_genuine_I4_checked':True,'result':receipt(stage/'receipts/SYNTHETIC_CPU_SOURCE_QA_RESULT.json'),
 'CUDA_executions':0,'scientific_fits':0,'predictive_scores':0,'GO_created':False})
custodies=[str((phase/'training_pipeline_v2_20261012/prepared_data'/dataset/'CUSTODY.json').relative_to(phase)) for dataset in ('wikics','amazon-photo')]
setup=run([python,'-B',str(runtime/'prepare_private_bootstrap_weights.py'),'--project-root',str(phase),'--custody',*custodies,
 '--source-bindings',str((stage/'SOURCE_BINDINGS.json').relative_to(phase)),'--output',str((stage/'fixed_TRAIN_weights').relative_to(phase))],'FIXED_TRAIN_TABLE_PREPARATION',180)
if setup['exit_code']!=0:sys.exit(1)
compile_result=run([python,'-B',str(runtime/'compile_private_bootstrap_matrix.py'),'--project-root',str(phase),'--custody',*custodies,
 '--source-bindings',str((stage/'SOURCE_BINDINGS.json').relative_to(phase)),'--runtime-package',str(runtime.relative_to(phase)),
 '--weights',str((stage/'fixed_TRAIN_weights/MANIFEST.json').relative_to(phase)),'--fixture-receipts',str((stage/'CPU_SOURCE_PROOF.json').relative_to(phase)),
 '--output',str((stage/'matrix').relative_to(phase))],'DISABLED_SOURCE_COMPILER',180)
if compile_result['exit_code']!=0:sys.exit(1)
# Controller-free exact metadata/source authority checks, no numerical imports.
import gzip,collections
plan=json.loads((stage/'matrix/PLAN.json').read_text());summary=json.loads((stage/'matrix/SUMMARY.json').read_text());counts=summary['counts']
assert plan['scientific_launch_enabled'] is False and summary['scientific_training_started'] is False and summary['GO_created'] is False
assert counts['atomic_fit_groups']==5184 and counts['all_run_records']==13872 and counts['full_budget_optimizer_updates']==25920000
assert counts['optimizer_starts_by_stage']=={'search':864,'comparative':4320} and counts['genuine_I4_banks']==720 and counts['bank_exports']==2880
cohorts=[];originals=0;weighted=0
for spec in plan['studies']:
 path=phase/spec['path'];study=json.loads(gzip.decompress(path.read_bytes()));dataset=study['dataset'];assert dataset in ('wikics','amazon-photo')
 assert study['study_id'].startswith('private-bootstrap-standardized-v1-')
 expected={(i,s) for i in range(20 if dataset=='wikics' else 10) for s in (77201,77203,77207)}
 exports=[r for r in study['runs'] if r['kind']=='ensemble_assembly'];fits=[r for r in study['runs'] if r['kind']=='model_fit']
 selectors=[r for r in study['runs'] if r['kind']=='selection'];assert len(selectors)==12 and all(len(r['config']['configurations'])==9 for r in selectors)
 for comparison in study['comparisons']:
  for method in comparison['references']+comparison['candidates']:assert {(r['split'],r['seed']) for r in exports if r['method']==method}==expected
 for fit in fits:
  c=fit['config'];assert fit['budget']=={'unit':'optimizer_updates','policy':'full_budget','maximum':5000}
  if c['is_original_gnnm']:originals+=1;assert c['members']==4 and c['fusion']=='mean_logits'
  if c['bootstrap_arm'].endswith('_bootstrap'):weighted+=1;assert c['members']==4 and c['bootstrap_weights']
 for bank in exports:
  if len(bank['config']['fit_parents'])==4:
   rows=[next(r for r in fits if r['run_id']==pid) for pid in bank['config']['fit_parents']]
   assert [r['body'] for r in rows]==[0,1,2,3] and all(r['config']['members']==1 for r in rows)
 cohorts.append({'dataset':dataset,'study_id':study['study_id'],'source_spec':receipt(path),'fit_groups':len(fits),'selectors':12,
  'served_methods':16,'matched_comparative_cohorts':len(expected),'original_GNNM_reference':True,'genuine_I4_body0_reused':True})
assert originals==252 and weighted==756
proof={'schema':'private-bootstrap-source-port-proof-v1','cohorts':cohorts,'counts':counts,'original_GNNM_fit_records':originals,'weighted_block_VJP_fit_records':weighted,
 'source_bindings':receipt(stage/'SOURCE_BINDINGS.json'),'source_inventory':receipt(stage/'SOURCE_INVENTORY.json'),'CPU_source_proof':receipt(stage/'CPU_SOURCE_PROOF.json'),
 'fixed_TRAIN_weight_tables':receipt(stage/'fixed_TRAIN_weights/MANIFEST.json'),'weight_tables_count':94,
 'source_protocol_changes':payload['contract']['new_protocol'],'new_matched_control_seed_change':payload['contract']['matched_corrected_I4_seed_adaptation'],
 'original_source_history_preserved':True,'heldout_target_tensors_decoded':False,'scores_or_current_scientific_outcomes_opened':False,
 'scientific_fits':0,'CUDA_executions':0,'GO_created':False,'source_exact_recipe':False,'scientific_resource_protocol_master_GO_gates_pending':True}
save(stage/'SOURCE_PORT_PROOF.json',proof)
resources={'schema':'private-bootstrap-unmeasured-resource-boundaries-v1','qualified':False,'actual_GPU_peaks':None,'runtime_ETA':None,
 'facts':['one native slow body/one actual AdamW per atomic fit','weighted arms:four route forwards/two retained-graph VJPs/one optimizer action per update',
 'GAT uses source attention/native maps;SAGE uses both native graph maps','fixed float64[4,TRAIN] table per paired seed/mask shared across weighted arms/backbones',
 'genuine I4 pays four separately selected fit groups;endpoint export currently sequentially reconstructs each body'],
 'unknowns':['constructor_and_forward_gradient_tape_CUDA_peaks','native_AdamW_moment_update_workspaces','selected_checkpoint_model_moment_private_rng_table_binding_storage_IO_restore',
 'full_graph_GAT_edge_attention_cost','SELECT_CAL_evaluate_costs','whole_master_disk_cost_and_queue_resource_admission'],
 'reservation_mib_placeholder':32768,'source_private_block_gradient_calls_planned':756*5000*2,'backbone_updates_planned':25920000,
 'scalar_CAL_updates_separate':2880*500,'scientific_fits':0,'CUDA_executions':0,'GO_created':False}
save(stage/'RESOURCE_UNKNOWNS.json',resources)
entry={'id':'private_bootstrap_native_v1','plan':receipt(stage/'matrix/PLAN.json'),'summary':receipt(stage/'matrix/SUMMARY.json'),
 'plan_hash':summary['plan_hash'],'coverage':[receipt(stage/'SOURCE_PORT_PROOF.json'),receipt(stage/'RESOURCE_UNKNOWNS.json'),receipt(stage/'HISTORY_SOURCE_CITATIONS.json')]}
save(stage/'MASTER_INPUT_ENTRY.json',entry)
save(stage/'SUMMARY.json',{'schema':'private-bootstrap-source-disabled-handoff-v1','entry':receipt(stage/'MASTER_INPUT_ENTRY.json'),'proof':receipt(stage/'SOURCE_PORT_PROOF.json'),
 'counts':counts,'canonical_methods':3,'source_exact_recipe':False,'scientific_fits':0,'CUDA_executions':0,'scores_emitted':0,'GO_created':False,'resource_qualification':'UNMEASURED_REQUIRES_ROOT'})
paths=[stage/n for n in ['SOURCE_BINDINGS.json','SOURCE_INVENTORY.json','HISTORY_SOURCE_CITATIONS.json','CPU_SOURCE_PROOF.json','SOURCE_PORT_PROOF.json','RESOURCE_UNKNOWNS.json','MASTER_INPUT_ENTRY.json','SUMMARY.json',
 'matrix/PLAN.json','matrix/SUMMARY.json','fixed_TRAIN_weights/MANIFEST.json','receipts/SYNTHETIC_CPU_SOURCE_QA_RESULT.json','receipts/FIXED_TRAIN_TABLE_PREPARATION_RESULT.json','receipts/DISABLED_SOURCE_COMPILER_RESULT.json']]
paths += [phase/r['path'] for r in plan['studies']]
print(json.dumps({'final_summary':json.loads((stage/'SUMMARY.json').read_text()),'mirror_files':[{'receipt':receipt(path),'data':base64.b64encode(path.read_bytes()).decode()} for path in paths]}),flush=True)
'''


def main():
 out=PACKAGE/'preparation'/LABEL
 if out.exists():raise FileExistsError('Preserve prior source preparation; use fresh repair receipt')
 out.mkdir();files=[]
 for rel in OWNED:
  raw=(PACKAGE/rel).read_bytes();files.append({'path':rel,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
 history_path=PACKAGE/'qa/comprehensive_coverage_port_queue_v1/METHOD_HISTORY_CLASSIFICATION_METADATA.json';history=json.loads(history_path.read_text())
 projection=[r for r in history['all_record_history'] if r.get('canonical_method_id','').startswith('private_bootstrap_')];assert len(projection)==3
 raw=history_path.read_bytes();citation={'path':'training_pipeline_v2_20261012/qa/comprehensive_coverage_port_queue_v1/METHOD_HISTORY_CLASSIFICATION_METADATA.json','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 payload={'files':files,'sources':SOURCES,'contract':CONTRACT,'history':{'schema':'Q21-private-bootstrap-history-source-citations-v1','history_projection_source':citation,
  'canonical_records':projection,'prior_scientific_family':'private_class_balanced_bootstrap18_closed_negative_20261010_v1','historical_adverse_closure_retained':True,
  'new_packet_not_an_outcome_reinterpretation':True,'exact_historical_execution_revision_custody_limit_retained':True,'scientific_outcome_values_read':False}}
 encoded=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
 for offset in range(0,len(encoded),24000):
  result=invoke(TRANSFER,[NAME,str(offset),encoded[offset:offset+24000]],out,'TRANSFER_'+str(offset),60)
  if result['exit_code']!=0:raise RuntimeError(result['stderr'])
 result=invoke(REMOTE,[NAME,hashlib.sha256(encoded.encode()).hexdigest()],out,'TRANSPORT',600)
 for line in result['stdout'].splitlines():
  if not line.startswith('{'):continue
  val=json.loads(line)
  for row in val.pop('mirror_files',[]):
   raw=base64.b64decode(row['data']);assert len(raw)==row['receipt']['bytes'] and hashlib.sha256(raw).hexdigest()==row['receipt']['sha256']
   target=out/'allocation_artifacts'/row['receipt']['path'];assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
  print(json.dumps(val,indent=2))
 if result['exit_code']!=0:print(result['stderr'],result['stdout'][-8000:]);raise RuntimeError('Retained Q21 source preparation failure; no scientific fits/CUDA/GO')


if __name__=='__main__':main()
