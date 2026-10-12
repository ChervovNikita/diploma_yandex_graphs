"""Materialize disabled Coauthor source-bound coverage; no numerical reruns."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib
from prepare_molecular_packets_allocation import GUARD, TRANSFER, invoke

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from compile_polyformer_practical_matrix import COAUTHOR_POLICY
DESTINATION='massive_polyformer_coauthor_practical_packet_20261012_v1'
LABEL='polyformer_coauthor_practical_allocation_20261012_v1'
OWNED=['compile_polyformer_practical_matrix.py','compile_polyformer_coauthor_practical_matrix.py',
 'tests/test_polyformer_coauthor_practical.py','docs/POLYFORMER_COAUTHOR_PRACTICAL_V1.md']

REMOTE=GUARD+r'''
transport=phase/(sys.argv[1]+'_transport');encoded=(transport/'payload.b64').read_text();assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)));stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
old=phase/'massive_polyformer_packet_20261012_v2';old_package=old/'sources/training_pipeline_v2_20261012'
package=stage/'sources/training_pipeline_v2_20261012';shutil.copytree(old_package,package,ignore=shutil.ignore_patterns('__pycache__'))
def save(path,value):
 assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');path.chmod(0o444)
def receipt(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return {'path':str(path.relative_to(phase)),'bytes':path.stat().st_size,'sha256':h.hexdigest()}
for row in payload['files']:
 relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
 raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 path=package/relative;path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():path.chmod(0o644)
 path.write_bytes(raw);path.chmod(0o444)
qa_path=old/'POLYFORMER_BOUNDARY_CPU_QA_RESULT.json';qa=json.loads(qa_path.read_text())
assert qa['exit_code']==0 and 'Ran 15 tests' in qa['stderr'] and '\nOK\n' in qa['stderr'] and 'skipped' not in qa['stderr']
unchanged=[]
for rel in ['engine.py','run_study.py','run_endpoint.py','spec.py','registry.py','launcher.py',
 'compile_polyformer_matrix.py','compile_polyformer_boundary_matrix.py',
 'adapters/polyformer.py','adapters/polyformer_boundary.py','adapters/polyformer_endpoints.py','adapters/classifier.py','adapters/node_losses.py','adapters/node_endpoints.py',
 'tests/test_polyformer.py','tests/test_polyformer_matrix.py','tests/test_polyformer_boundary.py']:
 assert (package/rel).read_bytes()==(old_package/rel).read_bytes();unchanged.append({'source':receipt(old_package/rel),'copy':receipt(package/rel)})
save(stage/'NUMERICAL_QA_INHERITANCE.json',{'schema':'polyformer-coauthor-source-QA-inheritance-v1','v2_QA_result':receipt(qa_path),
 'prior15_CPU_tests_passed':True,'skips':0,'unchanged_numerical_source_controller_compiler_APIs_and_tests':unchanged,
 'additional_numerical_tests':0,'additional_model_constructions':0,'scientific_fits':0,'CUDA_executions':0})
save(stage/'POLICY.json',payload['policy']);save(stage/'NATIVE_ANCHOR_METADATA.json',payload['native_anchor_metadata'])
save(stage/'SOURCE_INVENTORY.json',[receipt(p) for p in sorted(package.rglob('*')) if p.is_file()])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',GNNM_REFERENCE_PROJECT=str(phase))
tmp=stage/'engineering_tmp';tmp.mkdir();env['TMPDIR']=str(tmp)
def run(command,label,timeout):
 tick=time.monotonic();child=subprocess.Popen(command,cwd=package,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
 save(stage/(label+'_START.json'),{'pid':child.pid,'start_ticks':int(fields[19]),'command':command,'scientific_fits':0,'CUDA_executions':0})
 try:stdout,stderr=child.communicate(timeout=timeout)
 except subprocess.TimeoutExpired:child.kill();stdout,stderr=child.communicate();stderr+='\nBounded metadata preparation timeout retained.\n'
 value={'exit_code':child.returncode,'wall_seconds':time.monotonic()-tick,'stdout':stdout,'stderr':stderr,'scientific_fits':0,'CUDA_executions':0,'scores_emitted':0}
 save(stage/(label+'_RESULT.json'),value);print(json.dumps({'label':label,**value}),flush=True);return value
python=str(repo/'.venv/bin/python')
result=run([python,'-B','-m','unittest','discover','-s','tests','-p','test_polyformer_coauthor_practical.py','-v'],'COAUTHOR_POLICY_REFERENCE_METADATA_QA',180)
if result['exit_code']!=0:sys.exit(1)
assert 'Ran 2 tests' in result['stderr'] and '\nOK\n' in result['stderr'] and 'skipped' not in result['stderr']
result=run([python,'-B',str(package/'compile_polyformer_coauthor_practical_matrix.py'),'--project-root',str(phase),
 '--v2-packet',str(old.relative_to(phase)),'--runtime-prefix',str(package.relative_to(phase)),'--policy',str((stage/'POLICY.json').relative_to(phase)),
 '--output',str((stage/'matrix').relative_to(phase))],'DISABLED_COAUTHOR_COMPILER',300)
if result['exit_code']!=0:sys.exit(1)
# The remaining work is pure source/plan metadata qualification under the repo Python.
verify_code=r"""
import collections,gzip,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);stage=Path(sys.argv[2]);package=stage/'sources/training_pipeline_v2_20261012';sys.path.insert(0,str(package))
from compile_polyformer_practical_matrix import COAUTHOR_POLICY,practical_domain
from compile_polyformer_boundary_matrix import receipt,verify_receipt
from compile_polyformer_matrix import templates
from adapters.polyformer_boundary import M4,M1,validate_config
from adapters.polyformer_endpoints import independent_contracts
from spec import fingerprint,validate
policy=json.loads((stage/'POLICY.json').read_text());assert policy==COAUTHOR_POLICY
plan=json.loads((stage/'matrix/PLAN.json').read_text());summary=json.loads((stage/'matrix/SUMMARY.json').read_text())
assert fingerprint(plan)==summary['plan_hash'] and fingerprint(policy)==summary['policy_hash']
assert plan['scientific_launch_enabled'] is False and plan['human_GO_required'] is True and summary['GO_created'] is False
matrix=json.loads((package/'docs/MATRIX_AXES_DRAFT.json').read_text());counts=collections.Counter();stages=collections.Counter();cohorts=[];original_records=[];resources=[];seen={}
for row in plan['studies']:
 value=json.loads(gzip.decompress((root/row['path']).read_bytes()));assert fingerprint(value)==row['spec_hash'];validate(value)
 dataset=value['dataset'];assert dataset in policy['datasets'] and value['stage']=='development'
 assert value['data']['evidence']['split_roster']==list(range(10)) and value['data']['evidence']['search_split']==0
 assert value['data']['evidence']['unused_confirmation_claimed'] is False and value['data']['evidence']['author_replication'] is False
 recipe=next(r for r in matrix['baseline_recipe_catalogue']['rows'] if r['recipe_id']=='polyformer_mono_'+dataset)
 for binding in value['source']['files']:
  if binding['path'] not in seen:verify_receipt(root,binding);seen[binding['path']]=binding
  else:assert seen[binding['path']]==binding
 selectors={r['run_id']:r for r in value['runs'] if r['kind']=='selection'};assert len(selectors)==36
 assert {r['method'] for r in selectors.values()}=={t['method'] for t in templates()}|{M4,M1}
 runmap={r['run_id']:r for r in value['runs']};groups={}
 for selector in selectors.values():
  actual=selector['config']['configurations'];domain=practical_domain(recipe,selector['method'],policy)
  assert len(actual)==9 and [p['parameters'] for p in actual]==[p['parameters'] for p in domain]
  assert sum(p['source_anchor'] for p in actual)==1
  assert selector['parents']==[i for p in actual for i in p['run_ids']]
  for p in actual:
   assert len(p['run_ids'])==2
   for i in p['run_ids']:
    run=runmap[i];assert run['split']==0 and run['seed'] in policy['search_seeds']
    assert all(run['config'][key]==val for key,val in p['parameters'].items())
 for run in value['runs']:
  counts[run['kind']]+=1
  if run['kind']=='model_fit':
   c=run['config'];stages[c['stage']]+=1
   assert run['budget']=={'maximum':2000,'policy':'full_budget','unit':'optimizer_updates'}
   assert c['native']['hidden']==128 and c['native']['nlayer']==1
   if c['stage']=='comparative':assert c['selected_config_from'] in selectors and run['seed'] in policy['comparative_seeds'] and run['split'] in policy['comparative_splits']
   if c.get('independence_group'):groups.setdefault(c['independence_group'],[]).append(run)
   if run['method'] in (M4,M1):validate_config(c,run)
   if c.get('is_original_gnnm'):
    assert run['method']==M4 and c['members']==4 and c['factor_initialization']=='source_rsb_xavier' and c['fusion']=='mean_logits'
    original_records.append({'study_id':value['study_id'],'run_id':run['run_id'],'stage':c['stage'],'split':run['split'],'seed':run['seed'],'config_hash':fingerprint(c)})
 for group in groups.values():independent_contracts(value,group)
 assert len(groups)==570
 exports=[r for r in value['runs'] if r['kind']=='ensemble_assembly'];expected={(mask,seed) for mask in range(10) for seed in policy['comparative_seeds']}
 assert len({r['method'] for r in exports})==55
 for comparison in value['comparisons']:
  assert comparison['original_GNNM_reference']==M4 and M4 in comparison['references'] and M1+'_I4' in comparison['references']
  for method in comparison['references']+comparison['candidates']:
   assert {(r['split'],r['seed']) for r in exports if r['method']==method}==expected
 graph=value['data']['evidence']['model_graph'];feature_bytes=graph['nodes']*graph['features']*4;token_bytes=feature_bytes*(recipe['operator']['polynomial_order_K']+1)
 resources.append({'dataset':dataset,'native_float32_feature_bytes':feature_bytes,'one_body_K_plus1_token_bytes':token_bytes,
  'four_separately_built_bodies_token_bytes_if_not_shared':4*token_bytes,
  'formula_only_not_measured':True,'source_profile':value['resources']['profile'],
  'blockers':['CUDA_constructor_forward_gradient_tape_peak_unmeasured','native_optimizer_moments_update_kernels_unmeasured','selected_checkpoint_storage_IO_restore_unqualified',
   'separate_four_body_bank_export_residency_unqualified','SELECT_CAL_evaluate_resources_unqualified','source_estimated_time_and64GiB_reservation_provisional','whole_master_disk_resource_gate_and_human_GO']})
 cohorts.append({'dataset':dataset,'search_mask':0,'comparative_masks':list(range(10)),'search_seeds':policy['search_seeds'],'comparative_seeds':policy['comparative_seeds'],
  'procedures':36,'domain_points_each':9,'source_anchors':36,'served_methods':55,'independent_four_body_groups':570,'all_reference_cohorts_matched':True})
assert counts=={'model_fit':6876,'selection':72,'ensemble_assembly':3300,'calibration':3300,'evaluation':3300}
assert stages=={'search':1296,'comparative':5580}
assert len(original_records)==96 and sum(r['stage']=='search' for r in original_records)==36 and sum(r['stage']=='comparative' for r in original_records)==60
assert summary['totals']['all_run_records']==16848 and summary['totals']['full_budget_optimizer_updates']==13752000
proof={'schema':'polyformer-coauthor-practical-source-policy-reference-proof-v1','plan_hash':fingerprint(plan),'policy_hash':fingerprint(policy),'counts':dict(counts),
 'fit_stages':dict(stages),'formal_totals':summary['totals'],'cohorts':cohorts,'source_and_custody_files_rehashed':len(seen),
 'original_GNNM_fit_records':96,'original_GNNM_by_stage':{'search':36,'comparative':60},'original_GNNM_records':original_records,
 'native_group_LR_shape_operator_and_real_RSB_source_locked':True,'numerical_QA_inherited_by_exact_source_bytes':True,'new_metadata_policy_tests_passed':2,
 'model_constructors_executed':0,'scientific_fits':0,'CUDA_executions':0,'scores_emitted':0,'GO_created':False,'heldout_target_tensors_decoded':False,
 'prepared_label_files_hash_verified_only':True,'full_v2_author_and_private_packets_modified':False,'author_score_protocol_equality':False,
 'PLAN_file':receipt(root,stage.relative_to(root)/'matrix/PLAN.json'),'resource_status':'PROVISIONAL_REQUIRES_ROOT_QUALIFICATION'}
for name,obj in [('FINAL_SOURCE_POLICY_REFERENCE_PROOF.json',proof),('RESOURCE_BLOCKERS.json',{'schema':'coauthor-practical-unmeasured-resource-blockers-v1','datasets':resources,'runtime_estimate':None,'measured_peaks':None,'qualified':False,'scientific_fits':0,'CUDA_executions':0})]:
 path=stage/name;assert not path.exists();path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');path.chmod(0o444)
entry={'id':'practical_polyformer_coauthor_v1','plan':receipt(root,stage.relative_to(root)/'matrix/PLAN.json'),'summary':receipt(root,stage.relative_to(root)/'matrix/SUMMARY.json'),
 'plan_hash':fingerprint(plan),'coverage':[receipt(root,stage.relative_to(root)/'FINAL_SOURCE_POLICY_REFERENCE_PROOF.json'),receipt(root,stage.relative_to(root)/'RESOURCE_BLOCKERS.json')]}
path=stage/'MASTER_INPUT_ENTRY.json';assert not path.exists();path.write_text(json.dumps(entry,indent=2,sort_keys=True)+'\n');path.chmod(0o444)
print(json.dumps({k:v for k,v in proof.items() if k not in ('original_GNNM_records',)}))
"""
# Guarded transport Python only orchestrates; allocation repo Python runs API checks.
result=run([python,'-B','-c',verify_code,str(phase),str(stage)],'FINAL_SOURCE_POLICY_REFERENCE_VERIFY',300)
if result['exit_code']!=0:sys.exit(1)
save(stage/'PREPARATION_SUMMARY.json',{'schema':'coauthor-practical-disabled-source-preparation-v1','status':'prepared',
 'entry':receipt(stage/'MASTER_INPUT_ENTRY.json'),'source_reference_proof':receipt(stage/'FINAL_SOURCE_POLICY_REFERENCE_PROOF.json'),'resource_blockers':receipt(stage/'RESOURCE_BLOCKERS.json'),
 'numerical_QA_inheritance':receipt(stage/'NUMERICAL_QA_INHERITANCE.json'),'new_metadata_policy_tests':2,'repeated_numerical_tests':0,
 'full_v2_and_author_packets_preserved':True,'private12_queue_unchanged_disabled':True,'scientific_fits':0,'CUDA_executions':0,'scores_emitted':0,'GO_created':False})
paths=[stage/n for n in ['POLICY.json','NATIVE_ANCHOR_METADATA.json','NUMERICAL_QA_INHERITANCE.json','SOURCE_INVENTORY.json','MASTER_INPUT_ENTRY.json','FINAL_SOURCE_POLICY_REFERENCE_PROOF.json','RESOURCE_BLOCKERS.json','PREPARATION_SUMMARY.json','COAUTHOR_POLICY_REFERENCE_METADATA_QA_RESULT.json','DISABLED_COAUTHOR_COMPILER_RESULT.json','FINAL_SOURCE_POLICY_REFERENCE_VERIFY_RESULT.json','matrix/PLAN.json','matrix/SUMMARY.json']]
plan=json.loads((stage/'matrix/PLAN.json').read_text());paths += [phase/r['path'] for r in plan['studies']]
print(json.dumps({'final_summary':json.loads((stage/'PREPARATION_SUMMARY.json').read_text()),'mirror_files':[{'receipt':receipt(p),'data':base64.b64encode(p.read_bytes()).decode()} for p in paths]}),flush=True)
'''


def main():
    out=PACKAGE/'preparation'/LABEL
    assert out.exists()
    if (out/'OWNED_SOURCE_INVENTORY.json').exists():raise FileExistsError('Preserve previous materialization attempts')
    files=[]
    for relative in OWNED:
        raw=(PACKAGE/relative).read_bytes();files.append({'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
    (out/'OWNED_SOURCE_INVENTORY.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='data'} for r in files],indent=2)+'\n')
    payload={'destination':DESTINATION,'files':files,'policy':COAUTHOR_POLICY,'native_anchor_metadata':json.loads((out/'NATIVE_ANCHOR_METADATA.json').read_text())}
    encoded=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    for offset in range(0,len(encoded),24000):
        result=invoke(TRANSFER,[DESTINATION,str(offset),encoded[offset:offset+24000]],out,'TRANSFER_'+str(offset),60)
        if result['exit_code']!=0:raise RuntimeError(result['stderr'])
    result=invoke(REMOTE,[DESTINATION,hashlib.sha256(encoded.encode()).hexdigest()],out,'PREPARATION_TRANSPORT',600)
    for line in result['stdout'].splitlines():
        if not line.startswith('{'):continue
        value=json.loads(line)
        for row in value.pop('mirror_files',[]):
            raw=base64.b64decode(row['data']);assert len(raw)==row['receipt']['bytes'] and hashlib.sha256(raw).hexdigest()==row['receipt']['sha256']
            target=out/'allocation_artifacts'/row['receipt']['path'];assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        print(json.dumps(value,indent=2))
    if result['exit_code']!=0:print(result['stderr'],result['stdout'][-5000:]);raise RuntimeError('Retained blocked preparation; no fits/CUDA/GO')


if __name__=='__main__':main()
