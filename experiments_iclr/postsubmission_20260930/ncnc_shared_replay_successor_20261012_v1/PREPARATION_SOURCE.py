"""Exclusive source successor for249LinkPrototype fits; no scientific GO."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'training_pipeline_v2_20261012/preparation'))
from prepare_molecular_packets_allocation import GUARD,TRANSFER,invoke

REMOTE=GUARD+r'''
import copy,gzip,importlib
transport=phase/(sys.argv[1]+'_transport');encoded=(transport/'payload.b64').read_bytes()
assert hashlib.sha256(encoded).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))
out=phase/sys.argv[1];assert not out.exists();out.mkdir()
old=phase/'ncnc_completion_successor_root_20261012_v3'
def receipt(p):return {'path':str(p.relative_to(phase)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
shutil.copytree(old/'runtime',out/'runtime')
for row in payload['files']:
 p=out/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():p.chmod(0o644)
 raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];p.write_bytes(raw)
# This finite numerical engineering proof opens fabricated arrays and pinned
# source files only. Complete scientific datasets/targets are never loaded.
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
command=[str(repo/'.venv/bin/python'),'-B',str(out/'check_synthetic.py')]
tick=time.monotonic();child=subprocess.Popen(command,cwd=out,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
(out/'SYNTHETIC_OWNER_START.json').write_text(json.dumps({'pid':child.pid,'start_ticks':int(fields[19]),'command':command,'scope':'tiny_CPU_learning_law_noop_Adam_engineering','scientific_fits':0})+'\n')
try:stdout,stderr=child.communicate(timeout=300)
except subprocess.TimeoutExpired:
 child.kill();stdout,stderr=child.communicate();stderr+='\nFiniteCPUQA300secondtimeoutpreserved\n'
q={'exit_code':child.returncode,'stdout':stdout,'stderr':stderr,'wall_seconds':time.monotonic()-tick,'scientific_fits':0,'actual_optimizer_updates':0}
(out/'SYNTHETIC_CPU_TRANSPORT.json').write_text(json.dumps(q,indent=2,sort_keys=True)+'\n')
(out/'SYNTHETIC_OWNER_END.json').write_text(json.dumps({'pid':child.pid,'returncode':child.returncode,'child_reaped':True,'scientific_fits':0})+'\n')
assert child.returncode==0,stderr
qa=json.loads((out/'SYNTHETIC_CPU_QA.json').read_text());assert qa['status']=='PASS' and qa['actual_optimizer_maps']==0 and qa['scientific_fits']==0
sys.path.insert(0,str(out/'runtime'))
from spec import canonical,fingerprint,validate
from compile_master_plan import typed_counts
oldplan=json.loads((old/'matrix/PLAN.json').read_text());assert len(oldplan['studies'])==1
study=json.loads(gzip.decompress((phase/oldplan['studies'][0]['path']).read_bytes()))
assert study['dataset']=='ogbl-collab'
before=copy.deepcopy(study['runs']);eligible=[];excluded=[]
integration={k:receipt(out/v) for k,v in {'shared_factory':'runtime/adapters/link_prototypes.py',
 'shared_helper':'runtime/adapters/shared_source_replay.py','shared_store':'runtime/shared_replay_store.py'}.items()}
policy={'schema':'LinkPrototype-shared-source-replay-policy-v1','authority_scope':'prospective_scientific_TRAIN_integers',
 'store_root':str((out/'prospective_shared_integer_store').relative_to(phase)),
 'physical_admission_directory':str((out/'physical_runtime_admissions').relative_to(phase)),
 'integration_source_receipts':integration,'qualification_evidence_is_not_a_donor':True,
 'epochs':100,'source_master_seed':2026100401,'source_draw_per_new_shared_epoch':1,
 'source_generation_reuse_and_failure_costs_retained':True,'leases_and_failed_attempts_retained':True,
 'native_boundary_fits_eligible':False,'source_learning_hyperparameters_changed':False}
for r in study['runs']:
 if r['kind']!='model_fit':continue
 if 'link_prototype_arm' in r['config']:
  assert r['config']['training']['batch_size']==65536 and r['config']['train_minibatches_per_epoch']==17 and r['budget']['maximum']==100
  r['adapter']=str((out/'runtime/adapters/link_prototypes.py').relative_to(phase))+':make_adapter'
  r['config']['shared_source_replay']=copy.deepcopy(policy);eligible.append(r['run_id'])
 else:excluded.append(r['run_id'])
assert len(eligible)==249 and len(excluded)==126
for original,changed in zip(before,study['runs']):
 if original['run_id'] in eligible:
  normalized=copy.deepcopy(changed);normalized['adapter']=original['adapter'];normalized['config'].pop('shared_source_replay')
  assert normalized==original
 else:assert changed==original,'Nativeboundaryandendpointrecordsremainbyte-equivalent'
assert typed_counts([study])==typed_counts([{**study,'runs':before}])
old_source={str(p.relative_to(old/'runtime')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (old/'runtime').rglob('*.py')}
changedfiles=[]
for rel,sha in old_source.items():
 if hashlib.sha256((out/'runtime'/rel).read_bytes()).hexdigest()!=sha:changedfiles.append(rel)
assert changedfiles==['adapters/link_prototypes.py']
matrix=out/'matrix';matrix.mkdir();(matrix/'studies').mkdir()
sourcefiles=[receipt(out/p) for p in ['runtime/adapters/link_prototypes.py','runtime/adapters/shared_source_replay.py','runtime/shared_replay_store.py','LEARNING_LAW_AST_PROOF.json','SYNTHETIC_CPU_QA.json','SHARED_REPLAY_SCOPE.md']]
study['source']['files']+=sourcefiles
study['data']['evidence']['shared_source_replay_storage_successor']={'eligible_fits':249,'native_boundary_fits_excluded':126,
 'fixed_native_learning_law_unchanged':True,'source_epoch1_fullinput_qualification':receipt(phase/'shared_replay_epoch1_execution_root_20261012_v1/proof/RESULT.json'),
 'qualification_cache_reused_as_scientific_donor':False,'maximum_physical_device_namespaces':3,'physical_77_admission_pending':True}
validate(study);path=matrix/'studies'/(study['study_id']+'.json.gz');path.write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0))
plan=copy.deepcopy(oldplan);plan['studies']=[{'path':str(path.relative_to(phase)),'spec_hash':fingerprint(study)}]
plan['output_root']=str((matrix/'execution').relative_to(phase))
for gate in plan['global_gates'].values():gate['path']=str((matrix/'gate_receipts').relative_to(phase))+'/'+Path(gate['path']).name
for group in plan['groups']:group['assignment_status']='disabled;physicalruntime/disk/memoryanduserGOrequired'
summary=copy.deepcopy(json.loads((old/'matrix/SUMMARY.json').read_text()))
summary.update(schema='NCNC-shared-source-replay-exclusive-successor-v1',status='DISABLED_SOURCE_CPU_PARITY_PASS_PHYSICAL_RESOURCES_AND_GO_PENDING',plan_hash=fingerprint(plan),
 eligible_shared_replay_fits=249,excluded_native_boundary_fits=126,extra_scientific_fits=0,
 source_data_learning_hyperparameters_source_optimizer_work_changed=False,
 prospective_integer_payload_bytes_per_runtime_device=4716208000,
 maximum_three_qualified_device_integer_payload_bytes=14148624000,
 duplicate249provider_fit_integer_payload_before_bytes=1174335792000,
 above_counts_exclude_NPY_EPOCH_manifests_leases_costs_failed_attempts=True,
 no_compression_or_deletion_credited=True,all_old_sources_failures_preserved=True,GO=False)
lineage={'schema':'NCNC-shared-replay-exclusive-replacement-v1','exclusive_replacement_required':True,
 'predecessor_plan':receipt(old/'matrix/PLAN.json'),'predecessor_entry':receipt(old/'MASTER_INPUT_ENTRY.json'),
 'unchanged_native_boundary_fit_ids':excluded,'eligible_shared_replay_fit_ids':eligible,
 'all_noneligible_records_byte_equivalent':True,'eligible_numeric_configs_budget_parent_selector_unchanged':True,
 'counts_unchanged':True,'known_optimizer_step_calls':637500,'source_registration_not_execution':True}
for name,value in [('PLAN.json',plan),('SUMMARY.json',summary),('EXCLUSIVE_REPLACEMENT.json',lineage)]:
 (matrix/name).write_text(canonical(value)+'\n')
# Empty admission directory; templates are explicitly disabled, never authority.
admissions=out/'physical_runtime_admissions';admissions.mkdir()
runtime_template={'schema':'shared-source-replay-physical-admission-v1','admitted':False,'root_authorization_reference':None,
 'device_binding':None,'runtime':None,'integration_source_receipts':integration,
 'source_epoch1_qualification_receipt':study['data']['evidence']['shared_source_replay_storage_successor']['source_epoch1_fullinput_qualification'],
 'qualification_evidence_is_not_scientific_donor':True,'requires_threeGPU_destination_preflight':True}
(out/'PHYSICAL_RUNTIME_ADMISSION_TEMPLATE.json').write_text(canonical(runtime_template)+'\n')
coverage=[receipt(out/p) for p in ['LEARNING_LAW_AST_PROOF.json','SYNTHETIC_CPU_QA.json','SHARED_REPLAY_SCOPE.md','PHYSICAL_RUNTIME_ADMISSION_TEMPLATE.json']]+[receipt(matrix/'EXCLUSIVE_REPLACEMENT.json')]
entry={'id':'retained_link_completion_exclusive_successor','plan':receipt(matrix/'PLAN.json'),'summary':receipt(matrix/'SUMMARY.json'),'plan_hash':fingerprint(plan),'coverage':coverage}
(out/'MASTER_INPUT_ENTRY_WITH_COVERAGE.json').write_text(canonical(entry)+'\n')
manifest={'schema':'NCNC-shared-replay-source-successor-manifest-v1','files':[receipt(p) for p in sorted(out.rglob('*')) if p.is_file() and not any(part in ('fabricated_CPU_data','synthetic_shared_store','__pycache__') for part in p.relative_to(out).parts)],
 'scientific_fits':0,'actual_optimizer_maps':0,'source_qualified_fullinput_epoch1_proof_not_a_donor':True,'GO':False}
(out/'MANIFEST.json').write_text(canonical(manifest)+'\n')
for p in out.rglob('*'):
 if p.is_file():p.chmod(0o444)
print(json.dumps({'entry':entry,'manifest':receipt(out/'MANIFEST.json'),'summary':summary,'synthetic_QA':qa,'integration_source_receipts':integration,'scientific_fits':0,'GO':False}))
'''


def main():
    files=[]
    mapping={'link_prototypes.py':'runtime/adapters/link_prototypes.py','shared_source_replay.py':'runtime/adapters/shared_source_replay.py',
             'shared_replay_store.py':'runtime/shared_replay_store.py','check_synthetic.py':'check_synthetic.py',
             'LEARNING_LAW_AST_PROOF.json':'LEARNING_LAW_AST_PROOF.json','SHARED_REPLAY_SCOPE.md':'SHARED_REPLAY_SCOPE.md',
             'prepare_allocation.py':'PREPARATION_SOURCE.py'}
    for name,relative in mapping.items():
        path=HERE/name;raw=path.read_bytes()
        if path.suffix=='.py':ast.parse(raw)
        files.append({'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
    encoded=base64.b64encode(zlib.compress(json.dumps({'files':files}).encode(),9)).decode()
    for offset in range(0,len(encoded),48000):
        result=invoke(TRANSFER,[HERE.name,str(offset),encoded[offset:offset+48000]],HERE,'TRANSFER_'+str(offset),90)
        assert result['exit_code']==0
    result=invoke(REMOTE,[HERE.name,hashlib.sha256(encoded.encode()).hexdigest()],HERE,'PREPARATION',600)
    assert result['exit_code']==0,'Preserve failed packet and receipts; use separate successor'
    value=json.loads(result['stdout']);(HERE/'ALLOCATION_HANDOFF.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    for name,key in [('SYNTHETIC_CPU_QA.json','synthetic_QA'),('MASTER_INPUT_ENTRY_WITH_COVERAGE.json','entry')]:
        (HERE/name).write_text(json.dumps(value[key],indent=2,sort_keys=True)+'\n')
    print(json.dumps({'entry':value['entry'],'manifest':value['manifest'],'source_arms':value['synthetic_QA']['source_arms'],
      'eligible_shared_replay_fits':249,'excluded_native_boundary_fits':126,'scientific_fits':0,'GO':False}))

if __name__=='__main__':main()
