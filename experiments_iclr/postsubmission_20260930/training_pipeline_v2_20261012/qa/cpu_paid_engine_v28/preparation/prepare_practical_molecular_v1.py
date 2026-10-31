"""Re-enumerate a finite molecular study from reused, hashed real custody."""
import json
from pathlib import Path

from prepare_molecular_packets_allocation import GUARD, invoke

HERE = Path(__file__).resolve().parent / 'practical_molecular_20261012_v1'
REMOTE = GUARD + r'''
import gzip,importlib
stage=phase/'practical_molecular_packet_20261012_v1';assert not stage.exists();stage.mkdir()
qa=phase/'training_pipeline_v2_20261012/qa/cpu_molecular_practical_v23'
runtime=stage/'runtime';runtime.mkdir()
inventory=json.loads((qa/'SOURCE_INVENTORY.json').read_text())
for row in inventory:
 raw=(qa/row['path']).read_bytes()
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 target=runtime/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(raw);target.chmod(0o444)
def save(path,value):
 path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def receipt(path):
 raw=path.read_bytes()
 return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
save(stage/'SOURCE_INVENTORY.json',inventory)
qa_result=json.loads((qa/'RESULT.json').read_text());assert qa_result['exit_code']==0
save(stage/'CPU_QA_RESULT.json',qa_result)
policy={'schema':'molecular-practical-policy-v1','datasets':['ogbg-molhiv','ogbg-molbace'],
 'epoch_domains':[300],'coefficient_policy':'native_fixed',
 'numeric_search_points':[{'width':256,'depth':5,'weight_decay':0,'learning_rate':lr,'dropout':dropout}
   for lr in [.00025,.001,.004] for dropout in [.1,.3,.5]]}
save(stage/'PRACTICAL_POLICY.json',policy)
ancestor=phase/'massive_molecular_packet_20261012_v4/matrix'
oldplan=json.loads((ancestor/'PLAN.json').read_text());custodies=[]
for row in oldplan['studies']:
 original=json.loads(gzip.decompress((phase/row['path']).read_bytes()))
 if original['runs'][0]['budget']['maximum']!=300:continue
 train=next(r['config']['molecular_inputs']['train']['path'] for r in original['runs'] if r['kind']=='model_fit')
 manifest=phase/Path(train).parent/'MANIFEST.json'
 assert manifest.exists()
 assert json.loads(manifest.read_text())['dataset']==original['dataset']
 custodies.append(str(manifest))
 assert original['data']['evidence']['pack_descriptives']['train']['graphs'] in (32901,1210)
assert len(custodies)==2
prior=phase/'massive_molecular_packet_20261012_v1'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
command=[str(repo/'.venv/bin/python'),'-B',str(runtime/'compile_molecular_matrix.py'),
 '--project-root',str(phase),'--custody',*custodies,'--resources',str(prior/'RESOURCE_ESTIMATES.json'),
 '--exposure-decisions',str(prior/'EXPOSURE_DECISIONS.json'),'--host-bindings',str(prior/'HOST_BINDINGS.json'),
 '--output',str(stage/'base_matrix'),'--engine-prefix',str(runtime),
 '--epochs','300','--scalar-coefficients','native_fixed','--practical-policy',str(stage/'PRACTICAL_POLICY.json')]
tick=time.monotonic();result=subprocess.run(command,cwd=runtime,env=env,capture_output=True,text=True,timeout=300)
save(stage/'BASE_COMPILER_RESULT.json',{'command':command,'exit_code':result.returncode,'stdout':result.stdout,
 'stderr':result.stderr,'wall_seconds':time.monotonic()-tick,'scientific_fits':0})
assert result.returncode==0,result.stderr
sys.path.insert(0,str(runtime))
compiler=importlib.import_module('compile_molecular_boundary_matrix')
master=importlib.import_module('compile_master_plan')
baseplan=json.loads((stage/'base_matrix/PLAN.json').read_text())
bound=compiler.receipt(phase,'common_wrapper_native_source_20261010_v1/models.py')
assert (bound['bytes'],bound['sha256'])==compiler.BOUNDARY_PIN
native=compiler.receipt(phase,'source_snapshot/run_tabm.py')
assert (native['bytes'],native['sha256'])==compiler.NATIVE_TRAIN_PIN
source_files=[receipt(path) for path in sorted(runtime.rglob('*')) if path.is_file()]
source_files+=[receipt(stage/'PRACTICAL_POLICY.json'),receipt(stage/'CPU_QA_RESULT.json'),native,
 receipt(ancestor/'PLAN.json'),receipt(ancestor/'SUMMARY.json')]
prepared=[]
for row in baseplan['studies']:
 raw=(phase/row['path']).read_bytes();original=json.loads(gzip.decompress(raw))
 assert compiler.fingerprint(original)==row['spec_hash']
 prepared.append(compiler.append_boundary_reference(original,str(runtime.relative_to(phase)),bound,
  source_files+[receipt(phase/row['path'])],9))
output=stage/'matrix';output.mkdir();(output/'studies').mkdir()
plan=compiler.make_plan(prepared,str(output.relative_to(phase)),baseplan['host_bindings'])
for value,*_ in prepared:
 path=output/'studies'/(value['study_id']+'.json.gz')
 path.write_bytes(gzip.compress((compiler.canonical(value)+'\n').encode(),mtime=0))
 plan['studies'].append({'path':str(path.relative_to(phase)),'spec_hash':compiler.fingerprint(value)})
summary={'schema':'practical-molecular-disabled-v1','plan_hash':compiler.fingerprint(plan),
 'counts':master.typed_counts([value for value,*_ in prepared]),
 'per_study':[{'study_id':value['study_id'],'counts':detail,'boundary_banks':banks,'boundary_domain':domain}
   for value,detail,banks,domain in prepared],
 'policy':policy,'unrestricted_v4_unchanged':True,'data_reused_by_hash':True,
 'model_fits':450,'known_model_optimizer_calls':18090000,'scientific_training_runs':0,
 'actual_model_fits':0,'actual_predictive_scores':0,'GO_receipt_created':False,
 'resource_status':'PROVISIONAL_UNMEASURED_SOURCE_SHAPE_REQUIRES_ENGINEERING_QUALIFICATION'}
assert summary['counts']['totals']['atomic_fit_run_groups']==450
assert summary['counts']['model_optimizer_step_call_total']==18090000
save(output/'PLAN.json',plan);save(output/'SUMMARY.json',summary)
entry={'id':'practical_molecular_v1','plan':receipt(output/'PLAN.json'),'summary':receipt(output/'SUMMARY.json')}
packet=master.load_packet(phase,entry)
save(stage/'MASTER_INPUT_ENTRY.json',entry)
verification={'packet_loaded':True,'counts':packet['counts'],'references':packet['references'],
 'scientific_fits':0,'role_arrays_opened':0}
save(stage/'MASTER_INGEST_VERIFICATION.json',verification)
print(json.dumps({'entry':entry,'summary':summary,'master_ingest':verification}))
'''


def main():
    assert not HERE.exists()
    HERE.mkdir()
    transport = invoke(REMOTE, [], HERE, 'PREPARATION_TRANSPORT', 600)
    if transport['exit_code'] != 0:
        raise RuntimeError('Practical molecular preparation failed; transport retained')
    value = json.loads(transport['stdout'])
    for name, item in [('MASTER_INPUT_ENTRY.json', value['entry']), ('SUMMARY.json', value['summary']),
                       ('MASTER_INGEST_VERIFICATION.json', value['master_ingest'])]:
        (HERE/name).write_text(json.dumps(item, indent=2) + '\n')
    print(json.dumps({'prepared': True, 'fits': value['summary']['model_fits'],
                      'calls': value['summary']['known_model_optimizer_calls'], 'training': False}))


if __name__ == '__main__':
    main()
