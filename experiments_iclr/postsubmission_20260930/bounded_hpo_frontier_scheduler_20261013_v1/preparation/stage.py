"""Guarded synthetic scheduler checks and prior-matrix metadata DAG audit."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import zlib

OUT = Path(__file__).resolve().parents[1]
PHASE = OUT.parent
spec = importlib.util.spec_from_file_location('guarded_transport', PHASE / 'training_pipeline_v2_20261012/preparation/prepare_molecular_packets_allocation.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
transport = OUT / 'transport'
transport.mkdir(exist_ok=True)
files = []
for name in ('runtime/hpo_frontier.py', 'amend_controller.py', 'tests/check_frontier.py', 'preparation/stage.py'):
    raw = (OUT / name).read_bytes()
    files.append({'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'data': base64.b64encode(raw).decode()})
reference = PHASE / 'final_resource_budget_20261013_v1/unused_search_retention_successor_20261013_v1/runtime/unused_search_retention.py'
raw = reference.read_bytes()
files.append({'path': 'source_reference/unused_search_retention.py', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'data': base64.b64encode(raw).decode()})
code = helper.GUARD + r'''
import gzip,collections,importlib.util
payload=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])));stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
base=phase/'per_gpu_command_preparation_20261013_v1/runtime';shutil.copytree(base,stage/'runtime')
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts;raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];q=stage/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');records=[]
for label,args in [('NARROW_CONTROLLER_AMEND',[str(stage/'amend_controller.py'),str(stage/'runtime')]),('ACTUAL_REGISTRY_LAUNCHER_SYNTHETIC_QA',[str(stage/'tests/check_frontier.py'),'--controller',str(stage/'runtime')]),('COMPANION_REGRESSION_QA',[str(phase/'per_gpu_command_preparation_20261013_v1/tests/check_gpu_commands.py'),'--controller',str(stage/'runtime')])]:
 tick=time.monotonic();r=subprocess.run([str(repo/'.venv/bin/python'),'-B',*args],cwd=stage,env=env,capture_output=True,text=True,timeout=30);record={'label':label,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'wall_seconds':time.monotonic()-tick,'fits':0,'GO':False};(stage/(label+'_RESULT.json')).write_text(json.dumps(record,indent=2,sort_keys=True)+'\n');records.append(record)
 if r.returncode:print(json.dumps({'records':records}));sys.exit(1)
sys.dont_write_bytecode=True;sys.path.insert(0,str(stage/'runtime'))
from hpo_frontier import declare
spec=importlib.util.spec_from_file_location('reviewed_unused_search_reference',stage/'source_reference/unused_search_retention.py');retention=importlib.util.module_from_spec(spec);spec.loader.exec_module(retention)
mp=phase/'massive_practical_master_20261013_v24/PLAN.json';raw=mp.read_bytes();plan=json.loads(raw);canon=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode();assert hashlib.sha256(canon(plan)).hexdigest()=='8f691a4d7b9d833595d531b75b7cb78a3cba0cdb06d00015a6e15355e1883957'
placement={key:{'worker':group['gpu_assignment']} for group in plan['groups'] for key in group['run_keys']};studies={};runs={}
for row in plan['studies']:
 q=phase/row['path'];raw=q.read_bytes();study=json.loads(gzip.decompress(raw) if q.name.endswith('.gz') else raw);assert hashlib.sha256(canon(study)).hexdigest()==row['spec_hash']
 study['release']['checkpoint_retention_policy']['unused_search_binary_release']=retention.declare(phase,study);studies[study['study_id']]=study
 for run in study['runs']:runs[study['study_id']+'/'+run['run_id']]={'study':study['study_id'],'run':run}
policy=declare(runs,placement,studies);audit={'schema':'actual-v24-metadata-only-HPO-DAG-prospective-declaration-proof-v1','prior_plan_hash':'8f691a4d7b9d833595d531b75b7cb78a3cba0cdb06d00015a6e15355e1883957','cohorts':len(policy['cohorts']),'search_fit_members':len(policy['fit_memberships']),'multiple_cohort_memberships':sum(len(v)>1 for v in policy['fit_memberships'].values()),'maximum_open_cohorts_per_GPU':2,'largest_cohort_fit_parent_records_per_worker':policy['largest_cohort_fit_parent_records_per_worker'],'maximum_two_cohort_fit_parent_records_per_worker':policy['maximum_two_cohort_fit_parent_records_per_worker'],'cohort_policy_canonical_sha256':hashlib.sha256(canon(policy)).hexdigest(),'only_prospective_study_copies_amended_in_memory':True,'old_PLAN_and_studies_unchanged':True,'actual_registry_opened':False,'actual_arrays_decoded':0,'models':0,'updates':0,'fits':0,'GO':False};(stage/'ACTUAL_V24_METADATA_DAG_AUDIT.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
def receipt(q):
 raw=q.read_bytes();return {'path':str(q.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
manifest={'schema':'bounded-HPO-scheduler-preparation-v1','base_controller_manifest':receipt(base.parent/'MANIFEST.json'),'files':[receipt(q) for q in sorted((stage/'runtime').rglob('*')) if q.is_file()],'helper':receipt(stage/'runtime/hpo_frontier.py'),'amender':receipt(stage/'amend_controller.py'),'metadata_DAG_audit':receipt(stage/'ACTUAL_V24_METADATA_DAG_AUDIT.json'),'retention_validator_reference':receipt(stage/'source_reference/unused_search_retention.py'),'source_gates_learning_budgets_assignments_unchanged':True,'metadata_and_synthetic_only':True,'scientific_fits':0,'GO':False};(stage/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
for q in stage.rglob('*'):
 if q.is_file():q.chmod(0o444)
artifacts=[]
for name in ('MANIFEST.json','NARROW_CONTROLLER_AMEND_RESULT.json','ACTUAL_REGISTRY_LAUNCHER_SYNTHETIC_QA_RESULT.json','COMPANION_REGRESSION_QA_RESULT.json','ACTUAL_V24_METADATA_DAG_AUDIT.json'):
 q=stage/name;raw=q.read_bytes();artifacts.append({**receipt(q),'name':name,'data':base64.b64encode(raw).decode()})
print(json.dumps({'manifest':receipt(stage/'MANIFEST.json'),'helper':manifest['helper'],'amender':manifest['amender'],'metadata_DAG_audit':audit,'artifacts':artifacts}))
'''
payload = base64.b64encode(zlib.compress(json.dumps({'destination': OUT.name, 'files': files}).encode(), 9)).decode()
result = helper.invoke(code, [payload], transport, 'STAGE_BOUNDED_HPO_FRONTIER_METADATA_SYNTHETIC_ATTEMPT1', 120)
assert result['exit_code'] == 0, result['stdout']
value = json.loads(result['stdout'].strip())
for row in value.pop('artifacts'):
    raw = base64.b64decode(row['data'])
    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    target = OUT / row['name']
    assert not target.exists()
    target.write_bytes(raw)
    target.chmod(0o444)
(OUT / 'ALLOCATION_STAGE_RESULT.json').write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
print(json.dumps(value, sort_keys=True))
