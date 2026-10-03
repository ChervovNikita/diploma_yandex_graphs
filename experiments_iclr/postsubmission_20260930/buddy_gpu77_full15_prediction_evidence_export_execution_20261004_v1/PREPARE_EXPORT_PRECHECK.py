"""Prepare the source-pinned metadata/byte precheck for one post-primary evidence export."""
from pathlib import Path
import base64,hashlib,json,shlex

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
REPO='/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
old=PHASE/'buddy_gpu77_postfamily_heldout_evaluation_execution_20261004_v1'
admission=(HERE/'ROOT_PREDICTION_EXPORT_ADMISSION.json').read_bytes()
assert hashlib.sha256(admission).hexdigest()=='3057f8b7cac62278ec4dfcf8a149cc41d84671cc98157c0af5ffcdbacfc88623'
review=old/'ROOT_COMPLETE_HELDOUT_REVIEW.json'
assert hashlib.sha256(review.read_bytes()).hexdigest()=='f9242c4de9e0d0e1a72bad8f76890df63de0316b85d6a98e5ff6b05e79e2e368'
source=old/'PRECHECK_REMOTE_v2.py.txt';code=source.read_text()
code=code.replace(base64.b64encode((old/'ROOT_EVALUATION_ADMISSION.json').read_bytes()).decode(),base64.b64encode(admission).decode())
code=code.replace('48c4eb06863688d73e8b4c68458b2d494db1b615c7612486794c46013510125c','3057f8b7cac62278ec4dfcf8a149cc41d84671cc98157c0af5ffcdbacfc88623')
code=code.replace('ROOT_EVALUATION_ADMISSION.json','ROOT_PREDICTION_EXPORT_ADMISSION.json')
code=code.replace("here/'root_eval_v1'","here/'root_predictions_v1'").replace('root_eval_supervision_20261004_v1','root_predictions_supervision_20261004_v1')
code=code.replace("p/'EVALUATION_CLAIM.json',p/'EVALUATION_RECEIPT.json'","p/'EXPORT_CLAIM.json',p/'PREDICTIONS_RECEIPT.json'")
assert code.count('e.deny_existing_test_artifacts()')==1
code=code.replace('e.deny_existing_test_artifacts()',"assert e.TEST_SPLIT.is_file() and (e.CACHE/'test.pt').is_file() and (e.CACHE/'test_manifest.json').is_file()")
begin=code.index('evaluation_identity=dict(')
end=code.index('assert safe(e.ARCHIVE)',begin)
block=r'''
primary_path=e.EVAL_DIR/'EVALUATION_RECEIPT.json'
assert sha(primary_path)=='ac8d9e45ac87d8e8254fb3d1580e38d0884af7d675641932d59e9101b3447727'
assert sha(e.ADMISSION)=='48c4eb06863688d73e8b4c68458b2d494db1b615c7612486794c46013510125c'
primary=read(primary_path)
e.matches(primary,dict(schema='buddy77-postfamily-evaluation-receipt-v2',**identity,
 status='all15_locked_cells_scored_once',scoring_exit_code=0,scoring_argv=e.score_command(),test_payload_opened=True,
 scoring_attempted=True,other_jobs_stopped=False,family_lock_sha256=sha(e.LOCK),lock_audit_sha256=sha(e.LOCK_AUDIT)),'Reviewed complete primary evaluation')
assert len(primary['final_results'])==15
primary_rows={(row['arm'],row['seed']):row for row in primary['final_results']}
assert len(primary_rows)==15 and set(primary_rows)=={(row['arm'],row['seed']) for row in locked_rows}
bindings=[];primary_results=[]
for row in locked_rows:
 path=Path(row['run_directory'])/'final_test.json';digest=sha(path)
 assert digest==primary_rows[(row['arm'],row['seed'])]['result_sha256']
 result=read(path)
 e.matches(result,dict(arm=row['arm'],seed=row['seed'],family_lock_sha256=sha(e.LOCK),
  test_manifest_sha256=sha(e.CACHE/'test_manifest.json'),checkpoint_sha256=row['selected_checkpoint_sha256']),'Primary scalar result')
 guards.finite_number(result['hits50'],'Primary Hits50',upper=1.0)
 bindings.append(dict(arm=row['arm'],seed=row['seed'],result_sha256=digest));primary_results.append(descriptor(path))
test_manifest=read(e.CACHE/'test_manifest.json')
assert sha(e.CACHE/'test.pt')==test_manifest['test_sha256']=='011612668871657d4d3fff6dd6259c9da89d8d434cf1efc1f01cdf3748d977f1'
assert sha(e.CACHE/'test_manifest.json')=='442d5f4a6d861da6b97431f49197412df3d4ac957cb0769df61327dbd7e14a9a'
e.matches(primary['final_cache_qualification'],dict(test_manifest_sha256=sha(e.CACHE/'test_manifest.json'),
 test_cache_sha256=test_manifest['test_sha256'],full_candidate_order_and_shapes_qualified=True,test_topology_added_to_graph=False),'Primary final cache')
export_identity=dict(**identity,family_lock_sha256=sha(e.LOCK),lock_audit_sha256=sha(e.LOCK_AUDIT),
 primary_evaluation_receipt_sha256=sha(primary_path),primary_evaluation_admission_sha256=sha(e.ADMISSION),
 test_manifest_sha256=sha(e.CACHE/'test_manifest.json'),test_cache_sha256=test_manifest['test_sha256'],
 primary_result_bindings=bindings,replay_GPU_UUID='GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998')
x=load(here/'export_predictions77.py','sealed_export_admission_metadata_only')
x.runtime_mode_gate(payload)
x.matches(payload,dict(export_identity,schema='buddy77-prediction-export-admission-v2',decision='admitted',
 family_cells=15,export_once=True,extra_forward_per_cell=1,no_new_training=True,no_new_selection=True,
 other_jobs_stopped=False,prior_partial_fits_excluded=True),'Separate root prediction-export admission')
assert isinstance(payload['root_export_cost_decision'],str) and payload['root_export_cost_decision'].strip()
assert payload['root_independent_complete_heldout_review_sha256']=='f9242c4de9e0d0e1a72bad8f76890df63de0316b85d6a98e5ff6b05e79e2e368'
'''
code=code[:begin]+block+code[end:]
code=code.replace("evaluation_identity['scoring_GPU_UUID']","export_identity['replay_GPU_UUID']")
code=code.replace('identity=evaluation_identity','identity=export_identity')
code=code.replace("lock=descriptor(e.LOCK),lock_audit=descriptor(e.LOCK_AUDIT),audit_physical_terminal=descriptor(physical),", "lock=descriptor(e.LOCK),lock_audit=descriptor(e.LOCK_AUDIT),audit_physical_terminal=descriptor(physical),\n primary_evaluation=descriptor(primary_path),primary_admission=descriptor(e.ADMISSION),primary_results=primary_results,\n test_manifest=descriptor(e.CACHE/'test_manifest.json'),test_cache=descriptor(e.CACHE/'test.pt'),")
code=code.replace('buddy77_heldout_evaluation_metadata_precheck_v1','buddy77_prediction_export_metadata_precheck_v1')
code=code.replace('PASS_NO_EXISTING_EVALUATION_OR_TEST_ATTEMPT','PASS_FULL15_PRIMARY_NO_EXISTING_EXPORT_ATTEMPT')
code=code.replace('current_Torch_runtime_will_be_verified_once_by_existing_evaluation','current_Torch_runtime_will_be_verified_once_by_existing_export')
code=code.replace('TEST_payload_access=False','TEST_payload_bytes_hashed=True,TEST_payload_deserialized=False')
compile(code,'export_metadata_and_byte_precheck','exec')
with (HERE/'PRECHECK_REMOTE_v1.py.txt').open('x') as h:h.write(code)
command=PHASE/'gpu77_connection_recovery_v1/buddy_v4_prediction_export_precheck_20261004_v1_command.txt'
with command.open('x') as h:h.write(shlex.join(['/usr/bin/python3','-I','-S','-B','-c','import os;os.chdir('+repr(REPO)+');\n'+code])+'\n')
receipt=dict(schema='buddy77_export_precheck_preparation_v1',source_template_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 remote_source_sha256=hashlib.sha256(code.encode()).hexdigest(),command_sha256=hashlib.sha256(command.read_bytes()).hexdigest(),
 admission_sha256=hashlib.sha256(admission).hexdigest(),source_compilation='PASS',remote_executed=False)
with (HERE/'PRECHECK_PREPARATION.json').open('x') as h:json.dump(receipt,h,indent=2);h.write('\n')
print(json.dumps(receipt))
