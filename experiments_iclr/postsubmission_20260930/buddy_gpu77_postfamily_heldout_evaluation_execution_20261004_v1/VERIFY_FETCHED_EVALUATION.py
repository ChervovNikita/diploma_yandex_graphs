"""Verify full15 scalar closure and measured receipts without making predictive claims."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
fetch=json.loads((HERE/'MONITOR_RESULT_01.json').read_text())
precheck=json.loads((HERE/'PRECHECK_RESULT.json').read_text())
admission=json.loads((HERE/'ROOT_EVALUATION_ADMISSION.json').read_text())
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def finite(value):return type(value) in (int,float) and math.isfinite(value) and value>=0
assert fetch['status']=='ALL15_PHYSICALLY_COMPLETE' and fetch['all15_closure_established'] is True
assert fetch['final_scalar_files_present']==15 and fetch['physical_exit_code']==0
assert not fetch['missing'] and not fetch['oversized'] and len(fetch['files'])==29
assert all(row['observation']=='absent' for row in fetch['owned_handles'])
index={row['path'].removeprefix('experiments_iclr/postsubmission_20260930/'):row for row in fetch['files']}
for row in fetch['files']:
 path=HERE/row['local_path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
def local(relative):return HERE/index[relative]['local_path']
packet='buddy_gpu77_postfamily_eval_preparation_v4/'
supervision=packet+'root_eval_supervision_20261004_v1/'
output=packet+'root_eval_v1/'
physical=read(local(supervision+'PHYSICAL_TERMINAL.json'))
launch=read(local(supervision+'DETACHED_LAUNCH.json'))
runner=read(local(supervision+'RUNNER_STARTED.json'))
child=read(local(supervision+'CHILD_STARTED.json'))
claim=read(local(output+'EVALUATION_CLAIM.json'))
evaluation=read(local(output+'EVALUATION_RECEIPT.json'))
manifest_relative='buddy_complete_data_cache_preparation_v3/root_run_77_v1/cache/test_manifest.json'
test_manifest=read(local(manifest_relative));test_manifest_sha=sha(local(manifest_relative))
admission_sha=sha(HERE/'ROOT_EVALUATION_ADMISSION.json')
assert admission_sha=='48c4eb06863688d73e8b4c68458b2d494db1b615c7612486794c46013510125c'
assert sha(local(packet+'ROOT_EVALUATION_ADMISSION.json'))==admission_sha
assert evaluation['evaluation_admission_sha256']==claim['evaluation_admission_sha256']==physical['ROOT_EVALUATION_ADMISSION_sha256']==runner['ROOT_EVALUATION_ADMISSION_sha256']==admission_sha
assert launch['admission']['sha256']==admission_sha
assert sha(local(supervision+'run_heldout_evaluation_once.py'))==sha(HERE/'DETACHED_RUNNER_SOURCE.py')=='f74a141b9e4d8302b0d3456aa232255228d9ef29c3e704076c87bd6f3c365d06'
assert launch['runner_handle']['pid']==runner['identity']['pid']==3179181
assert launch['runner_handle']['start_ticks']==runner['identity']['start_ticks']==1721480102
assert child['identity']['pid']==3179184 and child['identity']['start_ticks']==1721480105
assert physical['argv']==child['argv']==child['identity']['argv']
assert physical['argv'][3:5]==['evaluate','--execute']
assert physical['exit_code']==evaluation['scoring_exit_code']==0
assert physical['evaluation_once'] is True and physical['no_retry'] is True and physical['official_TEST_access_authorized'] is True
assert physical['other_jobs_stopped'] is False and physical['namespace_isolation_used'] is False
assert physical['overlapping_internal_evaluation_intervals_not_added'] is True
assert evaluation['status']=='all15_locked_cells_scored_once' and evaluation['test_payload_opened'] is True and evaluation['scoring_attempted'] is True
assert evaluation['other_jobs_stopped'] is False and evaluation['original_scores_changed'] is False and evaluation['scientific_advantage_claimed'] is False
assert claim['status']=='in_progress' and claim['test_payload_opened'] is False and claim['scoring_attempted'] is False
assert all(evaluation[key]==value and claim[key]==value and admission[key]==value for key,value in precheck['identity'].items())
prior=PHASE/'buddy_gpu77_postfamily_lock_audit_execution_20261004_v1/fetched_registered_outputs_v1/root_lock_v1'
lock=read(prior/'FAMILY_LOCK.json')
assert sha(prior/'FAMILY_LOCK.json')==evaluation['family_lock_sha256']=='4d4041ad0d02a36c94bd9112f4e01029fb722eee75339a5743a0259435c33fd9'
assert sha(prior/'LOCK_AUDIT.json')==evaluation['lock_audit_sha256']=='2c9d8b13759c88cfb657782c3e6a22938388e91f171e8a67ab49616c9b392b38'
assert lock['runs']==precheck['locked_rows'] and len(lock['runs'])==15
assert len(evaluation['final_results'])==15
assert {(row['arm'],row['seed']) for row in evaluation['final_results']}=={(row['arm'],row['seed']) for row in lock['runs']}
receipt_results={(row['arm'],row['seed']):row for row in evaluation['final_results']}
scalars=[]
for row in lock['runs']:
 relative=str(Path(row['run_directory']).relative_to('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930')/'final_test.json')
 path=local(relative);result=read(path);receipt=receipt_results[(row['arm'],row['seed'])]
 assert result['arm']==row['arm'] and result['seed']==row['seed']
 assert result['family_lock_sha256']==evaluation['family_lock_sha256'] and result['test_manifest_sha256']==test_manifest_sha
 assert result['checkpoint_sha256']==row['selected_checkpoint_sha256']
 assert sha(path)==receipt['result_sha256'] and result['prediction_seconds']==receipt['prediction_seconds']
 assert finite(result['hits50']) and result['hits50']<=1 and finite(result['prediction_seconds'])
 scalars.append(dict(**result,result_sha256=sha(path),fetched_path=str(path.relative_to(HERE))))
assert sum(row['prediction_seconds'] for row in scalars)==evaluation['summed_prediction_seconds']
score_lines=local(output+'score.log').read_text().splitlines()
logged=[json.loads(line) for line in score_lines if line.startswith('{"arm":')]
assert len(logged)==15 and not any(row.get('existing_result_reused') for row in logged)
assert logged==[{key:row[key] for key in ('arm','seed','hits50','prediction_seconds','family_lock_sha256','test_manifest_sha256','checkpoint_sha256')} for row in scalars]
for name in ('CHILD.stdout.log','CHILD.stderr.log','RUNNER.stdout.log','RUNNER.stderr.log'):
 assert local(supervision+name).stat().st_size==0
official=evaluation['official_test_qualification'];final=evaluation['final_cache_qualification'];staging=evaluation['official_staging']
assert official['positive_rows']==46329 and official['negative_rows']==100000
assert official['combined_split_accessor_called'] is False and staging['other_archive_members_extracted'] is False
assert staging['archive_sha256']==evaluation['archive_sha256']==admission['archive_sha256']
assert staging['member']==evaluation['archive_member']=='collab/split/time/test.pt' and staging['bytes']==3404768
assert final['full_candidate_order_and_shapes_qualified'] is True and final['test_topology_added_to_graph'] is False
assert final['test_manifest_sha256']==test_manifest_sha and final['test_cache_sha256']==test_manifest['test_sha256']
assert test_manifest['cache_manifest_sha256']==evaluation['cache_manifest_sha256']
assert test_manifest['family_lock_sha256']==evaluation['family_lock_sha256'] and test_manifest['graph_policy']=='training_only_all_splits'
assert test_manifest['official_test_file_sha256']==staging['official_test_file_sha256']
assert all(test_manifest[key]==official[key] for key in ('positive_sha256','negative_sha256','pair_order_sha256'))
assert all(evaluation['training_costs'][key]==value for key,value in precheck['training_costs'].items())
assert evaluation['training_costs']['prior_partial_fits_excluded_from_scientific_totals'] is True
assert 0<evaluation['summed_prediction_seconds']<evaluation['scoring_subprocess_seconds']<evaluation['total_evaluation_wrapper_seconds']<physical['physical_wrapper_wall_seconds']
timing_keys=('selected_checkpoint_revalidation_seconds','archive_verification_and_test_extraction_seconds',
 'official_test_qualification_seconds','production_finalize_seconds','final_cache_qualification_seconds',
 'scoring_subprocess_seconds','summed_prediction_seconds','total_evaluation_wrapper_seconds')
assert all(finite(evaluation[key]) for key in timing_keys)
scalar_packet=dict(schema='buddy77_all15_fixed_selected_heldout_scalar_results_v1',UTC=datetime.now(timezone.utc).isoformat(),
 family_cells=15,optimizer_fits=24,completed_cell_epochs=1500,results=scalars,
 fixed_family_lock_sha256=evaluation['family_lock_sha256'],test_manifest_sha256=test_manifest_sha,
 all15_physical_and_stage_closure_verified=True,predictive_conclusions_pending_root_review=True,
 training_selection_unchanged=True,scientific_advantage_claimed=False)
with (HERE/'ALL15_SCALAR_RESULTS.json').open('x') as h:json.dump(scalar_packet,h,indent=2);h.write('\n')
verification=dict(schema='buddy77_full15_heldout_fetched_local_verification_v1',UTC=datetime.now(timezone.utc).isoformat(),status='PASS',
 family_cells=15,optimizer_fits=24,completed_cell_epochs=1500,all15_results_bind_exact_locked_checkpoints=True,
 all15_result_hashes_match_stage_receipt=True,all15_unchanged_logged_once=True,physical_exit_code=0,
 no_evaluation_retry=True,no_export=True,no_refit=True,no_reselection=True,scientific_source_changed=False,
 official_positive_rows=official['positive_rows'],official_negative_rows=official['negative_rows'],train_only_graph=True,
 physical_costs=physical,launch_client_costs={key:value for key,value in launch.items() if key.startswith('launch_client_')},
 successful_metadata_precheck_wall_seconds=precheck['precheck_wall_seconds'],
 successful_metadata_precheck_peak_RSS_bytes=precheck['precheck_peak_RSS_bytes'],
 prior_failed_metadata_adapter_cost='Not separately instrumented; receipt/source preserved. No evaluation or TEST opening occurred.',
 internal_evaluation_timings={key:evaluation[key] for key in timing_keys},
 nested_and_launch_tail_intervals_overlap_physical_wall_and_are_not_added=True,
 peak_RSS_scope_clarification='Owned evaluate child and its descendants, measured by RUSAGE_CHILDREN. Inherited receipt prose says audit descendants.',
 training_costs=evaluation['training_costs'],owned_handles=fetch['owned_handles'],all_fetched_descriptors=fetch['files'],
 EVALUATION_RECEIPT_sha256=sha(local(output+'EVALUATION_RECEIPT.json')),
 PHYSICAL_TERMINAL_sha256=sha(local(supervision+'PHYSICAL_TERMINAL.json')),
 all15_scalar_packet_sha256=sha(HERE/'ALL15_SCALAR_RESULTS.json'),predictive_conclusions_pending_root_review=True)
with (HERE/'LOCAL_VERIFICATION_RESULT.json').open('x') as h:json.dump(verification,h,indent=2);h.write('\n')
print(json.dumps({key:verification[key] for key in ('status','family_cells','optimizer_fits','completed_cell_epochs','physical_exit_code',
 'official_positive_rows','official_negative_rows','EVALUATION_RECEIPT_sha256','PHYSICAL_TERMINAL_sha256','all15_scalar_packet_sha256')}))
