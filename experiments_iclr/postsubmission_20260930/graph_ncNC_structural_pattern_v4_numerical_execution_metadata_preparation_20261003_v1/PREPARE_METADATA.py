"""Author disabled numerical execution metadata and path-bound support sources.

This utility uses stdlib file/hash/AST processing only. Its author is the V4
model-source reviewer; this is not an independent review of the new helper.
"""
from pathlib import Path
import ast, copy, datetime, difflib, hashlib, json

P=Path(__file__).resolve().parent
R=P.parent
OLD=R/'graph_ncNC_structural_pattern_normal_supervision_preparation_20261003_v2'
SUP=R/'graph_ncNC_structural_pattern_normal_supervision_preparation_20261003_v3'
DRIVER=R/'graph_ncNC_structural_pattern_pilot_preparation_20261003_v4'
MODEL_REVIEW=R/'graph_ncNC_structural_pattern_v4_independent_source_review_20261003_v1'
EXECUTION_NAME='graph_ncNC_structural_pattern_numerical_execution_root_20261003_v4'
REPO='/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
PHASE=REPO+'/experiments_iclr/postsubmission_20260930'
EXECUTION=PHASE+'/'+EXECUTION_NAME
PYTHON='/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12'
UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p): return {'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)}
def save(d,n,x): (d/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def verify(d,pin):
    assert sha(d/'MANIFEST.json')==pin
    m=json.loads((d/'MANIFEST.json').read_text())
    for x in m['files']:
        f=(d/x['path']).resolve();assert f.is_relative_to(d.resolve())
        assert sha(f)==x['sha256'] and f.stat().st_size==x.get('bytes',x.get('size'))
    return m
def seal(d,status):
    files=sorted(f for f in d.rglob('*') if f.is_file() and f.name not in ['MANIFEST.json','SEAL.json'])
    payload=[{'path':str(f.relative_to(d)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in files]
    save(d,'MANIFEST.json',{'schema':'ncnc_V4_disabled_numerical_preparation_manifest_v1','UTC':UTC,
         'status':status,'files':payload,'payload_count':len(payload),'payload_bytes':sum(x['bytes'] for x in payload)})
    for x in payload: assert sha(d/x['path'])==x['sha256']
    save(d,'SEAL.json',{'schema':'ncnc_V4_disabled_numerical_preparation_seal_v1','UTC':UTC,
         'manifest_sha256':sha(d/'MANIFEST.json'),'payload_count':len(payload),'payload_bytes':sum(x['bytes'] for x in payload),
         'execution_authorized':False,'GPU_or_project_execution':False,'independent_new_helper_review':False})
    return sha(d/'MANIFEST.json')

verify(DRIVER,'9021a598c7642bf428124de1230a078dfddbf3968095432354ee18a14541104c')
verify(OLD,'dd853261be34c61b472813d5f452d7c0fe1ecd78088da35b22cf5c4f7b4cba21')
verify(MODEL_REVIEW,'8f10818dc693e7ca4fe899de761a2ef370994cdbbe95c899e9b47f3338942357')
assert sha(MODEL_REVIEW/'SEAL.json')=='cc083ec570c81db9081ee5bea2d78cf0e08aa12a7102fc4e32820e706a54b363'
assert not SUP.exists() and not (R/EXECUTION_NAME).exists()
SUP.mkdir()
changes={
 'supervise_numerical.py':[
  ('graph_ncNC_structural_pattern_numerical_execution_root_20261003_v2',EXECUTION_NAME),
  ('graph_ncNC_structural_pattern_pilot_preparation_20261003_v3','graph_ncNC_structural_pattern_pilot_preparation_20261003_v4'),
  ('Only independently repaired V3 may be released','Only separately reviewed V4 may be released')],
 'numerical_child.py':[
  ('graph_ncNC_structural_pattern_pilot_preparation_20261003_v3','graph_ncNC_structural_pattern_pilot_preparation_20261003_v4')]}
diff=[]
for name,replacements in changes.items():
    before=(OLD/name).read_text();after=before
    for old,new in replacements:
        assert after.count(old)==1,(name,old,after.count(old))
        after=after.replace(old,new)
    (SUP/name).write_text(after)
    ast.parse(after,filename=str(SUP/name));compile(after,str(SUP/name),'exec')
    diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(OLD/name),tofile=str(SUP/name)))
(SUP/'V2_TO_V3_BINDING_ONLY.diff').write_text(''.join(diff))
(SUP/'CAPS.json').write_bytes((OLD/'CAPS.json').read_bytes())
caps=json.loads((OLD/'CAPS.json').read_text())['caps']
old_release=json.loads((R/'graph_ncNC_structural_pattern_numerical_execution_root_20261003_v2/ROOT_RELEASE_NUMERICAL.json').read_text())
assert caps==old_release['numerical_caps']
rt=R/'graph_ncNC_predictive_runtime_authority_root_20261003_v1/RUNTIME_AUTHORITY.json'
assert sha(rt)=='e63f602baa8a91e129fb4a0c0debd6ec282e7cd132be87eb66cd8c5cf0b7c882'
runtime=json.loads(rt.read_text())
assert runtime['interpreter_path']==PYTHON and runtime['interpreter_sha256']=='14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'
save(SUP,'BINDING_ADAPTATION.json',{'schema':'ncnc_V4_numerical_helper_author_adaptation_v1','UTC':UTC,
    'predecessor_manifest':ref(OLD/'MANIFEST.json'),'predecessor_review':ref(R/'graph_ncNC_structural_pattern_normal_supervision_v2_independent_source_review_20261003_v1/REVIEW.json'),
    'changes':changes,'caps_byte_identical':True,'interpreter_binding_unchanged':True,
    'process_supervision_and_peak_observer_logic_changed':False,
    'authoring_is_not_independent_review':True,'new_helper_independent_review_required':True,
    'source_import_or_execution':False})
save(SUP,'AUTHOR_STATIC_CHECK.json',{'schema':'author_stdlib_binding_checks_v1','UTC':UTC,
    'source_files_AST_compiled_without_execution':2,'exact_string_replacement_counts_checked':True,
    'caps_and_interpreter_unchanged':True,'independent_review':False,'launch_or_GPU_execution':False})
(SUP/'README.md').write_text('# V4 numerical support source successor\n\nSource only; authored after the independent V4 model review. This helper has not received an independent successor review.\n\nOnly the fixed execution-root/driver path bindings and one guard message differ from supervision v2. Caps, interpreter, process supervision, peak observer, route checks and ordinary-host behavior remain. Exact diff and author metadata checks are saved. These are author checks, not an independent review.\n\nA fresh disabled release descriptor is prepared separately. No source upload, root approval, numerical/full-graph execution, predictive fit or canonical edit occurred.\n')
SUP_SHA=seal(SUP,'SOURCE_ONLY_PATH_BOUND_HELPER_AUTHOR_PREPARED_INDEPENDENT_REVIEW_PENDING')

inv={'stage':'numerical','unit':'pair','base_seed':0,'output_directory':EXECUTION+'/numerical/run01'}
release=copy.deepcopy(old_release)
release.update({'example_only_NOT_AUTHORIZATION':True,'driver_manifest_sha256':sha(DRIVER/'MANIFEST.json'),
  'root_authorization_reference':None,'closure_output_directory':PHASE+'/graph_ncNC_structural_pattern_execution_root_20261003_v4/close/run01',
  'authorized_stages':[],'authorized_invocations':[],'example_invocations_NOT_AUTHORIZED':[inv],
  'supervision_manifest_sha256':SUP_SHA,
  'numerical_driver_path':PHASE+'/'+DRIVER.name+'/pattern_run.py',
  'source_review_status':'V4_MODEL_SOURCE_REVIEW_COMPLETED_NEW_HELPER_INDEPENDENT_REVIEW_AND_ROOT_APPROVAL_PENDING',
  'independent_reviews':[ref(MODEL_REVIEW/'REVIEW.json')],
  'predecessor_failure_custody':ref(DRIVER/'PRESERVED_FULL_GRAPH_FAILURES.json'),
  'source_custody':None,'source_upload_or_remote_rehash_completed':False,
  'launch_support_independent_review':None,'new_launch_support_review_required':True,
  'scientific_fits_authorized':False,'full_graph_authorized':False,
  'activation_checkpoint_source_review':{'manifest':ref(MODEL_REVIEW/'MANIFEST.json'),'seal':ref(MODEL_REVIEW/'SEAL.json')},
  'V4_numerical_parity_REQUIRED':True,'historical_V3_numerical_PASS_is_current_admission':False,
  'execution_root':EXECUTION,'supervision_output':EXECUTION+'/supervision/run01',
  'prepared_source_only_no_root_release':True,'predecessor_original_identity_and_costs_retained':True})
# Current-driver prior failures stay separate from cross-version history.
release['prior_failure_receipts']=[]
save(P,'ROOT_RELEASE_NUMERICAL_CANDIDATE.json',release)
save(P,'ROOT_NUMERICAL_ADMISSION_CANDIDATE.json',{'schema':'ncnc_pattern_V4_root_numerical_admission_candidate_v1','UTC':UTC,
    'decision':'DISABLED_PENDING_ROOT_INSPECTION_HELPER_REVIEW_AND_EXPLICIT_APPROVAL','authorization_reference':None,
    'candidate_driver_manifest_sha256':sha(DRIVER/'MANIFEST.json'),'candidate_supervision_manifest_sha256':SUP_SHA,
    'candidate_invocation':inv,'caps':caps,'selected_GPU':release['cuda_visible_devices'],
    'interpreter':{'path':PYTHON,'sha256':runtime['interpreter_sha256']},
    'new_helper_independent_review':None,'fresh_remote_source_custody':None,'fresh_physical_resource_admission':None,
    'ordinary_host_execution':True,'automatic_retry':False,'scientific_fits_authorized':False,
    'full_graph_authorized':False,'source_authorization_from_prior_release_carried_forward':False})

history=[]
for version in ['v1','v2']:
    folder=R/f'graph_ncNC_structural_pattern_numerical_execution_root_20261003_{version}'/'remote_receipts_run01'
    terminal=folder/'supervision/run01/SUPERVISOR_TERMINAL.json'
    t=json.loads(terminal.read_text());peaks=t['CUDA_peak_observation']
    original_release=folder/'ROOT_RELEASE_NUMERICAL.json'
    history.append({'kind':'numerical','original_execution_root':folder.parent.name,
        'driver_manifest_sha256':json.loads(original_release.read_text())['driver_manifest_sha256'],
        'terminal':ref(terminal),'original_release':ref(original_release),'status':t['status'],
        'supervisor_inclusive_wall_seconds':t['supervisor_inclusive_wall_seconds'],
        'cuda_peak_allocated_bytes':peaks['cuda_peak_allocated_bytes'],'cuda_peak_reserved_bytes':peaks['cuda_peak_reserved_bytes'],
        'current_V4_admission_or_state_donor':False})
    if t['qualification_receipt'] is not None:
        q=folder/'numerical/run01/QUALIFICATION.json'
        assert sha(q)==t['qualification_receipt']['sha256'];history[-1]['historical_qualification']=ref(q)
failures=json.loads((DRIVER/'PRESERVED_FULL_GRAPH_FAILURES.json').read_text())
for f in failures['failures']:
    history.append({'kind':'full_graph_cap_failure','original_identity':f['original_identity'],
         'attempt_version':f['attempt_version'],'status':f['status'],
         'supervisor_inclusive_wall_seconds':f['paid_supervisor_inclusive_wall_seconds'],
         'cuda_peak_allocated_bytes':f['CUDA_allocated_peak_bytes'],'cuda_peak_reserved_bytes':f['CUDA_reserved_peak_bytes'],
         'caps':f['caps'],'completed_first_J_updates':0,'evidence':f['evidence'],
         'current_V4_admission_or_state_donor':False})
    for e in f['evidence']:
        for kind in ['original','preserved_copy']:
            x=e[kind];assert sha(R/x['path'])==x['sha256'] and (R/x['path']).stat().st_size==x['bytes']
save(P,'PRIOR_ATTEMPTS_AND_COST_CUSTODY.json',{'schema':'V4_numerical_candidate_original_attempt_history_v1','UTC':UTC,
  'attempts':history,'supervisor_inclusive_wall_sum_seconds':sum(h['supervisor_inclusive_wall_seconds'] for h in history),
  'no_overlapping_child_or_driver_walls_added':True,'historical_PASS_not_current_receipt':True,
  'full_graph_failures_both_preserved':True,'cross_version_costs_separate_from_current_driver_prior_failures':True})
work={'numerical':{'inputs':'Existing fabricated16node/128feature/10record fixture; no real-graph load or project metric',
      'existing_replay_optimizer_updates':6,'new_saved_V3_V4_parity_optimizer_updates':8,'total_engineering_optimizer_updates':14,
      'new_parity_member_trajectory_updates':32,'arms':['J','F'],'updates_each_variant_per_arm':2,
      'extra_J_probes':['auxiliary_grad=False forward/RNG','empty captured query complete scorer forward/backward/RNG'],
      'existing_math_teacher_native_detach_serialized_replay_checks_retained':True,
      'parity_comparisons':['full forward/support/teacher order','all gradients','actual next model/Adam','exact all-RNG/training flags','caller-RNG restoration'],
      'all_probe_snapshot_comparison_and_recomputation_costs_paid':True},
    'unchanged_scientific_schedule':{'members':4,'width':64,'seed':0,'lambda':1,'epochs_per_arm':100,
      'train_batch_size':65536,'full_batches_per_epoch':17,'optimizer_updates_per_arm':1700,
      'VALID_selector_candidates_per_arm':100,'serving':'mean_raw_logits_private_own_completion',
      'teacher':'complete_TRAIN_observation_membership_only','source_zero':'unobserved_not_verified_nonlink',
      'authorized_by_this_packet':False},
    'future_full_graph_work_NOT_RELEASED':{'existing_native_optimizer_updates':34,'existing_replay_updates':6,
      'new_parity_updates':8,'total_engineering_updates':48,'complete_five_route_VALID_traversals':2,
      'support_or_batch_or_schedule_reduced':False},'numerical_caps_unchanged':caps,
    'full_graph_caps_not_adapted_or_released':True,'cap_changes':False}
save(P,'WORK_AND_SCHEDULE.json',work)
sources=[ref(DRIVER/'MANIFEST.json'),ref(DRIVER/'SEAL.json'),ref(DRIVER/'PILOT_PLAN.json'),
    ref(DRIVER/'V4_ENGINEERING_QUALIFICATION_PLAN.json'),ref(DRIVER/'PRESERVED_FULL_GRAPH_FAILURES.json'),
    ref(MODEL_REVIEW/'MANIFEST.json'),ref(MODEL_REVIEW/'SEAL.json'),ref(MODEL_REVIEW/'REVIEW.json'),
    ref(SUP/'MANIFEST.json'),ref(SUP/'SEAL.json'),ref(OLD/'MANIFEST.json'),ref(rt),
    ref(R/'graph_ncNC_structural_pattern_numerical_execution_root_20261003_v2/ROOT_RELEASE_NUMERICAL.json')]
save(P,'SOURCE_REVIEW_INTERPRETER_BINDINGS.json',{'schema':'V4_numerical_metadata_input_bindings_v1','UTC':UTC,
    'inputs':sources,'interpreter':{'path':PYTHON,'sha256':runtime['interpreter_sha256']},
    'author_role':'V4 model source review completed; new helper and execution metadata authored, not independently reviewed',
    'prior_supervision_review_is_historical_only':True,'current_helper_independent_review_required':True})
save(P,'ROOT_REPORTED_TRANSPORT_CONTEXT.json',{'schema':'root_reported_transport_context_v1','UTC':UTC,
    'root_reported_verified_push':'db0302db9991bbb5e167a94948ff6fe5f4199551',
    'root_reported_push_includes':'V4 model source',
    'root_inspected_model_REVIEW_sha256':'7682474372bee7162f969710ad4f4651cdb06cb1bbb507529d4a5826402f5322',
    'root_reported_77_safe_sync':'Stopped before SSH/science: MacLink reports The other Mac is not connected yet',
    'last_root_reported_authorized_77_metadata_snapshot':'20:14 jobs still advancing',
    'agent_fresh_remote_custody_or_resource_observation':False,'alternate_allocation_used':False,
    'source_only_preparation_continues':True})
argv=[PYTHON,'-B',PHASE+'/'+SUP.name+'/supervise_numerical.py','--root-release',EXECUTION+'/ROOT_RELEASE_NUMERICAL.json','--output',EXECUTION+'/supervision/run01']
save(P,'LAUNCH_SEQUENCE.json',{'schema':'disabled_V4_numerical_candidate_launch_sequence_v1','UTC':UTC,
    'status':'PREPARED_NOT_APPROVED_NOT_EXECUTED','candidate_release_descriptor':ref(P/'ROOT_RELEASE_NUMERICAL_CANDIDATE.json'),
    'steps':[
     'Root inspects V4 model-source review and independently reviews this newly authored path-bound numerical helper.',
     'Root arranges read-only custody verification/source transport for exact V4 driver, exact saved V3 sibling reference, helper, pinned dependencies, authorities and review inputs; no scientific command at this step.',
     'Root obtains fresh physical GPU/interpreter/route/resource admission under unchanged numerical caps and verifies all new remote output/log paths are absent.',
     'Only after explicit root approval, create fresh ROOT_NUMERICAL_ADMISSION.json and ROOT_RELEASE_NUMERICAL.json in the V4 execution root. Fill the approval reference, exact source/review/custody bindings and only the one numerical invocation; do not overwrite this sealed candidate.',
     'Dispatch the one ordinary detached supervisor below from the exact repository with the bound destination environment; no automatic retry.',
     'Inspect physical supervisor/child identities, inclusive terminal/CUDA/attempt receipts and current V4 QUALIFICATION with complete saved-V3 parity. Preserve failure and cost if it stops. A PASS does not authorize full_graph or fit.'
    ],'remote_execution_root':EXECUTION,'cwd':REPO,
    'environment':{'GNNM_SSH_DESTINATION':'shmelev@192.168.18.77','PYTHONDONTWRITEBYTECODE':'1'},
    'supervisor_argv_after_approval':argv,'stdout_stderr_log':EXECUTION+'/DETACHED_SUPERVISOR.log',
    'stdin':'DEVNULL','detached':'ordinary nohup/session dispatch as existing root model; root records physical dispatch identity',
    'output_paths_that_must_be_fresh':[EXECUTION+'/supervision/run01',EXECUTION+'/numerical/run01',EXECUTION+'/DETACHED_SUPERVISOR.log'],
    'execution_root_created_or_remote_freshness_checked':False,'command_executed':False,
    'full_graph_or_scientific_prediction_authorized':False})
(P/'LAUNCH_SEQUENCE.md').write_text('# Disabled V4 numerical launch preparation\n\nStatus: authored metadata and path-bound support source only. Root approval, independent new-helper review, fresh source custody, and physical admission are pending. No launch occurred.\n\nExact candidate descriptor: ROOT_RELEASE_NUMERICAL_CANDIDATE.json. It has null root approval and empty authorized stages/invocations. Exact launch argv, future output paths and ordering are in LAUNCH_SEQUENCE.json.\n\nThe sealed support successor changes fixed V3/v2 paths to V4 and one guard message, with unchanged numerical caps: wall1800s, host16GiB, allocated8GiB/reserved8GiB. Model-source review is complete; authoring/checking this helper does not independently review it.\n\nNumerical work is14engineering Adam updates:6existing replay +8new V3/V4 parity. The fabricated fixture and existing math/teacher/native/replay checks remain, with all extra probe/snapshot/comparison/recomputation cost charged. The frozen scientific schedule and future complete full-graph work are retained but unreleased.\n\nPrior attempts and exact supervisor costs remain separate by original identities: one numerical failure, historical V3 numerical PASS, and both full-graph cap failures. No historical PASS substitutes for current V4 qualification.\n\nAfter approval root creates new active admission/release files in the fresh V4 execution root, then uses the exact prepared supervisor argv. No sealed source/review/registry or candidate file is overwritten. Full-graph and scientific work require later separate admission.\n')
for f in P.glob('*.json'):json.loads(f.read_text())
ast.parse((P/'PREPARE_METADATA.py').read_text())
assert release['authorized_stages']==[] and release['authorized_invocations']==[] and release['root_authorization_reference'] is None
assert sha(DRIVER/'MANIFEST.json')=='9021a598c7642bf428124de1230a078dfddbf3968095432354ee18a14541104c'
assert sha(MODEL_REVIEW/'MANIFEST.json')=='8f10818dc693e7ca4fe899de761a2ef370994cdbbe95c899e9b47f3338942357'
save(P,'AUTHOR_METADATA_CHECK.json',{'schema':'disabled_metadata_author_check_v1','UTC':UTC,
    'candidate_release_disabled':True,'new_output_roots_absent_locally':True,'remote_freshness_unchecked':True,
    'caps_unchanged':True,'input_hashes_checked':True,'JSON_parsed_and_sources_AST_compiled_only':True,
    'independent_new_helper_review':False,'GPU_project_or_model_execution':False,'canonical_or_registry_mutation':False})
META_SHA=seal(P,'DISABLED_V4_NUMERICAL_METADATA_AUTHOR_PREPARED_ROOT_APPROVAL_AND_HELPER_REVIEW_PENDING')
print(json.dumps({'helper_manifest_sha256':SUP_SHA,'helper_seal_sha256':sha(SUP/'SEAL.json'),
    'metadata_manifest_sha256':META_SHA,'metadata_seal_sha256':sha(P/'SEAL.json'),
    'candidate_release_sha256':sha(P/'ROOT_RELEASE_NUMERICAL_CANDIDATE.json'),
    'remote_execution_root':EXECUTION,'candidate_authorized':False},indent=2))
