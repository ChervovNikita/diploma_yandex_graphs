"""Stdlib metadata review, complete setup accounting and root resource proposal."""
import ast,gzip,hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parents[1];P=OUT.parent
def receipt(path):
 raw=path.read_bytes();return {'path':str(path.relative_to(P)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
entries=json.loads((OUT/'MASTER_INPUT_ENTRIES.json').read_text());compiled=json.loads((OUT/'COMPILATION_METADATA.json').read_text());composition=json.loads((OUT/'ROOT_V14_COMPOSITION.json').read_text())
checks=[];studies=[];setups=[];resource_cases=[]
for family in ('peer','tour'):
 plan=json.loads((OUT/'matrix'/family/'PLAN.json').read_text());path=P/plan['studies'][0]['path'];study=json.loads(gzip.decompress(path.read_bytes()));studies.append(study)
 assert len({r['run_id'] for r in study['runs']})==len(study['runs']);checks.append(f'{family}: complete unique declared roster')
 assert plan['scientific_launch_enabled'] is False and plan['human_GO_required'] is True;checks.append(f'{family}: disabled no-GO plan')
 fits={r['run_id']:r for r in study['runs'] if r['kind']=='model_fit'};export=[r for r in study['runs'] if r['kind']=='ensemble_assembly']
 training_bodies=sum(r['config']['source_complete_body_constructors'] for r in fits.values());export_bodies=sum(fits[i]['config']['source_complete_body_constructors'] for r in export for i in r['config']['fit_parents'])
 training_opts=len(fits);export_opts=sum(len(r['config']['fit_parents']) for r in export)
 def scanner(arm):return family=='tour' and arm in ('PV1','WI4','V1','SW4','C4','R4')
 def phi(arm):return family=='peer' and arm in ('PA1','IA4','SA4','SM4','SX4','UX4')
 training_scan=sum(r['config']['source_complete_body_constructors'] if scanner(r['config']['meansage_arm']) else 0 for r in fits.values())
 export_scan=sum(fits[i]['config']['source_complete_body_constructors'] if scanner(fits[i]['config']['meansage_arm']) else 0 for r in export for i in r['config']['fit_parents'])
 training_phi=sum(r['config']['source_complete_body_constructors'] if phi(r['config']['meansage_arm']) else 0 for r in fits.values())
 export_phi=sum(fits[i]['config']['source_complete_body_constructors'] if phi(fits[i]['config']['meansage_arm']) else 0 for r in export for i in r['config']['fit_parents'])
 setups.append({'family':family,'training_full_body_constructors':training_bodies,'training_optimizer_constructors':training_opts,
  'export_selected_state_reconstruction_body_constructors':export_bodies,'export_unstepped_optimizer_constructors':export_opts,
  'selected_model_optimizer_restores':export_opts,'training_scanner_constructors':training_scan,'export_scanner_constructors':export_scan,
  'training_feedback_map_constructors':training_phi,'export_feedback_map_constructors':export_phi,'CAL_scalar_optimizer_constructors':len(export),
  'CAL_scalar_updates':500*len(export),'CAL_objective_forward_evaluations':1001*len(export),'actual_allocation_constructors':0,'scope':'declared_unexecuted_full_roster_setup_costs'})
 for arm in dict.fromkeys(r['config']['meansage_arm'] for r in fits.values()):
  body=3 if arm in ('I4','IA4','WI4') else 0
  run=next(r for r in fits.values() if r['split']==0 and r['seed']==98401 and r['body']==body and r['config']['meansage_arm']==arm)
  resource_cases.append({'case_id':'native-meansage-'+family+'-'+arm.lower(), 'study_id':study['study_id'],'run_id':run['run_id'],
   'study_path':str(path.relative_to(P)),'source_contract':'exact fixed native source adapter; GRAPH/TRAIN only with disclosed synthetic SELECT stub',
   'declared_body':body,'source_full_factory_body_constructors':run['config']['source_complete_body_constructors'],
   'source_optimizer_class':'torch.optim.AdamW' if arm=='original_gnnm' else 'instance concrete type TF1Adam, dynamically returned by pinned author.optimizer_class',
   'deny_optimizer_update':'patch concrete optimizer instance/class step after construction; generic torch.optim.Adam interception is insufficient for TF1Adam',
   'repetitions':2,'scientific_optimizer_updates':0,'scores':0,'resource_profile_qualified':False})
 checks.append(f'{family}: all comparison cohorts matched per allocation compiler')
 assert all(r['budget']['policy']=='native_selector_stop' and r['budget']['maximum']==100000 and r['selection']['ties']=='last' for r in fits.values());checks.append(f'{family}: native stopping and latest tied minimum')
 for r in fits.values():
  if r['config']['meansage_arm']=='UX4':assert r['config']['physical_slow_bodies']==4 and r['config']['source_optimizer_count']==1
 checks.append(f'{family}: no joint UX4 mislabeled as genuine I4')
for row in composition['files']:
 old=(OUT/'controller_v14_base'/row['path']).read_bytes();new=(OUT/'runtime'/row['path']).read_bytes()
 assert hashlib.sha256(old).hexdigest()==row['sha256'] and hashlib.sha256(new).hexdigest()==row['successor_sha256']
 names={'engine.py':{'_validate_spec','run_trajectory'},'run_study.py':{'execute'},'spec.py':{'validate'},'compile_master_plan.py':{'fit_semantics','typed_counts'}}[row['path']]
 oldnodes={getattr(n,'name',None):ast.dump(n,include_attributes=False) for n in ast.parse(old).body};newnodes={getattr(n,'name',None):ast.dump(n,include_attributes=False) for n in ast.parse(new).body}
 changed={n for n,v in oldnodes.items() if v!=newnodes[n]};assert changed==names
 checks.append(row['path']+': exact v14 and only declared AST functions changed; AMP/Single8/NSD/SeHGNN/curvature/growth retained')
assert compiled['combined_counts']['model_optimizer_step_call_total'] is None;assert compiled['combined_counts']['model_optimizer_step_call_minimum']==62730;assert compiled['combined_counts']['model_optimizer_step_call_maximum']==123000000
checks+=['bounded51..100000 source stopping counts; caps never exact executed updates','all12 available source conditions completed; no unavailable-source exclusion','Photo raw binary/reciprocal/one-self/136components/fresh private12tours verified GRAPH-only','original five excluded by frozen schema; original full TABM SAGE baseline included per new cohort']
proposal={'schema':'native-meansage-peer-tour-root-resource-proposal-v1','status':'PROPOSED_NOT_AUTHORIZED_OR_EXECUTED','allocation_only':True,'source_case_count':len(resource_cases),'cases':resource_cases,
 'callback_count_if_root_admitted':2*len(resource_cases),'root_review_required_before_actual_input_callbacks':True,
 'TRAIN_only':True,'real_SELECT_CAL_evaluation_targets_forbidden':True,'native_TF1Adam_dynamic_class_denial_required':True,
 'source_original_GNNM_equations_match_prior_literal_SAGE_reference':True,'qualified_profile_inheritance_only_by_root_byte_semantics_review':True,
 'actual_source_resource_callbacks_executed':0,'actual_source_optimizer_updates':0,'scientific_GO':False}
(OUT/'RESOURCE_PROPOSAL.json').write_text(json.dumps(proposal,indent=2,sort_keys=True)+'\n')
totals={k:sum(r[k] for r in setups) for k in setups[0] if type(setups[0][k])==int}
(OUT/'SOURCE_SETUP_COSTS.json').write_text(json.dumps({'schema':'native-meansage-full-declared-setup-costs-v1','families':setups,'totals':totals,
 'GRAPH_preparation':{'raw_incidence_CD_constructions':1,'private_seed_groups':3,'private_DFS_forests':12,'tours_reused_immutable_across10role_masks':True,'target_reads':0,'models_constructed':0,'optimizers_constructed':0},
 'higher_cost_factory_refactor_disclosed':'Each own I4 body fit/export reconstructs all4 exact source bodies then discards3; all constructors charged; one own optimizer only',
 'source_fit_selected_full_replay_stage_adaptation':'The selected model/optimizer/RNG is fully restored by the fit engine; the source selected full forward/SELECT-total replay is charged at label-free export using the same complete forward for role export',
 'scope':'planned_not_executed; exact setup count roster; no scientific optimizer kernels or scores yet'},indent=2,sort_keys=True)+'\n')
amend={'schema':'native-meansage-v14-stopping-cost-amendment-v2','base':'master_controller_source_20261012_v14/runtime','files':composition['files'],
 'amendment_script':receipt(OUT/'preparation/amend_controller.py'),'new_native_helper':receipt(OUT/'runtime/native_stopping.py'),
 'composition':'Apply amend_engine/amend_spec/amend_run/amend_compiler to current root text; all anchors checked. Copy native_stopping.py beside composed controller. Do not replace newer root controllers.',
 'source_completion':'native_selector_stop exclusively exact cap100000/p50/no0/latest tied own SELECT-total; actual completed52 versus cap100000 proof; stopped recovery adds zero updates',
 'full_budget_default_and_serialized_spec_unchanged':True,'bounds_not_executed_counts':True,'global_controller_edited':False,'GO':False}
(OUT/'CONTROLLER_TYPE_AMENDMENT.json').write_text(json.dumps(amend,indent=2,sort_keys=True)+'\n')
handoff={'schema':'native-meansage-peer-tour-source-completion-handoff-v1','status':'SOURCE_AND_FINITE_DISABLED_PACKET_PREPARED; ROOT_RESOURCE_AND_MASTER_ADMISSION_PENDING',
 'packet':OUT.name,'entries':entries,'canonical_ids':compiled['canonical_ids'],'counts':compiled['combined_counts'],
 'setup_costs':receipt(OUT/'SOURCE_SETUP_COSTS.json'),'source_bindings':receipt(OUT/'EXACT_SOURCE_BINDINGS.json'),'source_orchestration':receipt(OUT/'ORCHESTRATION_SOURCE_BINDINGS.json'),
 'native_adapter':receipt(OUT/'runtime/adapters/meansage_native.py'),'native_endpoints':receipt(OUT/'runtime/adapters/meansage_endpoints.py'),
 'controller_amendment':receipt(OUT/'CONTROLLER_TYPE_AMENDMENT.json'),'controller_composition':receipt(OUT/'ROOT_V14_COMPOSITION.json'),
 'synthetic_source_proof':receipt(OUT/'SYNTHETIC_CPU_PROOF.json'),'stopping_recovery_proof':receipt(OUT/'STOPPING_RECOVERY_CPU_PROOF.json'),
 'resource_proposal':receipt(OUT/'RESOURCE_PROPOSAL.json'),'metadata_checks_passed':len(checks),'metadata_checks':checks,
 'failures_preserved':['initial multiclass-flat exact-CAL assertion numerical assumption','first disabled compiler called project-relative master receipt API with absolute path; fixed metadata-only'],
 'scientific_fits':0,'actual_input_optimizer_updates':0,'actual_predictive_scores':0,'actual_scalar_CAL_updates':0,
 'unavailable_source_exclusions':[],'original_data_experiments_rerun':False,'original_GNNM_new_data_baseline':True,
 'historical_states_scores_authorization_inherited':False,'root_global_controller_master_modified':False,'NSD_diagnostics_or_corrections_repeated':False,'GO':False}
(OUT/'ALLOCATION_HANDOFF.json').write_text(json.dumps(handoff,indent=2,sort_keys=True)+'\n')
print(json.dumps({'metadata_checks':len(checks),'setup_totals':totals,'resource_cases':len(resource_cases),'science':False,'GO':False}))
