"""Exact finite source closure, costs, declared callbacks; no numerical imports."""
import ast,gzip,hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parents[1];PHASE=OUT.parent
def receipt(path):
 raw=path.read_bytes();return {'path':str(path.relative_to(PHASE)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(name,value):
 p=OUT/name
 if p.exists():raise FileExistsError('Preserve '+name)
 p.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def main():
 meta=json.loads((OUT/'COMPILATION_METADATA.json').read_text());audit=json.loads((OUT/'FINAL_TYPED_INGESTION.json').read_text());proof=json.loads((OUT/'SYNTHETIC_CPU_PROOF.json').read_text());plan=json.loads((OUT/'matrix/PLAN.json').read_text())
 studies=[json.loads(gzip.decompress((PHASE/r['path']).read_bytes())) for r in plan['studies']]
 assert audit['verified'] and audit['plan_hash']==meta['entry']['plan_hash'] and len(meta['canonical_ids'])==15 and proof['factories_checked']==18
 assert meta['counts']['totals']['source_complete_body_constructors']==723 and meta['counts']['totals']['source_optimizer_constructors']==723
 assert meta['counts']['model_optimizer_step_call_total'] is None and [meta['counts']['model_optimizer_step_call_minimum'],meta['counts']['model_optimizer_step_call_maximum']]==[721053,1443000]
 assert all(r['matched'] for values in meta['references'].values() for r in values)
 fits=[r for s in studies for r in s['runs'] if r['kind']=='model_fit'];exports=[(s,r) for s in studies for r in s['runs'] if r['kind']=='ensemble_assembly'];by_study={s['study_id']:{r['run_id']:r for r in s['runs']} for s in studies}
 assert len(fits)==624 and len(exports)==372
 export_fits=[by_study[s['study_id']][p] for s,r in exports for p in r['config']['fit_parents']]
 setup={'schema':'Physics-OIPG-literal-all-declared-setup-costs-v1','training_complete_model_constructors':sum(r['config']['physical_slow_bodies'] for r in fits),
  'training_actual_optimizer_constructors':sum(r['config']['source_optimizer_count'] for r in fits),'training_constructed_then_discarded_complete_models':0,
  'export_complete_model_constructors':sum(r['config']['physical_slow_bodies'] for r in export_fits),
  'export_actual_unstepped_optimizer_constructors':sum(r['config']['source_optimizer_count'] for r in export_fits),
  'selected_container_model_optimizer_private_global_RNG_restores_at_export':len(export_fits),
  'CAL_scalar_optimizer_constructors':372,'CAL_scalar_updates':186000,
  'source_construction_authority':'One complete TABM/meanSAGE object per actual owner; portable Session constructs one complete WikiPolynormer per own model, shared factor installation in the one OIPG body; no full model prototype or discarded-body stage. Each WikiBackbone explicitly calls native reset_parameters after native constructor, and both initialization programs are charged.',
  'factor_costs':'All input/output TABM boundary blocks or source-installed FactorLinear modules are included in complete constructor totals; source native/fast initialization and all member dropout stream construction remain literal.',
  'physics_detached_prefix_forward_upper_bound':meta['counts']['totals']['source_detached_prefix_forward_calls_upper_bound'],
  'OIPG_old_state_gradient_collections':meta['counts']['totals']['registered_OIPG_source_old_state_gradient_collections'],
  'source_input_hash_and_metadata_IO':'Every source/input authority is rechecked by runtime factory before load; label-free endpoint ID hashes before export, CAL/evaluation target hashes only in their gated endpoints. No fresh graph copy. These setup/IO costs are additional to TRAIN counters.',
  'selected_export_work':'Source selected checkpoints restore model, actual optimizer objects and own/global/private streams. Coupled Physics exports own-selected states while preserving live LAST companions. Native FP32 serving executes on original execution device before CPU float64 metric work.',
  'unknown_native_executed_updates_kept_bounded':True,'actual_input_constructors':0,'actual_input_optimizer_updates':0,'actual_scalar_CAL_updates':0,'actual_scores':0,'GO':False}
 write('SOURCE_SETUP_COSTS.json',setup)
 cases=[]
 for study in studies:
  seen=set()
  for run in study['runs']:
   if run['kind']!='model_fit' or run['config']['stage']!='comparative':continue
   c=run['config'];key=(c.get('physics_arm'),c.get('oipg_policy'),c.get('allocation_lambda'))
   if key in seen:continue
   seen.add(key)
   cases.append({'case_id':study['study_id']+'-'+run['run_id'],'study_id':study['study_id'],'study_path':next(r['path'] for r in plan['studies'] if Path(r['path']).name==study['study_id']+'.json.gz'),
    'run_id':run['run_id'],'source_arm':c.get('physics_arm',c.get('oipg_policy')),'allocation_lambda':c.get('allocation_lambda'),
    'repetitions':2,'complete_source_body_constructors':c['physical_slow_bodies'],'actual_source_optimizer_objects':c['source_optimizer_count'],'source_members':c['members'],
    'callback':'Exact source TRAIN programs at old weights; optimizer.step denied, source direct PolicyAdapter old-state gradient assignment allowed, full state/RNG preserved. SELECT replaced with synthetic stub; no actual SELECT/CAL/evaluation targets.',
    'source_roles':'GRAPH/TRAIN only; full immutable input; fresh case explicit seed; any source local transition separately preserves live end-local streams',
    'resource_profile_qualified':False,'callbacks_executed':0,'scientific_updates':0,'scores':0})
 # The selected comparative I/P/G defaults are .05; add declared .5 source cases
 # because both numeric search fits are part of the approved finite design.
 for policy in ('I','P','G'):
  study=studies[1];run=next(r for r in study['runs'] if r['config'].get('oipg_policy')==policy and r['config']['stage']=='search' and r['config']['allocation_lambda']==.5)
  prior=next(c for c in cases if c['study_id']==study['study_id'] and c['source_arm']==policy and c['allocation_lambda']==.05)
  cases.append({**prior,'case_id':study['study_id']+'-'+run['run_id'],'run_id':run['run_id'],'allocation_lambda':.5})
 proposal={'schema':'Physics-OIPG-finite-source-resource-proposal-v1','status':'PROPOSED_UNQUALIFIED_NOT_EXECUTED','allocation_only':True,'cases':cases,
  'source_cases':len(cases),'proposed_callbacks':2*len(cases),'actual_callbacks_executed':0,'root_resource_admission_required':True,'actual_input_optimizer_updates':0,'scores':0,'GO':False}
 write('RESOURCE_PROPOSAL.json',proposal)
 delta=json.loads((OUT/'CONTROLLER_TYPE_AMENDMENT.json').read_text());composition=json.loads((OUT/'CONTROLLER_COMPOSITION_METADATA.json').read_text());records=[]
 for row in delta['files']:
  before=(OUT/'controller_base'/row['file']).read_bytes();after=(OUT/'runtime'/row['file']).read_bytes();assert hashlib.sha256(before).hexdigest()==row['base_sha256'] and hashlib.sha256(after).hexdigest()==row['successor_sha256']
  records.append({**row,'changed_AST_functions':next(r['changed_AST_functions'] for r in composition['checks'] if r['file']==row['file'])})
 amendment={'schema':'Physics-OIPG-v17-composable-controller-handoff-v1','base':delta['base'],'files':records,
  'amendment_script':receipt(OUT/'preparation/amend_controller.py'),'helpers':[receipt(OUT/'runtime/tabm_stopping.py'),receipt(OUT/'runtime/oipg_typing.py')],
  'composition':'Apply functions to current controller text, copy exact helper bytes beside it; preserve every existing BSNN policy set member and standalone HGT if anchor; CONTROLLER_FILES insertion uses AST tuple boundaries. Never overwrite newer controller with own v17 successor.',
  'preserved_byte_identical_helpers':[receipt(OUT/'runtime'/name) for name in ('native_stopping.py','hgt_stopping.py','checkpoint_retention.py','retention_lifecycle.py')],
  'required_final_master_gate':{'name':'whole_bank_exports','source_stages':['bank_export'],'required_for_stages':['calibration']},
  'source_exporter':receipt(OUT/'runtime/adapters/source_endpoints.py'),'source_exporter_not_in_retention_allowlist':True,
  'full_selected_states_retained_until_root_exact_exporter_audit':True,'numerical_factory_proof_inherited':receipt(OUT/'SYNTHETIC_CPU_PROOF.json'),
  'endpoint_changes_since_factory_proof':'Stable native FP32 log-mean-probability proper-score pool, exact endpoint file receipts, schema/ID/dependency checks; no training model or gradient-program change; endpoint admission remains prospective.',
  'global_root_edited':False,'GO':False}
 write('CONTROLLER_HANDOFF.json',amendment)
 failures=[{'authority':receipt(OUT/n),'scope':'source/data/metadata preparation only; no actual-input models or updates'} for n in ('SOURCE_COPY_ATTEMPT1_FAILURE.json','DATA_PREPARATION_ATTEMPT1.json','DATA_PREPARATION_ATTEMPT2.json','COMPILER_INGESTION_RESULT.json')]
 failures += [{'authority':receipt(OUT/'transport/WIKICS_EXACT_GRAPH_PATH_AND_PHYSICS_INHERITANCE.json'),'scope':'outer transport syntax error before route/project operation; corrected exclusive retry retained'}]
 checks=['15literal canonical IDs joined exactly to remaining27 source path/hash authority','full raw-verified Physics/WikiCS immutable graphs;3/10fresh development role splits; no graph copies',
  '18literal synthetic source/control factories/83synthetic optimizer maps and500scalar maps passed; actual-input maps0','exact next-update model/optimizer/global/private/sampler replay; coupled inactive LAST states held separately from selected states',
  'OIPG lambda .05/.5 candidates complete1100 each before SELECT choice, separate fresh comparative seed, lambda0 programs retained, one O aliasfit',
  'native TABM cap5000/p300/per-ten saved-best; meanSAGE cap100000/p50/latest total loss; bounded quantities keep actual total null',
  'full native512WikiPolynormer1100/100localbest transition preserves live end-local RNG','source independent4_contrastive four own optimizers and unscaled own gradients plus4times fixed alignment/residual auxiliary; not genuine uncoupled I4',
  'exact frozen launcher.load_plan source binding and typed ingestion passed with numerical/provider imports, array reads and process/network denied',
  'all selected exports before CAL; all CAL before evaluation required; whole cohorts/failures preserved; resources unqualified','native/AMP/HGT/retention paths kept, controller amendment composes with BSNN policy']
 handoff={'schema':'Physics10-OIPG4-coupled1-allocation-source-completion-handoff-v1','status':'SOURCE_AND_ACTUAL_DATA_CLOSED_DISABLED_ROOT_RESOURCE_MASTER_ADMISSION_PENDING',
  'packet':OUT.name,'entry':meta['entry'],'canonical_ids':meta['canonical_ids'],'canonical_count':15,'counts':meta['counts'],'per_study_counts':meta['per_study_counts'],
  'canonical_source_join':receipt(OUT/'CANONICAL_SOURCE_COMPLETION.json'),'prospective_design':receipt(OUT/'DESIGN_DECISION.json'),'actual_data_custody':receipt(OUT/'DATA_CUSTODY.json'),
  'source_snapshots':receipt(OUT/'SOURCE_SNAPSHOT_RECEIPTS.json'),'source_adapter_receipts':[receipt(OUT/'runtime/adapters'/n) for n in ('physics_joint.py','oipg.py','source_common.py','source_endpoints.py')],
  'source_synthetic_proof':receipt(OUT/'SYNTHETIC_CPU_PROOF.json'),'exact_disabled_loader_audit':receipt(OUT/'FINAL_TYPED_INGESTION.json'),
  'controller_delta':receipt(OUT/'CONTROLLER_HANDOFF.json'),'source_setup_costs':receipt(OUT/'SOURCE_SETUP_COSTS.json'),'resource_proposal':receipt(OUT/'RESOURCE_PROPOSAL.json'),
  'resource_qualified':False,'actual_input_callbacks_executed':0,'source_preparation_failures_preserved':failures,
  'metadata_attempts':'first path-format failure retained; second successful provisional matrix and loader proof retained under metadata_revisions/attempt2; third adds exact full-budget constructor/old-state quantities only',
  'endpoint_authority':'Physics literal mean logits; OIPG native FP32 mean probabilities controls decisions; stable FP32 logsumexp(log_softmax)-logM defines proper-score input; common fixed final500 CAL after pool.',
  'compatible_original_GNNM_disclosure':'Actual source TABM SAGE2H128/5000p300 original boundary model on fresh roles; differs from512WikiPolynormer1100; no published-score replication claim.',
  'metadata_checks':checks,'metadata_checks_passed':len(checks),'actual_input_model_constructors':0,'actual_input_fits':0,'actual_input_optimizer_updates':0,'actual_scalar_CAL_updates':0,'actual_predictive_scores':0,
  'historical_fit_states_scores_or_authorization_inherited':False,'global_root_controller_master_modified':False,'GO':False}
 write('ALLOCATION_HANDOFF.json',handoff)
 print(json.dumps({'canonical_count':15,'fit_groups':624,'bank_exports':372,'complete_model_and_optimizer_constructors':723,'resource_cases':len(cases),'resource_callbacks_executed':0,'handoff':receipt(OUT/'ALLOCATION_HANDOFF.json'),'GO':False}))
if __name__=='__main__':main()
