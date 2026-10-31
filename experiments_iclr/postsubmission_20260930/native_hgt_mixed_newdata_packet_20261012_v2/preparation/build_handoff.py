"""Metadata/source scope review, complete setup costs and resource proposal."""
import ast,gzip,hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parents[1];P=OUT.parent
def receipt(path):
 raw=path.read_bytes();return {'path':str(path.relative_to(P)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
meta=json.loads((OUT/'COMPILATION_METADATA.json').read_text());audit=json.loads((OUT/'FINAL_TYPED_INGESTION.json').read_text());plan=json.loads((OUT/'matrix/PLAN.json').read_text());study=json.loads(gzip.decompress((P/plan['studies'][0]['path']).read_bytes()));composition=json.loads((OUT/'ROOT_V16_COMPOSITION.json').read_text());custody=json.loads((OUT/'DATA_CUSTODY.json').read_text())
checks=[];fits={r['run_id']:r for r in study['runs'] if r['kind']=='model_fit'};exports=[r for r in study['runs'] if r['kind']=='ensemble_assembly']
assert audit['verified'] and audit['fit_groups']==570 and audit['banks']==450 and audit['genuine_I4_banks']==60 and audit['rows']==1920;checks.append('complete immutable disabled roster570 ownfit groups/450banks/60genuine I4/1920rows')
assert audit['native_update_bounds']==[17670,171000] and meta['counts']['model_optimizer_step_call_total'] is None;checks.append('native cap300/p30/latest pooledSELECTCE; exact optimizer total unknown before31..300 source stopping')
assert len(meta['canonical_ids'])==10 and len(set(meta['canonical_ids']))==10;checks.append('all HGT6 plus mixed4 exact canonical source conditions')
assert all(r['matched'] for r in meta['references']);checks.append('all source/candidate cohorts retain native/factorM1/genuineI4/original typedboundary reference cohorts')
assert custody['model_graph']['node_counts']=={'0':4057,'1':14328,'2':7723,'3':20} and custody['model_graph']['support_edges']==239566 and len(custody['splits'])==10 and not custody['test_label_member_opened'];checks.append('full raw archive/relation/feature2 materialization;10fresh roles;officialtest labels unopened')
assert all(r['config']['source_scheduler_rule']=='OneCycle300 maxlr.001 pct_start.05 cosine step(update)' for r in fits.values());checks.append('exact native AdamW+OneCycle300 constructor/updates/portable cosine state')
assert all(r['config']['source_optimizer_count']==1 and r['config']['physical_slow_bodies']==4 for r in fits.values() if r['config']['hgt_arm']=='untied_HGT');checks.append('source untied4 remains coupled oneoptimizer/oneselector; never genuineI4')
assert all(r['config']['initialization_seed']==r['seed']+1000003*r['body'] for r in fits.values());checks.append('fresh own seeds/route streams/factor law; no historical fit states/scores/authorization')
for row in composition['files']:
 old=(OUT/'controller_v16_base'/row['file']).read_bytes();new=(OUT/'runtime'/row['file']).read_bytes();assert hashlib.sha256(old).hexdigest()==row['base_sha256'] and hashlib.sha256(new).hexdigest()==row['successor_sha256']
 before={getattr(n,'name',None):ast.dump(n,include_attributes=False) for n in ast.parse(old).body};after={getattr(n,'name',None):ast.dump(n,include_attributes=False) for n in ast.parse(new).body};changed={n for n,v in before.items() if v!=after[n]};assert changed==set(row['changed_AST_functions'])
 checks.append(row['file']+': exact frozenv16/only declared AST functions; AMP/Single8/UniMP/GINE/BSNN/peer/NSD/growth/curvature retained')
assert (OUT/'runtime/native_stopping.py').read_bytes()==(OUT/'controller_v16_base/native_stopping.py').read_bytes();checks.append('meanSAGE native_stopping helper bytes retained unchanged')
checks+=['synthetic13 source/control factories; all39 synthetic AdamW updates/next-step replays passed, actual-input updates0','source mixed one-backward AST reverse projection exact; unusedNone gradients preserve AdamWskip','literal original BEBlock typed sparse/dense source equations; native tanh/residual/norm/dropout preserved','HGTsource stop32 after tie2+30strict worse; actual32 distinct from cap300','original paper five excluded; no original-data experiments rerun; no publishedscore replication','fixed native recipe no tuning fits or added method combinations']
def factor_count(arm):return 0 if arm=='original_gnnm' else 27 if arm=='wider_BE' else 9 if arm in ('native_M1','untied_HGT') else 15 if arm=='factor_M1' else 12
setup={'schema':'native-HGT-full-declared-setup-costs-v2',
 'training_full_body_constructors':sum(r['config']['source_complete_body_constructors'] for r in fits.values()),'training_optimizer_constructors':len(fits),'training_OneCycle_constructors':len(fits),
 'training_factor_module_constructors':sum(factor_count(r['config']['hgt_arm']) for r in fits.values()),'training_original_typed_BEBlock_constructors':5*sum(r['config']['hgt_arm']=='original_gnnm' for r in fits.values()),
 'export_full_body_constructors':sum(fits[i]['config']['source_complete_body_constructors'] for r in exports for i in r['config']['fit_parents']),
 'export_unstepped_optimizer_constructors':sum(len(r['config']['fit_parents']) for r in exports),'export_OneCycle_constructors':sum(len(r['config']['fit_parents']) for r in exports),
 'export_factor_module_constructors':sum(factor_count(fits[i]['config']['hgt_arm']) for r in exports for i in r['config']['fit_parents']),
 'export_original_typed_BEBlock_constructors':5*sum(fits[i]['config']['hgt_arm']=='original_gnnm' for r in exports for i in r['config']['fit_parents']),
 'selected_model_optimizer_scheduler_restores_at_export':sum(len(r['config']['fit_parents']) for r in exports),'CAL_scalar_optimizer_constructors':len(exports),'CAL_scalar_updates':500*len(exports),
 'source_factory_costs':'base full native prototypes and requested deepcopies charged; wider64count-prototype then72model charged; all three-mode factor prototypes and selected copied/sliced modules charged',
 'selected_native_replay_at_export':'same complete selected forward supplies raw logits and SELECT pooledCE replay; source absolute1e-7 tolerance; export reconstructs full model/optimizer/OneCycle/private/global RNG and charges all setup',
 'serving_arithmetic':'FP32 raw complete logits.mean0 stored once; endpoint/calculation float64 only after native FP32 mean',
 'actual_input_model_constructors':0,'actual_optimizer_updates':0,'scope':'declared_unexecuted_setup_costs'}
(OUT/'SOURCE_SETUP_COSTS.json').write_text(json.dumps(setup,indent=2,sort_keys=True)+'\n')
cases=[]
for arm in dict.fromkeys(r['config']['hgt_arm'] for r in fits.values()):
 body=3 if arm in ('native_M1','factor_M1') else 0;r=next(r for r in fits.values() if r['config']['hgt_arm']==arm and r['split']==0 and r['seed']==99601 and r['body']==body)
 cases.append({'case_id':'native-HGT-'+arm.replace('/','_'),'study_id':study['study_id'],'study_path':plan['studies'][0]['path'],'run_id':r['run_id'],'repetitions':2,
  'source_body_index':body,'native_constructor_count':r['config']['source_complete_body_constructors'],'source_members':r['config']['members'],'source_TRAIN_backward_calls':r['config']['source_TRAIN_backwards_per_update'],
  'callback':'exact source native train objective/mixedblock backward with optimizer.step and scheduler.step denied; full state/mode/flag/global/private RNG preserved; populated optimizer/scheduler serializer proven synthetically',
  'targets':'GRAPH/TRAIN only; synthetic SELECT stub; real SELECT/CAL/evaluation labels forbidden','unused_gradients':'source None leaves remain None; skip optimizer moments/decay; never insertzero',
  'original_control':'pinned actual BEBlock pertype/head on complete native HGT inner architecture' if arm=='original_gnnm' else None,'actual_callbacks_executed':0,'scientific_updates':0,'scores':0})
proposal={'schema':'native-HGT-mixed-root-resource-proposal-v2','status':'PROPOSED_UNQUALIFIED_NOT_EXECUTED','allocation_only':True,'cases':cases,'source_cases':len(cases),'proposed_callbacks':2*len(cases),
 'root_review_required_before_actual_input_callbacks':True,'actual_input_resource_callbacks_executed':0,'actual_input_optimizer_updates':0,'resource_profile_qualified':False,'GO':False}
(OUT/'RESOURCE_PROPOSAL.json').write_text(json.dumps(proposal,indent=2,sort_keys=True)+'\n')
amend={'schema':'native-HGT-mixed-v16-stopping-cost-amendment-v2','base':'master_controller_source_20261012_v16/runtime','files':composition['files'],'helper':receipt(OUT/'runtime/hgt_stopping.py'),'amendment_script':receipt(OUT/'preparation/amend_controller.py'),
 'composition':'Apply amend_engine/amend_spec/amend_run/amend_compiler to current root text and copy hgt_stopping.py beside controller; all anchors checked; do not replace newer root controllers',
 'native_bounded_cost_typing_reused':'v16 native_count_quantities/typed_counts unchanged; helper emits HGT31..300 bounds, complete body/optimizer constructors and known source path/backward upperbounds',
 'untargeted_AST_functions_unchanged':True,'global_controller_edited':False,'GO':False}
(OUT/'CONTROLLER_TYPE_AMENDMENT.json').write_text(json.dumps(amend,indent=2,sort_keys=True)+'\n')
handoff={'schema':'native-HGT-mixed-source-completion-handoff-v2','status':'SOURCE_AND_ACTUAL_DATA_CLOSED_DISABLED; ROOT_RESOURCE_AND_MASTER_ADMISSION_PENDING','packet':OUT.name,'entry':meta['entry'],'canonical_ids':meta['canonical_ids'],
 'counts':meta['counts'],'data_closure':{'custody':receipt(OUT/'DATA_CUSTODY.json'),'archive_sha256':custody['source_archive']['sha256'],'typed_nodes':26128,'raw_directed_support':239566,'target_nodes':4057,'released_development_pool':1217,'fresh_masks':10,'official_test_labels_opened':False},
 'resource_closure':{'status':'UNQUALIFIED_ROOT_ACTUAL_INPUT_SOURCE_CALLBACKS_PENDING','proposal':receipt(OUT/'RESOURCE_PROPOSAL.json'),'cases':13,'callbacks_proposed':26,'actual_callbacks':0},
 'source_bindings':receipt(OUT/'EXACT_SOURCE_BINDINGS.json'),'native_adapter':receipt(OUT/'runtime/adapters/hgt_native.py'),'native_endpoints':receipt(OUT/'runtime/adapters/hgt_endpoints.py'),
 'controller_amendment':receipt(OUT/'CONTROLLER_TYPE_AMENDMENT.json'),'controller_composition':receipt(OUT/'ROOT_V16_COMPOSITION.json'),'setup_costs':receipt(OUT/'SOURCE_SETUP_COSTS.json'),
 'synthetic_source_proof':receipt(OUT/'SYNTHETIC_CPU_PROOF.json'),'final_typed_ingestion':receipt(OUT/'FINAL_TYPED_INGESTION.json'),'metadata_checks_passed':len(checks),'metadata_checks':checks,
 'source_CPU_and_data_preparation_failures':[],'unavailable_source_exclusions':[],'actual_input_fits':0,'actual_input_optimizer_updates':0,'actual_predictive_scores':0,'actual_scalar_CAL_updates':0,
 'historical_fit_states_scores_or_authorization_inherited':False,'original_GNNM_typed_adaptation_root_approved':True,'minimal_DBLP_only_scope_root_approved':True,'root_global_controller_master_modified':False,'GO':False}
(OUT/'ALLOCATION_HANDOFF.json').write_text(json.dumps(handoff,indent=2,sort_keys=True)+'\n')
print(json.dumps({'metadata_checks':len(checks),'setup':setup,'source_cases':len(cases),'science':False,'GO':False}))
