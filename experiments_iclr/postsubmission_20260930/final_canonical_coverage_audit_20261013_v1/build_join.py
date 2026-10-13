"""Reusable canonical recipe join. Reads frozen JSON/CSV metadata only."""
import argparse, collections, csv, gzip, hashlib, json
from pathlib import Path

P=Path(__file__).resolve().parents[1]
D=P/'training_pipeline_v2_20261012/docs'
H=Path(__file__).resolve().parent

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def receipt(p):
 p=p.resolve();raw=p.read_bytes();return {'path':str(p.relative_to(P)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def load(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def source_hashes(v):
 out=set()
 if isinstance(v,dict):
  if isinstance(v.get('sha256'),str):out.add(v['sha256'])
  for x in v.values():out.update(source_hashes(x))
 elif isinstance(v,list):
  for x in v:out.update(source_hashes(x))
 return out
def dotted(v,k):
 for x in k.split('.'):
  if not isinstance(v,dict) or x not in v:return None
  v=v[x]
 return v

# Every nonliteral ID mapping names the actual prepared method and, where
# available, the consumed selector field. These are scoped recipe mappings,
# never an assertion that all historical aliases share a training procedure.
MAP={
 'buddy_factorized4':(['buddy_shared_cache_factorized4'],{'buddy_arm':'factorized4'}),
 'ordinary_single_alignment_residual':(['molecular_single_contrastive'],{'arm':'single_contrastive','training.views_per_update':2}),
 'allmap_shared4_alignment_residual':(['molecular_be_init_contrastive','molecular_be_unit_contrastive'],{'training.views_per_update':2}),
 'allmap_factor_single_mean4_train_dropout':(['polyformer_equipped_own_mean4_dropout','polyformer_equipped_first_own_mean4_dropout'],{'train_views':4,'members':1}),
 'ordinary_single_mean4_train_dropout':(['polyformer_native_own_mean4_dropout'],{'train_views':4,'members':1}),
 'internal_be_single8_single8_attribution':(['internal_BE_single8_own','internal_BE_single8_alignment','internal_BE_single8_virtual_contrast'],{'mechanism_arm':'single8_attribution'}),
 'graph_relation_private_credit_independent_native_local_scorer':(['graph_relation_private_credit_independent_native_local_scorer_alphaF','graph_relation_private_credit_independent_native_local_scorer_relationJ'],{}),
 'sehgnn_joint_additive_joint_source_additive':(['ADD_PS'],{'sehgnn_arm':'ADD_PS'}),
 'sehgnn_joint_additive_native_source_view_controls':(['ADD_P','BE_P','BE_PS','native_single_P','rank1_single_P','rank4_single_P','independent_ADD_P','independent_ADD_PS','ADD_P_copied_dictionary','ADD_PS_neutral_all_source_average'],{}),
 'nsd_shared_incidence_assigned':(['nsd_shared_incidence_assigned_fixed','nsd_shared_incidence_assigned_resampled'],{}),
 'endpoint_frames_same_operation_single_and_i4':(['endpoint_capable_one_frame_native_M1','endpoint_capable_one_frame_genuine_I4','endpoint_capable_same_four_frames_native_M1'],{}),
 'masked_context_reconstruction_shared4_own':(['source_polyformer_shared4_own'],{'core_condition':'shared4_own'}),
 'masked_context_reconstruction_single_native':(['source_polyformer_native_single'],{'core_condition':'single_native'}),
}
for x in ('shared_own_only','native_pool_credit','source_view_supervision','uncoupled_source_contrast','common_cycle','assigned_source_supply'):
 name='COMMON_cycle' if x=='common_cycle' else x
 MAP['sehgnn_source_supply_'+x]=([name],{'sehgnn_arm':name})
for x in ('common_only','fixed_first_graph_pair','selected_graph_pair','selected_permuted_span','selected_random_span'):
 MAP['graph_curvature_initialization_'+x]=(['curvature_'+x],{'curvature_arm':x})
for x in ('b4','g1','gf1','gm4','gn4','gi16','gfi16'):
 MAP['private_gru_memory_'+x]=(['private_gru_standardized_'+x.upper()],{} if x in ('gi16','gfi16') else {'gru_arm':x.upper()})
for x in ('b1','b4','t16','s64','fs64','lm4'):
 MAP['landmark_memory_'+x]=(['landmark_standardized_'+x.upper()],{'landmark_arm':x.upper()})
MAP['landmark_memory_i4_s64']=(['landmark_standardized_I4[S64]'],{})
MAP['landmark_memory_i4_fs64']=(['landmark_standardized_I4[FS64]'],{})
for x in ('head_only','mixed_single','private4','middle_state_mean','bias_coordinate_single'):
 MAP['private_pt_continuation_'+x]=(['retained_pt_'+x],{'pt_arm':x})
MAP['private_pt_continuation_i4_mixed']=(['retained_pt_mixed_i4'],{})
MAP['private_pt_continuation_i4_bias_coordinate']=(['retained_pt_bias_coordinate_i4'],{})
for x in ('active_reversible_exp','pre_sigmoid_split','full_qk'):
 for own in ('single','be_init','independent4'):
  MAP['pre_sigmoid_qk_'+x+'_'+own]=(['native_polynormer_qk_'+x+'_'+own],{} if own=='independent4' else {'polynormer_qk_arm':x+'_'+own})

PROOF_FILES={
 'private_graph_rotation_shared4_coherent':'private_graph_rotation_q21_successor_20261012_v1/Q21_FINITE_BANK_EQUIVALENCE.json',
 'private_graph_rotation_shared4_paired_graph':'private_graph_rotation_q21_successor_20261012_v1/Q21_FINITE_BANK_EQUIVALENCE.json',
 'private_graph_rotation_shared4_rank1_lora_graph':'private_graph_rotation_q21_successor_20261012_v1/Q21_FINITE_BANK_EQUIVALENCE.json',
 'ncnc_factorized_private_completion4':'ncnc_shared_replay_successor_20261012_v3/LEARNING_LAW_AST_PROOF.json',
 'ncnc_factorized_pooled_after_clamp_4':'ncnc_shared_replay_successor_20261012_v3/LEARNING_LAW_AST_PROOF.json',
 'bsnn_deterministic_d2':'retained_final_five_batch_20261012_v1/bsnn_ready_v1/CLOSED_HANDOFF.json',
 'bsnn_cayley_d2':'native_bsnn_bayesian_composition_bridge_20261012_v1/ALLOCATION_HANDOFF.json',
 'bsnn_richer_cayley_d2_l4':'native_bsnn_bayesian_composition_bridge_20261012_v1/ALLOCATION_HANDOFF.json',
}
rotation=load(P/'private_graph_rotation_q21_successor_20261012_v1/Q21_FINITE_BANK_EQUIVALENCE.json')
for k,v in rotation['canonical_to_existing_methods'].items():MAP[k]=([v],{})
sehgnn_coverage=load(P/'sehgnn_amp_cycle_successor_20261012_v1/CANONICAL_COVERAGE.json')
for row in sehgnn_coverage['records']:
 PROOF_FILES[row['canonical_method_id']]='sehgnn_amp_cycle_successor_20261012_v1/CANONICAL_COVERAGE.json'

INACTIVE={
 'cognn_polynormer_externalmessage':('inactive gated source prototype; RUNTIME_ENABLED=False and no adopted predictive worker','cognn_polynormer_external_message_inactive_prototype_20261007_v1/prototype.py'),
 'forde_graph_gradient_author_identity_force':('source-only force primitive; explicit unadopted graph protocol with learning rate/duration unresolved','retained_final_five_batch_20261012_v1/forde_source_only_exclusion_v1/CLOSED_EXCLUSION.json'),
 'forde_dice_graph_regularization':('derivative oracle only; not a predictive benchmark fitter',None),
 'ar_fitted_screen_synthetic_autoregressive_laws':('synthetic autoregressive screen; no admitted node/link/graph benchmark method',None),
 'spectral_ppi_private_spectral_training':('wrong-allocation evidence and authorized-route exclusion; no access or rerun',None),
 'conditional_fast_factors_unimplemented_proposal':('architecture proposal without verified callable predictive pipeline',None),
 'ar_mixture_decoder_private_recurrent_schedule_laws':('injected-backend synthetic decoder prototype without predictive benchmark adapter',None),
 'ar_shared_transition_private_output_head_control':('synthetic shared-transition source control without predictive benchmark adapter',None),
 'e_fixed_feature_head_capacity_declared_nonlinear_head_controls':('Canonical label says nonlinear controls, but pinned PLAN/run implement BE continuation and affine residuals only; no nonlinear trainer/architecture/budget exists','WikiCS_E_fixed_feature_head_capacity_source_20261011_v1/PLAN.json'),
 'propagation_shared_centered_aggregation':('Explicit fixture-only implementation: permanently closed real pilot CLI, absent complete-data API, fixture-bound constructor; full predictive provider/finite protocol not adopted','propagation_cost_impl_v1/IMPLEMENTATION_STATUS.json'),
}

RESIDUAL={
 'graph_relation_private_credit_configured_relation12':('graph_relation_private_credit_source_20261008_v2/PROTOCOL.json','Four source gradient policies alphaF/allJ/phiJ/relationJ, beta.5, source512/L7local/L2global,1100/local100, two-view disjoint VJP recipients and one Adam. Selected independent-scorer procedures do not implement these partition policies.'),
 'graph_relation_private_credit_configured_relation18':('graph_relation_private_credit_source_20261008_v2/PROTOCOL.json','Exact grouped relation12 plus matched-reference panel must be unpacked from source relation18 pointers. No selected relation-credit gradient-policy DAG; aliases are not18 independent algorithms.'),
 'private_hop_private_hop_controls':('private_hop_credit_pubmed_fullfit_source_20261010_v1/ROSTER.json','Missing source full_aux/common_nonfull/allblock_missinghop/factor1_allview slots. Selected private_missinghop/shared4_own and plain factor1_native do not cover them; source2000cap/p250/seeds9101,9203,9307.'),
 'e_fixed_feature_head_capacity_affine_head':('WikiCS_E_fixed_feature_head_capacity_source_20261011_v1/PLAN.json','Literal21 fixed-feature E/M1/I4 BE-continuation/zero-affine banks,39 own head optimizers, AdamW.001/WD0,1000cap/p300. Missing fresh current-role endpoint-VALUE donor/feature extraction/head fitting DAG.'),
 'structural_pattern_exact_count_conditioned':('graph_count_conditioned_pattern_loss_prototype_preparation_20261004_v1/CONTRACT.json','Implemented differentiable J_K/J_K_sep/W_K conditional objective and TRAIN teacher law are absent from selected native NCNC callbacks. Needs exact native neural schedule/provider integration, not merely imported algebra.'),
 'structural_pattern_exact_support_bucketed':('exact_cb_support_bucket_implementation_hypothesis_preparation_20261004_v1/PLAN.json','Implemented bucketed execution preserves the same conditional objective and neural work; selected callbacks never invoke it. Missing exact complete-support integration; no extra scientific algorithm/fit count inferred.'),
 'structural_pattern_cardinality_single':('ncnc_cardinality_single_comparator_implementation_preparation_20261003_v1/IMPLEMENTATION_CONTRACT.json','Implemented make_single/train_epoch/complete VALID selector/D4-M4 serving, fixed seed0/100epochs/lambda1, native65536/drop-tail, FP64 full-count DP. Missing integrated fitter/DAG/new-role custody, fixed-bank M4 endpoint and complete-work budget.'),
 'structural_pattern_private_structural_interaction':('shared_NCN_private_structural_interaction_prototype_20261006_v3/FIXED_RECIPE.json','Implemented run/models.build,18 fixed source cells/60cycles/eval5,Adam.001/WD0, first max rounded4 MRR, structural_private/tied/informed single/I4/count-only/blind controls. Missing source structural message callback and finite DAG; preaggregation/private response are different operations.'),
}
for x in ('be_factor_refit','be_full_refit','ordinary_full_refit','single_full_refit'):
 RESIDUAL['direct_hidden_head_refit_'+x]=('Wiki24_direct12_head_refit_source_20261008_v2/FROZEN_CONVENTIONS.json','Callable heads.fit/collect_features/run_refits: frozen upstream, TRAIN-only mean member CE + .001/2 class-centered effective L2, CPU FP64 LBFGS max_iter1000/max_eval50000/no SELECT or restart. Missing fresh matched selected donors/feature extraction/postfit DAG.')
for x in ('self_reliability','graph_reliability'):
 RESIDUAL['learned_graph_serving_'+x]=('common_wrapper_graph_reliability_source_20261010_v2/FROZEN_SUPPORT.json','Callable operators.fit_fold/study.run: same held-fold excluded serving fitting, five folds,500 Adam.01 updates, fixed final endpoint; complete45 banks/855 control/fusion calls. Missing fresh selected bank ancestry/label-free neighbor exports/current role postfit DAG. Source encountered VALID crossfit is not whole-pipeline unused confirmation.')

def main():
 a=argparse.ArgumentParser();a.add_argument('--snapshot',required=True);a.add_argument('--output',required=True);a.add_argument('--master-proof');args=a.parse_args()
 out=Path(args.output);assert not out.exists();out.mkdir()
 inv=list(csv.DictReader((D/'METHOD_INVENTORY.csv').open()));cov={r['canonical_method_id']:r for r in csv.DictReader((D/'ALL_METHOD_COVERAGE_20261012.csv').open())};aliases=list(csv.DictReader((D/'METHOD_SOURCE_ALIASES.csv').open()))
 assert len(inv)==273 and len({r['canonical_method_id'] for r in inv})==273 and len(aliases)==361
 snap=load(Path(args.snapshot));by_alias=collections.defaultdict(list)
 for r in aliases:by_alias[r['canonical_method_id']].append(r)
 contracts=[]
 for s in snap['studies']:
  for g in s['records']:
   contracts.append((s,g,source_hashes(s['source'])|source_hashes(g['config'])))
 contract_hashes={(s['study_id'],g['recipe_hash']):shas for s,g,shas in contracts}
 all_prepared_source_hashes=set().union(*(ss for _,_,ss in contracts))
 rows=[];alias_rows=[];inputs=[receipt(D/'METHOD_INVENTORY.csv'),receipt(D/'ALL_METHOD_COVERAGE_20261012.csv'),receipt(D/'METHOD_SOURCE_ALIASES.csv'),receipt(Path(args.snapshot)),receipt(Path(__file__))]
 for row in inv:
  key=row['canonical_method_id'];old=cov[key];own_aliases=by_alias[key];names,requirements=MAP.get(key,([key],{}));source_pins={r['source_sha256'] for r in own_aliases}|{row['source_sha256']}
  factory_pins={v.rsplit('=',1)[-1] for v in old['current_factory_sha256'].split(';') if '=' in v}
  matches=[];rejections=[]
  proof=None
  if key in PROOF_FILES:proof=receipt(P/PROOF_FILES[key])
  for s,g,shas in contracts:
   if g['method'] not in names:continue
   if g['kind']=='model_fit' and any(dotted(g['config'],k)!=v for k,v in requirements.items()):continue
   source_match=bool(source_pins & shas);factory_match=bool(factory_pins & shas)
   if not (source_match or factory_match or proof):
    rejections.append({'study_id':s['study_id'],'method':g['method'],'reason':'matching label/selector but no exact source pin or prior source-mapping proof'});continue
   if not g['adapter'] or not isinstance(g['config'],dict):continue
   selector_fields={k:v for k,v in g['config'].items() if k.endswith('_arm') or k.endswith('_condition') or k in ('arm','condition','architecture','members','source_contract')}
   matches.append({'packet_id':s['packet_id'],'study_id':s['study_id'],'dataset':s['dataset'],'spec_hash':s['spec_hash'],'method':g['method'],'kind':g['kind'],'recipe_hash':g['recipe_hash'],'count':g['count'],'adapter':g['adapter'],'budget':g['budget'],'selection':g['selection'],'source_pin_match':source_match,'existing_qualified_factory_pin_match':factory_match,'source_mapping_proof':proof,'consumed_selector_fields':selector_fields,'required_selector_predicates':requirements,'fit_parents':g['config'].get('fit_parents'),'fusion':g['config'].get('fusion'),'seed_values':g['seed_values'],'declared_recipe_scope':g['config'].get('recipe_scope') or g['config'].get('provider_role_horizon_adaptation') or g['config'].get('source_contract') or 'exact config/budget/selector preserved in snapshot; whole archived alias equivalence not asserted'})
  predictive=row['grid_eligibility']=='predictive_candidate_after_migration'
  if key in INACTIVE:
   reason,path=INACTIVE[key];category='inactive_nonpredictive_exclusion';status='explicit_inactive_or_nonbenchmark_source';evidence=receipt(P/path) if path else {'inventory_source_entrypoint':row['source_entrypoint'],'source_sha256':row['source_sha256'],'eligibility':row['grid_eligibility'],'reason':row['known_fidelity_and_training_issues']}
  elif key in RESIDUAL:
   path,reason=RESIDUAL[key];category='missing_predictive_port';status='implemented_predictive_or_control_stage_missing_from_DAG';evidence=receipt(P/path)
  elif matches:
   category='scientific_runnable' if row['method_kind']=='scientific_variant' else 'reference_control';status='prepared_selected_scoped_recipe';reason='Exact selected config/budget/selector/adapter and source binding joined; scientific admission and GO remain root gates';evidence=None
  elif not predictive:
   category='reference_control';status='unselected_engineering_grouped_or_postfit_request';reason='No selected finite predictive DAG contract for this nonpredictive eligibility row; source request retained explicitly, not promoted or excluded by old history';evidence=None
  else:
   category='missing_predictive_port';status='no_exact_selected_recipe_join';reason='No exact selected recipe/source contract joined; source/history preserved';evidence=None
  for alias in own_aliases:
   occurrences=sorted({m['study_id'] for m in matches if alias['source_sha256'] in contract_hashes[(m['study_id'],m['recipe_hash'])]})
   alias_rows.append({**alias,'selected_source_code_studies':occurrences,'recipe_equality_to_historical_alias':'not_asserted','qualification_status':'source_code_pin_present_in_selected_scoped_recipe' if occurrences else 'historical_source_alias_retained_without_selected_recipe_equality','historical_evidence_promoted':False})
  rows.append({'canonical_method_id':key,'family':row['family'],'method_kind':row['method_kind'],'inventory_grid_eligibility':row['grid_eligibility'],'prior_coverage_status':old['coverage_status'],'category':category,'status':status,'reason':reason,'source_entrypoint':row['source_entrypoint'],'source_sha256':row['source_sha256'],'source_alias_ids':[r['method_id'] for r in own_aliases],'source_alias_count':len(own_aliases),'matches':matches,'rejected_name_only_candidates':rejections,'evidence':evidence,'historical_recipe_equality_claimed':False,'scientific_launch_ready':False,'new_fit_obligation_from_old_history':False})
 counts=dict(collections.Counter(r['category'] for r in rows));status_counts=dict(collections.Counter(r['status'] for r in rows));missing=[r['canonical_method_id'] for r in rows if r['category']=='missing_predictive_port'];unselected=[r['canonical_method_id'] for r in rows if r['status']=='unselected_engineering_grouped_or_postfit_request']
 report={'schema':'all273-selected-exact-recipe-coverage-join-v1','inputs':inputs,'snapshot_index':snap['index_receipt'],'study_count':snap['study_count'],'model_fit_groups':snap['model_fits'],'counts':counts,'status_counts':status_counts,'canonical_rows':rows,'source_alias_count':361,'genuine_missing_predictive_methods':missing,'unselected_nonpredictive_requests':unselected,'historical_alias_recipe_equality_not_asserted':True,'aliases_not_extra_methods':True,'history_based_exclusions':0,'source_label_alone_not_coverage':True,'scientific_runnable_definition':'Prepared selected predictive recipe only. Root source/resource/AMP/master/GO admission remains; no launched fit or quality claim.','all273_accounted':len(rows)==273,'no_scientific_training_or_optimizer_imports':True,'heldout_array_reads':0,'scientific_fits':0,'optimizer_updates':0,'GO':False}
 if args.master_proof:report['final_master_binding']=load(Path(args.master_proof));inputs.append(receipt(Path(args.master_proof)))
 (out/'COVERAGE_JOIN.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 (out/'ALL_SOURCE_RECIPE_ALIASES.json.gz').write_bytes(gzip.compress(canonical(alias_rows).encode(),mtime=0))
 reference_contracts=[{'packet_id':s['packet_id'],'study_id':s['study_id'],'method':g['method'],'recipe_hash':g['recipe_hash'],'config':g['config'],'count':g['count']} for s,g,_ in contracts if g['kind']=='ensemble_assembly' and ('I4' in g['method'] or 'independent' in g['method'] or len(g['config'].get('fit_parents',[]))>1)]
 (out/'EXACT_MATCHED_REFERENCE_CONTRACTS.json.gz').write_bytes(gzip.compress(canonical(reference_contracts).encode(),mtime=0))
 rules={'schema':'reviewed-selected-recipe-authority-v1','master_hash':report.get('final_master_binding',{}).get('master_plan_hash'),'snapshot_receipt':receipt(Path(args.snapshot)),'rules':[{k:r[k] for k in ('canonical_method_id','category','status','matches')} for r in rows],'future_reuse':'Only these exact recipe hashes/source bindings are reviewed. A changed objective/selector/fusion/ownership/ancestry needs its own source-port proof; names alone never extend coverage.'}
 (out/'SELECTED_RECIPE_AUTHORITY.json.gz').write_bytes(gzip.compress(canonical(rules).encode(),mtime=0))
 with (out/'ALL273_COVERAGE.csv').open('w') as stream:
  fields=['canonical_method_id','family','method_kind','inventory_grid_eligibility','category','status','source_alias_count','selected_packet_ids','selected_study_ids','selected_recipe_hashes','source_binding_verified','reason']
  writer=csv.DictWriter(stream,fields);writer.writeheader()
  for r in rows:writer.writerow({**{k:r[k] for k in fields if k in r},'selected_packet_ids':';'.join(sorted({m['packet_id'] for m in r['matches']})),'selected_study_ids':';'.join(sorted({m['study_id'] for m in r['matches']})),'selected_recipe_hashes':';'.join(sorted({m['recipe_hash'] for m in r['matches']})),'source_binding_verified':bool(r['matches'])})
 summary={k:report[k] for k in ('schema','study_count','model_fit_groups','counts','status_counts','genuine_missing_predictive_methods','unselected_nonpredictive_requests','all273_accounted','history_based_exclusions','scientific_fits','optimizer_updates','heldout_array_reads','GO')}
 (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
