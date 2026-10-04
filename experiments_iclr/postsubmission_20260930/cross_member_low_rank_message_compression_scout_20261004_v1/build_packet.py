"""Package a bounded theoretical assessment; no numerical/model execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
UTC=datetime.now(timezone.utc).isoformat()

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)), 'bytes':p.stat().st_size,'sha256':sha(p)}
def save(n,x):(HERE/n).write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n')

prior_names=[
 'literature_memory/index_v50/LITERATURE_INDEX.json',
 'literature_memory/index_v50/ROOT_ADOPTION_NOTES.md',
 'literature_memory/index_v50/SEAL.json',
 'member_subspace_messages_v1/REPORT.md',
 'member_subspace_messages_v1/low_rank_communication_assessment_v1/ASSESSMENT.md',
 'member_subspace_messages_v1/low_rank_communication_assessment_v1/EVIDENCE.json',
 'member_subspace_messages_v1/graph_replica_analysis_v1/ANALYSIS.md',
 'adaptive_sharing_novelty_v1/REPORT.md',
 'shared_message_closest_prior_v1/REPORT.md',
 'efficient_paths/CANDIDATE.md',
 'efficient_paths/LITERATURE.md']
refs=[ref(ROOT/n) for n in prior_names]
assert refs[0]['sha256']=='62b5dd473aea9573be7039e4981861609111db88df658d9acd742949ce699fe5'
index=json.loads((ROOT/prior_names[0]).read_text())
ids=['1604.03539','1705.08142','2003.02436','2006.16362','2304.02806','2002.06715','2410.24210','2210.09184','2409.16670','2607.10077','2004.11198','2405.03401']
selected=[{'record_number_zero_based':i,'record':p} for i,p in enumerate(index['paper_records']) if any(q in str(p.get('canonical_id','')) for q in ids)]
save('REUSED_INDEX_RECORDS.json',selected)
evidence=json.loads((ROOT/'member_subspace_messages_v1/low_rank_communication_assessment_v1/EVIDENCE.json').read_text())
papers=[]
for p in evidence['primary_sources']:
 binding=p['saved_text']; source=ROOT/binding['path']
 assert source.stat().st_size==binding['bytes'] and sha(source)==binding['sha256']
 papers.append({'versioned_id':p['version'],'title':p['title'],'primary_url':p['primary_url'],
   'saved_primary_binding':binding,'prior_recorded_scope_locations':[q.get('section') for q in p['inspected_passages']],
   'current_read_status':'cached analytical conclusion/scope metadata reuse; original carrier not semantically reopened',
   'new_primary_method_read':False,'scope_extension':False,'full_paper_certification':False,
   'scope_reference':ref(ROOT/'member_subspace_messages_v1/low_rank_communication_assessment_v1/EVIDENCE.json')})
save('REUSED_PRIOR_SCOPES.json',papers)
save('PRIOR_CONTEXT.json',{'UTC':UTC,'source_index':refs[0],'consulted_cached_sources':refs,
 'consultation_order':'v50 before cached mechanism comparisons; no new primary retrievals',
 'exact_internal_predecessor':['member_subspace_messages_v1/REPORT.md','member_subspace_messages_v1/low_rank_communication_assessment_v1/ASSESSMENT.md'],
 'extra_locator_only_context':'Graph-conditioned low-rank scout heading/summary lines were located but not method-reread; no conclusion or source scope added from that locator.',
 'historical_engineering_summaries_incidentally_exposed':True,'historical_values_or_gates_adopted':False})
save('IDENTITY_AND_ACCOUNTING.json',{'UTC':UTC,'source_index':refs[0],
 'cached_phase_primary_identities':ids,'matching_index_conclusion_records':len(selected),
 'index_absence_is_unread_claim':False,'new_public_requests':0,'new_primary_method_papers':0,
 'new_primary_scope_extensions':0,'deliberate_original_primary_carrier_rereads':0,
 'cached_primary_excerpt_display_documents':1,'cached_primary_excerpt_display_was_truncated':True,
 'cached_excerpt_document':refs[5],'possible_cached_paper_exposures_in_document':10,
 'zero_repeated_primary_semantic_exposure_certified':False,
 'full_paper_certifications':0,'author_code_reads':0,'scientific_model_or_numerical_runs':0,
 'project_source_eligibility_reads':0,'data_labels_predictions_weights_checkpoint_server_reads':0,
 'historical_engineering_cost_summaries_exposed_not_adopted':True,
 'new_agents':0,'current_study_or_queue_changes':0,'canonical_index_ledger_status_root_client_edits':0})
save('HYPOTHESIS_LIMITS_AND_CONTROLS.json',{'UTC':UTC,'candidate_status':'duplicate internal family; attributed prospective utility only',
 'operator':'P tensor (11^T/M+UU^T), with U global over nodes/features, U^T1=0 and U^TU=I',
 'rank_zero':'previous shared mean message','rank_full':'R=M-1 gives exact supplied-state messages',
 'private_trajectories_retained':True,'generic_commutation_or_spectral_lipschitz_theorem_novelty':False,
 'quality_hypothesis':'Task-relevant neighbor contrasts occupy a stable low-dimensional member subspace; preset R>0 retains useful complementarity lost at R0 at lower total paid cost than exact.',
 'failure_conditions':['Low-energy task signal discarded by global PCA','Rare node/feature contexts need incompatible member directions','Private maps/nonlinear recurrence amplify discarded error','Global basis changes or becomes stale across layers/contexts','Member-dependent P or node-dependent mixing invalidates simple factorization','Projecting before nonlinear pre-normalization changes the proposed operator'],
 'paid_costs':{'sparse_forward':'O((R+1)E d) versus O(M E d)','centering':'O(M N d)','encoding_reconstruction':'O(M R N d)','dynamic_input_gram_and_eigendecomposition':'O(M^2 N d+M^3)','private_states_and_dense_local_work':'remain full member width','backward':'compressed P^T plus declared basis/encoding/reconstruction derivatives','optimal_propagated_gram':'computing all full P Delta_m first spends full graph cost'},
 'required_controls':['efficient exact full member messages','R0 mean messages','full R=M-1 exact operator check at supplied states','same-rank fixed centered/group basis when basis learning/adaptation is claimed','competent ordinary alternative at same propagated-channel budget with all other costs separately accounted'],
 'falsifier':'Reject utility in the declared setting if R1 fails prospective deployed quality retention or loses the quality/cost tradeoff to R0 and equal-budget alternatives after all basis/projection/private costs. Matching a fixed basis rejects necessary learned orientation.',
 'controls_are_execution_adoption':False,'implementation_eligibility_owned_by_other_source_agent':True,
 'hardware_limit_is_rejection_reason':False})
save('CONCLUSIONS.json',{'UTC':UTC,'decision':'NO-GO for distinct new mechanism: exact duplicate of saved internal member-subspace communication family.',
 'exact_internal_candidate_duplicate_verified':True,'exact_published_complete_architecture_duplicate_verified':False,
 'global_novelty_clearance':False,'new_trainable_principle_established':False,
 'remaining_attributable_question':'Prospective deployed quality/cost utility of low-rank member-axis neighbor communication with full private local trajectories.',
 'standard_algebra_subspace_spectral_lipschitz_results_are_prior':True,
 'predictive_quality_preservation_proved_or_observed':False,'end_to_end_cost_advantage_proved_or_observed':False,
 'new_scoped_primary_method_reads':0,'new_scope_extensions':0,'full_paper_certifications':0,
 'blanket_zero_repeated_primary_semantic_exposure_claim':False,
 'scientific_execution_adopted':False,'implementation_adopted':False,'current_studies_changed':False,
 'report':'REPORT.txt','accounting':'IDENTITY_AND_ACCOUNTING.json','limits_and_controls':'HYPOTHESIS_LIMITS_AND_CONTROLS.json'})
save('VERIFICATION.json',{'UTC':UTC,'v50_pin_verified':True,'cached_primary_binding_digests_mechanically_verified':len(papers),
 'cached_context_binding_count':len(refs),'current_project_source_semantic_audit':False,
 'original_primary_carriers_semantically_opened':False,'incidental_cached_primary_excerpt_repeat_exposure_declared':True,
 'historical_engineering_values_not_adopted':True,'packet_only_writes':True,'scientific_imports_or_runs':False,
 'report_word_count':len((HERE/'REPORT.txt').read_text().split())})
payload=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name not in ['MANIFEST.json','SEAL.json'])
save('MANIFEST.json',{'schema':'sha256_manifest_v1','UTC':UTC,'files':[{'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in payload],'file_count':len(payload),'execution_authorized':False})
save('SEAL.json',{'schema':'sha256_seal_v1','UTC':UTC,'manifest':ref(HERE/'MANIFEST.json'),'report':ref(HERE/'REPORT.txt'),'conclusions':ref(HERE/'CONCLUSIONS.json'),'source_index':refs[0],'execution_authorized':False})
print(json.dumps({'manifest_sha256':sha(HERE/'MANIFEST.json'),'seal_sha256':sha(HERE/'SEAL.json'),'report_word_count':len((HERE/'REPORT.txt').read_text().split()),'new_primary_method_reads':0},indent=2))
