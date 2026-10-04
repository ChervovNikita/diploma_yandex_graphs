"""Seal a DOI-specific public-access assessment without model/source execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
UTC=datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)}
def save(n,x):(HERE/n).write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n')

names=[
 'literature_memory/index_v51/LITERATURE_INDEX.json','literature_memory/index_v51/ROOT_ADOPTION_NOTES.md','literature_memory/index_v51/SEAL.json',
 'graph_member_filter_quality_gap_20261003_v1/REPORT.md',
 'graph_member_filter_quality_gap_20261003_v1/discovery/GENN_ACCESS.json',
 'graph_member_filter_quality_gap_20261003_v1/discovery/GENN_CROSSREF.json',
 'graph_member_filter_quality_gap_20261003_v1/CANDIDATE_DISPOSITIONS.json',
 'closest_graph_ensemble_sources_v1/RETRIEVAL_MANIFEST.json',
 'closest_graph_ensemble_sources_v1/sources/genn_openalex.json',
 'accuracy_first_graph_ensemble_direction_20261004_v1/DECISION.md',
 'accuracy_first_graph_ensemble_direction_20261004_v1/AMENDMENT_LABEL_VISIBILITY_AND_EXPOSURE.md']
refs=[ref(ROOT/n) for n in names]
assert refs[0]['sha256']=='9b2e4d1f3e55a0b58646c01e91ce4cbaa8b853b7dfe791fd35482c896bf03b1a'
save('PRIOR_CONTEXT.json',{'UTC':UTC,'consulted_cached_sources':refs,'source_index':refs[0],
 'consultation_order':'v51 and previous conclusions/access receipts before new public discovery',
 'retained_hypothesis_sources':refs[-2:],'superseding_label_visibility_and_exposure_amendment_governs':True,
 'source_implementation_feasibility_not_audited':True,'closed_historical_summary_exposure_not_adopted':True})

old=json.loads((ROOT/'closest_graph_ensemble_sources_v1/RETRIEVAL_MANIFEST.json').read_text())
old_rows=[]
def walk(q):
 if isinstance(q,dict):
  if any('genn' in str(v).lower() or '102461' in str(v) for v in q.values() if not isinstance(v,(list,dict))):
   old_rows.append({k:q[k] for k in ['name','url','retrieved_at_utc','status','final_url','path','bytes','sha256','evidence_class','error'] if k in q})
  else:
   for v in q.values():walk(v)
 elif isinstance(q,list):
  for v in q:walk(v)
walk(old)
save('PREVIOUS_ACCESS_HISTORY.json',{'old_receipts':old_rows,
 'recent_abstract_route_receipt':json.loads((ROOT/names[4]).read_text()),
 'publisher_and_exact_title_arxiv_failures_not_identically_retried':True,
 'old_openalex_open_access':json.loads((ROOT/names[8]).read_text())['open_access']})

index=json.loads((ROOT/names[0]).read_text())
matches=[i for i,p in enumerate(index['paper_records']) if '10.1016/j.inffus.2024.102461' in json.dumps(p)]
save('IDENTITY_CHECK.json',{'DOI':'10.1016/j.inffus.2024.102461','title':'Graph ensemble neural network',
 'authors':['Rui Duan','Chungang Yan','Junli Wang','Changjun Jiang'],
 'venue':'Information Fusion','volume':'110','article':'102461','publication':'October 2024',
 'matching_v51_method_record_indices':matches,'previous_phase_metadata_access_history_preserved':True,
 'absence_from_method_records_is_unread_or_novelty_claim':False,'other_article_methods_accessed':False})

receipts=[json.loads(p.read_text()) for p in sorted((HERE/'discovery').glob('*.receipt.json'))]
for r in receipts:
 p=HERE/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
assert len(receipts)==25
dispositions={
 'google_exact_title.html':'redirect shell; no usable search results',
 'bing_exact_title_pdf.html':'unrelated search results; no usable matching carrier',
 'semantic_scholar_doi.json':'exact identity; CLOSED and empty openAccessPdf URL',
 'openaire_doi.json':'timeout; repository availability unresolved',
 'arxiv_doi.xml':'zero DOI query entries; not a global absence certificate',
 'ddg_exact_title.html':'access challenge, not solved',
 'yahoo_exact_title.html':'HTTP500 discovery failure',
 'github_exact_title.json':'four unrelated candidates; not certified target-author repositories or method-read',
 'rui_duan_orcid_person.xml':'known first author; no researcher URL',
 'rui_duan_semantic_author.json':'known first author; homepage null',
 'google_scholar_exact_title.html':'HTTP403 discovery failure',
 'baidu_exact_title.html':'redirected to access challenge, not solved',
 'chungang_yan_orcid_person.xml':'known coauthor; no researcher URL',
 'changjun_jiang_orcid_person.xml':'known corresponding author; no researcher URL',
 'github_rui_duan_users.json':'ten name matches; name alone does not certify authorship'}
save('DISCOVERY_DISPOSITIONS.json',[
 {'request':r,'disposition':dispositions.get(r['name'],'public professional profile only; no verified article association/copy'),
  'primary_method_read':False} for r in receipts])
save('BROWSER_ATTEMPT.json',{'in_app_browser_create':'unavailable','browser_inventory':[],
 'search_page_was_not_opened':True,'challenge_solved':False,'paywall_or_access_control_bypassed':False})
save('SEARCH_ACCOUNTING.json',{'public_http_get_requests':len(receipts),
 'http_status_counts':{str(k):sum(r.get('status')==k for r in receipts) for k in [200,202,403,500,None]},
 'requests_are_discovery_or_author_professional_metadata_only':True,
 'primary_pdf_or_fulltext_requests':0,'legitimate_matching_primary_carriers_recovered':0,
 'public_challenges_not_solved':True,'prior_failed_identical_publisher_requests_repeated':0,
 'global_absence_or_exhaustive_coverage_claim':False,'direct_author_messages':0})
save('READ_SCOPES.json',{'papers':[],'target_identity_metadata_only':'doi:10.1016/j.inffus.2024.102461',
 'new_scoped_primary_method_reads':0,'new_primary_scope_extensions':0,'primary_carrier_rereads':0,
 'full_paper_certifications':0,'author_code_audits':0,
 'unresolved_requested_sections':['method','parameter sharing','continuing member trajectories','graph views','loss/objective','serving pooling'],
 'secondary_hgen_characterization_not_primary_method_evidence':True,
 'incidental_exposure':'Requested old analytical report includes closed historical study summaries; not adopted. No current partial outcomes opened.'})
save('PAPER_CONCLUSIONS.json',[{'canonical_id':'doi:10.1016/j.inffus.2024.102461',
 'title':'Graph ensemble neural network','authors':['Rui Duan','Chungang Yan','Junli Wang','Changjun Jiang'],
 'read_status':'metadata and access-route resolution only; primary method unavailable in bounded legitimate search',
 'new_primary_method_read':False,'exact_method_scope':None,
 'sharing':'unresolved','member_trajectories':'unresolved','graph_views':'unresolved','loss':'unresolved','pooling':'unresolved',
 'comparison_with_gnnm':'Cannot establish exact duplication or an architectural distinction without primary method content.',
 'comparison_with_tied_untied_hypothesis':'Cannot clear or establish overlap with tying effects under matched native-plus-semantic-view CE, bank exposure, selector and probability pool.',
 'novelty_clearance':False,'public_benefits_transferred':False,'full_paper_certification':False}])
save('CONCLUSIONS.json',{'UTC':UTC,'decision':'PRIMARY_ACCESS_LIMIT_PRESERVED',
 'legitimate_matching_primary_fulltext_found':False,'global_copy_absence_claim':False,
 'closest_prior_blind_spot_resolved_at_method_level':False,'blind_spot_documented_with_fresh_bounded_discovery':True,
 'parameter_sharing_member_trajectories_graph_views_loss_pooling_remain_unresolved':True,
 'exact_duplicate_or_architectural_distinction_established':False,'novelty_clearance':False,
 'retained_hypothesis':'Attributed pooled-quality effect of tying versus otherwise equivalent untied private trajectories under identical native-plus-assigned semantic view CE, full TRAIN labels, bank exposure/checkpoint selector and fixed probability pool.',
 'superseding_visibility_amendment_pinned':True,'new_pilot_or_execution_admission':False,
 'new_primary_method_reads':0,'scientific_runs':0,'server_dataset_partial_outcome_accesses':0,
 'canonical_or_existing_packet_edits':0,'report':'REPORT.txt'})
save('VERIFICATION.json',{'UTC':UTC,'v51_pin_verified':True,'prior_context_binding_count':len(refs),
 'all_new_response_digests_verified':True,'new_public_request_count':len(receipts),
 'new_primary_method_reads':0,'current_partial_outcomes_unopened':True,'no_other_article_method_access':True,
 'no_challenge_or_paywall_bypass':True,'new_folder_only_writes':True,
 'report_word_count':len((HERE/'REPORT.txt').read_text().split())})
payload=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name not in ['MANIFEST.json','SEAL.json'])
save('MANIFEST.json',{'schema':'sha256_manifest_v1','UTC':UTC,'files':[{'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in payload],'file_count':len(payload),'execution_authorized':False})
save('SEAL.json',{'schema':'sha256_seal_v1','UTC':UTC,'manifest':ref(HERE/'MANIFEST.json'),'report':ref(HERE/'REPORT.txt'),'conclusions':ref(HERE/'CONCLUSIONS.json'),'source_index':refs[0],'execution_authorized':False})
print(json.dumps({'manifest_sha256':sha(HERE/'MANIFEST.json'),'seal_sha256':sha(HERE/'SEAL.json'),'payload_files':len(payload),'new_primary_method_reads':0,'public_get_requests':len(receipts)},indent=2))
