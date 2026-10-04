"""Adopt two already saved scoped conclusions; mechanically verify, never parse primary papers."""
from pathlib import Path
from datetime import datetime, timezone
import copy, hashlib, json, os

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
PREV=ROOT/'literature_memory/index_v49'
SOURCE=ROOT/'member_private_typed_path_quality_gap_scout_20261004_v1'
assert not (OUT/'LITERATURE_INDEX.json').exists(), 'Immutable output already exists'
UTC=datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)}
def dump(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
def check_ref(r):
 p=ROOT/r['path'];assert p.is_file() and sha(p)==r['sha256'],r['path']
 if 'bytes' in r:assert p.stat().st_size==r['bytes'],r['path']
 return ref(p)
def verify_packet(p):
 m=json.loads((p/'MANIFEST.json').read_text());s=json.loads((p/'SEAL.json').read_text())
 expected=s.get('manifest_sha256',s.get('manifest',{}).get('sha256'))
 assert sha(p/'MANIFEST.json')==expected,p
 payload=[]
 for r in m['files']:
  q=p/r['path'];assert q.is_file() and sha(q)==r['sha256'] and q.stat().st_size==r['bytes'],q
  payload.append(ref(q))
 assert {r['path'] for r in m['files']}=={str(q.relative_to(p)) for q in p.rglob('*') if q.is_file() and q.name not in ['MANIFEST.json','SEAL.json']},p
 return {'packet':str(p.relative_to(ROOT)),'manifest':ref(p/'MANIFEST.json'),'seal':ref(p/'SEAL.json'),'payloads_verified':payload,'payload_count':len(payload),'payload_bytes':sum(r['bytes'] for r in payload)}
def metrics(x):
 cat={r['path'] for r in x['existing_packets']};cons={r['conclusion_file'] for r in x['paper_records'] if 'conclusion_file' in r}
 scopes={r['read_scope_reference']['path'] for r in x['paper_records'] if 'read_scope_reference' in r}
 legacy={r['read_scope_file_reference']['path'] for r in x['paper_records'] if 'read_scope_file_reference' in r}
 groups=x['canonical_identifier_normalization']['groups']
 return {'conclusion_records':len(x['paper_records']),'normalized_paper_identifiers':sum(g['kind']=='paper' for g in groups),'software_documentation_identifiers':sum(g['kind']!='paper' for g in groups),'catalog_entries':len(x['existing_packets']),'unique_catalog_document_paths':len(cat),'unique_conclusion_source_documents':len(cons),'unique_referenced_document_paths':len(cat|cons),'unique_scope_reference_document_paths':len(scopes),'unique_scope_reference_document_paths_including_legacy_field_alias':len(scopes|legacy)}

packet_checks=[verify_packet(PREV),verify_packet(SOURCE)]
assert sha(PREV/'LITERATURE_INDEX.json')=='6af95565489025d814c7b68bd390ca97c937cea3d3cce76af991287aa9e91a5a'
assert sha(SOURCE/'MANIFEST.json')=='96dcfef976d671e3b856e01d5580ee12d096513bc369f6701472b68b5f82f0d1'
prior=json.loads((PREV/'LITERATURE_INDEX.json').read_text());index=copy.deepcopy(prior)
catalog_checks=[check_ref(r) for r in prior['existing_packets']]
record_bindings=[]
for r in prior['paper_records']:
 if 'conclusion_file' in r:record_bindings.append(check_ref({'path':r['conclusion_file'],'sha256':r['conclusion_file_sha256']}))
 for k in ['read_scope_reference','read_scope_file_reference']:
  if k in r:record_bindings.append(check_ref(r[k]))
context=json.loads((SOURCE/'PRIOR_CONTEXT.json').read_text())
context_checks=[check_ref(r) for r in context['consulted_sources']]
papers=json.loads((SOURCE/'PAPER_CONCLUSIONS.json').read_text())
scoped=json.loads((SOURCE/'READ_SCOPES.json').read_text())
account=scoped['read_accounting']
assert account=={'new_scoped_primary_paper_identities':2,'new_primary_scope_extensions':0,'repeated_primary_method_reads':0,'indexed_primary_rereads':0,'full_paper_certifications':0,'author_code_audits':0}
assert len(papers)==len(scoped['papers'])==2
identities=json.loads((SOURCE/'SELECTED_IDENTITIES.json').read_text())['papers']
groups=index['canonical_identifier_normalization']['groups'];added=[]
for paper in papers:
 s=next(s for s in scoped['papers'] if s['key']==paper['scope_key'])
 canonical=paper['canonical_id'];assert canonical==s['canonical_id']
 assert not any(g['normalized_identifier']==canonical for g in groups)
 assert paper['primary']==s['primary'];check_ref(s['primary'])
 assert not s['full_paper_read'] and not s['proofs_certified'] and not s['numeric_results_adopted'] and not s['author_code_read']
 metadata=next(m for m in identities if paper['versioned_id'] in m['id'])
 citation={k:metadata[k] for k in ['id','title','authors','published','updated']}
 assert citation['title']==paper['title']==s['title'] and citation['authors']==s['authors']
 pos=len(index['paper_records']);scope_key=hashlib.sha256(json.dumps({'canonical_id':canonical,'versioned_id':paper['versioned_id'],'scope':s['semantic_scope']},sort_keys=True).encode()).hexdigest()
 r={'canonical_id':canonical,'normalized_identifier':canonical,'conclusion':copy.deepcopy(paper),'conclusion_file':str((SOURCE/'PAPER_CONCLUSIONS.json').relative_to(ROOT)),'conclusion_file_sha256':sha(SOURCE/'PAPER_CONCLUSIONS.json'),'citation_metadata':citation,'read_scope_reference':{**ref(SOURCE/'READ_SCOPES.json'),'paper_canonical_id':canonical,'scope_key':s['key']},'compact_primary_scope':copy.deepcopy(s),'exact_read_scope':copy.deepcopy(s['semantic_scope']),'scope_deduplication_key_sha256':scope_key,'primary_payload_reference':copy.deepcopy(s['primary']),'source_packet_manifest_reference':ref(SOURCE/'MANIFEST.json'),'source_packet_seal_reference':ref(SOURCE/'SEAL.json'),'source_report_reference':ref(SOURCE/'REPORT.txt'),'full_paper_read':False,'genuinely_new_scoped_paper_identity_in_source_packet':True,'integration_pass_new_semantic_read':False,'integration_pass_primary_reread':False,'integration_read_status':'Previously completed scoped method read adopted without new primary read','global_novelty_clearance':False,'predictive_adoption':False,'execution_authorized':False}
 index['paper_records'].append(r)
 groups.append({'normalized_identifier':canonical,'kind':'paper','raw_canonical_identifiers':[canonical,'arxiv:'+paper['versioned_id']],'explicit_aliases':[],'record_indices':[pos]})
 added.append({'record_index':pos,'canonical_id':canonical,'normalized_identifier':canonical,'scope':copy.deepcopy(s['semantic_scope']),'scope_deduplication_key_sha256':scope_key,'full_paper_read':False,'genuinely_new_scoped_paper_identity_in_source_packet':True,'integration_new_primary_reads':0})
catalog_names=['PAPER_CONCLUSIONS.json','READ_SCOPES.json','CONCLUSIONS.json','REPORT.txt','POSTERIOR_SERVING_PRIOR_NOTE.txt','LITERATURE_MEMORY_RECORD.json','PHASE_IDENTITY_CHECK.json','PRIOR_CONTEXT.json','REUSED_INDEX_RECORDS.json','SELECTED_IDENTITIES.json','SEARCH_ACCOUNTING.json','DISCOVERY_LOG.json','RETRIEVAL_fastgtn.json','RETRIEVAL_mug.json','VERIFICATION.json','MANIFEST.json','SEAL.json']
for name in catalog_names:
 index['existing_packets'].append({**ref(SOURCE/name),'kind':'stored_scoped_typed_path_negative_novelty_document','scope':'Saved two scoped method conclusions and bounded negative novelty/utility limits. Posterior note reuses prior conclusions; no added paper read, method or execution adoption.'})
old_metrics=metrics(prior);new_metrics=metrics(index)
index['integration_v50_predecessor_v49_snapshot']={'index_reference':ref(PREV/'LITERATURE_INDEX.json'),'actual_recomputed_metrics':old_metrics,'read_accounting':copy.deepcopy(prior['read_accounting']),'latest_adoption':copy.deepcopy(prior['latest_adoption']),'lineage_metadata':{k:copy.deepcopy(prior[k]) for k in ['schema','created_UTC','predecessor_index','predecessor_index_sha256']}}
adoption={'UTC':UTC,'status':'SEALED_ADDITIVE_SCOPED_NEGATIVE_NOVELTY_MEMORY','predecessor':ref(PREV/'LITERATURE_INDEX.json'),'source_packet_manifest':ref(SOURCE/'MANIFEST.json'),'source_packet_seal':ref(SOURCE/'SEAL.json'),'added_records':added,'new_memory_records':2,'new_canonical_paper_identity_groups':2,'new_catalog_entries':len(catalog_names),'source_packet_new_scoped_method_reads':2,'source_packet_scope_extensions':0,'source_packet_repeated_primary_method_reads':0,'source_packet_full_paper_certifications':0,'integration_pass_new_primary_reads':0,'integration_pass_primary_rereads':0,'integration_pass_author_source_semantic_reads':0,'integration_public_requests':0,'prior_records_groups_scopes_aliases_history_and_limits_preserved':True,'negative_typed_path_mechanism_finding_adopted':True,'posterior_gate_generic_and_no_new_information_when_observations_determined_by_backbone_input':True,'method_adoption':False,'predictive_adoption':False,'novelty_clearance':False,'new_proposal_or_launch_adoption':False,'research_ledger_or_status_updated':False,'execution_authorized':False}
index.update(schema='literature-memory-index-v50',created_UTC=UTC,predecessor_index=str((PREV/'LITERATURE_INDEX.json').relative_to(ROOT)),predecessor_index_sha256=sha(PREV/'LITERATURE_INDEX.json'),latest_adoption=copy.deepcopy(adoption))
index['post_v50_append']=copy.deepcopy(adoption)
index['canonical_identifier_normalization']['post_v50_scope_append']={'UTC':UTC,'added_record_indices':[209,210],'new_paper_identities':[r['canonical_id'] for r in papers],'old_groups_preserved_exactly':len(prior['canonical_identifier_normalization']['groups']),'title_only_alias_merges':0,'deduplication':'Exact normalized arXiv identifiers and version-pinned saved metadata. No new DOI alias or title-only merge.','integration_primary_reads':0}
index['typed_path_negative_novelty_and_posterior_limits_v50']={'source_report':ref(SOURCE/'REPORT.txt'),'source_posterior_note':ref(SOURCE/'POSTERIOR_SERVING_PRIOR_NOTE.txt'),'scope':'Negative attribution/utility check only; two new saved method records. No new graph mechanism or initialization rule found. Persistent private nonlinear utility remains untested. Generic graph-likelihood responsibilities are prior; conditioning on observations determined by full graph input adds no information.','new_primary_reads_in_this_integration':0,'new_paper_records_from_posterior_addendum':0,'novelty_or_predictive_or_execution_adoption':False}
index['read_accounting'].update(new_metrics)
index['read_accounting'].update(state='ROOT_DELEGATED_ADDITIVE_SCOPED_NEGATIVE_NOVELTY_MEMORY',historical_path_catalog_note='All v49 records and histories retained. Two previously completed new scoped methods (GTN/FastGTN and MUG) added as negative novelty evidence. Integration adds zero primary reads; posterior addendum is cached prior/derived limits only.',latest_index_growth=2,latest_packet_new_scoped_primary_reads=2,latest_packet_genuinely_new_scoped_paper_identities=2,latest_packet_new_paper_identity_groups=2,latest_packet_previously_completed_scoped_read_adoptions=2,latest_packet_scoped_primary_method_events=2,latest_packet_full_primary_reads=0,latest_packet_qualified_scope_extensions=0,latest_packet_repeated_primary_method_scope_events=0,latest_packet_retained_primary_revisits=0,latest_packet_previously_read_identity_omissions_reconciled=0,latest_packet_first_scoped_method_identity=[r['canonical_id'] for r in papers],latest_packet_bounded_catalog_proposals=0,latest_adoption_packets=1,latest_accounting_correction_reference='literature_memory/index_v50/VERIFICATION.json',integration_pass_new_primary_reads=0,integration_pass_full_primary_reads=0,integration_pass_retained_primary_revisits=0,integration_pass_author_source_semantic_reads=0,integration_pass_metadata_identity_checks=2)
mutable={'schema','created_UTC','predecessor_index','predecessor_index_sha256','latest_adoption','read_accounting','canonical_identifier_normalization','paper_records','existing_packets'}
preserve={'first_209_records_unchanged':index['paper_records'][:209]==prior['paper_records'],'first_343_catalog_entries_unchanged':index['existing_packets'][:343]==prior['existing_packets'],'all_160_prior_groups_unchanged':groups[:160]==prior['canonical_identifier_normalization']['groups'],'all_other_prior_top_level_fields_unchanged':all(index[k]==v for k,v in prior.items() if k not in mutable),'all_prior_normalization_metadata_unchanged':all(index['canonical_identifier_normalization'][k]==v for k,v in prior['canonical_identifier_normalization'].items() if k!='groups'),'latest_accounting_adoption_and_lineage_snapshotted_exactly':index['integration_v50_predecessor_v49_snapshot']['read_accounting']==prior['read_accounting'] and index['integration_v50_predecessor_v49_snapshot']['latest_adoption']==prior['latest_adoption']}
assert all(preserve.values()),preserve
assert new_metrics['conclusion_records']==211 and new_metrics['normalized_paper_identifiers']==160
dump('LITERATURE_INDEX.json',index);dump('ADOPTION.json',adoption)
unique_refs={ (r['path'],r['sha256']):r for r in record_bindings }
verification={'UTC':UTC,'status':'PASS','write_scope':'Only newly created literature_memory/index_v50 files.','predecessor_metrics_recomputed':old_metrics,'current_memory_metrics_recomputed':new_metrics,'preservation_checks':preserve,'input_packet_checks':packet_checks,'predecessor_catalog_entries_verified':len(catalog_checks),'predecessor_unique_catalog_paths_verified':len({r['path'] for r in catalog_checks}),'bound_predecessor_conclusion_scope_binding_occurrences_verified':len(record_bindings),'bound_predecessor_unique_conclusion_scope_path_hash_bindings_verified':len(unique_refs),'bound_predecessor_conclusion_scope_references':[unique_refs[k] for k in sorted(unique_refs)],'consulted_inputs_and_reused_notes_verified':context_checks,'source_payloads_verified':packet_checks[1]['payload_count'],'source_payload_bytes_verified':packet_checks[1]['payload_bytes'],'incremental_reading_counts_in_source_packet':account,'integration_semantic_read_counts':{'new_primary_methods':0,'primary_rereads':0,'author_source':0,'full_papers':0},'new_source_requests':0,'network_actions':False,'mechanical_digesting_is_not_semantic_primary_rereading':True,'raw_primary_text_or_abstracts_copied':False,'new_canonical_identity_duplicates':0,'title_only_identity_merges':0,'same_paper_scopes_separate':True,'cumulative_read_totals_certified':False,'negative_novelty_conclusions_only':True,'posterior_addendum_paper_records_added':0,'root_clients_status_ledger_or_other_agents_directories_changed':False,'execution_authorized':False}
dump('VERIFICATION.json',verification)
note=f'''# Index v50 root adoption notes

Created {UTC}. Predecessor: `literature_memory/index_v49/LITERATURE_INDEX.json`, SHA256 `{sha(PREV/'LITERATURE_INDEX.json')}`. Source scout manifest: `{sha(SOURCE/'MANIFEST.json')}`.

Exactly two previously completed scoped method conclusions are adopted: GTN/FastGTN (`arxiv:2106.06218`, v1; PDF pages 3-6, Sections 3.2-3.5, Eqs. 2-21) and MUG (`arxiv:2602.22645`, v1; PDF pages 3-5, Definitions 3.1-3.3 and Method, Eqs. 1-14). Their separate saved semantic boundaries, incidental exposures, visual checks, citations, source hashes and limits remain bound. These are two new scoped paper identities in the source scout, with zero scope extensions, repeats or full-paper certifications. This integration adds zero primary reads, rereads, author-source reads, public requests or full-paper certifications. Memory growth is not additional reading. Cumulative read totals remain uncertified.

The adopted finding is negative: no new typed-path propagation or initialization mechanism was found. GTN already combines learned ordered paths, shared dense maps and nonlinear channels; MUG already runs one shared encoder across path views. Persistent private nonlinear trajectories retain an attributed, untested quality question, extending the prior filter review. No new method, proposal, predictive advantage or launch is adopted.

The posterior-serving addendum is bound as a limitation, adding no paper record or read. Likelihood-derived graph responsibilities have classical graph-mixture ancestry. If the full graph input already determines the structural observations, gating adds no new graph information. Its possible value concerns a restricted readout or optimization and remains untested. The complete neural composition/global novelty is unresolved in the saved limited scopes. No generic Bayes theorem novelty is adopted.

Current memory: {new_metrics['conclusion_records']} conclusion records, {new_metrics['normalized_paper_identifiers']} normalized paper identities, 2 software identities, {new_metrics['catalog_entries']} catalog entries / {new_metrics['unique_catalog_document_paths']} unique catalog paths, {new_metrics['unique_conclusion_source_documents']} conclusion documents, and {new_metrics['unique_referenced_document_paths']} catalog/conclusion paths. Scope documents: {new_metrics['unique_scope_reference_document_paths']}, or {new_metrics['unique_scope_reference_document_paths_including_legacy_field_alias']} including the legacy field alias. Seventeen scoped/provenance documents are appended; raw paper text and abstracts are not copied into the index.

All v49 records, catalog entries, identity groups, aliases, citations, scopes, failures, exclusions and limits are preserved exactly, with exact latest-adoption/accounting/lineage snapshots. The prior manifest/seal and four payloads, all 343 prior catalog entries, all 304 prior conclusion/scope binding occurrences, source manifest/seal and all {packet_checks[1]['payload_count']} source payloads, and consulted input digests verify. Primary PDF/extraction/render files were mechanically hashed only. v49 remains byte-identical and immutable. No canonical ledger/status, root client or other agent directory was changed. **execution_authorized=false**.
'''
(OUT/'ROOT_ADOPTION_NOTES.md').write_text(note)
for r in catalog_checks+record_bindings+context_checks:check_ref(r)
assert sha(PREV/'LITERATURE_INDEX.json')=='6af95565489025d814c7b68bd390ca97c937cea3d3cce76af991287aa9e91a5a'
payload=sorted(p for p in OUT.iterdir() if p.is_file())
manifest={'schema':'literature-memory-index-manifest-v50','UTC':UTC,'files':[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in payload],'predecessor':ref(PREV/'LITERATURE_INDEX.json'),'source_packet_manifest':ref(SOURCE/'MANIFEST.json'),'execution_authorized':False}
dump('MANIFEST.json',manifest)
seal={'schema':'literature-memory-index-seal-v50','UTC':UTC,'status':'SEALED_SCOPED_NEGATIVE_NOVELTY_CONCLUSIONS_ONLY','index_sha256':sha(OUT/'LITERATURE_INDEX.json'),'manifest_sha256':sha(OUT/'MANIFEST.json'),'predecessor_sha256':sha(PREV/'LITERATURE_INDEX.json'),'source_packet_manifest_sha256':sha(SOURCE/'MANIFEST.json'),'new_memory_records':2,'new_canonical_paper_groups':2,'source_packet_scoped_method_reads':2,'integration_new_primary_reads':0,'new_full_paper_certifications':0,'memory_totals':new_metrics,'immutable_files_mode':'0444','immutable_directory_mode':'0555','execution_authorized':False}
dump('SEAL.json',seal)
for p in OUT.iterdir():os.chmod(p,0o444)
os.chmod(OUT,0o555)
print(json.dumps({'index_sha256':sha(OUT/'LITERATURE_INDEX.json'),'manifest_sha256':sha(OUT/'MANIFEST.json'),'seal_sha256':sha(OUT/'SEAL.json'),'memory_totals':new_metrics,'preservation_checks':preserve,'integration_new_primary_reads':0},indent=2))
