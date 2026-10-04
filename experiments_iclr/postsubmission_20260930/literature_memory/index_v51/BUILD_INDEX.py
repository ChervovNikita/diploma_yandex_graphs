"""Add one completed scoped conclusion to v50; no primary rereading or science actions."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import os

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREV = ROOT / 'literature_memory/index_v50'
SOURCE = ROOT / 'label_conditioned_shared_latent_link_scout_20261004_v1'
assert not (OUT / 'LITERATURE_INDEX.json').exists(), 'Immutable output already exists'
UTC = datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


def check_ref(row):
    path = ROOT / row['path']
    assert path.is_file() and sha(path) == row['sha256'], row['path']
    assert 'bytes' not in row or path.stat().st_size == row['bytes'], row['path']
    return ref(path)


def metrics(value):
    catalog = {row['path'] for row in value['existing_packets']}
    conclusions = {row['conclusion_file'] for row in value['paper_records'] if 'conclusion_file' in row}
    scopes = {row['read_scope_reference']['path'] for row in value['paper_records'] if 'read_scope_reference' in row}
    legacy = {row['read_scope_file_reference']['path'] for row in value['paper_records'] if 'read_scope_file_reference' in row}
    groups = value['canonical_identifier_normalization']['groups']
    return {
        'conclusion_records': len(value['paper_records']),
        'normalized_paper_identifiers': sum(row['kind'] == 'paper' for row in groups),
        'software_documentation_identifiers': sum(row['kind'] != 'paper' for row in groups),
        'catalog_entries': len(value['existing_packets']),
        'unique_catalog_document_paths': len(catalog),
        'unique_conclusion_source_documents': len(conclusions),
        'unique_referenced_document_paths': len(catalog | conclusions),
        'unique_scope_reference_document_paths': len(scopes),
        'unique_scope_reference_document_paths_including_legacy_field_alias': len(scopes | legacy),
    }


prior_pin = ref(PREV / 'LITERATURE_INDEX.json')
assert prior_pin['sha256'] == '62b5dd473aea9573be7039e4981861609111db88df658d9acd742949ce699fe5'
prior = json.loads((PREV / 'LITERATURE_INDEX.json').read_text())
index = copy.deepcopy(prior)
prior_manifest = json.loads((PREV / 'MANIFEST.json').read_text())
prior_seal = json.loads((PREV / 'SEAL.json').read_text())
assert sha(PREV / 'MANIFEST.json') == prior_seal['manifest_sha256']
prior_index_entry = next(row for row in prior_manifest['files'] if row['path'] == 'LITERATURE_INDEX.json')
assert prior_index_entry['sha256'] == prior_pin['sha256'] and prior_index_entry['bytes'] == prior_pin['bytes']
source_manifest = json.loads((SOURCE / 'MANIFEST.json').read_text())
source_seal = json.loads((SOURCE / 'SEAL.json').read_text())
assert sha(SOURCE / 'MANIFEST.json') == source_seal['manifest']['sha256']
assert sha(SOURCE / 'MANIFEST.json') == '79ef345beeed8c9f9b9f737d0102e86a12e0bd979d7a24ad8432a5c693d92df3'
source_rows = {row['path']: row for row in source_manifest['files']}
catalog_names = [
    'PAPER_CONCLUSIONS.json', 'READ_SCOPES.json', 'CONCLUSIONS.json', 'REPORT.txt',
    'HYPOTHESIS_AND_LIMITS.json', 'LITERATURE_MEMORY_RECORD.json', 'PHASE_IDENTITY_CHECK.json',
    'PRIOR_CONTEXT.json', 'REUSED_INDEX_RECORDS.json', 'SEARCH_ACCOUNTING.json',
    'DISCOVERY_LOG.json', 'RETRIEVAL_mmsb.json', 'SOURCE_SEMANTICS_SCOPE.json',
    'ADDITIONAL_SOURCE_AUDIT_CONTEXT.json', 'VERIFICATION.json',
    'discovery/exact_mmsb_dgi.json', 'MANIFEST.json', 'SEAL.json',
]
source_document_checks = []
for name in catalog_names:
    actual = ref(SOURCE / name)
    if name not in ('MANIFEST.json', 'SEAL.json'):
        saved = source_rows[name]
        assert actual['bytes'] == saved['bytes'] and actual['sha256'] == saved['sha256'], name
    source_document_checks.append(actual)

scoped = json.loads((SOURCE / 'READ_SCOPES.json').read_text())
papers = json.loads((SOURCE / 'PAPER_CONCLUSIONS.json').read_text())
account = scoped['read_accounting']
assert len(papers) == len(scoped['papers']) == 1
assert account == {
    'author_code_audits': 0, 'full_paper_certifications': 0, 'indexed_primary_rereads': 0,
    'limited_project_source_file_scopes': 3, 'new_primary_scope_extensions': 0,
    'new_scoped_primary_paper_identities': 1, 'repeated_primary_method_reads': 0,
}
paper = papers[0]
scope = scoped['papers'][0]
canonical = paper['canonical_id']
assert canonical == scope['canonical_id'] == 'arxiv:0705.4485'
assert paper['versioned_id'] == scope['versioned_id'] == '0705.4485v1'
assert paper['primary'] == scope['primary']
assert not any(scope[key] for key in ('full_paper_read', 'proofs_certified', 'numeric_results_adopted', 'author_code_read'))
groups = index['canonical_identifier_normalization']['groups']
assert not any(row['normalized_identifier'] == canonical for row in groups)
assert not any(row['normalized_identifier'] == 'arxiv:1809.10341' for row in groups)
metadata_rows = json.loads((SOURCE / 'discovery/exact_mmsb_dgi.json').read_text())
metadata = next(row for row in metadata_rows if row['id'].endswith('/0705.4485v1'))
citation = {key: metadata[key] for key in ('id', 'title', 'authors', 'published', 'updated')}
assert citation['title'] == paper['title'] == scope['title']
assert citation['authors'] == paper['authors'] == scope['authors']
position = len(index['paper_records'])
scope_key = hashlib.sha256(json.dumps({
    'canonical_id': canonical, 'versioned_id': paper['versioned_id'], 'scope': scope['semantic_scope'],
}, sort_keys=True).encode()).hexdigest()
record = {
    'canonical_id': canonical, 'normalized_identifier': canonical,
    'conclusion': copy.deepcopy(paper),
    'conclusion_file': str((SOURCE / 'PAPER_CONCLUSIONS.json').relative_to(ROOT)),
    'conclusion_file_sha256': sha(SOURCE / 'PAPER_CONCLUSIONS.json'),
    'citation_metadata': citation,
    'citation_metadata_reference': ref(SOURCE / 'discovery/exact_mmsb_dgi.json'),
    'read_scope_reference': {**ref(SOURCE / 'READ_SCOPES.json'), 'paper_canonical_id': canonical, 'scope_key': scope['key']},
    'compact_primary_scope': copy.deepcopy(scope), 'exact_read_scope': copy.deepcopy(scope['semantic_scope']),
    'scope_deduplication_key_sha256': scope_key, 'primary_payload_reference': copy.deepcopy(scope['primary']),
    'source_packet_manifest_reference': ref(SOURCE / 'MANIFEST.json'),
    'source_packet_seal_reference': ref(SOURCE / 'SEAL.json'),
    'source_report_reference': ref(SOURCE / 'REPORT.txt'),
    'full_paper_read': False, 'genuinely_new_scoped_paper_identity_in_source_packet': True,
    'integration_pass_new_semantic_read': False, 'integration_pass_primary_reread': False,
    'integration_read_status': 'Previously completed scoped method read adopted without new primary read',
    'global_novelty_clearance': False, 'predictive_adoption': False, 'execution_authorized': False,
}
index['paper_records'].append(record)
groups.append({
    'normalized_identifier': canonical, 'kind': 'paper',
    'raw_canonical_identifiers': [canonical, 'arxiv:0705.4485v1'],
    'explicit_aliases': [], 'record_indices': [position],
})
for name in catalog_names:
    index['existing_packets'].append({
        **ref(SOURCE / name), 'kind': 'stored_scoped_label_conditioned_latent_negative_novelty_document',
        'scope': 'One completed MMSB scoped conclusion, reused DGI/indexed priors and cached source-audit limits. No new reading, mechanism, predictive or execution adoption in integration.',
    })

context = json.loads((SOURCE / 'PRIOR_CONTEXT.json').read_text())
audit = json.loads((SOURCE / 'ADDITIONAL_SOURCE_AUDIT_CONTEXT.json').read_text())
audit_refs = [check_ref(row) for row in audit['consulted_cached_sources']]
dgi_report = next(row for row in context['consulted_cached_sources'] if row['path'] == 'continuous_method_gap_search_v1/round2_heterophily_uncertainty/REPORT.md')
dgi_metadata = next(row for row in metadata_rows if row['id'].endswith('/1809.10341v2'))
dgi_reuse = {
    'canonical_id': 'arxiv:1809.10341', 'versioned_id': '1809.10341v2', 'title': 'Deep Graph Infomax',
    'citation_metadata': {key: dgi_metadata[key] for key in ('id', 'title', 'authors', 'published', 'updated')},
    'saved_prior_scope': ['Sections 3.1-3.4', 'Section 4.2', 'Appendix C'],
    'cached_report_reference': copy.deepcopy(dgi_report), 'cached_report_locator_line': 15,
    'scope_provenance': ref(SOURCE / 'PRIOR_CONTEXT.json'),
    'read_status': 'Previously read in phase; cached conclusion reused, not a new read or scope extension',
    'index_v50_canonical_group_present': False, 'index_v51_new_record_or_group_added': False,
    'new_primary_retrievals': 0, 'new_primary_reads': 0, 'primary_rereads': 0,
    'semantic_read_performed_in_integration': False,
}
old_metrics, new_metrics = metrics(prior), metrics(index)
assert new_metrics['conclusion_records'] == 212 and new_metrics['normalized_paper_identifiers'] == 161
assert new_metrics['software_documentation_identifiers'] == 2
assert new_metrics['catalog_entries'] == old_metrics['catalog_entries'] + len(catalog_names)
added = [{
    'record_index': position, 'canonical_id': canonical, 'normalized_identifier': canonical,
    'scope': copy.deepcopy(scope['semantic_scope']), 'scope_deduplication_key_sha256': scope_key,
    'full_paper_read': False, 'genuinely_new_scoped_paper_identity_in_source_packet': True,
    'integration_new_primary_reads': 0,
}]
adoption = {
    'UTC': UTC, 'status': 'SEALED_ADDITIVE_SCOPED_NEGATIVE_NOVELTY_MEMORY',
    'predecessor': prior_pin, 'source_packet_manifest': ref(SOURCE / 'MANIFEST.json'),
    'source_packet_seal': ref(SOURCE / 'SEAL.json'), 'added_records': added,
    'new_memory_records': 1, 'new_canonical_paper_identity_groups': 1, 'new_catalog_entries': len(catalog_names),
    'source_packet_new_scoped_method_reads': 1, 'source_packet_scope_extensions': 0,
    'source_packet_repeated_primary_method_reads': 0, 'source_packet_full_paper_certifications': 0,
    'source_packet_cached_index_conclusion_records_reused': 12,
    'dgi_reused_not_new_record_or_read': copy.deepcopy(dgi_reuse),
    'integration_pass_new_primary_reads': 0, 'integration_pass_primary_rereads': 0,
    'integration_pass_author_source_semantic_reads': 0, 'integration_pass_project_source_semantic_reads': 0,
    'integration_public_requests': 0, 'prior_records_groups_scopes_aliases_history_and_limits_preserved': True,
    'negative_latent_compatibility_novelty_finding_adopted': True,
    'full_TRAIN_serving_degeneracy_adopted': True, 'full_TRAIN_teacher_bayes_factor': 1,
    'method_adoption': False, 'predictive_adoption': False, 'novelty_clearance': False,
    'new_proposal_or_launch_adoption': False, 'research_ledger_or_status_updated': False,
    'execution_authorized': False,
}
index['integration_v51_predecessor_v50_snapshot'] = {
    'index_reference': prior_pin, 'actual_recomputed_metrics': old_metrics,
    'read_accounting': copy.deepcopy(prior['read_accounting']),
    'latest_adoption': copy.deepcopy(prior['latest_adoption']),
    'lineage_metadata': {key: copy.deepcopy(prior[key]) for key in ('schema', 'created_UTC', 'predecessor_index', 'predecessor_index_sha256')},
}
index.update(schema='literature-memory-index-v51', created_UTC=UTC,
             predecessor_index=prior_pin['path'], predecessor_index_sha256=prior_pin['sha256'],
             latest_adoption=copy.deepcopy(adoption))
index['post_v51_append'] = copy.deepcopy(adoption)
index['canonical_identifier_normalization']['post_v51_scope_append'] = {
    'UTC': UTC, 'added_record_indices': [position], 'new_paper_identities': [canonical],
    'old_groups_preserved_exactly': len(prior['canonical_identifier_normalization']['groups']),
    'title_only_alias_merges': 0, 'deduplication': 'Exact normalized arXiv identity and version-pinned saved metadata. No new DOI alias or title-only merge.',
    'dgi_reuse_adds_no_group': True, 'integration_primary_reads': 0,
}
index['label_conditioned_latent_negative_novelty_and_serving_limits_v51'] = {
    'source_report': ref(SOURCE / 'REPORT.txt'), 'source_hypothesis_and_limits': ref(SOURCE / 'HYPOTHESIS_AND_LIMITS.json'),
    'source_audit_context': ref(SOURCE / 'ADDITIONAL_SOURCE_AUDIT_CONTEXT.json'), 'cached_source_audit_references': audit_refs,
    'finding': 'Reject score/distribution/Bayes/graph-generative/graph-contrastive mechanism novelty: qJ/qF=M*dot(rL,rR) is a soft latent-compatibility score. Exact complete neural composition duplicate is not certified and does not confer novelty clearance.',
    'remaining_question': 'Untested predictive inductive bias of normalized count-conditioned residual identity-law responsibility features tied to existing private graph filters.',
    'generative_and_conditional_discriminative_objectives_are_distinct': True,
    'class_conditional_negative_reconstruction_does_not_directly_repel_responsibilities': True,
    'slot_normalized_loss_difference_is_not_literal_log_bayes_factor': True,
    'full_TRAIN_serving_fact': 'On unmasked complete TRAIN, residual teacher-positive subsets are empty even if candidate supports are nonempty: kL=kR=0, all conditional member likelihoods are one, responsibilities are uniform, qJ/qF=1. Query-edge-only masking remains deterministic after endpoint exclusion.',
    'successor_serving_requirement': 'A nontrivial structural posterior/Bayes-factor gate needs a separately fixed inference mask/view/support and observable reconstruction protocol, with charged graph passes and likelihood work. No protocol adopted.',
    'current_studies_original_mean_serving_unchanged': True, 'new_paper_records_from_source_audit': 0,
    'new_primary_reads_in_integration': 0, 'novelty_or_predictive_or_execution_adoption': False,
    'dgi_reuse': dgi_reuse,
}
index['read_accounting'].update(new_metrics)
index['read_accounting'].update(
    state='ROOT_DELEGATED_ADDITIVE_SCOPED_NEGATIVE_NOVELTY_MEMORY',
    historical_path_catalog_note='All v50 records and histories retained. One completed MMSB scoped method added as negative novelty evidence. DGI is reused from the prior phase, with no added record/read. Source-audit serving limit is cached. Integration adds zero primary or project-source reads.',
    latest_index_growth=1, latest_packet_new_scoped_primary_reads=1,
    latest_packet_genuinely_new_scoped_paper_identities=1, latest_packet_new_paper_identity_groups=1,
    latest_packet_previously_completed_scoped_read_adoptions=1, latest_packet_scoped_primary_method_events=1,
    latest_packet_full_primary_reads=0, latest_packet_qualified_scope_extensions=0,
    latest_packet_repeated_primary_method_scope_events=0, latest_packet_retained_primary_revisits=0,
    latest_packet_previously_read_identity_omissions_reconciled=0,
    latest_packet_first_scoped_method_identity=[canonical], latest_packet_bounded_catalog_proposals=0,
    latest_adoption_packets=1, latest_accounting_correction_reference='literature_memory/index_v51/VERIFICATION.json',
    latest_packet_source_identity_and_citation_document_scopes=1,
    latest_packet_dgi_cached_prior_reuses=1, latest_packet_cached_index_conclusion_records_reused=12,
    integration_pass_new_primary_reads=0, integration_pass_primary_method_reads=0,
    integration_pass_full_primary_reads=0, integration_pass_retained_primary_revisits=0,
    integration_pass_author_source_semantic_reads=0, integration_pass_project_source_semantic_reads=0,
    integration_pass_metadata_identity_checks=2,
)
mutable = {'schema', 'created_UTC', 'predecessor_index', 'predecessor_index_sha256', 'latest_adoption',
           'read_accounting', 'canonical_identifier_normalization', 'paper_records', 'existing_packets'}
preservation = {
    'all_211_prior_records_unchanged': index['paper_records'][:len(prior['paper_records'])] == prior['paper_records'],
    'all_360_prior_catalog_entries_unchanged': index['existing_packets'][:len(prior['existing_packets'])] == prior['existing_packets'],
    'all_162_prior_identity_groups_unchanged': groups[:len(prior['canonical_identifier_normalization']['groups'])] == prior['canonical_identifier_normalization']['groups'],
    'all_other_prior_top_level_fields_unchanged': all(index[key] == value for key, value in prior.items() if key not in mutable),
    'all_prior_normalization_metadata_unchanged': all(index['canonical_identifier_normalization'][key] == value for key, value in prior['canonical_identifier_normalization'].items() if key != 'groups'),
    'latest_accounting_adoption_and_lineage_snapshotted_exactly': index['integration_v51_predecessor_v50_snapshot']['read_accounting'] == prior['read_accounting'] and index['integration_v51_predecessor_v50_snapshot']['latest_adoption'] == prior['latest_adoption'],
}
assert all(preservation.values()), preservation
dump('LITERATURE_INDEX.json', index)
dump('ADOPTION.json', adoption)
verification = {
    'UTC': UTC, 'status': 'PASS_ADDITIVE_STRUCTURE_AND_SELECTED_DOCUMENT_PINS',
    'write_scope': 'Only newly created literature_memory/index_v51 files.',
    'predecessor_metrics_recomputed': old_metrics, 'current_memory_metrics_recomputed': new_metrics,
    'preservation_checks': preservation, 'predecessor_manifest': ref(PREV / 'MANIFEST.json'),
    'predecessor_seal': ref(PREV / 'SEAL.json'), 'source_document_pins_checked': source_document_checks,
    'cached_source_audit_pins_checked': audit_refs,
    'verification_scope': 'Predecessor index/manifest/seal and adopted documentary pins; no recursive requalification of prior catalog or primary carriers.',
    'incremental_reading_counts_in_source_packet': account,
    'integration_semantic_read_counts': {'new_primary_methods': 0, 'primary_rereads': 0, 'author_source': 0, 'project_source': 0, 'full_papers': 0},
    'source_packet_dgi_primary_retrievals_or_rereads': 0, 'dgi_new_record_or_group_added': False,
    'primary_pdf_extraction_and_render_content_not_opened': True,
    'source_requests_or_network_actions': 0, 'raw_primary_text_or_abstracts_copied': False,
    'new_canonical_identity_duplicates': 0, 'title_only_identity_merges': 0,
    'cumulative_read_totals_certified': False, 'negative_novelty_conclusions_only': True,
    'new_source_audit_or_runtime_qualification_performed': False,
    'source_status_ledger_root_client_or_existing_studies_changed': False, 'execution_authorized': False,
}
dump('VERIFICATION.json', verification)
note = f'''# Index v51 adoption conclusions

One completed scoped paper conclusion is added to v50: **Mixed membership stochastic blockmodels**, Edoardo M. Airoldi, David M. Blei, Stephen E. Fienberg and Eric P. Xing, **arXiv:0705.4485v1**, posted 30 May 2007. Exact saved scope: PDF pages **4–7**, Section **2** and Section **2.1 Modeling sparsity**, Figure 1, the complete initiator/receiver generative procedure, Eq. **(1)** and sparsity/missing-interaction model. The scope stops before Section 2.2. Inference algorithms, proofs, applications, results, author code and later/journal versions were not audited. Incidental case-study/locator exposure and all original scope limits remain in the record.

**DGI, arXiv:1809.10341v2, is reused rather than read again.** Its saved prior phase scope is Sections **3.1–3.4**, Section **4.2** and **Appendix C**, recorded at `continuous_method_gap_search_v1/round2_heterophily_uncertainty/REPORT.md:15`. It was absent from v50's canonical groups but already method-read in that phase. This adoption adds no DGI paper record, identity group, retrieval, reread or scope-extension credit. Twelve existing index conclusions are also cached reuse.

## Conclusions adopted

The novelty finding is negative. The shared-versus-independent density ratio reduces exactly to **qJ/qF = M·dot(rL,rR)**, a soft latent-compatibility score. MMSB supplies context-dependent initiator/receiver roles and graph-link compatibility; cached DGI supplies graph joint/product discrimination ancestry. No new score, distribution, Bayes theorem, graph-generative or graph-contrastive mechanism is established. The limited scopes do not certify an exact duplicate of the complete neural composition, and this does not confer novelty clearance. The remaining question is an untested predictive inductive bias from normalized count-conditioned residual identity-law responsibility features tied to existing private filters. Generative class reconstruction and conditional label BCE remain different objectives; negative marginal reconstruction does not directly repel responsibilities.

The sealed source audit supplies an exact serving limit: on **unmasked complete TRAIN**, residual teacher-positive subsets are empty even when candidate supports are nonempty. Thus **kL=kR=0**, every conditional member likelihood is **1**, responsibilities are **uniform**, and **qJ/qF=1**. Removing only the query edge remains deterministic after endpoint exclusion. Native predicted completion weights do not alter this teacher-law result. A nontrivial posterior/Bayes-factor serving rule needs a separately fixed mask/view/support and observable reconstruction protocol, with its graph passes and likelihood work charged. No such protocol or execution is adopted; the frozen studies keep their original mean-score serving. The slot-normalized loss difference is log Bayes factor divided by the slot denominator, not a literal unscaled Bayes factor.

## Exact accounting

- Source scout: **1 new scoped primary method identity/read**, MMSB; **0 scope extensions**, **0 repeated-primary reads**, **0 full-paper certifications**, **0 author-code audits**. Its prior three limited project-source snippet scopes are preserved as source history, not new paper reads.
- This integration: **0 new primary reads, rereads, author/project-source semantic reads, public requests or full-paper certifications**. Copying saved scope and conclusions is not reading.
- Index growth: **1 conclusion record**, **1 normalized paper identity group**, **{len(catalog_names)} documentary catalog entries**. Current memory: **{new_metrics['conclusion_records']} records**, **{new_metrics['normalized_paper_identifiers']} normalized paper identities**, **{new_metrics['software_documentation_identifiers']} software identities**, **{new_metrics['catalog_entries']} catalog entries / {new_metrics['unique_catalog_document_paths']} unique catalog paths**, **{new_metrics['unique_conclusion_source_documents']} conclusion documents**, and **{new_metrics['unique_referenced_document_paths']} catalog/conclusion paths**. Scope documents: **{new_metrics['unique_scope_reference_document_paths']}**, or **{new_metrics['unique_scope_reference_document_paths_including_legacy_field_alias']}** including the legacy field alias. These are memory counts; cumulative read totals remain uncertified.

All **211 prior records, 360 prior catalog entries and 162 prior identity groups** are preserved exactly, along with their scopes, aliases, history, failures and limits. Prior accounting, latest adoption and lineage are snapshotted before updating v51. Only the predecessor index/manifest/seal, adopted documentary pins and cached source-audit references were mechanically checked; prior primary carriers were not reopened or recursively requalified. No raw primary text or abstracts were copied. Canonical status/ledger, source packets, existing studies and server/queue state are unchanged. **execution_authorized=false**.

Predecessor index SHA-256: `{prior_pin['sha256']}`. Scout manifest SHA-256: `{sha(SOURCE / 'MANIFEST.json')}`. Source audit manifest SHA-256: `e0efe7ae8b1c3f1cbe84686b2578dbe63f0c0ccc5dd0f7c41f6ce7264f9777be`.
'''
(OUT / 'ROOT_ADOPTION_NOTES.md').write_text(note)
assert ref(PREV / 'LITERATURE_INDEX.json') == prior_pin
for row in source_document_checks + audit_refs:
    check_ref(row)
manifest = {
    'schema': 'literature-memory-index-manifest-v51', 'UTC': UTC,
    'files': [{'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in sorted(OUT.iterdir()) if path.is_file()],
    'predecessor': prior_pin, 'source_packet_manifest': ref(SOURCE / 'MANIFEST.json'), 'execution_authorized': False,
}
dump('MANIFEST.json', manifest)
seal = {
    'schema': 'literature-memory-index-seal-v51', 'UTC': UTC,
    'status': 'SEALED_SCOPED_NEGATIVE_NOVELTY_CONCLUSIONS_ONLY',
    'index_sha256': sha(OUT / 'LITERATURE_INDEX.json'), 'manifest_sha256': sha(OUT / 'MANIFEST.json'),
    'predecessor_sha256': prior_pin['sha256'], 'source_packet_manifest_sha256': sha(SOURCE / 'MANIFEST.json'),
    'new_memory_records': 1, 'new_canonical_paper_groups': 1, 'source_packet_scoped_method_reads': 1,
    'integration_new_primary_reads': 0, 'new_full_paper_certifications': 0, 'memory_totals': new_metrics,
    'immutable_files_mode': '0444', 'immutable_directory_mode': '0555', 'execution_authorized': False,
}
dump('SEAL.json', seal)
for path in OUT.iterdir():
    os.chmod(path, 0o444)
os.chmod(OUT, 0o555)
print(json.dumps({
    'index_sha256': seal['index_sha256'], 'manifest_sha256': seal['manifest_sha256'],
    'seal_sha256': sha(OUT / 'SEAL.json'), 'memory_totals': new_metrics,
    'preservation_checks': preservation, 'integration_new_primary_reads': 0,
}, indent=2))
