"""Append five sealed saved method scopes and a non-reading protocol amendment.

Metadata transformation only: primary/discovery payloads must never be opened.
The root agent owns adoption; this builder grants no scientific authority.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
PREV = BASE / 'literature_memory/index_v69'
ADOPTION = BASE / 'literature_memory/index_v69_root_adoption_20261005_v1/ROOT_ADOPTION.json'
SOURCE = BASE / 'graph_aware_saved_prediction_aggregation_prior_scout_20261005_v1'
AMEND = BASE / 'graph_aware_aggregation_development_reuse_pairwise_amendment_20261005_v1'
PREV_INDEX_PIN = 'e7d3049ff23d8f870e2e25ea04876c6073bc2f8c56f0e33c67f40e7e630a884c'
PREV_MANIFEST_PIN = '6321ad41e9a1cadcbf553b4de7c4d8bb3532fef8bc63728ec739b50538a1c65c'
PREV_SEAL_PIN = 'ce67a59b4f46ca6c15028bcd4d85c12ef5c836e9ebe3e3f589ad65b168bc465e'
ADOPTION_PIN = '409e55b2001243752d4ffe834f2cb9d32cd6804b2f4a9349b022904ed60ccf20'
SOURCE_MANIFEST_PIN = '04159e6f5a61170d4bc1c8ea99006ddec735c10b025056542c645d63d7ee92ed'
SOURCE_SEAL_PIN = 'bce901c7a3c8f18f71aea50d9f591ec427324da01966dfb471f37e45eb534451'
SOURCE_REPORT_PIN = '47f8475669e60dccf7998e79e7d88b80b027812f276f4d2fa2a462ba68cfde40'
SOURCE_CONCLUSIONS_PIN = '6299ce10a387523b8f4982ed27f4f9de12ee3ef19e247d378bddf29bc0e1ee4d'
AMEND_MANIFEST_PIN = '92856813d863cb765c885817b286fe495237619de846a108703cfecf8312b12e'
AMEND_SEAL_PIN = '76806027cfc9137aeedc2cc468066313bc97ef8fc526ffb0ac4c1a95299c58c6'
AMEND_REPORT_PIN = '5ad2a8a4efec7a21007b08333b412b24f5649277b868edd4a69904a8de646716'
SAFE_SOURCE_METADATA = ('SCOPED_CONCLUSIONS.json', 'READ_SCOPES.json',
    'READ_ACCOUNTING.json', 'SOURCE_BINDINGS.json', 'VERIFICATION.json', 'REPORT.md',
    'DISCOVERY_RECEIPTS.json', 'SECOND_RETRIEVAL_RECEIPTS.json',
    'THIRD_RETRIEVAL_RECEIPTS.json', 'FOURTH_RETRIEVAL_RECEIPT.json',
    'INITIAL_DISCOVERY_DECLARATION.json')
SAFE_AMEND_METADATA = ('AMENDMENT_CONCLUSIONS.json', 'READ_ACCOUNTING.json',
    'SOURCE_BINDINGS.json', 'VERIFICATION.json', 'REPORT.md')
KEYS = ['gamlp_v3', 'meta_des_v1', 'calibration_2206', 'gats_v1', 'moenp_v3']
IDS = ['arxiv:2108.10097', 'arxiv:1810.01270', 'arxiv:2206.01570',
    'arxiv:2210.06391', 'arxiv:2412.00418']
VERSIONS = ['2108.10097v3', '1810.01270v1', '2206.01570v1', '2210.06391v1', '2412.00418v3']
MAX_BYTES = 2_000_000


def ref(path):
    path = Path(path)
    assert path.resolve().is_relative_to(BASE) and not path.is_symlink()
    # Explicitly reject all current and historical primary/discovery directories.
    assert 'primary' not in path.parts and 'discovery' not in path.parts
    assert path.name not in ('PRIMARY_SCOPES.json', 'PUBLIC_SOURCE_SCOPES.json', 'AUTHOR_REPOSITORY_README.md')
    assert not path.name.endswith('_PRIMARY_SCOPES.json')
    assert path.suffix not in ('.pdf', '.html', '.xml', '.source', '.pt', '.pkl', '.npy', '.jsonl')
    if path.is_relative_to(SOURCE):
        assert path.name in SAFE_SOURCE_METADATA + ('MANIFEST.json', 'SEAL.json')
    if path.is_relative_to(AMEND):
        assert path.name in SAFE_AMEND_METADATA + ('MANIFEST.json', 'SEAL.json')
    raw = path.read_bytes()
    return {'path': str(path.relative_to(BASE)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def save(name, value):
    with (HERE / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def normalized(value):
    return {re.sub(r'v\d+$', '', part.strip().lower()) if part.strip().lower().startswith('arxiv:')
        else part.strip().lower() for part in value.split(';') if part.strip()}


def metrics(index):
    catalog = {row['path'] for row in index['existing_packets']}
    conclusions = {row['conclusion_file'] for row in index['paper_records'] if isinstance(row.get('conclusion_file'), str)}
    scopes = {row['read_scope_reference']['path'] for row in index['paper_records'] if 'read_scope_reference' in row}
    legacy = {row['read_scope_file_reference']['path'] for row in index['paper_records'] if 'read_scope_file_reference' in row}
    groups = index['canonical_identifier_normalization']['groups']
    return {'conclusion_records': len(index['paper_records']),
        'normalized_paper_identifiers': sum(row['kind'] == 'paper' for row in groups),
        'software_documentation_identifiers': sum(row['kind'] != 'paper' for row in groups),
        'catalog_entries': len(index['existing_packets']), 'unique_catalog_document_paths': len(catalog),
        'unique_conclusion_source_documents': len(conclusions), 'unique_referenced_document_paths': len(catalog | conclusions),
        'unique_scope_reference_document_paths': len(scopes),
        'unique_scope_reference_document_paths_including_legacy_field_alias': len(scopes | legacy)}


def authenticate_predecessor():
    manifest_ref = ref(PREV / 'MANIFEST.json')
    assert manifest_ref['sha256'] == PREV_MANIFEST_PIN
    rows = json.loads((PREV / 'MANIFEST.json').read_text())['files']
    verified = [manifest_ref]
    for row in rows:
        member = Path(row['path'])
        assert not member.is_absolute() and '..' not in member.parts
        got = ref(PREV / member)
        assert (got['bytes'], got['sha256']) == (row['bytes'], row['sha256'])
        verified.append(got)
    seal_ref = ref(PREV / 'SEAL.json')
    assert seal_ref['sha256'] == PREV_SEAL_PIN
    seal = json.loads((PREV / 'SEAL.json').read_text())
    assert seal['manifest_sha256'] == PREV_MANIFEST_PIN and seal['payload_files'] == len(rows)
    assert ref(PREV / 'LITERATURE_INDEX.json')['sha256'] == PREV_INDEX_PIN
    adoption_ref = ref(ADOPTION)
    assert adoption_ref['sha256'] == ADOPTION_PIN
    adoption = json.loads(ADOPTION.read_text())
    assert adoption['root_adopted'] is True
    assert adoption['index_sha256'] == PREV_INDEX_PIN
    assert adoption['manifest_sha256'] == PREV_MANIFEST_PIN and adoption['seal_sha256'] == PREV_SEAL_PIN
    assert adoption['all_249_records_preserved'] is True
    assert adoption['new_paper_records_or_read_credit'] == 0
    return verified + [seal_ref, adoption_ref]


def authenticate_packet(folder, manifest_pin, seal_pin, safe_names, report_pin):
    manifest_ref, seal_ref = ref(folder / 'MANIFEST.json'), ref(folder / 'SEAL.json')
    assert manifest_ref['sha256'] == manifest_pin and seal_ref['sha256'] == seal_pin
    rows = json.loads((folder / 'MANIFEST.json').read_text())['files']
    assert len({row['path'] for row in rows}) == len(rows)
    assert all(not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts for row in rows)
    declared = {row['path']: row for row in rows}
    seal = json.loads((folder / 'SEAL.json').read_text())
    assert seal['manifest_sha256'] == manifest_pin and seal['payload_files'] == len(rows)
    assert seal['execution_authority_granted'] is False
    verified = [manifest_ref, seal_ref]
    for name in safe_names:
        got = ref(folder / name)
        row = declared[name]
        assert (got['bytes'], got['sha256']) == (row['bytes'], row['sha256'])
        verified.append(got)
    assert ref(folder / 'REPORT.md')['sha256'] == report_pin
    return declared, verified, seal


def declaration(folder, declared, name):
    row = declared[name]
    return {'path': str((folder / name).relative_to(BASE)), 'bytes': row['bytes'], 'sha256': row['sha256'],
        'binding_status': 'Inherited authenticated sealed-manifest declaration; payload not reopened or independently rehashed in integration',
        'payload_accessed_in_integration': False}


def main():
    assert {path.name for path in HERE.iterdir()} == {'BUILD_INDEX.py'}
    now = datetime.now(timezone.utc).isoformat()
    predecessor_refs = authenticate_predecessor()
    declared, source_refs, source_seal = authenticate_packet(SOURCE, SOURCE_MANIFEST_PIN,
        SOURCE_SEAL_PIN, SAFE_SOURCE_METADATA, SOURCE_REPORT_PIN)
    amend_declared, amend_refs, amend_seal = authenticate_packet(AMEND, AMEND_MANIFEST_PIN,
        AMEND_SEAL_PIN, SAFE_AMEND_METADATA, AMEND_REPORT_PIN)
    assert source_seal['new_primary_unique_identities'] == 4 and source_seal['existing_identity_version_followups'] == 1
    assert source_seal['new_full_paper_reads'] == 0
    assert source_seal['scientific_execution'] is False and source_seal['target_endpoint_or_payload_access'] is False
    assert amend_seal['new_primary_reads'] == amend_seal['new_public_retrievals'] == 0 and amend_seal['target_access'] is False
    assert ref(SOURCE / 'SCOPED_CONCLUSIONS.json')['sha256'] == SOURCE_CONCLUSIONS_PIN
    previous = json.loads((PREV / 'LITERATURE_INDEX.json').read_text())
    before = metrics(previous)
    assert before == {key: previous['read_accounting'][key] for key in before}
    assert (before['conclusion_records'], before['normalized_paper_identifiers'], before['software_documentation_identifiers']) == (249, 197, 2)
    conclusions = json.loads((SOURCE / 'SCOPED_CONCLUSIONS.json').read_text())
    scopes = json.loads((SOURCE / 'READ_SCOPES.json').read_text())
    accounting = json.loads((SOURCE / 'READ_ACCOUNTING.json').read_text())
    source_bindings = json.loads((SOURCE / 'SOURCE_BINDINGS.json').read_text())
    initial = json.loads((SOURCE / 'INITIAL_DISCOVERY_DECLARATION.json').read_text())
    source_check = json.loads((SOURCE / 'VERIFICATION.json').read_text())
    amendment = json.loads((AMEND / 'AMENDMENT_CONCLUSIONS.json').read_text())
    amend_accounting = json.loads((AMEND / 'READ_ACCOUNTING.json').read_text())
    amend_bindings = json.loads((AMEND / 'SOURCE_BINDINGS.json').read_text())
    amend_check = json.loads((AMEND / 'VERIFICATION.json').read_text())
    candidates = conclusions['paper_conclusions']
    assert [row['key'] for row in candidates] == [row['key'] for row in scopes['scopes']] == KEYS
    assert [row['canonical_id'] for row in candidates] == IDS
    assert [row['versioned_id'] for row in candidates] == VERSIONS
    assert scopes['new_primary_scoped_events'] == accounting['new_completed_primary_method_scope_events'] == 5
    assert scopes['new_unique_primary_identities'] == accounting['new_primary_unique_paper_identities_in_packet'] == 4
    assert scopes['existing_identity_version_followups'] == accounting['new_existing_identity_version_followup_scopes'] == 1
    assert accounting['new_full_paper_reads'] == accounting['new_author_code_semantic_reads'] == accounting['repeated_exact_primary_scopes'] == 0
    assert accounting['public_retrievals'] == 16 and accounting['raw_body_retained_public_retrievals'] == 13
    assert accounting['public_metadata_search_queries'] == 8 and accounting['initial_metadata_response_bodies_not_retained'] == 3
    assert accounting['JKNet_DAGNN_original_primary_reads'] == 0
    assert accounting['target_data_model_scores_history_checkpoint_payload_reads'] == accounting['target_endpoint_contacts'] == 0
    assert accounting['scientific_source_execution'] is False and accounting['aggregation_fit_or_backbone_training'] is False
    assert initial['raw_hash_claimed'] is False and initial['raw_response_retained'] is False
    assert len(initial['public_metadata_queries']) == 3 and all('sha256' not in row for row in initial['public_metadata_queries'])
    assert source_check['scope_locator_counts_checked'] is True and source_check['saved_index69_pin_authenticated'] is True
    assert source_bindings['saved_index']['sha256'] == PREV_INDEX_PIN
    assert conclusions['candidate_count'] == 1 and conclusions['new_gate_or_execution_authority'] is False
    assert conclusions['novelty_or_breakthrough_established'] is False
    for key in ('new_primary_method_reads', 'new_full_paper_reads', 'new_public_retrievals',
                'new_public_searches', 'new_paper_identity_credit', 'primary_rereads',
                'target_endpoint_contacts', 'target_model_data_score_history_checkpoint_payload_reads'):
        assert amend_accounting[key] == 0
    assert amend_accounting['scientific_execution'] is False and amend_accounting['new_unseen_TEST_custody_established'] is False
    assert amend_bindings['index69']['sha256'] == PREV_INDEX_PIN
    assert amend_bindings['original_primary_payloads_reopened_or_rehashed'] is False
    for old_ref in amend_bindings['original_packet_safe_metadata']:
        assert ref(BASE / old_ref['path']) == old_ref
    assert amend_check['TEST_contract_relaxed'] is False and amend_check['target_access_or_execution'] is False
    assert amendment['new_GATE_TEST_or_launch_authority'] is False
    assert amendment['same_context_conditioned_pooling_hypothesis_not_second_candidate'] is True
    assert amendment['novelty_or_predictive_gain_certified'] is False
    assert len(previous['unresolved_primary_metadata_leads']) == 3

    receipts = []
    for name in ('DISCOVERY_RECEIPTS.json', 'SECOND_RETRIEVAL_RECEIPTS.json',
                 'THIRD_RETRIEVAL_RECEIPTS.json', 'FOURTH_RETRIEVAL_RECEIPT.json'):
        value = json.loads((SOURCE / name).read_text())
        receipts.extend(value if isinstance(value, list) else [value])
    assert len(receipts) == 13
    unopened = [declaration(SOURCE, declared, name) for name in sorted(declared) if name not in SAFE_SOURCE_METADATA]
    primary_declarations = [row for row in unopened if '/primary/' in row['path']]
    assert len(primary_declarations) == 15
    version_docs = []
    for pos, (paper, scope) in enumerate(zip(candidates, scopes['scopes'])):
        assert scope['key'] == paper['key'] and scope['canonical_id'] == paper['canonical_id']
        assert scope['versioned_id'] == paper['versioned_id']
        assert normalized('arxiv:' + scope['versioned_id']) == normalized(scope['canonical_id'])
        assert scope['full_paper_read'] is False and scope['author_code_read'] is False
        assert scope['numeric_results_adopted'] is False and scope['proofs_or_figure_pixels_audited'] is False
        assert scope['new_unique_primary_identity_in_packet'] == paper['new_paper_identity_in_packet'] == (pos < 4)
        assert scope['followup_of_existing_identity'] == (pos == 4)
        assert scope['existing_index69_record_indices'] == ([] if pos < 4 else [101])
        payloads = [declaration(SOURCE, declared, 'primary/' + scope['key'] + suffix)
            for suffix in ('.source', '.blocks.json', '.SELECTED_METHOD.json')]
        for field, expected in [('source_reference', payloads[0]), ('extracted_blocks_reference', payloads[1])]:
            assert scope[field] == {key: expected[key] for key in ('path', 'bytes', 'sha256')}
        source_binding = next(row for row in source_bindings['primary_sources'] if row['path'] == payloads[0]['path'])
        assert source_binding == scope['source_reference']
        receipt = next(row for row in receipts if row['id'] == scope['key'])
        assert receipt['role'] == 'primary' and receipt['http_status'] == 200
        assert (receipt['bytes'], receipt['sha256']) == (payloads[0]['bytes'], payloads[0]['sha256'])
        doc = {'canonical_id': scope['canonical_id'], 'exact_primary_id': 'arxiv:' + scope['versioned_id'],
            'saved_scope_metadata': copy.deepcopy(scope), 'scope_payload_declarations': payloads,
            'locator_metadata_reference': {**ref(SOURCE / 'READ_SCOPES.json'), 'selector': f'scopes/{pos}'},
            'saved_retrieval_metadata': copy.deepcopy(receipt),
            'retrieval_hash_status': 'Saved source receipt corroborated by authenticated manifest declaration; retained raw body not independently reopened or rehashed in integration',
            'primary_text_payload_accessed_in_integration': False}
        doc['scope_deduplication_key_sha256'] = hashlib.sha256(compact({key: doc[key] for key in
            ('canonical_id', 'exact_primary_id', 'saved_scope_metadata', 'scope_payload_declarations')})).hexdigest()
        version_docs.append(doc)
    assert len({doc['scope_deduplication_key_sha256'] for doc in version_docs}) == 5

    save('SOURCE_BINDINGS.json', {'UTC': now, 'predecessor_files': predecessor_refs,
        'adopted_v69_root_adoption_reference': ref(ADOPTION), 'source_files': source_refs, 'amendment_files': amend_refs,
        'source_manifest_sha256': SOURCE_MANIFEST_PIN, 'source_seal_sha256': SOURCE_SEAL_PIN,
        'amendment_manifest_sha256': AMEND_MANIFEST_PIN, 'amendment_seal_sha256': AMEND_SEAL_PIN,
        'safe_source_metadata_members_authenticated': len(SAFE_SOURCE_METADATA),
        'safe_amendment_metadata_members_authenticated': len(SAFE_AMEND_METADATA),
        'unopened_source_manifest_declarations': unopened,
        'source_primary_discovery_selected_passage_or_extraction_utility_payloads_opened': False,
        'source_primary_or_discovery_payload_hashes_recomputed': 0,
        'source_prior_model_data_score_history_checkpoint_bindings_dereferenced': False,
        'source_completed_primary_method_scope_events': 5, 'source_unique_new_paper_identities': 4,
        'source_existing_identity_version_followups': 1, 'source_full_paper_reads': 0,
        'source_public_retrievals_historical_only': 16, 'source_public_metadata_searches_historical_only': 8,
        'source_retained_public_response_bodies': 13,
        'source_initial_metadata_response_bodies_unretained': 3, 'unretained_response_hashes_invented': False,
        'amendment_reading_retrieval_identity_or_result_credit': 0,
        'integration_searches': 0, 'integration_retrievals': 0, 'integration_primary_reads': 0,
        'integration_primary_rereads': 0, 'integration_full_paper_reads': 0, 'integration_author_source_reads': 0,
        'integration_target_payload_accesses': 0, 'integration_scientific_execution': False,
        'only_new_index_v70_subtree_written': True})

    index = copy.deepcopy(previous)
    current = ['schema', 'created_UTC', 'latest_adoption', 'read_accounting', 'predecessor_index', 'predecessor_index_sha256']
    old_groups = previous['canonical_identifier_normalization']['groups']
    moe_group_pos = next(pos for pos, group in enumerate(old_groups) if group['normalized_identifier'] == IDS[4])
    assert moe_group_pos == 105 and old_groups[moe_group_pos]['record_indices'] == [101]
    index['integration_v70_predecessor_v69_snapshot'] = {
        **{key: copy.deepcopy(previous[key]) for key in current},
        'index_reference': ref(PREV / 'LITERATURE_INDEX.json'), 'manifest_reference': ref(PREV / 'MANIFEST.json'),
        'seal_reference': ref(PREV / 'SEAL.json'), 'root_adoption_reference': ref(ADOPTION),
        'only_extended_existing_group': {'group_index': moe_group_pos, 'group': copy.deepcopy(old_groups[moe_group_pos])}}
    index.update(schema='literature-memory-index-v70', created_UTC=now,
        predecessor_index='literature_memory/index_v69/LITERATURE_INDEX.json', predecessor_index_sha256=PREV_INDEX_PIN)
    groups = index['canonical_identifier_normalization']['groups']
    added = []
    for pos, (paper, doc) in enumerate(zip(candidates, version_docs)):
        identity = paper['canonical_id']
        existing = [group for group in old_groups if identity in set().union(*(normalized(value) for value in
            [group['normalized_identifier']] + group.get('raw_canonical_identifiers', []) + group.get('explicit_aliases', [])))]
        assert len(existing) == (0 if pos < 4 else 1)
        if pos < 4:
            assert all(identity not in normalized(row.get('canonical_id', '')) for row in previous['paper_records'])
        else:
            assert existing[0]['normalized_identifier'] == identity and existing[0]['record_indices'] == [101]
        position = len(index['paper_records'])
        index['paper_records'].append({'canonical_id': doc['exact_primary_id'], 'normalized_identifier': identity,
            'conclusion': copy.deepcopy(paper),
            'conclusion_file': str((SOURCE / 'SCOPED_CONCLUSIONS.json').relative_to(BASE)),
            'conclusion_file_sha256': SOURCE_CONCLUSIONS_PIN, 'conclusion_source_selector': f'paper_conclusions/{pos}',
            'read_scope_reference': {**ref(SOURCE / 'READ_SCOPES.json'), 'selector': f'scopes/{pos}',
                'scope_kind': 'Authenticated saved locator metadata; primary source/extracted/selected text not reopened'},
            'exact_read_scope': [copy.deepcopy(doc)], 'source_packet': SOURCE.name,
            'source_packet_binding_reference': ref(HERE / 'SOURCE_BINDINGS.json'),
            'source_packet_manifest_reference': ref(SOURCE / 'MANIFEST.json'), 'source_packet_seal_reference': ref(SOURCE / 'SEAL.json'),
            'scope_deduplication_key_sha256': doc['scope_deduplication_key_sha256'],
            'scope_deduplication_key_schema': 'normalized identity + exact arXiv version + saved locators + sealed unopened payload declarations',
            'read_status': 'Previously completed bounded method scope adopted; zero integration primary reads; not a full-paper read',
            'scoped_method_read': True, 'full_paper_read': False, 'author_source_read': False,
            'new_paper_identity': pos < 4, 'existing_identity_version_followup': pos == 4,
            'predecessor_record_indices': [] if pos < 4 else [101],
            'numeric_result_transfer': False, 'predictive_adoption': False, 'global_novelty_clearance': False,
            'execution_authorized': False, 'integration_pass_new_primary_reads': 0,
            'integration_pass_primary_reread': False, 'version_followup_is_additional_paper_identity': False})
        if pos < 4:
            groups.append({'normalized_identifier': identity, 'kind': 'paper',
                'raw_canonical_identifiers': [identity, doc['exact_primary_id']], 'explicit_aliases': [], 'record_indices': [position]})
        else:
            group = groups[moe_group_pos]
            assert doc['exact_primary_id'] not in group['raw_canonical_identifiers']
            group['raw_canonical_identifiers'].append(doc['exact_primary_id'])
            group['record_indices'].append(position)
        added.append({'canonical_id': identity, 'exact_canonical_id': doc['exact_primary_id'],
            'record_index': position, 'new_identity': pos < 4, 'existing_identity_version_followup': pos == 4,
            'source_completed_bounded_scope_adopted': True, 'full_paper_read': False,
            'scope_deduplication_key_sha256': doc['scope_deduplication_key_sha256']})

    for row, kind in ([(row, 'saved_graph_aggregation_scope_metadata') for row in source_refs]
        + [(row, 'sealed_manifest_declared_primary_payload_unopened_in_integration') for row in primary_declarations]
        + [(row, 'nonreading_development_reuse_pairwise_protocol_amendment') for row in amend_refs]):
        assert all(old['path'] != row['path'] for old in index['existing_packets'])
        index['existing_packets'].append({**row, 'kind': kind, 'source_bindings_reference': ref(HERE / 'SOURCE_BINDINGS.json')})

    index['graph_aware_saved_aggregation_scopes_v70'] = {
        'source_conclusions_preserved': copy.deepcopy(conclusions),
        'source_read_accounting_preserved': copy.deepcopy(accounting),
        'source_initial_discovery_declaration_preserved': copy.deepcopy(initial),
        'source_conclusion_reference': ref(SOURCE / 'SCOPED_CONCLUSIONS.json'),
        'source_report_reference': ref(SOURCE / 'REPORT.md'),
        'adopted_records': added,
        'original_memo_protocol_claims_are_historical_and_corrected_by': ref(AMEND / 'REPORT.md'),
        'original_MoE_NP_v1_record101_and_earlier_router_decisions_unchanged': True,
        'all_routes_already_compute_no_route_compute_savings_claim': True,
        'GAMLP_caches_propagation_and_storage_costs_charged': True,
        'JKNet_DAGNN_original_primary_reads': 0,
        'MoE_NP_v3_incidental_public_table_paragraph38_numbers_not_adopted': True,
        'new_direct_full_operator_matches': 0, 'new_ready_successors': 0,
        'demonstrated_superiority': False, 'global_novelty_clearance': False, 'execution_authority_granted': False}
    index['graph_aware_aggregation_protocol_amendment_v70'] = {
        'evidence_kind': 'Saved non-reading correction and explicit link-task adaptation of the same unconfirmed pooling hypothesis',
        'conclusions_preserved': copy.deepcopy(amendment), 'read_accounting_preserved': copy.deepcopy(amend_accounting),
        'conclusion_reference': ref(AMEND / 'AMENDMENT_CONCLUSIONS.json'), 'report_reference': ref(AMEND / 'REPORT.md'),
        'manifest_reference': ref(AMEND / 'MANIFEST.json'), 'seal_reference': ref(AMEND / 'SEAL.json'),
        'superseded_original_protocol_claims': ['All four label roles must be reserved before fitting to permit any honest final test',
            'Absent already-retained hidden states or confirmation/controls scientifically rejects the utility hypothesis'],
        'effective_protocol': {
            'VALID_may_select_backbone_checkpoints_and_fit_or_select_aggregator_as_disclosed_development': True,
            'performance_on_that_reused_VALID_is_biased_development_evidence': True,
            'aggregator_only_crossvalidation_erases_backbone_selection_reuse': False,
            'fully_frozen_pipeline_and_comparators_may_use_genuinely_untouched_TEST_under_authorized_protocol': True,
            'all_four_roles_must_be_reserved_before_any_fitting_if_final_TEST_is_clean': False,
            'frozen_TEST_contracts_unchanged': True, 'new_unseen_TEST_custody_established': False,
            'capable_same_evidence_single_required_comparator_not_automatic_utility_rejection': True,
            'missing_caches_controls_confirmation_may_require_new_representative_fits': True,
            'resource_scarcity_is_scheduling_not_scientific_rejection': True,
            'new_aggregation_rule_must_not_silently_replace_fixed_pool_estimand': True},
        'explicit_link_context': {
            'task': 'Citeseer-HeaRT link queries', 'support': 'Fixed permitted TRAIN support with unchanged target-fact and label-derived exclusions',
            'frozen_node_states': 'h_v; independently learned member coordinates remain separate unless aligned',
            'neighbor_mean': 'mu_v is declared normalized neighbor mean of frozen states on that fixed support',
            'symmetric_pair_features': '[h_u*h_v,h_u+h_v,abs(h_u-h_v),mu_u+mu_v,abs(mu_u-mu_v),(h_u-mu_u)*(h_v-mu_v)]',
            'additional_features': 'Link-MoE structural heuristics plus own-query logits z_m(q) and residuals z_m(q)-mean_j z_j(q)',
            'gate': 'alpha_m(q)=softmax_m small_gate(pair_context)',
            'served_score': 'z_fused(q)=sum_m alpha_m(q) z_m(q)',
            'complete_candidate_universe_unchanged': True, 'propagation_between_arbitrary_candidate_query_batches': False,
            'nearest_unchanged_operator': 'Link-MoE structural/endpoint gate with weighted expert-logit sum on same frozen bank and eligible labels',
            'expanded_evidence_same_gate_control_required': True,
            'same_context_conditioned_pooling_hypothesis_not_second_candidate': True},
        'new_reading_identity_or_result_credit': 0, 'new_gate_TEST_or_launch_authority': False,
        'demonstrated_superiority': False, 'global_novelty_clearance': False, 'new_ready_successor': False}

    assert index['paper_records'][:249] == previous['paper_records']
    assert compact(index['paper_records'][:249]) == compact(previous['paper_records'])
    assert index['existing_packets'][:len(previous['existing_packets'])] == previous['existing_packets']
    assert index['unresolved_primary_metadata_leads'] == previous['unresolved_primary_metadata_leads']
    for pos, group in enumerate(old_groups):
        if pos != moe_group_pos:
            assert groups[pos] == group
        else:
            assert {key: value for key, value in groups[pos].items() if key not in ('record_indices', 'raw_canonical_identifiers')} == {
                key: value for key, value in group.items() if key not in ('record_indices', 'raw_canonical_identifiers')}
            assert groups[pos]['record_indices'] == group['record_indices'] + [253]
            assert groups[pos]['raw_canonical_identifiers'] == group['raw_canonical_identifiers'] + ['arxiv:2412.00418v3']
    assert {key: value for key, value in index['canonical_identifier_normalization'].items() if key != 'groups'} == {
        key: value for key, value in previous['canonical_identifier_normalization'].items() if key != 'groups'}
    for key in previous:
        if key not in current + ['paper_records', 'existing_packets', 'canonical_identifier_normalization']:
            assert index[key] == previous[key], key
    assert len(groups) == len({group['normalized_identifier'] for group in groups})
    covered = [number for group in groups for number in group['record_indices']]
    assert set(covered) == set(range(254)) and len(covered) == 254
    totals = metrics(index)
    assert (totals['conclusion_records'], totals['normalized_paper_identifiers'], totals['software_documentation_identifiers']) == (254, 201, 2)
    assert totals['catalog_entries'] - before['catalog_entries'] == 35
    account = copy.deepcopy(previous['read_accounting'])
    account.update(totals, state='PROSPECTIVE_SAVED_GRAPH_AGGREGATION_SCOPE_ADOPTION_PENDING_ROOT_REVIEW',
        historical_path_catalog_note='All249 v69 records, all existing identity fields/list prefixes, all3 metadata leads and historical decisions/events preserved. Append five saved bounded scopes: four new identities and MoE-NP v3 in existing group. One non-reading protocol amendment bound separately. Six current metadata fields and original extended MoE-NP group snapshotted exactly. Zero integration primary/discovery payload reads, retrievals or execution.',
        latest_packet_new_scoped_primary_reads=5, latest_packet_scoped_primary_method_events=5,
        latest_packet_full_primary_reads=0, latest_packet_previously_completed_scoped_read_adoptions=5,
        latest_packet_new_paper_identity_groups=4, latest_packet_source_primary_version_documents=5,
        latest_packet_existing_identity_version_followup_scopes=1,
        latest_packet_version_followup_additional_paper_identities=0, latest_packet_repeated_exact_scopes=0,
        latest_packet_bounded_author_source_repositories=0, latest_packet_author_code_semantic_file_scopes=0,
        latest_packet_bounded_author_source_scope_events=0, latest_packet_source_repository_commits=0,
        latest_packet_source_code_line_ranges=0, latest_packet_nonreading_amendment_events=1,
        latest_packet_nonreading_amendment_read_credit=0,
        integration_pass_searches=0, integration_pass_retrievals=0, integration_pass_new_primary_reads=0,
        integration_pass_full_primary_reads=0, integration_pass_primary_method_reads=0,
        integration_pass_author_source_semantic_reads=0, integration_pass_experimental_score_artifact_reads=0,
        integration_pass_raw_primary_reads=0, integration_pass_selected_primary_text_scope_reads=0)
    index['read_accounting'] = account
    index['latest_adoption'] = {'UTC': now, 'status': 'PROSPECTIVE_PENDING_ROOT_REVIEW',
        'predecessor': ref(PREV / 'LITERATURE_INDEX.json'), 'adopted_predecessor_root_reference': ref(ADOPTION),
        'added_records': added, 'added_paper_identity_groups': IDS[:4],
        'existing_identity_version_followup': {'normalized_identifier': IDS[4], 'prior_record_indices': [101], 'added_record_index': 253},
        'source_completed_bounded_primary_scope_events': 5, 'source_full_paper_reads': 0,
        'source_new_unique_paper_identity_credit': 4, 'source_nonreading_amendment_events': 1,
        'nonreading_amendment_read_credit': 0,
        'integration_searches': 0, 'integration_primary_reads': 0, 'integration_primary_rereads': 0,
        'integration_raw_primary_reads': 0, 'integration_selected_passage_reads': 0, 'integration_retrievals': 0,
        'original_MGL_manuscript_lead_ACM403_and_source_only_event_preserved': True,
        'source_binding_reference': ref(HERE / 'SOURCE_BINDINGS.json'),
        'novelty_or_numeric_or_predictive_or_execution_adoption': False}
    encoded = compact(index) + b'\n'
    assert json.loads(encoded) == index and len(encoded) < MAX_BYTES
    save('SIZE_LIMIT_CHECK.json', {'UTC': now, 'index_bytes': len(encoded), 'decimal_2MB_limit': MAX_BYTES,
        'within_limit': True, 'serialization': 'Full schema-compatible compact UTF-8 JSON, sorted object keys',
        'JSON_roundtrip_equal': True, 'measured_before_index_write': True, 'history_discarded': False,
        'new_raw_primary_or_selected_passage_text_embedded': False,
        'predecessor_content_including_legacy_embedded_fields_preserved': True})
    with (HERE / 'LITERATURE_INDEX.json').open('xb') as stream:
        stream.write(encoded)
    save('DELTA.json', {'UTC': now, 'prospective': True,
        'predecessor_index': ref(PREV / 'LITERATURE_INDEX.json'), 'successor_index': ref(HERE / 'LITERATURE_INDEX.json'),
        'adopted_predecessor_root_reference': ref(ADOPTION), 'before': before, 'after': totals,
        'metric_deltas': {key: totals[key] - before[key] for key in totals},
        'added_paper_records': 5, 'added_normalized_paper_groups': 4, 'added_software_groups': 0,
        'saved_bounded_primary_method_scopes_adopted': 5, 'existing_identity_version_followups': 1,
        'source_new_unique_paper_identity_credit': 4, 'source_new_full_paper_reads': 0,
        'nonreading_protocol_amendments': 1, 'nonreading_amendment_reading_identity_result_credit': 0,
        'added_records': added, 'original_extended_existing_group_snapshot_reference':
            'integration_v70_predecessor_v69_snapshot/only_extended_existing_group',
        'unresolved_primary_metadata_leads_before_after': [3, 3],
        'integration_searches': 0, 'integration_retrievals': 0, 'integration_primary_reads': 0,
        'integration_primary_rereads': 0, 'integration_author_source_reads': 0,
        'integration_primary_discovery_or_selected_text_payload_reads': 0,
        'source_claims_rewritten': False, 'canonical_status_or_ledger_modified': False})
    save('VERIFICATION.json', {'UTC': now, 'status': 'PASS_METADATA_ONLY_PROSPECTIVE_PENDING_ROOT_REVIEW',
        'predecessor_index_sha256': PREV_INDEX_PIN, 'predecessor_manifest_sha256': PREV_MANIFEST_PIN,
        'predecessor_seal_sha256': PREV_SEAL_PIN, 'root_adoption_sha256': ADOPTION_PIN,
        'source_manifest_sha256': SOURCE_MANIFEST_PIN, 'source_seal_sha256': SOURCE_SEAL_PIN,
        'amendment_manifest_sha256': AMEND_MANIFEST_PIN, 'amendment_seal_sha256': AMEND_SEAL_PIN,
        'all249_predecessor_records_canonical_bytes_equal': True,
        'predecessor_record_canonical_bytes_sha256': hashlib.sha256(compact(previous['paper_records'])).hexdigest(),
        'all198_unextended_groups_preserved_exactly': True,
        'MoE_NP_group105_only_additive_raw_version_and_record_index_tails': True,
        'MoE_NP_prior_v1_record101_preserved_exactly': True,
        'original_extended_group_snapshotted_exactly': True,
        'all_old_normalization_rules_and_metadata_preserved_exactly': True,
        'all3_unresolved_metadata_leads_preserved_exactly': True,
        'predecessor_catalog_prefix_preserved': True, 'all_other_predecessor_fields_preserved': True,
        'changed_current_metadata_snapshotted_exactly': current,
        'history_source_events_MGL_event_and_all_decisions_preserved': True,
        'safe_source_metadata_members_hashes_verified': len(SAFE_SOURCE_METADATA),
        'safe_amendment_metadata_members_hashes_verified': len(SAFE_AMEND_METADATA),
        'source_primary_discovery_selected_text_and_extraction_utility_payloads_not_reopened_or_rehashed': True,
        'selected_primary_text_hashes_recomputed': 0,
        'five_saved_scope_locators_and_receipt_manifest_declarations_authenticated': True,
        'four_new_identities_and_one_existing_identity_version_followup': True,
        'one_nonreading_protocol_amendment_adds_zero_reading_identity_result_credit': True,
        'amendment_binds_effective_development_reuse_link_context_and_comparator_corrections': True,
        'initial_three_unretained_metadata_bodies_have_no_invented_hashes': True,
        'no_title_only_or_unverified_alias_merges': True, 'record_to_group_coverage_unique_complete': True,
        'recomputed_metrics': totals, 'source_bindings_reference': ref(HERE / 'SOURCE_BINDINGS.json'),
        'index_bytes': len(encoded), 'within_decimal_2MB': True, 'full_JSON_roundtrip_equal': True,
        'prior_uncertified_cumulative_read_flags_preserved': True,
        'integration_searches': 0, 'integration_retrievals': 0, 'integration_primary_reads': 0,
        'integration_primary_rereads': 0, 'integration_author_source_reads': 0,
        'new_full_paper_reads': 0, 'no_demonstrated_superiority_global_novelty_or_ready_successor': True,
        'no_new_arm_gate_or_fixed_study_change': True,
        'edited_only_new_index_v70_subtree': True, 'canonical_status_or_ledger_modified': False})
    notes = f'''# Prospective literature index v70

Prepared for independent root verification/adoption from root-adopted v69. The 249 existing records remain exactly equal in canonical JSON bytes. All 199 existing groups retain their order and old fields/values: 198 groups are exactly unchanged, while MoE-NP group 105 receives only the `arxiv:2412.00418v3` raw version and record 253 tails. Its original group is retained exactly in `integration_v70_predecessor_v69_snapshot`, alongside the six replaced current metadata fields. Record 101's v1 qualification is unchanged historical evidence.

Append five saved bounded primary-method scopes: GAMLP v3, META-DES v1, ratio-binned scaling v1, GATS v1 and MoE-NP v3. The first four create four normalized paper groups; MoE-NP v3 extends its existing identity. The totals are 254 records, 201 paper groups and 2 software groups. No full-paper, original JKNet/DAGNN, author-code, numerical-result, runtime or proof credit is added. The saved source's 16 public retrievals and 8 metadata searches remain historical accounting, with 13 retained bodies and 3 initial unretained Crossref metadata bodies that have no invented hashes.

The separate saved protocol amendment is non-reading evidence, with zero reading/identity/result credit. Preserve the original aggregation memo and link the corrected effective protocol: reused VALID may select checkpoints and fit/select aggregation as disclosed development; its performance is biased development evidence, and aggregator-only cross-validation cannot undo backbone-selection reuse. A fully frozen pipeline/comparators may use genuinely untouched TEST under its authorized protocol. No requirement to reserve four roles before fitting is imposed when final TEST is clean. Frozen TEST gates remain unchanged; no unseen TEST custody or access authority is established.

The same unconfirmed pooling hypothesis now has an explicit Citeseer-HeaRT link interface: frozen node states and normalized neighbor means on permitted fixed TRAIN support, symmetric endpoint/neighbor contrasts, structural heuristics and own-query logits/residuals, then a softmax-weighted logit sum over every already-computed route. Complete candidate universes and target-fact/support exclusions remain fixed; no propagation occurs across arbitrary query batches. Link-MoE's unchanged structural/endpoint gate on the same bank and an expanded-evidence same-gate baseline remain necessary controls. A capable same-evidence joint single is a comparator, not automatic rejection of utility. Missing caches, controls or confirmation may require representative fits; resources are a scheduling issue. All acquisition, propagation, storage and inference work remains charged.

All 3 metadata leads, original MGL manuscript/ACM403 history, the source-only MGL event, prior decisions and source histories are preserved exactly. There is no new predictive result, novelty clearance, ready successor, arm, gate, launch or scientific execution authority.

Integration authenticates 11 safe source metadata members and 5 safe amendment members plus their pinned manifests/seals, predecessor files and root adoption. Primary HTML/source bodies, extracted blocks, selected method passages, discovery bodies and extraction utility are not reopened or rehashed; their saved hashes are inherited authenticated manifest declarations. Integration performs zero public searches/retrievals, primary rereads, target payload access or scientific execution. Only this new index_v70 subtree is written.

The full JSON is {len(encoded):,} bytes, below the decimal 2,000,000-byte cap, with exact parsed roundtrip equality. Root owns independent review and adoption.
'''
    with (HERE / 'ROOT_ADOPTION_NOTES.md').open('x', encoding='utf-8') as stream:
        stream.write(notes)
    for row in predecessor_refs + source_refs + amend_refs:
        assert ref(BASE / row['path']) == row
    manifest_rows = [{'path': path.name, 'bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(HERE.iterdir()) if path.is_file()]
    save('MANIFEST.json', {'UTC': now, 'prospective': True, 'files': manifest_rows})
    save('SEAL.json', {'manifest_sha256': ref(HERE / 'MANIFEST.json')['sha256'], 'payload_files': len(manifest_rows),
        'saved_primary_scope_events_adopted': 5, 'new_paper_identity_groups': 4,
        'existing_identity_version_followups': 1, 'nonreading_protocol_amendments': 1,
        'new_full_paper_reads': 0, 'integration_primary_reads': 0,
        'source_primary_discovery_payloads_reopened_or_rehashed': False,
        'scientific_execution_or_authority': False, 'canonical_status_ledger_or_prior_index_edits': False})
    print(json.dumps({**totals, 'index_bytes': len(encoded),
        'index_sha256': ref(HERE / 'LITERATURE_INDEX.json')['sha256'],
        'manifest_sha256': ref(HERE / 'MANIFEST.json')['sha256'], 'seal_sha256': ref(HERE / 'SEAL.json')['sha256'],
        'saved_primary_scope_events_adopted': 5, 'new_paper_identity_groups': 4,
        'existing_identity_version_followups': 1, 'nonreading_protocol_amendments': 1,
        'status': 'PROSPECTIVE_READY_FOR_ROOT_REVIEW'}))


if __name__ == '__main__':
    main()
