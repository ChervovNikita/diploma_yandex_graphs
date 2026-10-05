"""Append a sealed saved MGL author-source event without reopening primary code."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
PREV = BASE / 'literature_memory/index_v68'
ADOPTION = BASE / 'literature_memory/index_v68_root_adoption_20261005_v1/ROOT_ADOPTION.json'
SOURCE = BASE / 'mgl_public_primary_scope_resolution_20261005_v1'
PREV_INDEX_PIN = '9baf1a82d51fcb6a6e29e8fb8e43dda605e0d7f15d3085e2b0129195f1362360'
PREV_MANIFEST_PIN = '128918ef4dca7ab126b5d022bc9acb4b291b6f9994a291274b0e23027a5f681e'
PREV_SEAL_PIN = 'a1acfd046b9fad6722f23d0ff78e4440c06a37129db130c34aa99ed68827e963'
ADOPTION_PIN = 'd2b36b0bbe8d56dfb06a7e0de9796b5bcd1632c508e1dbbbae56eb53a8b28ab0'
SOURCE_MANIFEST_PIN = '7fef1ab06970e4c92bdf4ccdea35c2a202324f49d3dd48486d5cdb445541ed9f'
SOURCE_SEAL_PIN = '75e6ffab659261f4e6867e1f0fab5d49b79c3e9a4817840668a722d30e8f696d'
SAFE_SOURCE_METADATA = ('SCOPED_CONCLUSIONS.json', 'READ_ACCOUNTING.json',
    'SOURCE_ASSOCIATION.json', 'SOURCE_BINDINGS.json', 'VERIFICATION.json',
    'PUBLIC_SOURCE_RETRIEVAL.json', 'REPORT.md')
MGL_ID = 'doi:10.1145/3580305.3599428'
COMMIT = 'c887b11676e7c80f108941aa7ce3cd2d35194f9c'
MAX_BYTES = 2_000_000


def ref(path):
    path = Path(path)
    assert path.resolve().is_relative_to(BASE) and not path.is_symlink()
    assert path.name not in ('PUBLIC_SOURCE_SCOPES.json', 'AUTHOR_REPOSITORY_README.md', 'PRIMARY_SCOPES.json')
    assert not path.name.endswith('_PRIMARY_SCOPES.json')
    assert path.suffix not in ('.pdf', '.html', '.xml', '.pt', '.pkl', '.npy', '.jsonl')
    raw = path.read_bytes()
    return {'path': str(path.relative_to(BASE)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def save(name, value):
    with (HERE / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


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
    refs = [manifest_ref]
    for row in rows:
        path = Path(row['path'])
        assert not path.is_absolute() and '..' not in path.parts
        got = ref(PREV / path)
        assert (got['bytes'], got['sha256']) == (row['bytes'], row['sha256'])
        refs.append(got)
    seal_ref = ref(PREV / 'SEAL.json')
    assert seal_ref['sha256'] == PREV_SEAL_PIN
    seal = json.loads((PREV / 'SEAL.json').read_text())
    assert seal['manifest_sha256'] == PREV_MANIFEST_PIN and seal['payload_files'] == len(rows)
    assert ref(PREV / 'LITERATURE_INDEX.json')['sha256'] == PREV_INDEX_PIN
    adoption_ref = ref(ADOPTION)
    assert adoption_ref['sha256'] == ADOPTION_PIN
    adoption = json.loads(ADOPTION.read_text())
    assert adoption['status'] == 'ADOPT_SAVED_SCOPED_LITERATURE_INDEX_V68'
    assert adoption['index']['sha256'] == PREV_INDEX_PIN
    assert adoption['manifest']['sha256'] == PREV_MANIFEST_PIN and adoption['seal']['sha256'] == PREV_SEAL_PIN
    assert (adoption['records'], adoption['paper_groups'], adoption['software_groups']) == (249, 197, 2)
    assert adoption['MGL_metadata_only'] is True
    return refs + [seal_ref, adoption_ref]


def authenticate_source():
    manifest_ref = ref(SOURCE / 'MANIFEST.json')
    seal_ref = ref(SOURCE / 'SEAL.json')
    assert manifest_ref['sha256'] == SOURCE_MANIFEST_PIN and seal_ref['sha256'] == SOURCE_SEAL_PIN
    rows = json.loads((SOURCE / 'MANIFEST.json').read_text())['files']
    assert len({row['path'] for row in rows}) == len(rows)
    declared = {row['path']: row for row in rows}
    seal = json.loads((SOURCE / 'SEAL.json').read_text())
    assert seal['manifest_sha256'] == SOURCE_MANIFEST_PIN
    assert seal['primary_paper_method_reads'] == seal['full_paper_reads'] == seal['new_paper_identity_read_credit'] == 0
    assert seal['bounded_public_author_repository_scope_events'] == 1 and seal['semantic_source_files'] == 3
    assert seal['pinned_author_repository_commit'] == COMMIT
    for key in ('primary_manuscript_retrieved_or_read', 'publication_or_runtime_equivalence_certified',
                'new_ready_successor_or_comparison_prepared', 'global_absence_or_novelty_clearance',
                'outcome_dataset_history_checkpoint_read', 'scientific_execution_or_source_repair',
                'allocation_18_77_MacLink_contact', 'canonical_status_ledger_index_or_fixed_study_edit',
                'execution_authority_granted'):
        assert seal[key] is False
    refs = [manifest_ref, seal_ref]
    for name in SAFE_SOURCE_METADATA:
        got = ref(SOURCE / name)
        row = declared[name]
        assert (got['bytes'], got['sha256']) == (row['bytes'], row['sha256'])
        refs.append(got)
    return declared, refs


def declaration(declared, name):
    row = declared[name]
    return {'path': str((SOURCE / name).relative_to(BASE)), 'bytes': row['bytes'], 'sha256': row['sha256'],
        'binding_status': 'Inherited authenticated sealed-manifest declaration; author-source/identity payload not reopened or independently rehashed in integration',
        'payload_accessed_in_integration': False}


def main():
    assert {path.name for path in HERE.iterdir()} == {'BUILD_INDEX.py'}
    now = datetime.now(timezone.utc).isoformat()
    predecessor_refs = authenticate_predecessor()
    declared, source_refs = authenticate_source()
    previous = json.loads((PREV / 'LITERATURE_INDEX.json').read_text())
    before = metrics(previous)
    assert before == {key: previous['read_accounting'][key] for key in before}
    assert (before['conclusion_records'], before['normalized_paper_identifiers'], before['software_documentation_identifiers']) == (249, 197, 2)
    conclusion = json.loads((SOURCE / 'SCOPED_CONCLUSIONS.json').read_text())
    accounting = json.loads((SOURCE / 'READ_ACCOUNTING.json').read_text())
    association = json.loads((SOURCE / 'SOURCE_ASSOCIATION.json').read_text())
    bindings = json.loads((SOURCE / 'SOURCE_BINDINGS.json').read_text())
    receipts = json.loads((SOURCE / 'PUBLIC_SOURCE_RETRIEVAL.json').read_text())
    source_check = json.loads((SOURCE / 'VERIFICATION.json').read_text())
    assert conclusion['canonical_paper_metadata_id'] == association['canonical_paper_metadata_id'] == MGL_ID
    assert conclusion['pinned_commit'] == association['pinned_commit'] == COMMIT
    assert conclusion['semantic_source_files'] == ['model.py', 'train.py', 'load_data.py']
    assert conclusion['primary_paper_method_read'] is False and conclusion['full_paper_read'] is False
    assert conclusion['new_paper_identity_reading_credit'] == 0
    assert conclusion['new_bounded_author_repository_scope_events'] == 1 and conclusion['public_author_source_semantic_files'] == 3
    assert conclusion['new_direct_full_operator_matches'] == conclusion['new_ready_successors'] == conclusion['new_comparisons_prepared'] == 0
    assert accounting['new_primary_paper_method_reads'] == accounting['new_full_paper_reads'] == accounting['new_paper_identity_reading_credit'] == 0
    assert accounting['new_bounded_public_author_repository_scope_events'] == 1 and accounting['new_public_author_source_semantic_file_scopes'] == 3
    assert accounting['selected_source_line_ranges'] == 17
    assert accounting['selected_source_lines_by_file'] == {'model.py': 326, 'train.py': 221, 'load_data.py': 117}
    assert association['publication_to_code_algorithm_equivalence_independently_established'] is False
    assert association['primary_manuscript_retrieved_or_read'] is False
    assert source_check['no_source_repair_or_program_execution'] is True
    assert source_check['primary_paper_method_or_full_read_credit'] == source_check['new_paper_identity_credit'] == 0
    assert bindings['no_dataset_history_checkpoint_or_outcome_payload_bindings'] is True
    assert len([row for row in previous['unresolved_primary_metadata_leads'] if row['canonical_metadata_identifier'] == MGL_ID]) == 1
    assert all(MGL_ID not in (row.get('canonical_id'), row.get('normalized_identifier')) for row in previous['paper_records'])
    assert all(MGL_ID != row['normalized_identifier'] for row in previous['canonical_identifier_normalization']['groups'])

    scope_ref = declaration(declared, 'PUBLIC_SOURCE_SCOPES.json')
    identity_ref = declaration(declared, 'AUTHOR_REPOSITORY_README.md')
    saved_scope = conclusion['selected_scope_reference']
    assert (saved_scope['bytes'], saved_scope['sha256']) == (scope_ref['bytes'], scope_ref['sha256'])
    assert Path(saved_scope['path']).resolve() == (SOURCE / 'PUBLIC_SOURCE_SCOPES.json').resolve()
    assert (bindings['README_identity_reference']['bytes'], bindings['README_identity_reference']['sha256']) == (identity_ref['bytes'], identity_ref['sha256'])
    compact_scopes = copy.deepcopy(bindings['public_claimed_author_code_bindings'])
    assert len(compact_scopes) == 3
    for row in compact_scopes:
        receipt = next(item for item in receipts['events'] if item['path'] == row['source_path'])
        assert row['pinned_commit'] == receipt['pinned_commit'] == COMMIT
        assert row['raw_body_sha256'] == receipt['sha256'] and row['git_blob_sha1'] == receipt['git_blob_sha1']
        assert receipt['git_tree_blob_match'] is True and row['raw_body_retained'] is False
    assert sum(len(row['selected_scope_ranges']) for row in compact_scopes) == 17
    event_id = 'author-source:weicy15/MGL@' + COMMIT + ':model_train_load_data-scoped'
    assert event_id not in json.dumps(previous)
    event = {
        'source_event_id': event_id,
        'event_kind': 'first_bounded_claimed_author_implementation_scope_for_known_metadata_only_lead',
        'paper_identity_association': {'canonical_metadata_id': MGL_ID, 'title': conclusion['title'],
            'existing_paper_record_indices': [], 'normalized_paper_group_created': False,
            'prior_metadata_lead_reference': {'index': ref(PREV / 'LITERATURE_INDEX.json'),
                'selector': 'unresolved_primary_metadata_leads/2'},
            'association_status': association['association_status'],
            'publication_to_code_algorithm_equivalence_established': False},
        'repository': conclusion['repository'], 'pinned_commit': COMMIT,
        'compact_source_scopes': compact_scopes,
        'conclusion': copy.deepcopy(conclusion),
        'conclusion_reference': ref(SOURCE / 'SCOPED_CONCLUSIONS.json'),
        'read_scope_reference': scope_ref,
        'README_identity_reference': identity_ref,
        'source_association_reference': ref(SOURCE / 'SOURCE_ASSOCIATION.json'),
        'source_retrieval_reference': ref(SOURCE / 'PUBLIC_SOURCE_RETRIEVAL.json'),
        'source_report_reference': ref(SOURCE / 'REPORT.md'),
        'new_paper_identity': False, 'new_primary_paper_method_read': False, 'full_paper_read': False,
        'previously_completed_author_repository_scope_events_adopted': 1,
        'previously_completed_author_source_semantic_file_scopes_adopted': 3,
        'integration_pass_new_semantic_read': False, 'integration_pass_new_primary_read': False,
        'dataset_or_score_or_checkpoint_bytes_read': False, 'source_execution': False,
        'predictive_results_adopted': False, 'demonstrated_superiority': False,
        'global_novelty_clearance': False, 'new_ready_successor': False,
        'execution_authorized': False}
    save('SOURCE_BINDINGS.json', {
        'UTC': now, 'predecessor_files': predecessor_refs, 'files': source_refs,
        'source_manifest_sha256': SOURCE_MANIFEST_PIN, 'source_seal_sha256': SOURCE_SEAL_PIN,
        'adopted_v68_root_adoption_reference': ref(ADOPTION),
        'safe_source_metadata_members_authenticated': len(SAFE_SOURCE_METADATA),
        'source_unopened_payload_declarations': [scope_ref, identity_ref],
        'source_selected_author_code_or_README_identity_payloads_opened': False,
        'source_selected_author_text_hashes_recomputed': 0,
        'original_public_code_bodies_retrieved_or_opened': False,
        'integration_searches': 0, 'integration_retrievals': 0, 'integration_primary_reads': 0,
        'integration_primary_rereads': 0, 'integration_full_paper_reads': 0,
        'integration_author_source_semantic_reads': 0,
        'source_completed_author_repository_scope_events': 1,
        'source_completed_author_code_semantic_files': 3,
        'source_primary_paper_method_reads': 0, 'source_full_paper_reads': 0, 'source_new_paper_identity_credit': 0,
        'source_read_accounting_reference': ref(SOURCE / 'READ_ACCOUNTING.json'),
        'source_association_and_qualification_claims_inherited_not_independently_reassessed': True,
        'source_prior_index_or_experimental_bindings_not_dereferenced_in_integration': True,
        'source_dataset_tree_entries_and_logging_code_not_semantically_reopened': True,
        'edited_only_new_index_v69_subtree': True})

    index = copy.deepcopy(previous)
    current = ['schema', 'created_UTC', 'latest_adoption', 'read_accounting', 'predecessor_index', 'predecessor_index_sha256']
    index['integration_v69_predecessor_v68_snapshot'] = {
        **{key: copy.deepcopy(previous[key]) for key in current},
        'index_reference': ref(PREV / 'LITERATURE_INDEX.json'),
        'manifest_reference': ref(PREV / 'MANIFEST.json'), 'seal_reference': ref(PREV / 'SEAL.json'),
        'root_adoption_reference': ref(ADOPTION)}
    index.update(schema='literature-memory-index-v69', created_UTC=now,
        predecessor_index='literature_memory/index_v68/LITERATURE_INDEX.json', predecessor_index_sha256=PREV_INDEX_PIN)
    for row in source_refs:
        assert all(old['path'] != row['path'] for old in index['existing_packets'])
        index['existing_packets'].append({**row, 'kind': 'saved_MGL_author_source_scope_metadata',
            'source_bindings_reference': ref(HERE / 'SOURCE_BINDINGS.json'),
            'scope': 'One previously completed claimed-author repository scope, three semantic code files, zero paper reading/identity credit; zero integration source or primary rereading.'})
    for row, kind in [(scope_ref, 'sealed_manifest_declared_author_source_scope_payload_unopened_in_integration'),
                      (identity_ref, 'sealed_manifest_declared_README_identity_payload_unopened_in_integration')]:
        assert all(old['path'] != row['path'] for old in index['existing_packets'])
        index['existing_packets'].append({**row, 'kind': kind, 'source_bindings_reference': ref(HERE / 'SOURCE_BINDINGS.json')})
    index['public_author_source_scope_upgrades_v69'] = [event]
    index['MGL_author_repository_scope_limits_v69'] = {
        'source_event_reference': {'field': 'public_author_source_scope_upgrades_v69', 'index': 0, 'source_event_id': event_id},
        'source_conclusion_reference': ref(SOURCE / 'SCOPED_CONCLUSIONS.json'),
        'source_association_reference': ref(SOURCE / 'SOURCE_ASSOCIATION.json'),
        'source_read_accounting_preserved': copy.deepcopy(accounting),
        'source_association_preserved': copy.deepcopy(association),
        'closest_ancestry': conclusion['closest_ancestry'],
        'source_roles': copy.deepcopy(conclusion['source_roles']),
        'one_supported_configuration_gap': copy.deepcopy(conclusion['one_supported_configuration_gap']),
        'implementation_qualification_limits': copy.deepcopy(conclusion['implementation_qualification_limits']),
        'manuscript_still_unread': True, 'publication_runtime_operator_equivalence_unverified': True,
        'original_MGL_metadata_lead_and_ACM403_history_preserved': True,
        'MGL_paper_record_group_or_read_credit_added': False,
        'new_direct_full_operator_matches': 0, 'new_ready_successors': 0, 'demonstrated_superiority': False,
        'global_absence_proof': False, 'global_novelty_clearance': False,
        'no_new_arm_grid_comparison_gate_threshold_or_fixed_study_change': True,
        'allowed_statement': conclusion['allowed_statement'],
        'scope_limits': 'Saved source-only ancestry: item-cooccurrence auxiliary supervision, temporary plain-gradient graph-generator response, query recommendation loss, global Adam commit and adaptation-free prediction function. Private Adam carry/recomputed commitment absent in saved source. One author-source event/three files; zero paper reading or new identity credit and zero integration semantic reads. Source defects and publication equivalence remain unqualified.'}
    assert index['paper_records'] == previous['paper_records']
    assert compact(index['paper_records']) == compact(previous['paper_records'])
    assert index['canonical_identifier_normalization'] == previous['canonical_identifier_normalization']
    assert index['unresolved_primary_metadata_leads'] == previous['unresolved_primary_metadata_leads']
    assert index['existing_packets'][:len(previous['existing_packets'])] == previous['existing_packets']
    for key in previous:
        if key not in current + ['existing_packets']:
            assert index[key] == previous[key], key
    assert set(number for group in index['canonical_identifier_normalization']['groups'] for number in group['record_indices']) == set(range(249))
    totals = metrics(index)
    assert (totals['conclusion_records'], totals['normalized_paper_identifiers'], totals['software_documentation_identifiers']) == (249, 197, 2)
    account = copy.deepcopy(previous['read_accounting'])
    account.update(totals, state='PROSPECTIVE_SAVED_MGL_AUTHOR_SOURCE_ADOPTION_PENDING_ROOT_REVIEW',
        historical_path_catalog_note='All249 v68 records, all197 paper/2 software groups, all3 metadata leads, source events, decisions and history preserved exactly. Append only one saved MGL claimed-author repository event and its catalog/limits; no paper record/group/read credit. Six current metadata fields snapshotted exactly. Zero integration retrieval, primary/source rereading or execution.',
        latest_packet_new_scoped_primary_reads=0, latest_packet_full_primary_reads=0,
        latest_packet_previously_completed_scoped_read_adoptions=0, latest_packet_new_paper_identity_groups=0,
        latest_packet_source_primary_version_documents=0, latest_packet_version_followup_additional_paper_identities=0,
        latest_packet_bounded_author_source_repositories=1, latest_packet_author_code_semantic_file_scopes=3,
        latest_packet_bounded_author_source_scope_events=1, latest_packet_source_repository_commits=1,
        latest_packet_source_code_line_ranges=17,
        integration_pass_searches=0, integration_pass_retrievals=0, integration_pass_new_primary_reads=0,
        integration_pass_full_primary_reads=0, integration_pass_primary_method_reads=0,
        integration_pass_author_source_semantic_reads=0, integration_pass_experimental_score_artifact_reads=0,
        integration_pass_raw_primary_reads=0, integration_pass_selected_primary_text_scope_reads=0)
    index['read_accounting'] = account
    index['latest_adoption'] = {'UTC': now, 'status': 'PROSPECTIVE_PENDING_ROOT_REVIEW',
        'predecessor': ref(PREV / 'LITERATURE_INDEX.json'), 'adopted_predecessor_root_reference': ref(ADOPTION),
        'previous_records_groups_leads_preserved': True, 'added_records': [], 'added_paper_identity_groups': [],
        'added_author_source_events': [event_id], 'source_completed_author_repository_scope_events': 1,
        'source_completed_author_semantic_file_scopes': 3, 'source_primary_method_reads': 0,
        'source_full_paper_reads': 0, 'source_new_paper_identity_read_credit': 0,
        'integration_searches': 0, 'integration_primary_reads': 0, 'integration_primary_rereads': 0,
        'integration_author_source_semantic_reads': 0, 'integration_raw_primary_reads': 0, 'integration_retrievals': 0,
        'MGL_manuscript_still_unread': True, 'MGL_prior_metadata_lead_unchanged': True,
        'source_binding_reference': ref(HERE / 'SOURCE_BINDINGS.json'),
        'novelty_or_numeric_or_predictive_or_execution_adoption': False}

    encoded = compact(index) + b'\n'
    assert json.loads(encoded) == index and len(encoded) <= MAX_BYTES
    save('SIZE_LIMIT_CHECK.json', {'UTC': now, 'index_bytes': len(encoded), 'decimal_2MB_limit': MAX_BYTES,
        'within_limit': True, 'serialization': 'Full schema-compatible compact UTF-8 JSON, sorted object keys',
        'JSON_roundtrip_equal': True, 'measured_before_index_write': True, 'history_discarded': False,
        'new_raw_author_code_or_primary_text_embedded': False,
        'predecessor_content_including_legacy_embedded_fields_preserved': True, 'publisher_modified': False})
    with (HERE / 'LITERATURE_INDEX.json').open('xb') as stream:
        stream.write(encoded)
    save('DELTA.json', {'UTC': now, 'prospective': True,
        'predecessor_index': ref(PREV / 'LITERATURE_INDEX.json'), 'successor_index': ref(HERE / 'LITERATURE_INDEX.json'),
        'adopted_predecessor_root_reference': ref(ADOPTION), 'before': before, 'after': totals,
        'metric_deltas': {key: totals[key] - before[key] for key in totals},
        'added_paper_records': 0, 'added_normalized_paper_groups': 0, 'added_software_groups': 0,
        'saved_author_repository_events_adopted': 1, 'saved_author_source_semantic_file_scopes_adopted': 3,
        'new_paper_identity_reading_credit': 0, 'new_primary_method_or_full_read_credit': 0,
        'MGL_metadata_lead_count_unchanged': True, 'unresolved_primary_metadata_leads_before_after': [3, 3],
        'source_event_id': event_id, 'integration_searches': 0, 'integration_retrievals': 0,
        'integration_primary_reads': 0, 'integration_primary_rereads': 0,
        'integration_author_source_semantic_reads': 0, 'integration_selected_source_payload_reads': 0,
        'source_claims_rewritten': False, 'canonical_status_or_ledger_modified': False})
    save('VERIFICATION.json', {'UTC': now, 'status': 'PASS_METADATA_ONLY_PROSPECTIVE_PENDING_ROOT_REVIEW',
        'predecessor_index_sha256': PREV_INDEX_PIN, 'predecessor_manifest_sha256': PREV_MANIFEST_PIN,
        'predecessor_seal_sha256': PREV_SEAL_PIN, 'root_adoption_sha256': ADOPTION_PIN,
        'source_manifest_sha256': SOURCE_MANIFEST_PIN, 'source_seal_sha256': SOURCE_SEAL_PIN,
        'all249_predecessor_records_canonical_bytes_equal': True,
        'predecessor_record_canonical_bytes_sha256': hashlib.sha256(compact(previous['paper_records'])).hexdigest(),
        'all197_paper_and2_software_group_order_values_rules_preserved': True,
        'all3_unresolved_metadata_leads_preserved_exactly': True,
        'predecessor_catalog_prefix_preserved': True, 'all_other_predecessor_fields_preserved': True,
        'changed_current_metadata_snapshotted_exactly': current,
        'history_source_events_and_decisions_preserved': True,
        'safe_source_metadata_members_hashes_verified': len(SAFE_SOURCE_METADATA),
        'source_author_code_or_identity_payloads_not_reopened_or_rehashed': True,
        'selected_source_text_hashes_recomputed': 0, 'saved_scope_locators_and_git_receipt_claims_authenticated': True,
        'new_author_source_event_count': 1, 'associated_semantic_source_files': 3,
        'no_new_paper_records_groups_or_read_credit': True,
        'no_title_only_or_unverified_alias_merges': True,
        'manuscript_runtime_operator_equivalence_not_certified': True,
        'recomputed_metrics': totals, 'source_bindings_reference': ref(HERE / 'SOURCE_BINDINGS.json'),
        'index_bytes': len(encoded), 'within_decimal_2MB': True, 'full_JSON_roundtrip_equal': True,
        'prior_uncertified_cumulative_read_flags_preserved': True,
        'integration_searches': 0, 'integration_retrievals': 0, 'integration_primary_reads': 0,
        'integration_primary_rereads': 0, 'integration_author_source_semantic_reads': 0,
        'new_full_paper_reads': 0, 'no_demonstrated_superiority_global_novelty_or_ready_successor': True,
        'no_new_arm_gate_or_fixed_study_change': True,
        'edited_only_new_index_v69_subtree': True, 'canonical_status_or_ledger_modified': False})
    notes = f'''# Prospective literature index v69

Prepared for independent root verification/adoption from adopted v68. All 249 conclusion records, 197 normalized paper groups, 2 software groups and 3 unresolved primary metadata leads remain unchanged, with exact canonical-byte equality of every record and preserved catalog prefixes, source events, decisions and history. The six replaced current metadata fields are retained exactly in `integration_v69_predecessor_v68_snapshot`; adopted v68 is bound to its root adoption metadata.

Append only the saved author-source event for MGL (DOI10.1145/3580305.3599428), repository `weicy15/MGL` at `{COMMIT}`. It contributes one previously completed bounded claimed-author repository event across three semantic files and17 line ranges. It adds zero paper records, normalized identities, primary-paper method reads or full-paper reads. The original MGL metadata-only manuscript lead and prior ACM403 remain unchanged historical evidence, linked to this narrower source event. Exact-title/official-README/profile-first-author association is a documented source claim, not independently established publication-to-code equivalence.

The saved construction provides close recommendation/meta-learning component ancestry: TRAIN item-cooccurrence support, temporary plain-gradient graph-generator response, query recommendation loss, one global Adam outer commit, and a prediction function without optimizer adaptation. It does not implement retained private Adam learners with recomputed private commitment. Saved state_dict/reLU-path and runtime-source defects remain explicit; no numerical derivative, execution, publication/runtime/operator equivalence, demonstrated superiority, global novelty clearance or ready successor is established.

Integration performs zero retrievals/searches, zero primary or author-code rereads, and zero execution. Seven safe saved metadata/conclusion/report members plus source manifest/seal, all predecessor packet members and root-adoption metadata were authenticated. Public author code scope and README identity payloads were neither reopened nor rehashed: their locators/hashes are inherited sealed-manifest declarations. Original Git-blob checks and deleted-body receipts remain source claims. No raw new code or primary text is embedded.

The full JSON is {len(encoded):,} bytes, below the decimal 2,000,000-byte limit, with parsed roundtrip equality. Only this new index_v69 subtree was edited; canonical status/ledger, previous indices, frozen studies and experimental endpoints are untouched. Root will independently review and adopt.
'''
    with (HERE / 'ROOT_ADOPTION_NOTES.md').open('x', encoding='utf-8') as stream:
        stream.write(notes)
    for row in predecessor_refs + source_refs:
        got = ref(BASE / row['path'])
        assert (got['bytes'], got['sha256']) == (row['bytes'], row['sha256'])
    manifest_rows = [{'path': path.name, 'bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(HERE.iterdir()) if path.is_file()]
    save('MANIFEST.json', {'UTC': now, 'prospective': True, 'files': manifest_rows})
    save('SEAL.json', {'manifest_sha256': ref(HERE / 'MANIFEST.json')['sha256'], 'payload_files': len(manifest_rows)})
    print(json.dumps({**totals, 'index_bytes': len(encoded),
        'index_sha256': ref(HERE / 'LITERATURE_INDEX.json')['sha256'],
        'manifest_sha256': ref(HERE / 'MANIFEST.json')['sha256'], 'seal_sha256': ref(HERE / 'SEAL.json')['sha256'],
        'saved_author_source_events_adopted': 1, 'new_paper_reading_credit': 0,
        'status': 'PROSPECTIVE_READY_FOR_ROOT_REVIEW'}))


if __name__ == '__main__':
    main()
