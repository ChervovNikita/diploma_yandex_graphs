"""Append sealed saved conclusions and declarations; never open primary scopes."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
PREV = BASE / 'literature_memory/index_v67'
SOURCE = BASE / 'persistent_optimizer_history_shared_graph_literature_20261005_v1'
PREV_INDEX_PIN = 'a0a7679813bb1eef1970012a62014f460f7df530a87c221454a631bbc605e511'
PREV_MANIFEST_PIN = 'cdefdd1592a70f9b35c7a97debae1aa334c1fef01516aa9d486c23742f5c9af6'
PREV_SEAL_PIN = '3f4d7434d70f504e7fee94174b65bfa48eea4842724108f87c25e8339598cf8d'
SOURCE_MANIFEST_PIN = '07b29f9e50b8f2c813de4805d94d6561e4e5dd3f971105aa5bd088c316322697'
SOURCE_SEAL_PIN = '0ad976d3852f0b30b2d87b9d275eba8e09e5f9815d890b30c4998bb51dd9cd17'
SAFE_SOURCE_METADATA = (
    'PAPER_CONCLUSIONS.json', 'READ_ACCOUNTING.json', 'DISCOVERY_DEDUP.json',
    'PRIMARY_RETRIEVAL.json', 'SOURCE_BINDINGS.json', 'VERIFICATION.json',
    'UNRESOLVED_COMPARISON.json', 'SCOPE_SELECTION.json', 'REPORT.md')
MAX_BYTES = 2_000_000


def ref(path):
    path = Path(path)
    assert path.resolve().is_relative_to(BASE) and not path.is_symlink()
    assert path.name != 'PRIMARY_SCOPES.json'
    assert not path.name.endswith('_PRIMARY_SCOPES.json')
    assert path.suffix not in ('.pdf', '.html', '.xml', '.pt', '.npy', '.jsonl')
    raw = path.read_bytes()
    return {'path': str(path.relative_to(BASE)), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def save(name, value):
    with (HERE / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def normalized(value):
    result = set()
    for item in value.split(';'):
        item = item.strip().lower()
        if item.startswith('arxiv:'):
            item = re.sub(r'v\d+$', '', item)
        if item:
            result.add(item)
    return result


def metrics(index):
    catalog = {row['path'] for row in index['existing_packets']}
    conclusions = {row['conclusion_file'] for row in index['paper_records']
                   if isinstance(row.get('conclusion_file'), str)}
    scopes = {row['read_scope_reference']['path'] for row in index['paper_records']
              if 'read_scope_reference' in row}
    legacy = {row['read_scope_file_reference']['path'] for row in index['paper_records']
              if 'read_scope_file_reference' in row}
    groups = index['canonical_identifier_normalization']['groups']
    return {'conclusion_records': len(index['paper_records']),
            'normalized_paper_identifiers': sum(row['kind'] == 'paper' for row in groups),
            'software_documentation_identifiers': sum(row['kind'] != 'paper' for row in groups),
            'catalog_entries': len(index['existing_packets']),
            'unique_catalog_document_paths': len(catalog),
            'unique_conclusion_source_documents': len(conclusions),
            'unique_referenced_document_paths': len(catalog | conclusions),
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
    return verified + [seal_ref]


def authenticate_source_metadata():
    manifest_ref = ref(SOURCE / 'MANIFEST.json')
    seal_ref = ref(SOURCE / 'SEAL.json')
    assert manifest_ref['sha256'] == SOURCE_MANIFEST_PIN and seal_ref['sha256'] == SOURCE_SEAL_PIN
    rows = json.loads((SOURCE / 'MANIFEST.json').read_text())['files']
    assert len({row['path'] for row in rows}) == len(rows)
    declared = {row['path']: row for row in rows}
    seal = json.loads((SOURCE / 'SEAL.json').read_text())
    assert seal['manifest_sha256'] == SOURCE_MANIFEST_PIN
    assert seal['new_bounded_primary_identities'] == seal['new_primary_version_documents'] == 2
    assert seal['full_paper_reads'] == 0
    for key in ('earlier_primary_scopes_reopened', 'raw_public_bodies_retained',
                'fixed30_39_changed', 'scientific_execution', 'allocation_18_77_MacLink_contact',
                'new_arm_grid_gate_threshold_or_execution_authority', 'global_absence_or_novelty_clearance'):
        assert seal[key] is False
    verified = [manifest_ref, seal_ref]
    for name in SAFE_SOURCE_METADATA:
        got = ref(SOURCE / name)
        row = declared[name]
        assert (got['bytes'], got['sha256']) == (row['bytes'], row['sha256'])
        verified.append(got)
    return declared, verified


def declared_scope(declared, selector):
    row = declared['PRIMARY_SCOPES.json']
    return {'path': str((SOURCE / 'PRIMARY_SCOPES.json').relative_to(BASE)),
            'bytes': row['bytes'], 'sha256': row['sha256'], 'selector': selector,
            'binding_status': 'Inherited authenticated sealed-manifest declaration; selected primary text payload not reopened or independently rehashed in integration',
            'payload_accessed_in_integration': False, 'full_paper_read': False}


def main():
    assert {path.name for path in HERE.iterdir()} == {'BUILD_INDEX.py'}
    now = datetime.now(timezone.utc).isoformat()
    predecessor_refs = authenticate_predecessor()
    declared, source_refs = authenticate_source_metadata()
    previous = json.loads((PREV / 'LITERATURE_INDEX.json').read_text())
    before = metrics(previous)
    assert before == {key: previous['read_accounting'][key] for key in before}
    assert (before['conclusion_records'], before['normalized_paper_identifiers'],
            before['software_documentation_identifiers']) == (247, 195, 2)
    conclusions = json.loads((SOURCE / 'PAPER_CONCLUSIONS.json').read_text())
    accounting = json.loads((SOURCE / 'READ_ACCOUNTING.json').read_text())
    dedup = json.loads((SOURCE / 'DISCOVERY_DEDUP.json').read_text())
    receipts = json.loads((SOURCE / 'PRIMARY_RETRIEVAL.json').read_text())
    source_check = json.loads((SOURCE / 'VERIFICATION.json').read_text())
    source_bindings = json.loads((SOURCE / 'SOURCE_BINDINGS.json').read_text())
    comparison = json.loads((SOURCE / 'UNRESOLVED_COMPARISON.json').read_text())
    selection = json.loads((SOURCE / 'SCOPE_SELECTION.json').read_text())
    candidates = conclusions['primary_method_scopes']
    assert [row['canonical_id'] for row in candidates] == ['arxiv:2312.02204', 'arxiv:2309.10376']
    assert [row['exact_primary_id'] for row in candidates] == ['arxiv:2312.02204v1', 'arxiv:2309.10376v1']
    assert conclusions['new_primary_identity_groups'] == accounting['new_distinct_primary_identity_groups'] == 2
    assert accounting['new_bounded_primary_method_papers'] == accounting['new_primary_version_documents_read'] == 2
    assert accounting['selected_method_containers'] == 8
    assert accounting['new_full_paper_reads'] == accounting['previously_saved_primary_method_rereads'] == 0
    assert accounting['known_paper_new_identity_credit'] == 0
    assert conclusions['new_direct_full_operator_matches'] == conclusions['new_supported_successors'] == 0
    assert conclusions['global_absence_proof'] is False and conclusions['fixed_study_changes'] is False
    assert source_check['fixed30_39_designs_preserved'] is True
    assert source_check['no_new_arm_grid_gate_threshold_or_launch'] is True
    assert dedup['selected_identifiers_absent_from_index67'] is True
    assert dedup['MGL_HTTP403_method_read'] is False
    assert comparison['maximum_comparisons_retained'] == 1
    assert comparison['preparation_or_execution_authorized'] is False
    assert comparison['new_arm_grid_gate_threshold_or_fixed_study_change'] is False

    version_docs = []
    document_keys = set()
    for pos, row in enumerate(candidates):
        assert normalized(row['exact_primary_id']) == normalized(row['canonical_id'])
        scope_ref = declared_scope(declared, 'selected_method_documents/' + row['key'])
        saved = row['source_scope_text_reference']
        assert (saved['bytes'], saved['sha256']) == (scope_ref['bytes'], scope_ref['sha256'])
        assert saved['selector'] == scope_ref['selector']
        assert Path(saved['path']).resolve() == (SOURCE / 'PRIMARY_SCOPES.json').resolve()
        binding = next(item for item in source_bindings['new_public_primary_scope_bindings']
                       if item['exact_primary_id'] == row['exact_primary_id'])
        receipt = next(item for item in receipts['events'] if item['key'] == row['key'])
        assert receipt['status'] == 200 and receipt['temporary_path_deleted'] is True
        assert receipt['sha256'] == binding['raw_HTTP_body_sha256']
        doc = {'canonical_id': row['canonical_id'], 'exact_primary_id': row['exact_primary_id'],
               'scope_locators': copy.deepcopy(row['scope_locators']),
               'selected_containers': copy.deepcopy(binding['selected_containers']),
               'selected_container_count': len(binding['selected_containers']),
               'scope_locator_source': f'PAPER_CONCLUSIONS.json:primary_method_scopes/{pos}/scope_locators',
               'scope_payload_reference': scope_ref,
               'saved_retrieval_metadata': {key: receipt[key] for key in
                    ('url', 'final_url', 'UTC', 'bytes', 'sha256', 'status')}}
        doc['saved_retrieval_metadata']['hash_status'] = 'Original deleted-raw-body receipt claim, not independently retrieved or rehashed in integration'
        key = hashlib.sha256(compact({key: doc[key] for key in
            ('canonical_id', 'exact_primary_id', 'scope_locators', 'scope_payload_reference')})).hexdigest()
        assert key not in document_keys
        document_keys.add(key)
        doc['scope_deduplication_key_sha256'] = key
        version_docs.append(doc)
    assert sum(row['selected_container_count'] for row in version_docs) == 8
    assert len({row['exact_primary_id'] for row in version_docs}) == 2

    save('SOURCE_BINDINGS.json', {
        'UTC': now, 'predecessor_files': predecessor_refs, 'files': source_refs,
        'source_manifest_sha256': SOURCE_MANIFEST_PIN, 'source_seal_sha256': SOURCE_SEAL_PIN,
        'safe_source_metadata_members_authenticated': len(SAFE_SOURCE_METADATA),
        'source_scope_payload_declarations': [row['scope_payload_reference'] for row in version_docs],
        'source_manifest_authenticated_all_payloads_not_rehashed': True,
        'selected_primary_scope_payloads_opened': False, 'selected_primary_text_hashes_recomputed': 0,
        'raw_primary_bodies_opened': False, 'integration_searches': 0, 'integration_retrievals': 0,
        'integration_primary_reads': 0, 'integration_primary_rereads': 0, 'integration_full_paper_reads': 0,
        'source_completed_bounded_primary_method_papers': 2, 'source_completed_version_documents': 2,
        'source_selected_containers': 8, 'source_full_paper_reads': 0, 'reused_primary_new_credit': 0,
        'source_accounting_reference': ref(SOURCE / 'READ_ACCOUNTING.json'),
        'source_incidental_public_exposure': copy.deepcopy(accounting['incidental_public_exposure']),
        'incidental_public_claims_not_reassessed_or_adopted_by_integration': True,
        'source_prior_fixed_plan_or_outcome_bindings_not_dereferenced_in_integration': True,
        'integrity_checks_of_saved_metadata_only': True,
        'edited_only_new_index_v68_subtree': True})

    index = copy.deepcopy(previous)
    current = ['schema', 'created_UTC', 'latest_adoption', 'read_accounting',
               'predecessor_index', 'predecessor_index_sha256']
    snapshots = current + ['unresolved_primary_metadata_leads']
    index['integration_v68_predecessor_v67_snapshot'] = {
        **{key: copy.deepcopy(previous[key]) for key in snapshots},
        'index_reference': ref(PREV / 'LITERATURE_INDEX.json'),
        'manifest_reference': ref(PREV / 'MANIFEST.json'), 'seal_reference': ref(PREV / 'SEAL.json')}
    index.update(schema='literature-memory-index-v68', created_UTC=now,
                 predecessor_index='literature_memory/index_v67/LITERATURE_INDEX.json',
                 predecessor_index_sha256=PREV_INDEX_PIN)
    groups = index['canonical_identifier_normalization']['groups']
    added = []
    for pos, row in enumerate(candidates):
        identity = next(iter(normalized(row['canonical_id'])))
        assert all(identity not in normalized(value) for group in groups for value in
                   [group['normalized_identifier']] + group.get('raw_canonical_identifiers', []) + group.get('explicit_aliases', []))
        assert all(identity not in normalized(item.get('canonical_id', '')) for item in previous['paper_records'])
        docs = [copy.deepcopy(doc) for doc in version_docs if doc['canonical_id'] == identity]
        key = hashlib.sha256(compact({'normalized_identifier': identity, 'version_document_scopes': docs})).hexdigest()
        position = len(index['paper_records'])
        index['paper_records'].append({
            'canonical_id': row['exact_primary_id'], 'normalized_identifier': identity,
            'conclusion': copy.deepcopy(row),
            'conclusion_file': str((SOURCE / 'PAPER_CONCLUSIONS.json').relative_to(BASE)),
            'conclusion_file_sha256': ref(SOURCE / 'PAPER_CONCLUSIONS.json')['sha256'],
            'conclusion_source_selector': f'primary_method_scopes/{pos}',
            'read_scope_reference': {**ref(SOURCE / 'PAPER_CONCLUSIONS.json'),
                'selector': f'primary_method_scopes/{pos}/scope_locators',
                'scope_kind': 'Authenticated saved locator metadata; selected primary text not reopened'},
            'exact_read_scope': docs, 'source_packet': SOURCE.name,
            'source_packet_binding_reference': ref(HERE / 'SOURCE_BINDINGS.json'),
            'source_retrieval_reference': ref(SOURCE / 'PRIMARY_RETRIEVAL.json'),
            'scope_deduplication_key_sha256': key,
            'scope_deduplication_key_schema': 'normalized paper identity + exact arXiv version + saved locators and sealed payload declaration selector',
            'read_status': 'Previously completed bounded source method scope; zero integration searches, retrievals or primary reads; not a full-paper read',
            'scoped_method_read': True, 'full_paper_read': False, 'author_source_read': False,
            'numeric_result_transfer': False, 'predictive_adoption': False, 'global_novelty_clearance': False,
            'execution_authorized': False, 'integration_pass_new_primary_reads': 0,
            'integration_pass_primary_reread': False, 'version_followup_is_additional_paper_identity': False})
        groups.append({'normalized_identifier': identity, 'kind': 'paper',
                       'raw_canonical_identifiers': [identity, row['exact_primary_id']],
                       'explicit_aliases': [], 'record_indices': [position]})
        added.append({'canonical_id': identity, 'exact_canonical_id': row['exact_primary_id'],
                      'record_index': position, 'new_identity': True, 'bounded_version_documents': 1,
                      'full_paper_read': False, 'source_packet': SOURCE.name,
                      'scope_deduplication_key_sha256': key})

    for row in source_refs:
        assert all(old['path'] != row['path'] for old in index['existing_packets'])
        index['existing_packets'].append({**row, 'kind': 'saved_optimizer_history_and_graph_scope_metadata',
            'source_bindings_reference': ref(HERE / 'SOURCE_BINDINGS.json'),
            'scope': 'Two saved paper identities and two v1 method documents; no integration primary reading or novelty/predictive/execution adoption.'})
    # Both records select different containers inside one unread scope file.
    payload_ref = {key: value for key, value in version_docs[0]['scope_payload_reference'].items() if key != 'selector'}
    assert all(old['path'] != payload_ref['path'] for old in index['existing_packets'])
    index['existing_packets'].append({**payload_ref,
        'selectors': [doc['scope_payload_reference']['selector'] for doc in version_docs],
        'kind': 'sealed_manifest_declared_selected_scope_payload_unopened_in_integration',
        'source_bindings_reference': ref(HERE / 'SOURCE_BINDINGS.json')})

    mgl = next(row for row in selection['selected_candidates'] if row['key'] == 'MGL')
    assert mgl['canonical_id'] == 'doi:10.1145/3580305.3599428'
    mgl_receipt = next(row for row in receipts['events'] if row['key'] == 'MGL')
    assert '403' in mgl_receipt['error']
    mgl_rows = [row for row in dedup['metadata_rows'] if mgl['canonical_id'] in row['normalized_identifiers']]
    assert mgl_rows and all(row['scope_read_in_this_task'] is False for row in mgl_rows)
    assert all(row['canonical_metadata_identifier'] != mgl['canonical_id'] for row in previous['unresolved_primary_metadata_leads'])
    lead = {'canonical_metadata_identifier': mgl['canonical_id'], 'doi': '10.1145/3580305.3599428',
            'title': mgl['title'], 'year': mgl_rows[0]['year'],
            'status': 'Source packet discovery/abstract metadata only; primary public PDF route returned HTTP403',
            'primary_url_attempted': mgl['url'], 'full_paper_read': False,
            'scoped_primary_method_read': False, 'paper_record_added': False,
            'normalized_paper_group_added': False, 'new_identity_reading_credit': 0,
            'source_reference': {**ref(SOURCE / 'SCOPE_SELECTION.json'), 'selector': 'selected_candidates/1'},
            'retrieval_reference': {**ref(SOURCE / 'PRIMARY_RETRIEVAL.json'), 'selector': 'events/1'},
            'consequence': 'Relevant graph virtual-update lead remains unresolved. No verified private optimizer-history, derivative or deployment mechanism; not negative novelty evidence.'}
    index['unresolved_primary_metadata_leads'].append(lead)

    assert index['paper_records'][:247] == previous['paper_records']
    assert compact(index['paper_records'][:247]) == compact(previous['paper_records'])
    assert index['existing_packets'][:len(previous['existing_packets'])] == previous['existing_packets']
    assert groups[:len(previous['canonical_identifier_normalization']['groups'])] == previous['canonical_identifier_normalization']['groups']
    assert index['unresolved_primary_metadata_leads'][:-1] == previous['unresolved_primary_metadata_leads']
    assert {key: value for key, value in index['canonical_identifier_normalization'].items() if key != 'groups'} == {
        key: value for key, value in previous['canonical_identifier_normalization'].items() if key != 'groups'}
    for key in previous:
        if key not in current + ['paper_records', 'existing_packets', 'canonical_identifier_normalization', 'unresolved_primary_metadata_leads']:
            assert index[key] == previous[key], key
    assert set(number for group in groups for number in group['record_indices']) == set(range(249))
    for identity in ['arxiv:2312.02204', 'arxiv:2309.10376']:
        assert sum(group['normalized_identifier'] == identity for group in groups) == 1
    totals = metrics(index)
    assert (totals['conclusion_records'], totals['normalized_paper_identifiers'], totals['software_documentation_identifiers']) == (249, 197, 2)
    account = copy.deepcopy(previous['read_accounting'])
    account.update(totals, state='PROSPECTIVE_SAVED_OPTIMIZER_HISTORY_SCOPE_ADOPTION_PENDING_ROOT_REVIEW',
        historical_path_catalog_note='All 247 v67 records, 195 paper and 2 software groups, catalog prefixes, source events, decision linkages and historical fields retained. Append two previously completed v1 method identities. Six replaced current fields and pre-append unresolved-lead list snapshotted exactly; MGL remains metadata-only. Zero integration searches/retrievals/primary reads.',
        latest_packet_new_scoped_primary_reads=2, latest_packet_full_primary_reads=0,
        latest_packet_previously_completed_scoped_read_adoptions=2, latest_packet_new_paper_identity_groups=2,
        latest_packet_source_primary_version_documents=2, latest_packet_version_followup_additional_paper_identities=0,
        integration_pass_searches=0, integration_pass_new_primary_reads=0, integration_pass_full_primary_reads=0,
        integration_pass_primary_method_reads=0, integration_pass_author_source_semantic_reads=0,
        integration_pass_experimental_score_artifact_reads=0, integration_pass_retrievals=0,
        integration_pass_raw_primary_reads=0, integration_pass_selected_primary_text_scope_reads=0)
    index['read_accounting'] = account
    index['latest_adoption'] = {'UTC': now, 'status': 'PROSPECTIVE_PENDING_ROOT_REVIEW',
        'predecessor': ref(PREV / 'LITERATURE_INDEX.json'), 'previous_records_preserved': True,
        'added_records': added, 'added_metadata_only_leads': [lead],
        'source_completed_new_scoped_method_papers': 2, 'source_bounded_version_documents': 2,
        'source_full_paper_reads': 0, 'integration_searches': 0, 'integration_primary_reads': 0,
        'integration_primary_rereads': 0, 'integration_raw_primary_reads': 0, 'integration_retrievals': 0,
        'reused_primary_new_read_credit': 0, 'version_followup_extra_paper_identities': 0,
        'source_binding_reference': ref(HERE / 'SOURCE_BINDINGS.json'),
        'novelty_or_numeric_or_predictive_or_execution_adoption': False}
    index['persistent_optimizer_history_shared_graph_prior_limits_v68'] = {
        'source_conclusions_reference': ref(SOURCE / 'PAPER_CONCLUSIONS.json'),
        'source_report_reference': ref(SOURCE / 'REPORT.md'),
        'source_verification_reference': ref(SOURCE / 'VERIFICATION.json'),
        'learned_federated_optimizer': copy.deepcopy(candidates[0]),
        'COLA_v1': copy.deepcopy(candidates[1]),
        'saved_unresolved_comparison': copy.deepcopy(comparison),
        'comparison_reference': ref(SOURCE / 'UNRESOLVED_COMPARISON.json'),
        'metadata_only_MGL_lead': copy.deepcopy(lead),
        'source_read_accounting_preserved': copy.deepcopy(accounting),
        'source_conclusions_preserved_without_rewrite': True,
        'new_direct_full_operator_matches': 0, 'new_supported_method_successors': 0,
        'new_arm_grid_gate_threshold_or_fixed_study_change': False,
        'allowed_statement': 'Carried moment/Ada optimizer accumulators and carried EMA graph teachers are scoped component ancestry. No direct active full-operator match, ready successor, predictive utility, global absence proof or novelty clearance. The single inherited history-alignment falsifier remains conceptual and unready.',
        'scope_limits': 'Two identities, two v1 method documents, eight selected source containers, zero source full-paper reads and zero integration searches/retrievals/primary reads. Title differences do not create identities; later-version or journal equivalence unverified. Selected-text hashes inherited from sealed declarations, not recomputed.'}

    encoded = compact(index) + b'\n'
    assert json.loads(encoded) == index and len(encoded) <= MAX_BYTES
    save('SIZE_LIMIT_CHECK.json', {'UTC': now, 'index_bytes': len(encoded), 'decimal_2MB_limit': MAX_BYTES,
        'within_limit': True, 'serialization': 'Full schema-compatible compact UTF-8 JSON, sorted object keys',
        'JSON_roundtrip_equal': True, 'measured_before_index_write': True, 'history_discarded': False,
        'new_raw_texts_embedded': False, 'predecessor_content_including_legacy_embedded_fields_preserved': True,
        'publisher_modified': False})
    with (HERE / 'LITERATURE_INDEX.json').open('xb') as stream:
        stream.write(encoded)
    save('DELTA.json', {'UTC': now, 'prospective': True,
        'predecessor_index': ref(PREV / 'LITERATURE_INDEX.json'), 'successor_index': ref(HERE / 'LITERATURE_INDEX.json'),
        'before': before, 'after': totals, 'metric_deltas': {key: totals[key] - before[key] for key in totals},
        'added_records': added, 'metadata_only_leads_added': [lead],
        'unresolved_metadata_leads_before': len(previous['unresolved_primary_metadata_leads']),
        'unresolved_metadata_leads_after': len(index['unresolved_primary_metadata_leads']),
        'previously_completed_bounded_method_papers_adopted': 2, 'previously_completed_version_documents': 2,
        'source_selected_method_containers': 8, 'version_followup_extra_paper_identity_credit': 0,
        'integration_searches': 0, 'integration_primary_reads': 0, 'integration_primary_rereads': 0,
        'integration_raw_primary_reads': 0, 'integration_selected_primary_scope_payload_reads': 0,
        'integration_retrievals': 0, 'new_full_paper_reads': 0, 'reused_primary_read_credit': 0,
        'source_claims_rewritten': False, 'canonical_status_or_ledger_modified': False})
    save('VERIFICATION.json', {'UTC': now, 'status': 'PASS_METADATA_ONLY_PROSPECTIVE_PENDING_ROOT_REVIEW',
        'predecessor_index_sha256': PREV_INDEX_PIN, 'predecessor_manifest_sha256': PREV_MANIFEST_PIN,
        'predecessor_seal_sha256': PREV_SEAL_PIN, 'source_manifest_sha256': SOURCE_MANIFEST_PIN,
        'source_seal_sha256': SOURCE_SEAL_PIN, 'predecessor_records_preserved': 247,
        'predecessor_record_prefix_canonical_bytes_sha256': hashlib.sha256(compact(previous['paper_records'])).hexdigest(),
        'all247_predecessor_record_canonical_bytes_equal': True,
        'predecessor_group_order_and_values_preserved': True, 'predecessor_catalog_prefix_preserved': True,
        'predecessor_unresolved_metadata_lead_prefix_preserved': True, 'all_other_predecessor_fields_preserved': True,
        'changed_current_metadata_snapshotted_exactly': snapshots,
        'history_source_events_and_decision_linkages_preserved': True,
        'safe_source_metadata_members_hashes_verified': len(SAFE_SOURCE_METADATA),
        'source_manifest_and_seal_authenticated': True,
        'source_selected_primary_text_payloads_not_reopened': True,
        'source_selected_text_hashes_not_recomputed': True,
        'saved_scope_locator_and_retrieval_metadata_authenticated': True,
        'two_new_normalized_paper_identity_groups': True, 'two_exact_arxiv_v1_documents': True,
        'primary_title_metadata_difference_creates_no_extra_identity': True,
        'unverified_cross_scheme_aliases_or_title_only_merges': 0,
        'MGL_metadata_only_no_record_group_or_credit': True,
        'group_indices_cover_all249_records': True,
        'new_scope_document_deduplication_keys': sorted(document_keys), 'added_records': added,
        'recomputed_metrics': totals, 'source_bindings_reference': ref(HERE / 'SOURCE_BINDINGS.json'),
        'index_bytes': len(encoded), 'within_decimal_2MB': True, 'full_JSON_roundtrip_equal': True,
        'new_raw_texts_not_embedded': True, 'prior_uncertified_cumulative_read_flags_preserved': True,
        'integration_searches': 0, 'integration_primary_reads': 0, 'integration_primary_rereads': 0,
        'integration_raw_primary_reads': 0, 'integration_retrievals': 0, 'new_full_paper_reads': 0,
        'no_numeric_predictive_novelty_execution_adoption': True,
        'no_new_arm_gate_or_fixed_study_change': True,
        'edited_only_new_index_v68_subtree': True, 'canonical_status_or_ledger_modified': False})
    notes = f'''# Prospective literature index v68

Prepared for independent root verification and adoption. This successor contains 249 conclusion records, 197 normalized paper groups and 2 software groups. All 247 v67 records, all 195 prior paper/2 software groups, their order and values, catalog prefixes, source events, decision linkages and historical fields are preserved. The six replaced current metadata fields and the complete pre-append unresolved-lead list are retained exactly in `integration_v68_predecessor_v67_snapshot`.

Only two authenticated saved conclusion rows from `persistent_optimizer_history_shared_graph_literature_20261005_v1` are appended: `arxiv:2312.02204v1` and `arxiv:2309.10376v1`. Exact versions map to one new normalized identity each. The first paper's primary v1 title differs from current metadata; this creates no extra identity and later-version equivalence is not assumed. COLA conclusions apply only to its read v1; no unverified journal or title-only merge is made. MGL (`doi:10.1145/3580305.3599428`) is appended solely to unresolved metadata leads after its source primary route returned403, with no paper record, group or reading credit.

The source author completed 2 bounded paper method scopes across 2 version documents and 8 selected containers, with 0 full-paper reads. This integration performs 0 searches, 0 retrievals, 0 new primary reads, 0 primary rereads, 0 raw primary reads and 0 full-paper reads. Nine safe saved metadata/conclusion/report members, source manifest/seal and all predecessor packet members were authenticated. The shared selected-primary scope file was never reopened or rehashed; its two selectors and hashes are inherited sealed-manifest declarations. Original deleted-body receipt hashes remain claims from the authenticated source metadata.

The learned optimizer explicitly carries moment/Ada accumulators and meta-learns decay coefficients with PES gradient estimates. COLA carries EMA teacher weights, detaches lookup/support embeddings, trains by direct contrastive loss and fits a test-support classifier. Their unchanged saved conclusions establish no direct full active-operator match, ready successor, predictive benefit, global absence proof or novelty clearance. The one already saved history-alignment question remains a conceptual falsifier, with coordinate/trajectory mismatch and generic Adam sensitivity preventing adoption. No arm, grid, gate, threshold or fixed30/39 change is adopted.

The complete JSON is {len(encoded):,} bytes, below the decimal 2,000,000-byte limit, with parsed roundtrip equality and exact canonical-byte preservation of the 247-record prefix. Prior uncertified cumulative-reading flags remain unchanged. All work is confined to this new index_v68 subtree; index67, canonical status/ledger, frozen studies, publisher and experimental systems are untouched. SOURCE_BINDINGS, DELTA, VERIFICATION and SIZE_LIMIT_CHECK document the exact append and its limits.
'''
    with (HERE / 'ROOT_ADOPTION_NOTES.md').open('x', encoding='utf-8') as stream:
        stream.write(notes)
    for row in predecessor_refs + source_refs:
        actual = ref(BASE / row['path'])
        assert (actual['bytes'], actual['sha256']) == (row['bytes'], row['sha256'])
    manifest_rows = [{'path': path.name, 'bytes': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                     for path in sorted(HERE.iterdir()) if path.is_file()]
    save('MANIFEST.json', {'UTC': now, 'prospective': True, 'files': manifest_rows})
    save('SEAL.json', {'manifest_sha256': ref(HERE / 'MANIFEST.json')['sha256'],
                      'payload_files': len(manifest_rows)})
    print(json.dumps({**totals, 'index_bytes': len(encoded),
        'index_sha256': ref(HERE / 'LITERATURE_INDEX.json')['sha256'],
        'manifest_sha256': ref(HERE / 'MANIFEST.json')['sha256'],
        'seal_sha256': ref(HERE / 'SEAL.json')['sha256'],
        'unresolved_metadata_leads': len(index['unresolved_primary_metadata_leads']),
        'status': 'PROSPECTIVE_READY_FOR_ROOT_REVIEW'}))


if __name__ == '__main__':
    main()
