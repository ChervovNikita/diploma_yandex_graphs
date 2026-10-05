"""Append two sealed literature packets using saved metadata only."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
PREV = BASE / 'literature_memory/index_v70'
ADOPTION = BASE / 'literature_memory/index_v70_root_adoption_20261005_v1/ROOT_ADOPTION.json'
A = BASE / 'cross_member_hidden_graph_error_aggregation_prior_scout_20261005_v1'
B = BASE / 'dense_graph_ensemble_error_correction_scout_20261005_v1'
PINS = {
    'index': '280bf60e10f6bf68d5b22f19e25d0ffdad10b9f155b125eba5b48e2b5a5cb73d',
    'manifest': 'a39240b367f7b9b7968e22c2e99151e4bfc0d76dd05ec698fda18b088def628b',
    'seal': '31cdf26cebf812447e5a97a215f113beaa8e372a4479cdfabc40e449ee155193',
    'adoption': '1b496f2f49c6fe7d6546083145a0c177037c673927106a42f4e68e2a45c031e5',
    'A_manifest': 'e5333199245e0ac13a317df54d667962b50ad0fda80a8fa343a6addcdeea8a9c',
    'A_seal': '36b8863f300382c3379f9f9d951b113be4dfddf8be904d4ebcaf7e9b2fe12fef',
    'B_manifest': 'e8c7f27e86bec1ed733927c2d4275b5b0283e65cdc45f83bbf0fb3cadff62f01',
    'B_seal': 'dac4b03b00598ff1939c7541364bc75dd2839dbdbce37d002703992cfd40a336'}
MAX_BYTES = 2_000_000


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def ref(path):
    path = Path(path)
    assert path.resolve().is_relative_to(BASE) and not path.is_symlink()
    assert not {'primary', 'sources', 'evidence', 'discovery'} & set(path.parts)
    assert path.suffix not in ('.pdf', '.png', '.html', '.xml', '.pt', '.npy', '.pkl', '.jsonl')
    assert path.name not in ('PRIMARY_SCOPES.json', 'PUBLIC_SOURCE_SCOPES.json', 'AUTHOR_REPOSITORY_README.md')
    assert not path.name.endswith('_PRIMARY_SCOPES.json')
    if path.is_relative_to(A) or path.is_relative_to(B):
        assert path.parent in (A, B) and path.name not in ('PARSE_PRIMARY.py', 'SHA256SUMS')
    raw = path.read_bytes()
    return {'path': str(path.relative_to(BASE)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def read(folder, name):
    ref(folder / name)
    return json.loads((folder / name).read_text())


def save(name, value):
    with (HERE / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def normalized(value):
    return {re.sub(r'v\d+$', '', s.strip().lower()) if s.strip().lower().startswith('arxiv:')
        else s.strip().lower() for s in value.split(';') if s.strip()}


def metrics(index):
    catalog = {r['path'] for r in index['existing_packets']}
    conclusions = {r['conclusion_file'] for r in index['paper_records'] if isinstance(r.get('conclusion_file'), str)}
    scopes = {r['read_scope_reference']['path'] for r in index['paper_records'] if 'read_scope_reference' in r}
    legacy = {r['read_scope_file_reference']['path'] for r in index['paper_records'] if 'read_scope_file_reference' in r}
    groups = index['canonical_identifier_normalization']['groups']
    return {'conclusion_records': len(index['paper_records']),
        'normalized_paper_identifiers': sum(g['kind'] == 'paper' for g in groups),
        'software_documentation_identifiers': sum(g['kind'] != 'paper' for g in groups),
        'catalog_entries': len(index['existing_packets']), 'unique_catalog_document_paths': len(catalog),
        'unique_conclusion_source_documents': len(conclusions), 'unique_referenced_document_paths': len(catalog | conclusions),
        'unique_scope_reference_document_paths': len(scopes),
        'unique_scope_reference_document_paths_including_legacy_field_alias': len(scopes | legacy)}


def declaration(folder, declared, name):
    row = declared[name]
    return {'path': str((folder / name).relative_to(BASE)), 'bytes': row['bytes'], 'sha256': row['sha256'],
        'binding_status': 'Inherited authenticated sealed-manifest declaration; payload not reopened or independently rehashed in integration',
        'payload_accessed_in_integration': False}


def authenticate():
    refs = [ref(PREV / 'MANIFEST.json'), ref(PREV / 'SEAL.json'), ref(ADOPTION)]
    assert [r['sha256'] for r in refs] == [PINS['manifest'], PINS['seal'], PINS['adoption']]
    manifest = read(PREV, 'MANIFEST.json')
    for row in manifest['files']:
        assert not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts
        got = ref(PREV / row['path'])
        assert (got['bytes'], got['sha256']) == (row['bytes'], row['sha256'])
        refs.append(got)
    assert read(PREV, 'SEAL.json')['manifest_sha256'] == PINS['manifest']
    adoption = json.loads(ADOPTION.read_text())
    assert adoption['accepted'] is True and adoption['index_sha256'] == PINS['index']
    assert (adoption['record_total'], adoption['paper_group_total'], adoption['software_groups']) == (254, 201, 2)
    assert ref(PREV / 'LITERATURE_INDEX.json')['sha256'] == PINS['index']
    packets = []
    for folder, key in ((A, 'A'), (B, 'B')):
        assert ref(folder / 'MANIFEST.json')['sha256'] == PINS[key + '_manifest']
        assert ref(folder / 'SEAL.json')['sha256'] == PINS[key + '_seal']
        declared = {r['path']: r for r in read(folder, 'MANIFEST.json')['files']}
        assert len(declared) == len(read(folder, 'MANIFEST.json')['files'])
        assert read(folder, 'SEAL.json')['manifest_sha256'] == PINS[key + '_manifest']
        safe = [name for name in declared if len(Path(name).parts) == 1 and name not in ('PARSE_PRIMARY.py', 'SHA256SUMS')]
        verified = [ref(folder / 'MANIFEST.json'), ref(folder / 'SEAL.json')]
        for name in safe:
            got = ref(folder / name)
            assert (got['bytes'], got['sha256']) == (declared[name]['bytes'], declared[name]['sha256'])
            verified.append(got)
        unopened = [declaration(folder, declared, name) for name in sorted(declared) if name not in safe]
        packets.append((declared, verified, unopened))
    return refs, packets


def main():
    assert {p.name for p in HERE.iterdir()} == {'BUILD_INDEX.py'}
    now = datetime.now(timezone.utc).isoformat()
    predecessor_refs, packets = authenticate()
    da, ra, ua = packets[0]
    db, rb, ub = packets[1]
    old = read(PREV, 'LITERATURE_INDEX.json')
    before = metrics(old)
    assert before == {k: old['read_accounting'][k] for k in before}
    ac, asc = read(A, 'PAPER_CONCLUSIONS.json'), read(A, 'READ_SCOPES.json')
    bc, bsc = read(B, 'PAPER_CONCLUSIONS.json'), read(B, 'READ_SCOPES.json')
    metadata = read(A, 'PRIMARY_METADATA.json')
    comparator = read(A, 'COMPARATOR_SPEC.json')
    reuse = read(B, 'REUSED_CONCLUSIONS.json')
    limits = read(B, 'PROTOCOL_LIMITS.json')
    ledger = read(B, 'QUERY_RETRIEVAL_LEDGER.json')
    assert ac['new_scoped_primary_method_reads'] == asc['new_primary_scoped_method_identities'] == 1
    assert asc['retained_targeted_primary_method_revisits'] == 2
    assert asc['retained_official_source_revisits'] == asc['new_author_source_local_scope_extensions'] == 1
    assert asc['new_author_source_identities'] == 0
    assert bc['new_scoped_primary_identities'] == bsc['new_primary_identities_with_scoped_reads'] == 3
    assert bsc['redundant_primary_revisits'] == 1
    assert ac['new_full_paper_reads'] == asc['new_full_paper_reads'] == bc['new_full_paper_reads'] == bsc['new_full_paper_reads'] == 0
    assert ac['implementation_or_training_performed'] is False and ac['numerical_results_adopted'] is False
    assert all(bc['limits'][k] == 0 for k in ('new_experiments', 'models_or_scientific_payloads_opened', 'compute_endpoints_contacted'))
    assert bc['limits']['global_novelty_clearance'] is False and bc['limits']['predictive_gain_established'] is False
    assert limits['scientific_payload_access_authorized_by_this_packet'] is False and limits['training_authorized_by_this_packet'] is False
    assert comparator['shortlist_count'] == 1
    assert reuse['index_sha256'] == old['integration_v70_predecessor_v69_snapshot']['index_reference']['sha256']
    reused_selectors = []
    for row in reuse['records']:
        matches = [n for n, saved in enumerate(old['paper_records']) if saved.get('canonical_id') == row['canonical_id']
            and saved.get('conclusion_file') == row['conclusion_file'] and saved.get('conclusion') == row['conclusion']]
        assert matches and row['binding_matches_index'] is True
        reused_selectors.append({'canonical_id': row['canonical_id'], 'index_record_indices': matches,
            'conclusion_file': row['conclusion_file'], 'conclusion_sha256': row['conclusion_sha256'], 'reuse_kind': row['reuse_kind']})
    assert len(reused_selectors) == 23
    ffl = ac['conclusions'][0]
    assert ffl['id'] == 'FFL' and ffl['canonical_id'] == asc['new_primary'][0]['canonical_id'] == metadata['canonical_id']
    new_scopes = [(A, da, ffl, asc['new_primary'][0], 'conclusions/0', 'new_primary/0',
        ['primary/1904.09058v2.pdf', 'primary/1904.09058v2.txt', 'evidence/ffl_scoped_passages.txt', 'evidence/ffl-02.png', 'evidence/ffl-03.png'])]
    for pos, row in enumerate(bc['papers']):
        scope = bsc['scopes'][pos]
        assert row['canonical_id'] == scope['canonical_id'] and row['key'] == scope['key']
        names = [str(Path(scope[field]).relative_to(B.name)) for field in ('source_file', 'blocks_file', 'passages_file')]
        for name, hash_field in zip(names, ('source_sha256', 'blocks_sha256', 'passages_sha256')):
            assert db[name]['sha256'] == scope[hash_field]
        new_scopes.append((B, db, row, scope, f'papers/{pos}', f'scopes/{pos}', names))
    assert da['primary/1904.09058v2.pdf']['sha256'] == asc['new_primary'][0]['primary_sha256']
    assert [next(iter(normalized(row[2]['canonical_id']))) for row in new_scopes] == [
        'arxiv:1904.09058', 'arxiv:2405.03401', 'arxiv:2411.13700', 'arxiv:2605.20271']
    catalog_payloads = [declaration(folder, declared, name) for folder, declared, _, _, _, _, names in new_scopes for name in names]
    cs = bsc['scopes'][3]
    assert cs['new_identity_count'] == cs['new_unique_method_scope_count'] == 0
    for field in ('source_file', 'blocks_file', 'passages_file'):
        catalog_payloads.append(declaration(B, db, str(Path(cs[field]).relative_to(B.name))))
    assert len(catalog_payloads) == 17
    save('SOURCE_BINDINGS.json', {'UTC': now, 'predecessor_files': predecessor_refs,
        'adopted_v70_root_adoption_reference': ref(ADOPTION), 'source_files': ra + rb,
        'unopened_source_manifest_declarations': ua + ub,
        'all_safe_packet_metadata_members_authenticated': True,
        'primary_extracted_passage_visual_discovery_or_author_source_payloads_opened_or_rehashed': False,
        'prior_external_bindings_not_dereferenced': True,
        'new_saved_primary_scopes_adopted': 4, 'new_unique_identity_groups': 4,
        'source_targeted_retained_primary_revisits': 2, 'source_redundant_primary_revisits': 1,
        'source_official_author_source_revisits': 1, 'source_new_local_author_source_scope_extensions': 1,
        'source_new_author_source_identities': 0, 'source_full_paper_reads': 0,
        'integration_searches': 0, 'integration_retrievals': 0, 'integration_primary_reads': 0,
        'integration_author_source_reads': 0, 'integration_target_payload_accesses': 0,
        'integration_scientific_execution': False, 'only_new_index_v71_subtree_written': True})
    new = copy.deepcopy(old)
    current = ['schema', 'created_UTC', 'latest_adoption', 'read_accounting', 'predecessor_index', 'predecessor_index_sha256']
    new['integration_v71_predecessor_v70_snapshot'] = {
        **{k: copy.deepcopy(old[k]) for k in current + ['unresolved_primary_metadata_leads']},
        'index_reference': ref(PREV / 'LITERATURE_INDEX.json'), 'manifest_reference': ref(PREV / 'MANIFEST.json'),
        'seal_reference': ref(PREV / 'SEAL.json'), 'root_adoption_reference': ref(ADOPTION)}
    new.update(schema='literature-memory-index-v71', created_UTC=now,
        predecessor_index='literature_memory/index_v70/LITERATURE_INDEX.json', predecessor_index_sha256=PINS['index'])
    added = []
    groups = new['canonical_identifier_normalization']['groups']
    for folder, declared, row, scope, selector, scope_selector, names in new_scopes:
        identity = next(iter(normalized(row['canonical_id'])))
        alias = ['doi:' + metadata['doi'].lower()] if folder == A else []
        for wanted in [identity] + alias:
            assert all(wanted not in normalized(value) for group in groups for value in
                [group['normalized_identifier']] + group.get('raw_canonical_identifiers', []) + group.get('explicit_aliases', []))
        payloads = [declaration(folder, declared, name) for name in names]
        dedup = hashlib.sha256(compact({'identity': identity, 'version': row['canonical_id'], 'saved_scope': scope, 'payload_declarations': payloads})).hexdigest()
        position = len(new['paper_records'])
        new['paper_records'].append({'canonical_id': row['canonical_id'], 'normalized_identifier': identity,
            'conclusion': copy.deepcopy(row), 'conclusion_file': str((folder / 'PAPER_CONCLUSIONS.json').relative_to(BASE)),
            'conclusion_file_sha256': ref(folder / 'PAPER_CONCLUSIONS.json')['sha256'], 'conclusion_source_selector': selector,
            'read_scope_reference': {**ref(folder / 'READ_SCOPES.json'), 'selector': scope_selector},
            'exact_read_scope': copy.deepcopy(scope), 'unopened_payload_declarations': payloads,
            'source_packet': folder.name, 'source_packet_binding_reference': ref(HERE / 'SOURCE_BINDINGS.json'),
            'source_packet_manifest_reference': ref(folder / 'MANIFEST.json'), 'source_packet_seal_reference': ref(folder / 'SEAL.json'),
            'scope_deduplication_key_sha256': dedup, 'scoped_method_read': True, 'full_paper_read': False,
            'author_source_read': False, 'integration_primary_read': False, 'new_identity': True,
            'numeric_results_adopted': False, 'global_novelty_clearance': False, 'execution_authorized': False})
        groups.append({'normalized_identifier': identity, 'kind': 'paper', 'raw_canonical_identifiers': [identity, row['canonical_id']],
            'explicit_aliases': alias, 'record_indices': [position]})
        added.append({'canonical_id': identity, 'exact_primary_id': row['canonical_id'], 'record_index': position,
            'scope_deduplication_key_sha256': dedup, 'new_identity': True, 'full_paper_read': False})
    for row in ra + rb + catalog_payloads:
        assert all(old_row['path'] != row['path'] for old_row in new['existing_packets'])
        new['existing_packets'].append({**row, 'kind': 'saved_metadata' if row in ra + rb else 'sealed_unopened_payload_declaration',
            'source_bindings_reference': ref(HERE / 'SOURCE_BINDINGS.json')})
    new['saved_hidden_fusion_and_graph_error_scope_events_v71'] = {
        'cross_member_conclusions_preserved': ac, 'cross_member_read_accounting_and_all_reuse_scopes_preserved': asc,
        'cross_member_comparator_spec_reference': ref(A / 'COMPARATOR_SPEC.json'),
        'cross_member_pinned_FFL_identity_metadata': metadata,
        'dense_scout_conclusions_preserved': bc, 'dense_scout_read_accounting_and_all_reuse_scopes_preserved': bsc,
        'dense_protocol_limits_preserved': limits, 'dense_reused_conclusions_reference': ref(B / 'REUSED_CONCLUSIONS.json'),
        'dense_all23_exact_prior_record_reuse_selectors': reused_selectors,
        'dense_stale_DIVE_limit_and_initial_CS_lookup_failure_preserved': {k: reuse[k] for k in ('stale_limit_warning', 'lookup_limitation')},
        'all_reused_or_redundant_events_add_zero_identity_or_unique_method_scope_credit': True,
        'official_CS_local_source_extension_adds_zero_new_author_identity': True,
        'cross_stitch_and_sluice_saved_local_scopes_not_new_direct_index_records_or_reads': True,
        'incidental_public_result_exposures_not_adopted': True,
        'source_public_retrieval_accounting': {'cross_member_retained_receipts': len(read(A, 'RETRIEVAL.json')),
            'dense_ledger_attempts': len(ledger['entries']), 'dense_ledger_HTTP_errors': sum('error' in r for r in ledger['entries']),
            'all_historical_source_task_events_not_integration_contacts': True},
        'new_full_paper_reads': 0, 'numeric_results_adopted': False, 'scientific_execution': False,
        'global_novelty_clearance': False, 'new_ready_successor': False, 'execution_authority_granted': False}
    lead_id = 'doi:10.1016/j.inffus.2024.102461'
    assert all(r['canonical_metadata_identifier'] != lead_id for r in old['unresolved_primary_metadata_leads'])
    failed_pos = next(n for n, r in enumerate(ledger['entries']) if r['key'] == 'graph_ensemble_publisher')
    new['unresolved_primary_metadata_leads'].append({'canonical_metadata_identifier': lead_id,
        'title': 'Graph ensemble neural network', 'publication_metadata': 'Crossref metadata reports October 2024',
        'status': 'Metadata only; publisher retrieval returned HTTP 403; no abstract or primary method read',
        'source_report_reference': ref(B / 'REPORT.md'),
        'retrieval_reference': {**ref(B / 'QUERY_RETRIEVAL_LEDGER.json'), 'selector': f'entries/{failed_pos}'},
        'primary_method_read': False, 'full_paper_read': False, 'paper_record_added': False,
        'normalized_paper_group_added': False, 'new_identity_reading_credit': 0})
    new['unresolved_title_only_metadata_locators_v71'] = [{'title_locator': title,
        'source_report_reference': ref(B / 'REPORT.md'), 'status': 'Saved abstract/metadata lead; method unresolved',
        'canonical_identifier_invented': False, 'primary_method_read': False, 'paper_record_or_group_added': False}
        for title in ('GETS', 'Adapt/Agree/Aggregate')]
    assert new['paper_records'][:254] == old['paper_records']
    assert compact(new['paper_records'][:254]) == compact(old['paper_records'])
    assert groups[:len(old['canonical_identifier_normalization']['groups'])] == old['canonical_identifier_normalization']['groups']
    assert {k: v for k, v in new['canonical_identifier_normalization'].items() if k != 'groups'} == {
        k: v for k, v in old['canonical_identifier_normalization'].items() if k != 'groups'}
    assert new['existing_packets'][:len(old['existing_packets'])] == old['existing_packets']
    assert new['unresolved_primary_metadata_leads'][:3] == old['unresolved_primary_metadata_leads']
    for key in old:
        if key not in current + ['paper_records', 'existing_packets', 'canonical_identifier_normalization', 'unresolved_primary_metadata_leads']:
            assert new[key] == old[key], key
    assert len(groups) == len({g['normalized_identifier'] for g in groups})
    covered = [n for g in groups for n in g['record_indices']]
    assert len(covered) == len(set(covered)) == 258 and set(covered) == set(range(258))
    after = metrics(new)
    assert (after['conclusion_records'], after['normalized_paper_identifiers'], after['software_documentation_identifiers']) == (258, 205, 2)
    # Replace current per-packet counters as a whole; their complete prior values
    # remain in the exact predecessor read_accounting snapshot above.
    account = {k: copy.deepcopy(v) for k, v in old['read_accounting'].items() if not k.startswith('latest_packet_')}
    account.update(after, state='PROSPECTIVE_SAVED_HIDDEN_FUSION_GRAPH_ERROR_ADOPTION_PENDING_ROOT_REVIEW',
        latest_packet_new_scoped_primary_reads=4, latest_packet_scoped_primary_method_events=4,
        latest_packet_previously_completed_scoped_read_adoptions=4,
        latest_packet_new_paper_identity_groups=4, latest_packet_source_primary_version_documents=4,
        latest_packet_existing_identity_version_followup_scopes=0, latest_packet_version_followup_additional_paper_identities=0,
        latest_packet_full_primary_reads=0, latest_packet_targeted_retained_primary_revisits=2,
        latest_packet_redundant_primary_revisits_zero_unique_scope_credit=1,
        latest_packet_bounded_author_source_repositories=0, latest_packet_author_code_semantic_file_scopes=0,
        latest_packet_bounded_author_source_scope_events=0, latest_packet_author_source_revisits=1,
        latest_packet_local_author_source_scope_extensions=1, latest_packet_new_author_source_identities=0,
        latest_packet_source_repository_commits=0, latest_packet_source_code_line_ranges=0,
        latest_packet_nonreading_amendment_events=0, latest_packet_nonreading_amendment_read_credit=0,
        integration_pass_searches=0, integration_pass_retrievals=0, integration_pass_new_primary_reads=0,
        integration_pass_full_primary_reads=0, integration_pass_primary_method_reads=0,
        integration_pass_author_source_semantic_reads=0, integration_pass_experimental_score_artifact_reads=0,
        integration_pass_raw_primary_reads=0, integration_pass_selected_primary_text_scope_reads=0,
        historical_path_catalog_note='All254 prior records, all201 paper/2 software groups and all prior history preserved exactly. Four saved scoped identities appended; targeted/redundant PCL/C&S and official-source reuse events add zero identity credit. Existing three unresolved leads preserved; one new DOI metadata-only lead and two title-only unresolved locators recorded. No payload reopening/retrieval/scientific execution; current metadata and leads snapshotted.')
    new['read_accounting'] = account
    new['latest_adoption'] = {'UTC': now, 'status': 'PROSPECTIVE_PENDING_ROOT_REVIEW', 'predecessor': ref(PREV / 'LITERATURE_INDEX.json'),
        'root_adopted_predecessor_reference': ref(ADOPTION), 'added_records': added,
        'source_completed_new_scoped_identities': 4, 'source_targeted_primary_revisits': 2,
        'source_redundant_primary_revisits': 1, 'source_author_source_revisit_and_local_extension': 1,
        'new_author_source_identity_credit': 0, 'new_full_paper_reads': 0,
        'added_metadata_only_DOI_leads': [lead_id], 'added_title_only_unresolved_locators': ['GETS', 'Adapt/Agree/Aggregate'],
        'integration_primary_or_author_source_reads': 0, 'integration_retrievals': 0,
        'numeric_predictive_novelty_ready_successor_or_execution_adoption': False,
        'source_bindings_reference': ref(HERE / 'SOURCE_BINDINGS.json')}
    encoded = compact(new) + b'\n'
    assert len(encoded) < MAX_BYTES and json.loads(encoded) == new
    save('SIZE_LIMIT_CHECK.json', {'index_bytes': len(encoded), 'decimal_2MB_limit': MAX_BYTES,
        'within_limit': True, 'JSON_roundtrip_equal': True, 'lossy_removal_or_history_rewrite': False,
        'raw_new_primary_or_selected_text_embedded': False, 'serialization': 'Compact complete UTF-8 JSON; sorted keys'})
    with (HERE / 'LITERATURE_INDEX.json').open('xb') as stream:
        stream.write(encoded)
    save('DELTA.json', {'UTC': now, 'prospective': True, 'before': before, 'after': after,
        'metric_deltas': {k: after[k] - before[k] for k in after}, 'added_records': added,
        'unresolved_DOI_metadata_leads_before_after': [3, 4], 'title_only_metadata_locators_added': 2,
        'new_unique_method_scope_identity_adoptions': 4, 'full_paper_reads': 0,
        'source_targeted_primary_revisits': 2, 'source_redundant_primary_revisits': 1,
        'source_official_author_source_revisit': 1, 'source_local_author_source_extension': 1,
        'integration_primary_reads_or_retrievals': 0, 'canonical_status_ledger_or_prior_index_edits': False})
    save('VERIFICATION.json', {'UTC': now, 'status': 'PASS_METADATA_ONLY_PROSPECTIVE_PENDING_ROOT_REVIEW',
        'pins': PINS, 'all254_prior_records_canonical_bytes_equal': True,
        'all201_paper_and2_software_groups_exactly_preserved': True,
        'all_prior_normalization_rules_aliases_catalog_prefix_events_and_decisions_preserved': True,
        'all3_prior_unresolved_metadata_leads_exactly_preserved': True,
        'all23_dense_reuse_records_equal_prior_records': True,
        'all_changed_current_metadata_and_prior_leads_snapshotted_exactly': True,
        'four_new_scoped_identities_only': True, 'zero_new_identity_credit_for_all_reuse_revisit_events': True,
        'FFL_explicit_DOI_alias_from_saved_primary_identity_metadata_only': True,
        'Cross_stitch_Sluice_GETS_Adapt_Agree_Aggregate_not_new_paper_records': True,
        'new_DOI_lead_metadata_only_no_method_record_or_group': True,
        'source_payloads_not_reopened_or_rehashed': True, 'integration_primary_reads_or_retrievals': 0,
        'new_numerical_result_global_novelty_ready_successor_or_execution_claim': False,
        'index_bytes': len(encoded), 'within_decimal_2MB': True, 'JSON_roundtrip_equal': True,
        'lossy_history_removal': False, 'recomputed_metrics': after, 'only_new_index_v71_subtree_written': True})
    notes = f'''# Prospective literature index v71

Prepared from root-adopted index70. All 254 prior records, 201 paper groups, 2 software groups, normalization rules, aliases, catalog prefixes, historical decisions, MGL source event and protocol amendment remain exactly preserved. The six replaced current metadata fields and prior unresolved leads are snapshotted.

Append four saved scopes: FFL v2 (one complete-method bounded scope), E2GNN v1, collaborative CTR fusion v1 and multi-head Nadaraya-Watson v1 (restricted scalar positive-simplex setup and covariance algebra only). Totals: 258 records, 205 paper groups and 2 software groups. FFL's explicit ICPR DOI association comes from sealed source metadata; no title-only merge is inferred. Zero new full-paper or numerical-result credit.

Separately preserve source-task accounting: two targeted retained PCL/C&S primary revisits; one retained pinned C&S official-source revisit and local propagation-function extension; one redundant C&S HTML retrieval/reinspection; 23 scout indexed conclusion reuses. All add zero new identity/unique method-scope credit. Cross-stitch and sluice reuse saved local scopes without adding direct index records. Neither source's public result exposure nor runtime/source equivalence is adopted.

Existing three metadata leads remain intact. Append metadata-only Graph ensemble neural network DOI10.1016/j.inffus.2024.102461, whose publisher retrieval returned HTTP403, and retain GETS and Adapt/Agree/Aggregate as unresolved title locators without invented identifiers or method credit. Their failures are not absence evidence.

Both literature suggestions remain attributed, unconfirmed utility questions. Frozen hidden-state nonlinear fusion is established ancestry; its capable same-evidence single description is not automatic rejection. Graph-local error cross-moments are reconstructible from local diagonal competence and all pairwise disagreement fields. Same supervision/context/control opportunities, whole evaluation-fold residual/C&S masking, retrospective VALID dependence and all executed-route costs remain explicit. No node-to-link recipe, novelty, gain, ready successor, frozen-study change or execution authority is established. The existing development-reuse amendment remains authoritative.

Only safe saved packet metadata/conclusions and their manifests/seals were authenticated. Primary, extracted, selected-passage, visual, discovery and author-code payloads were not opened or rehashed; external prior bindings were not dereferenced. Zero integration retrievals, target payload accesses or scientific executions. Only this fresh subtree was written.

Full JSON: {len(encoded):,} bytes, below the decimal 2,000,000-byte cap, with parsed roundtrip equality and no lossy removal/history rewrite. Root owns independent adoption.
'''
    with (HERE / 'ROOT_ADOPTION_NOTES.md').open('x') as stream:
        stream.write(notes)
    for row in predecessor_refs + ra + rb:
        assert ref(BASE / row['path']) == row
    files = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(HERE.iterdir()) if p.is_file()]
    save('MANIFEST.json', {'UTC': now, 'prospective': True, 'files': files})
    save('SEAL.json', {'manifest_sha256': ref(HERE / 'MANIFEST.json')['sha256'], 'payload_files': len(files),
        'saved_new_scoped_identities': 4, 'source_targeted_primary_revisits': 2, 'source_redundant_primary_revisits': 1,
        'source_author_source_revisit_and_local_extension': 1, 'new_full_paper_reads': 0,
        'integration_primary_author_source_reads_or_retrievals': 0, 'source_payloads_reopened_or_rehashed': False,
        'scientific_execution_or_authority': False, 'canonical_status_ledger_or_prior_index_edits': False})
    print(json.dumps({**after, 'index_bytes': len(encoded),
        'index_sha256': ref(HERE / 'LITERATURE_INDEX.json')['sha256'], 'manifest_sha256': ref(HERE / 'MANIFEST.json')['sha256'],
        'seal_sha256': ref(HERE / 'SEAL.json')['sha256'], 'status': 'PROSPECTIVE_READY_FOR_ROOT_REVIEW'}))


if __name__ == '__main__':
    main()
