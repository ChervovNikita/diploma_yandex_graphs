"""Lossless append of sealed GETS/A3-GCN scopes; never reopen source payloads."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
PREV = BASE / 'literature_memory/index_v71'
ADOPTION = BASE / 'literature_memory/index_v71_root_adoption_20261005_v1/ROOT_ADOPTION.json'
SOURCE = BASE / 'dense_graph_ensemble_unread_primary_closure_20261005_v1'
PINS = {
    'index': '107eac912ddf95d8e5e50d78903a005c40c57d8e61691571c3c760b250933b49',
    'manifest': 'd18507d89a68bd957857f7484cf875310f39192c377628660b79e4ef4ae1035c',
    'seal': '520f2d34c07cb15470c6713f30eb75f8783b3f9c7aab9893261bda5e352535b8',
    'adoption': '9682c866995be1032bef3d36a945f9fee9931c398ab2ba8aaabcebfaa79a7f6c',
    'source_manifest': 'd3e827ea417d05dcd69b1b42f04005c16f6506a5ba1ee5514938822462bf0936',
    'source_seal': '9b6a9d73ee86849c9e0c39f764643548baacd7e7a8642427f47625f171899a06'}
COMMIT = '4403410fbf730eae9c5d2018d8554a6b645ba50f'
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
    if path.is_relative_to(SOURCE):
        assert path.parent == SOURCE and path.name != 'PARSE_PRIMARY.py'
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


def declaration(declared, path):
    name = str(Path(path).relative_to(SOURCE.name)) if str(path).startswith(SOURCE.name + '/') else str(path)
    row = declared[name]
    return {'path': str((SOURCE / name).relative_to(BASE)), 'bytes': row['bytes'], 'sha256': row['sha256'],
        'binding_status': 'Inherited authenticated sealed-manifest declaration; payload not reopened or independently rehashed in integration',
        'payload_accessed_in_integration': False}


def authenticate():
    refs = [ref(PREV / 'MANIFEST.json'), ref(PREV / 'SEAL.json'), ref(ADOPTION)]
    assert [r['sha256'] for r in refs] == [PINS['manifest'], PINS['seal'], PINS['adoption']]
    for row in read(PREV, 'MANIFEST.json')['files']:
        assert not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts
        got = ref(PREV / row['path'])
        assert (got['bytes'], got['sha256']) == (row['bytes'], row['sha256'])
        refs.append(got)
    assert read(PREV, 'SEAL.json')['manifest_sha256'] == PINS['manifest']
    adoption = json.loads(ADOPTION.read_text())
    assert adoption['root_adopted'] is True and adoption['index_sha256'] == PINS['index']
    assert (adoption['record_total'], adoption['paper_groups'], adoption['software_groups']) == (258, 205, 2)
    assert ref(PREV / 'LITERATURE_INDEX.json')['sha256'] == PINS['index']
    assert ref(SOURCE / 'MANIFEST.json')['sha256'] == PINS['source_manifest']
    assert ref(SOURCE / 'SEAL.json')['sha256'] == PINS['source_seal']
    manifest = read(SOURCE, 'MANIFEST.json')
    declared = {r['path']: r for r in manifest['files']}
    assert len(declared) == len(manifest['files']) == 54
    assert all(not Path(name).is_absolute() and '..' not in Path(name).parts for name in declared)
    seal = read(SOURCE, 'SEAL.json')
    assert seal['manifest_sha256'] == PINS['source_manifest'] and seal['file_count'] == 54
    assert seal['new_method_work_identities'] == 2 and seal['old_primary_revisits'] == seal['full_paper_reads'] == 0
    assert seal['scientific_execution'] is False and seal['predictive_adoption'] is False and seal['experimental_admission'] is False
    safe = [name for name in declared if len(Path(name).parts) == 1 and name != 'PARSE_PRIMARY.py']
    source_refs = [ref(SOURCE / 'MANIFEST.json'), ref(SOURCE / 'SEAL.json')]
    for name in safe:
        got = ref(SOURCE / name)
        assert (got['bytes'], got['sha256']) == (declared[name]['bytes'], declared[name]['sha256'])
        source_refs.append(got)
    assert len(safe) == 16
    unopened = [declaration(declared, name) for name in sorted(declared) if name not in safe]
    return refs, declared, source_refs, unopened


def main():
    assert {p.name for p in HERE.iterdir()} == {'BUILD_INDEX.py'}
    now = datetime.now(timezone.utc).isoformat()
    predecessor_refs, declared, source_refs, unopened = authenticate()
    old = read(PREV, 'LITERATURE_INDEX.json')
    before = metrics(old)
    assert before == {k: old['read_accounting'][k] for k in before}
    conclusion, scopes = read(SOURCE, 'PAPER_CONCLUSIONS.json'), read(SOURCE, 'READ_SCOPES.json')
    source_binding = read(SOURCE, 'SOURCE_BINDINGS.json')
    source_check = read(SOURCE, 'VERIFICATION.json')
    old_scout = read(SOURCE, 'OLD_SCOUT_PRESERVED.json')
    assert old_scout['all_match'] is True and old_scout['old_primary_scopes_reopened'] is False
    assert source_binding['author_source_commit'] == scopes['author_source']['commit'] == COMMIT
    assert scopes['new_work_identities_with_primary_method_reads'] == conclusion['new_method_work_identities'] == 2
    assert scopes['linked_published_metadata_upgrades'] == scopes['unresolved_metadata_only_work_identities'] == 1
    assert scopes['new_full_paper_reads'] == scopes['old_indexed_primary_scopes_reopened_in_followup'] == conclusion['old_primary_revisits'] == 0
    assert conclusion['new_full_paper_reads'] == 0 and conclusion['predictive_gain_established'] is False
    assert conclusion['global_novelty_clearance'] is False and conclusion['experimental_admission'] is False
    assert source_check['source_executed'] is False and source_check['scientific_payload_access'] is False
    assert source_check['GETS_class_vector_scale_discrepancy_recorded'] is True
    assert source_check['A3_high_confidence_set_vs_class_agreement_distinction_recorded'] is True
    assert source_check['GENNN_unread_status_preserved'] is True
    assert [p['canonical_id'] for p in conclusion['papers']] == ['arXiv:2410.09570v2', 'arXiv:2503.17842v1']
    assert [s['key'] for s in scopes['scopes']] == ['gets', 'aaa_preprint', 'aaa_published_preview', 'genn_locator']
    payloads_by_scope = []
    for pos, scope in enumerate(scopes['scopes']):
        payloads = []
        for path_field, hash_field in (('source_file', 'source_sha256'), ('blocks_file', 'blocks_sha256'),
                ('passages_file', 'passages_sha256'), ('algorithm_file', 'algorithm_sha256')):
            if path_field in scope:
                d = declaration(declared, scope[path_field])
                assert d['sha256'] == scope[hash_field]
                payloads.append(d)
        payloads_by_scope.append(payloads)
        assert scope['full_paper_read'] is False
        assert scope['complete_relevant_paper_method_read'] == (pos < 2)
    author = copy.deepcopy(scopes['author_source'])
    assert len(author['read_files']) == 4 and author['executed'] is False
    author_payloads = []
    for row in author['read_files']:
        for field, hash_field in (('path', 'sha256'), ('passages_path', 'passages_sha256')):
            d = declaration(declared, row[field])
            assert d['sha256'] == row[hash_field]
            author_payloads.append(d)
    semantic_files = [r for r in author['read_files'] if r['repo_path'] != 'README.md']
    assert len(semantic_files) == 3
    semantic_ranges = sum(len(r['line_ranges']) for r in semantic_files)
    semantic_lines = sum(hi - lo + 1 for r in semantic_files for lo, hi in r['line_ranges'])
    assert (semantic_ranges, semantic_lines) == (5, 540)
    source_primary_bindings = source_binding['paper_scopes']
    for scope, saved in zip(scopes['scopes'], source_primary_bindings):
        assert scope['canonical_id'] == saved['canonical_id'] and scope['source_sha256'] == saved['source_sha256']
    payload_catalog = [r for rows in payloads_by_scope for r in rows] + author_payloads
    assert len(payload_catalog) == 19 and len({r['path'] for r in payload_catalog}) == 19
    save('SOURCE_BINDINGS.json', {'UTC': now, 'predecessor_files': predecessor_refs,
        'adopted_v71_root_adoption_reference': ref(ADOPTION), 'source_files': source_refs,
        'unopened_source_manifest_declarations': unopened,
        'safe_source_metadata_members_authenticated': 16,
        'new_saved_primary_method_work_identities': 2, 'saved_bounded_author_repository_events': 1,
        'saved_author_semantic_files': 3, 'saved_author_semantic_line_ranges': 5,
        'saved_author_semantic_lines': 540, 'saved_README_identity_files': 1,
        'pinned_author_commit': COMMIT, 'linked_published_metadata_upgrades': 1,
        'source_old_primary_revisits': 0, 'source_full_paper_reads': 0,
        'primary_author_code_extracted_passage_discovery_or_visual_payloads_reopened_or_rehashed': False,
        'old_scout_payload_bindings_not_dereferenced': True,
        'integration_searches': 0, 'integration_retrievals': 0, 'integration_primary_reads': 0,
        'integration_author_source_reads': 0, 'integration_target_payload_accesses': 0,
        'integration_scientific_execution': False, 'only_new_index_v72_subtree_written': True})
    new = copy.deepcopy(old)
    current = ['schema', 'created_UTC', 'latest_adoption', 'read_accounting', 'predecessor_index', 'predecessor_index_sha256']
    new['integration_v72_predecessor_v71_snapshot'] = {
        **{k: copy.deepcopy(old[k]) for k in current}, 'index_reference': ref(PREV / 'LITERATURE_INDEX.json'),
        'manifest_reference': ref(PREV / 'MANIFEST.json'), 'seal_reference': ref(PREV / 'SEAL.json'),
        'root_adoption_reference': ref(ADOPTION)}
    new.update(schema='literature-memory-index-v72', created_UTC=now,
        predecessor_index='literature_memory/index_v71/LITERATURE_INDEX.json', predecessor_index_sha256=PINS['index'])
    added = []
    groups = new['canonical_identifier_normalization']['groups']
    for pos, paper in enumerate(conclusion['papers']):
        identity = next(iter(normalized(paper['canonical_id'])))
        aliases = ['doi:10.1007/s13042-026-03047-y'] if pos == 1 else []
        for wanted in [identity] + aliases:
            assert all(wanted not in normalized(value) for group in groups for value in
                [group['normalized_identifier']] + group.get('raw_canonical_identifiers', []) + group.get('explicit_aliases', []))
        scope = scopes['scopes'][pos]
        assert scope['canonical_id'] == paper['canonical_id']
        position = len(new['paper_records'])
        key = hashlib.sha256(compact({'identity': identity, 'version': paper['canonical_id'],
            'saved_scope': scope, 'unopened_payload_declarations': payloads_by_scope[pos]})).hexdigest()
        new['paper_records'].append({'canonical_id': paper['canonical_id'], 'normalized_identifier': identity,
            'conclusion': copy.deepcopy(paper), 'conclusion_file': str((SOURCE / 'PAPER_CONCLUSIONS.json').relative_to(BASE)),
            'conclusion_file_sha256': ref(SOURCE / 'PAPER_CONCLUSIONS.json')['sha256'], 'conclusion_source_selector': f'papers/{pos}',
            'read_scope_reference': {**ref(SOURCE / 'READ_SCOPES.json'), 'selector': f'scopes/{pos}'},
            'exact_read_scope': copy.deepcopy(scope), 'unopened_payload_declarations': copy.deepcopy(payloads_by_scope[pos]),
            'source_packet': SOURCE.name, 'source_packet_binding_reference': ref(HERE / 'SOURCE_BINDINGS.json'),
            'source_packet_manifest_reference': ref(SOURCE / 'MANIFEST.json'), 'source_packet_seal_reference': ref(SOURCE / 'SEAL.json'),
            'scope_deduplication_key_sha256': key, 'scoped_method_read': True, 'full_paper_read': False,
            'saved_bounded_author_source_scope_adopted': pos == 0, 'integration_primary_read': False,
            'integration_author_source_read': False, 'new_identity': True,
            'numeric_results_adopted': False, 'global_novelty_clearance': False, 'execution_authorized': False})
        groups.append({'normalized_identifier': identity, 'kind': 'paper', 'raw_canonical_identifiers': [identity, paper['canonical_id']],
            'explicit_aliases': aliases, 'record_indices': [position]})
        added.append({'canonical_id': identity, 'exact_primary_id': paper['canonical_id'], 'record_index': position,
            'new_identity': True, 'scope_deduplication_key_sha256': key, 'full_paper_read': False})
    for row in source_refs + payload_catalog:
        assert all(r['path'] != row['path'] for r in new['existing_packets'])
        new['existing_packets'].append({**row, 'kind': 'saved_metadata' if row in source_refs else 'sealed_unopened_payload_declaration',
            'source_bindings_reference': ref(HERE / 'SOURCE_BINDINGS.json')})
    new['GETS_pinned_author_source_scope_event_v72'] = {
        'event_kind': 'Previously completed bounded author repository implementation scope attached to new GETS paper identity',
        'paper_record_index': 258, 'paper_normalized_identifier': 'arxiv:2410.09570',
        'saved_author_source_scope_preserved': author, 'unopened_source_and_passage_declarations': author_payloads,
        'semantic_files': 3, 'semantic_line_ranges': semantic_ranges, 'semantic_lines': semantic_lines,
        'README_identity_files': 1, 'source_conclusion_reference': {**ref(SOURCE / 'PAPER_CONCLUSIONS.json'), 'selector': 'papers/0'},
        'paper_source_discrepancies_preserved': copy.deepcopy(conclusion['papers'][0]['source_prose_discrepancies']),
        'source_equations_preserved': conclusion['papers'][0]['exact_source_equations'],
        'official_accepted_version_method_equivalence_certified': False,
        'source_accuracy_preservation_or_conditional_execution_certified': False,
        'author_source_execution_or_integration_reread': False, 'new_additional_paper_identity_credit': 0}
    new['dense_graph_unread_primary_closure_scope_events_v72'] = {
        'source_paper_conclusions_reference': ref(SOURCE / 'PAPER_CONCLUSIONS.json'),
        'source_READ_SCOPES_reference': ref(SOURCE / 'READ_SCOPES.json'),
        'source_verification_preserved': source_check,
        'new_primary_method_work_identities': 2, 'linked_published_metadata_upgrades': 1,
        'source_old_primary_revisits': 0, 'source_full_paper_reads': 0,
        'old_scout_preservation_claim_and_bindings_preserved': old_scout,
        'old_scout_or_primary_bindings_dereferenced_in_integration': False,
        'A3_linked_published_preview_scope_preserved': copy.deepcopy(scopes['scopes'][2]),
        'A3_linked_published_metadata_preserved': copy.deepcopy(conclusion['linked_published_metadata_only']),
        'A3_published_method_scope_or_preprint_equivalence_certified': False,
        'GENNN_remaining_primary_method_gap_preserved': copy.deepcopy(conclusion['unresolved_primary_leads']),
        'GENNN_metadata_scope_preserved': copy.deepcopy(scopes['scopes'][3]),
        'incidental_public_result_exposures_not_adopted': True,
        'new_direct_complete_operator_match_or_absence_proof': False,
        'numeric_results_global_novelty_ready_successor_or_execution_adopted': False}
    assert [r['title_locator'] for r in old['unresolved_title_only_metadata_locators_v71']] == ['GETS', 'Adapt/Agree/Aggregate']
    new['historical_title_locator_resolution_events_v72'] = [
        {'historical_field': 'unresolved_title_only_metadata_locators_v71', 'historical_index': pos,
            'title_locator': title, 'resolved_to_method_record_index': 258 + pos,
            'resolved_to_exact_primary_id': conclusion['papers'][pos]['canonical_id'],
            'resolution_scope': 'Exact bounded preprint method only; publication-version qualifications retained',
            'original_title_locator_entry_unchanged': True}
        for pos, title in enumerate(('GETS', 'Adapt/Agree/Aggregate'))]
    gen_id = 'doi:10.1016/j.inffus.2024.102461'
    gen_pos = next(n for n, r in enumerate(old['unresolved_primary_metadata_leads']) if r['canonical_metadata_identifier'] == gen_id)
    new['GENNN_metadata_upgrade_and_remaining_method_gap_v72'] = {
        'canonical_metadata_identifier': gen_id, 'original_unresolved_lead_index': gen_pos,
        'original_lead_preserved_exactly': True,
        'saved_metadata_and_route_conclusions': copy.deepcopy(conclusion['unresolved_primary_leads'][0]),
        'metadata_scope_reference': {**ref(SOURCE / 'READ_SCOPES.json'), 'selector': 'scopes/3'},
        'resolved_metadata': 'DOI/title/authors/Information Fusion issue and public Elsevier coredata coverDate 2024-10-31',
        'primary_first_online_date_independently_confirmed': False,
        'abstract_read': False, 'primary_method_read': False, 'new_paper_record_or_group_added': False,
        'absence_or_novelty_clearance': False}
    published_id = 'doi:10.1007/s13042-026-03047-y'
    assert all(r['canonical_metadata_identifier'] != published_id for r in old['unresolved_primary_metadata_leads'])
    new['unresolved_primary_metadata_leads'].append({
        'canonical_metadata_identifier': published_id,
        'title': conclusion['papers'][1]['title'], 'linked_known_work_identifier': 'arxiv:2503.17842',
        'status': 'Linked published successor metadata/abstract preview only; full published method and exact preprint equivalence remain unqualified',
        'scope_reference': {**ref(SOURCE / 'READ_SCOPES.json'), 'selector': 'scopes/2'},
        'linked_method_record_index': 259, 'published_method_read': False, 'full_paper_read': False,
        'method_version_equivalence_certified': False, 'additional_work_identity_or_method_credit': 0,
        'paper_record_added_for_published_version': False, 'normalized_paper_group_added_for_published_version': False})
    new['current_metadata_locator_dispositions_v72'] = {
        'old_GETS_and_A3_title_locators_resolved_to_bounded_preprint_method_records': 2,
        'unresolved_title_only_locators_after_resolution_events': 0,
        'unresolved_DOI_primary_metadata_leads': 5,
        'GENNN_primary_method_unresolved': True, 'A3_published_version_method_equivalence_unresolved': True,
        'original_leads_and_all_failure_histories_preserved': True}
    assert new['paper_records'][:258] == old['paper_records']
    assert compact(new['paper_records'][:258]) == compact(old['paper_records'])
    assert groups[:len(old['canonical_identifier_normalization']['groups'])] == old['canonical_identifier_normalization']['groups']
    assert {k: v for k, v in new['canonical_identifier_normalization'].items() if k != 'groups'} == {
        k: v for k, v in old['canonical_identifier_normalization'].items() if k != 'groups'}
    assert new['existing_packets'][:len(old['existing_packets'])] == old['existing_packets']
    assert new['unresolved_primary_metadata_leads'][:4] == old['unresolved_primary_metadata_leads']
    assert new['unresolved_title_only_metadata_locators_v71'] == old['unresolved_title_only_metadata_locators_v71']
    for key in old:
        if key not in current + ['paper_records', 'existing_packets', 'canonical_identifier_normalization', 'unresolved_primary_metadata_leads']:
            assert new[key] == old[key], key
    assert len(groups) == len({g['normalized_identifier'] for g in groups})
    covered = [n for g in groups for n in g['record_indices']]
    assert len(covered) == len(set(covered)) == 260 and set(covered) == set(range(260))
    after = metrics(new)
    assert (after['conclusion_records'], after['normalized_paper_identifiers'], after['software_documentation_identifiers']) == (260, 207, 2)
    account = {k: copy.deepcopy(v) for k, v in old['read_accounting'].items() if not k.startswith('latest_packet_')}
    account.update(after, state='PROSPECTIVE_SAVED_GETS_A3_PRIMARY_CLOSURE_PENDING_ROOT_REVIEW',
        latest_packet_new_scoped_primary_reads=2, latest_packet_scoped_primary_method_events=2,
        latest_packet_previously_completed_scoped_read_adoptions=2, latest_packet_new_paper_identity_groups=2,
        latest_packet_source_primary_version_documents=2, latest_packet_full_primary_reads=0,
        latest_packet_targeted_retained_primary_revisits=0, latest_packet_redundant_primary_revisits_zero_unique_scope_credit=0,
        latest_packet_bounded_author_source_repositories=1, latest_packet_author_code_semantic_file_scopes=3,
        latest_packet_bounded_author_source_scope_events=1, latest_packet_README_identity_file_scopes=1,
        latest_packet_source_repository_commits=1, latest_packet_source_code_line_ranges=5,
        latest_packet_source_semantic_code_lines=540, latest_packet_linked_published_metadata_upgrades=1,
        integration_pass_searches=0, integration_pass_retrievals=0, integration_pass_new_primary_reads=0,
        integration_pass_full_primary_reads=0, integration_pass_primary_method_reads=0,
        integration_pass_author_source_semantic_reads=0, integration_pass_experimental_score_artifact_reads=0,
        integration_pass_raw_primary_reads=0, integration_pass_selected_primary_text_scope_reads=0,
        historical_path_catalog_note='All258 prior records, all205 paper/2 software groups and all history preserved exactly. Append GETS/A3 preprint method scopes and saved pinned GETS author-source event; zero integration rereads. Old targeted/redundant scope counts remain historical. Both title locators resolved by additive events; GENNN unread method and A3 unread published-method equality remain explicit. No lossy history rewrite; exact current metadata snapshot.')
    new['read_accounting'] = account
    new['latest_adoption'] = {'UTC': now, 'status': 'PROSPECTIVE_PENDING_ROOT_REVIEW', 'predecessor': ref(PREV / 'LITERATURE_INDEX.json'),
        'root_adopted_predecessor_reference': ref(ADOPTION), 'added_records': added,
        'source_completed_new_scoped_method_identities': 2, 'source_completed_author_repository_scope_events': 1,
        'source_completed_author_semantic_files': 3, 'source_README_identity_scopes': 1,
        'source_old_primary_revisits': 0, 'source_full_paper_reads': 0,
        'resolved_historical_title_only_locators': 2, 'GENNN_still_primary_unread': True,
        'A3_linked_published_metadata_not_additional_method_identity': True,
        'integration_primary_author_source_reads_or_retrievals': 0,
        'numeric_predictive_novelty_ready_successor_or_execution_adoption': False,
        'source_bindings_reference': ref(HERE / 'SOURCE_BINDINGS.json')}
    encoded = compact(new) + b'\n'
    assert len(encoded) < MAX_BYTES and json.loads(encoded) == new
    save('SIZE_LIMIT_CHECK.json', {'index_bytes': len(encoded), 'decimal_2MB_limit': MAX_BYTES,
        'headroom_bytes': MAX_BYTES - len(encoded), 'within_limit': True, 'JSON_roundtrip_equal': True,
        'lossy_removal_or_history_rewrite': False, 'raw_new_primary_or_source_text_embedded': False,
        'serialization': 'Complete compact UTF-8 JSON, sorted object keys, separators comma/colon without optional whitespace; ensure_ascii=False; one trailing newline',
        'formatting_only_compaction_preserves_all_JSON_values_types_and_array_order': True,
        'same_lossless_serialization_as_predecessor': True})
    with (HERE / 'LITERATURE_INDEX.json').open('xb') as stream:
        stream.write(encoded)
    save('DELTA.json', {'UTC': now, 'prospective': True, 'before': before, 'after': after,
        'metric_deltas': {k: after[k] - before[k] for k in after}, 'added_records': added,
        'unresolved_DOI_metadata_leads_before_after': [4, 5],
        'historical_title_locator_entries_preserved': 2, 'title_locators_resolved_by_additive_events': 2,
        'effective_unresolved_title_only_locators': 0, 'new_unique_method_identity_adoptions': 2,
        'saved_GETS_author_repository_events_adopted': 1, 'saved_GETS_semantic_files': 3,
        'saved_GETS_README_identity_files': 1, 'source_old_primary_revisits': 0, 'source_full_paper_reads': 0,
        'integration_primary_author_source_reads_or_retrievals': 0,
        'canonical_status_ledger_or_prior_index_edits': False})
    save('VERIFICATION.json', {'UTC': now, 'status': 'PASS_METADATA_ONLY_PROSPECTIVE_PENDING_ROOT_REVIEW',
        'pins': PINS, 'all258_prior_records_canonical_bytes_equal': True,
        'all205_paper_and2_software_groups_exactly_preserved': True,
        'all_prior_rules_aliases_catalog_prefix_events_decisions_and_scouts_preserved': True,
        'all4_prior_unresolved_metadata_leads_exactly_preserved': True,
        'both_prior_title_locator_entries_preserved_and_resolutions_additive': True,
        'all_changed_current_metadata_snapshotted_exactly': True,
        'two_new_primary_method_work_scopes_only': True, 'GETS_three_semantic_files_one_README_identity_scope_only': True,
        'GETS_pinned_commit_source_discrepancy_and_acceptance_version_limits_preserved': True,
        'A3_preprint_method_vs_published_preview_scope_and_identity_credit_separated': True,
        'GENNN_metadata_resolved_only_no_abstract_method_or_absence_claim': True,
        'source_old_primary_revisit_count_zero_prior_revisit_history_preserved': True,
        'source_payloads_not_reopened_or_rehashed': True, 'integration_primary_reads_or_retrievals': 0,
        'new_numerical_result_global_novelty_ready_successor_or_execution_claim': False,
        'index_bytes': len(encoded), 'within_decimal_2MB': True, 'JSON_roundtrip_equal': True,
        'lossless_compact_serialization_preserves_semantic_history': True,
        'recomputed_metrics': after, 'only_new_index_v72_subtree_written': True})
    notes = f'''# Prospective literature index v72

Prepared from root-adopted index71. All 258 prior records, 205 paper groups, 2 software groups, old aliases/rules, catalog prefixes, scout conclusions/revisit counts, MGL event, protocol amendments and failure histories remain exactly preserved. Six replaced current metadata fields are snapshotted exactly.

Append GETS v2 and A3-GCN v1 relevant method scopes: 260 records, 207 paper groups and 2 software groups. Source task: two new method work identities, zero full-paper reads and zero old primary revisits. Separately bind one pinned GETS author repository event at `{COMMIT}`: three semantic code files, five ranges/540 lines, plus one README identity file. This is saved bounded static-source evidence, with no integration reread or execution.

GETS paper/source qualifications remain explicit: class-vector positive scaling can change argmax; all experts execute despite top-k output weights; source gate inputs differ from paper prose; source fits on VALID and selects calibration loss on TRAIN after base VALID selection. No accuracy-preservation, conditional-execution efficiency, official accepted-version equivalence or runtime claim is certified. A3 uses augmented independent GCNs, confidence-set overlap and a separate consensus-label GCN; confidence-set Jaccard is not class agreement. Native source and exact published-method equality remain unqualified.

GETS/A3's original title-only metadata locators remain unchanged historical entries; additive resolution events point to the new bounded preprint method records. The A3 published DOI is an explicit linked-work alias and metadata/abstract preview, not an additional work or method read. Its full published method/equivalence gap is a separate linked metadata lead. All four original unresolved DOI leads remain intact; GENNN's title/authors/issue metadata is upgraded without abstract or method credit. GENNN remains unread, with all failed routes and unknown equivalence preserved. No global novelty or absence conclusion follows from unsuccessful retrieval.

Safe source conclusions, locators, receipts and manifest/seal were authenticated. Primary/source bodies, extracted blocks, selected passages, author files and discovery bodies were not opened or rehashed. Source claims are inherited, not independently reassessed; prior scout payload bindings were not dereferenced. Zero integration retrievals, target payload accesses or scientific execution. Only this fresh subtree was written. No canonical/source edits or execution authority.

Complete JSON is {len(encoded):,} bytes, with {MAX_BYTES - len(encoded):,} bytes below the strict decimal 2,000,000-byte publisher cap. Encoding is the same lossless compact UTF-8 JSON as index71: sorted object keys, comma/colon separators without optional whitespace, unescaped Unicode and one trailing newline. Parsed values/types/array order roundtrip exactly; no semantic history was removed or rewritten. Root owns final review/adoption/publication.
'''
    with (HERE / 'ROOT_ADOPTION_NOTES.md').open('x') as stream:
        stream.write(notes)
    for row in predecessor_refs + source_refs:
        assert ref(BASE / row['path']) == row
    files = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(HERE.iterdir()) if p.is_file()]
    save('MANIFEST.json', {'UTC': now, 'prospective': True, 'files': files})
    save('SEAL.json', {'manifest_sha256': ref(HERE / 'MANIFEST.json')['sha256'], 'payload_files': len(files),
        'saved_new_primary_method_work_identities': 2, 'saved_author_repository_events': 1,
        'saved_author_semantic_files': 3, 'saved_README_identity_files': 1,
        'source_old_primary_revisits': 0, 'new_full_paper_reads': 0,
        'integration_primary_author_source_reads_or_retrievals': 0, 'source_payloads_reopened_or_rehashed': False,
        'lossy_history_rewrite': False, 'scientific_execution_or_authority': False,
        'canonical_status_source_ledger_or_prior_index_edits': False})
    print(json.dumps({**after, 'index_bytes': len(encoded), 'headroom_decimal_bytes': MAX_BYTES - len(encoded),
        'index_sha256': ref(HERE / 'LITERATURE_INDEX.json')['sha256'], 'manifest_sha256': ref(HERE / 'MANIFEST.json')['sha256'],
        'seal_sha256': ref(HERE / 'SEAL.json')['sha256'], 'status': 'PROSPECTIVE_READY_FOR_ROOT_REVIEW'}))


if __name__ == '__main__':
    main()
