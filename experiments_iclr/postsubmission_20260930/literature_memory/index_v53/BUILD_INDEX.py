"""Append two completed accuracy-oriented scopes. Integration reads no primary method."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
P = HERE.parent.parent
PREV = P / 'literature_memory/index_v52'
PACKET = P / 'accuracy_ensemble_quality_prior_gap_20261004_v1'
EXPECTED_MANIFEST = '63dd695d13e543fe2e2b9cb44c9ad33e8232594bd358d21af150d8ca5157f5c7'
EXPECTED_IDS = {'arxiv:2504.12627', 'arxiv:2204.06625'}


def ref(path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(P)), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def save(name, obj):
    with (HERE / name).open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def metrics(index):
    catalog = {r['path'] for r in index['existing_packets']}
    conclusions = {r['conclusion_file'] for r in index['paper_records']
                   if isinstance(r.get('conclusion_file'), str)}
    scopes = {r['read_scope_reference']['path'] for r in index['paper_records']
              if 'read_scope_reference' in r}
    legacy = {r['read_scope_file_reference']['path'] for r in index['paper_records']
              if 'read_scope_file_reference' in r}
    groups = index['canonical_identifier_normalization']['groups']
    return {'conclusion_records': len(index['paper_records']),
            'normalized_paper_identifiers': sum(g['kind'] == 'paper' for g in groups),
            'software_documentation_identifiers': sum(g['kind'] != 'paper' for g in groups),
            'catalog_entries': len(index['existing_packets']),
            'unique_catalog_document_paths': len(catalog),
            'unique_conclusion_source_documents': len(conclusions),
            'unique_referenced_document_paths': len(catalog | conclusions),
            'unique_scope_reference_document_paths': len(scopes),
            'unique_scope_reference_document_paths_including_legacy_field_alias': len(scopes | legacy)}


def normalized(raw):
    raw = raw.strip().lower()
    if raw.startswith('arxiv:'):
        return re.sub(r'v\d+$', '', raw)
    return raw


def main():
    utc = datetime.now(timezone.utc).isoformat()
    previous = json.loads((PREV / 'LITERATURE_INDEX.json').read_text())
    prior_manifest = json.loads((PREV / 'MANIFEST.json').read_text())
    prior_seal = json.loads((PREV / 'SEAL.json').read_text())
    assert ref(PREV / 'MANIFEST.json')['sha256'] == prior_seal['manifest_sha256']
    for row in prior_manifest['files']:
        actual = ref(PREV / row['path'])
        assert actual['bytes'] == row['bytes'] and actual['sha256'] == row['sha256']
    source_manifest = json.loads((PACKET / 'MANIFEST.json').read_text())
    assert ref(PACKET / 'MANIFEST.json')['sha256'] == EXPECTED_MANIFEST
    for row in source_manifest['files']:
        actual = ref(PACKET / row['path'])
        assert actual['bytes'] == row['bytes'] and actual['sha256'] == row['sha256']

    # Source bytes are hashed only. Saved scopes/conclusions are integrated, not reread methods.
    scope_doc = json.loads((PACKET / 'READ_SCOPES.json').read_text())
    conclusion_doc = json.loads((PACKET / 'PAPER_CONCLUSIONS.json').read_text())
    passages = json.loads((PACKET / 'PRIMARY_PASSAGES.json').read_text())
    metadata = json.loads((PACKET / 'PRIMARY_METADATA.json').read_text())
    assert scope_doc['new_scoped_primary_paper_identities'] == 2
    assert scope_doc['new_full_paper_reads'] == scope_doc['new_proof_reads'] == 0
    assert scope_doc['new_author_source_reads'] == 0
    assert len(scope_doc['scopes']) == len(conclusion_doc['papers']) == len(passages) == 2
    assert {p['canonical_id'] for p in conclusion_doc['papers']} == EXPECTED_IDS
    assert {s['canonical_id'] for s in scope_doc['scopes']} == EXPECTED_IDS
    assert not conclusion_doc['queued_comparison_modified']
    assert not conclusion_doc['implementation_or_execution_adopted']
    old_groups = previous['canonical_identifier_normalization']['groups']
    old_ids = set()
    for group in old_groups:
        for raw in ([group['normalized_identifier']] + group.get('raw_canonical_identifiers', [])
                    + group.get('explicit_aliases', [])):
            old_ids.add(normalized(raw))
    assert EXPECTED_IDS.isdisjoint(old_ids)
    assert metrics(previous)['conclusion_records'] == 213
    assert metrics(previous)['normalized_paper_identifiers'] == 162
    assert metrics(previous)['software_documentation_identifiers'] == 2

    index = copy.deepcopy(previous)
    replaced = ('schema', 'created_UTC', 'latest_adoption', 'read_accounting',
                'predecessor_index', 'predecessor_index_sha256')
    index['integration_v53_predecessor_v52_snapshot'] = {
        **{k: copy.deepcopy(previous.get(k)) for k in replaced},
        'index_reference': ref(PREV / 'LITERATURE_INDEX.json'),
        'manifest_reference': ref(PREV / 'MANIFEST.json'),
        'seal_reference': ref(PREV / 'SEAL.json'),
        'actual_recomputed_metrics': metrics(previous)}
    index['schema'] = 'literature-memory-index-v53'
    index['created_UTC'] = utc
    index['predecessor_index'] = str((PREV / 'LITERATURE_INDEX.json').relative_to(P))
    index['predecessor_index_sha256'] = ref(PREV / 'LITERATURE_INDEX.json')['sha256']
    added = []
    new_groups = []
    for paper in conclusion_doc['papers']:
        key = paper['key']
        scope = next(s for s in scope_doc['scopes'] if s['key'] == key)
        passage = next(s for s in passages if s['key'] == key)
        canonical = paper['canonical_id']
        assert normalized('arxiv:' + scope['requested_versioned_id']) == canonical
        assert 'arxiv:' + scope['requested_versioned_id'].lower().split('v')[0] == canonical
        assert metadata[key]['requested_versioned_id'] == paper['requested_versioned_id'] == scope['requested_versioned_id']
        assert paper['full_paper_read'] is scope['full_paper_read'] is False
        assert not paper['global_novelty_clearance'] and not paper['predictive_utility_established']
        assert not scope['proof_read'] and not scope['author_source_read']
        assert ref(PACKET / scope['source_path'])['sha256'] == scope['source_sha256'] == paper['primary_sha256'] == passage['source_sha256']
        assert ref(PACKET / scope['blocks_path'])['sha256'] == scope['blocks_sha256'] == passage['blocks_sha256']
        assert ref(PACKET / scope['equation_path'])['sha256'] == scope['equation_sha256']
        blocks = json.loads((PACKET / scope['blocks_path']).read_text())
        equations = json.loads((PACKET / scope['equation_path']).read_text())
        authorized_indices = set(scope['additional_exact_paragraph_blocks'])
        for lo, hi in scope['paragraph_and_caption_block_ranges_inclusive']:
            authorized_indices.update(range(lo, hi + 1))
        for b in passage['passages']:
            assert b == blocks[b['index']] and b['index'] in authorized_indices
        assert passage['equations'] == equations
        assert scope['display_equations_read'] == [e['ids'][1] for e in equations]
        payload = {'canonical': canonical, 'version': scope['requested_versioned_id'],
                   'ranges': scope['paragraph_and_caption_block_ranges_inclusive'],
                   'additional_exact_paragraph_blocks': scope['additional_exact_paragraph_blocks'],
                   'display_equations_read': scope['display_equations_read']}
        dedup = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        assert not any(r.get('scope_deduplication_key_sha256') == dedup for r in index['paper_records'])
        position = len(index['paper_records'])
        scope_ref = ref(PACKET / 'READ_SCOPES.json'); scope_ref['selector'] = key
        record = {'canonical_id': canonical, 'normalized_identifier': canonical,
                  'requested_versioned_id': paper['requested_versioned_id'],
                  'conclusion': copy.deepcopy(paper), 'compact_primary_scope': copy.deepcopy(scope),
                  'exact_read_scope': copy.deepcopy(scope),
                  'conclusion_file': str((PACKET / 'PAPER_CONCLUSIONS.json').relative_to(P)),
                  'conclusion_file_sha256': ref(PACKET / 'PAPER_CONCLUSIONS.json')['sha256'],
                  'read_scope_reference': scope_ref, 'source_packet': PACKET.name,
                  'source_packet_manifest_reference': ref(PACKET / 'MANIFEST.json'),
                  'source_report_reference': ref(PACKET / 'REPORT.md'),
                  'primary_payload_reference': ref(PACKET / scope['source_path']),
                  'primary_metadata_reference': {**ref(PACKET / 'PRIMARY_METADATA.json'), 'selector': key},
                  'scope_deduplication_key_sha256': dedup,
                  'full_paper_read': False, 'proof_audit': False,
                  'global_novelty_clearance': False, 'predictive_adoption': False,
                  'execution_authorized': False, 'numeric_result_transfer': False,
                  'integration_pass_primary_reread': False,
                  'integration_pass_new_semantic_read': False,
                  'integration_pass_new_primary_reads': 0,
                  'integration_read_status': 'Completed saved scoped conclusion adopted; integration adds zero primary reads',
                  'genuinely_new_scoped_paper_identity_in_source_packet': True}
        index['paper_records'].append(record)
        group = {'normalized_identifier': canonical, 'kind': 'paper',
                 'raw_canonical_identifiers': [canonical, 'arxiv:' + scope['requested_versioned_id'].lower()],
                 'record_indices': [position], 'explicit_aliases': []}
        # No extra cross-scheme alias inferred from a title or discovery hit.
        new_groups.append(group)
        index['canonical_identifier_normalization']['groups'].append(group)
        added.append({'canonical_id': canonical, 'record_index': position,
                      'scope_deduplication_key_sha256': dedup, 'full_paper_read': False,
                      'integration_new_primary_reads': 0, 'scope': copy.deepcopy(scope)})

    for name in ('REPORT.md', 'PAPER_CONCLUSIONS.json', 'READ_SCOPES.json', 'PRIMARY_PASSAGES.json',
                 'PRIMARY_METADATA.json', 'MEMORY_REUSE.json', 'RETRIEVAL.json', 'DISCOVERY.json', 'MANIFEST.json'):
        index['existing_packets'].append({**ref(PACKET / name),
            'kind': 'stored_scoped_accuracy_shared_ensemble_prior',
            'scope': 'Two completed scoped primary reads, saved exact scopes and metadata; integration has zero primary reads and no scientific adoption.'})
    index['canonical_identifier_normalization']['post_v53_scope_append'] = {
        'UTC': utc, 'new_paper_identities': sorted(EXPECTED_IDS),
        'added_record_indices': [r['record_index'] for r in added],
        'old_groups_preserved_exactly': len(old_groups), 'title_only_alias_merges': 0,
        'new_cross_scheme_aliases_inferred': 0, 'integration_primary_reads': 0,
        'deduplication': 'Exact normalized arXiv IDs and version-pinned saved metadata; no title-only merging.'}
    index['latest_adoption'] = {'UTC': utc, 'source': ref(PACKET / 'MANIFEST.json'),
        'added_records': added, 'predecessor': ref(PREV / 'LITERATURE_INDEX.json'),
        'previous_records_and_groups_preserved': True,
        'source_completed_new_scoped_reads': 2, 'source_full_paper_reads': 0,
        'integration_primary_reads': 0, 'novelty_or_predictive_or_execution_adoption': False}
    index['post_v53_append'] = copy.deepcopy(index['latest_adoption'])
    index['accuracy_shared_ensemble_prior_limits_v53'] = {
        'root_reported_control_decision': 'Root is adding an all-TRAIN ordinary native GNNM control before comparison freezing.',
        'conditional_followup': 'A shared-encoder/private-head architecture comparator remains conditional on a positive core screen supporting a private-trajectory claim.',
        'comparison_freeze_or_execution_admitted_by_this_index': False,
        'complete_current_protocol_equivalence_established': False,
        'global_novelty_clearance': False, 'predictive_gain_established': False,
        'report_reference': ref(PACKET / 'REPORT.md'),
        'underlying_DPOSE_cited_lead_primary_method_read': False,
        'underlying_DPOSE_extra_paper_record_added': False}

    account = copy.deepcopy(previous['read_accounting'])
    account.update(metrics(index))
    account.update({'latest_accounting_correction_reference': 'literature_memory/index_v53/VERIFICATION.json',
        'latest_index_growth': 2, 'latest_adoption_packets': 1,
        'latest_packet_genuinely_new_scoped_paper_identities': 2,
        'latest_packet_new_paper_identity_groups': 2, 'latest_packet_new_scoped_primary_reads': 2,
        'latest_packet_previously_completed_scoped_read_adoptions': 2,
        'latest_packet_scoped_primary_method_events': 2,
        'latest_packet_first_scoped_method_identity': sorted(EXPECTED_IDS),
        'latest_packet_full_primary_reads': 0,
        'latest_packet_cached_index_conclusion_records_reused': 9,
        'latest_packet_dgi_cached_prior_reuses': 0,
        'latest_packet_source_identity_and_citation_document_scopes': 2,
        'integration_pass_metadata_identity_checks': 2,
        'integration_pass_primary_method_reads': 0, 'integration_pass_new_primary_reads': 0,
        'integration_pass_full_primary_reads': 0, 'integration_pass_retained_primary_revisits': 0,
        'integration_pass_author_source_semantic_reads': 0,
        'integration_pass_project_source_semantic_reads': 0,
        'full_paper_read_total_certified': False,
        'cumulative_scoped_or_full_read_totals_certified': False,
        'historical_path_catalog_note': 'All213 v52 records,164 existing identity groups and all history preserved. Two previously completed scoped reads appended;215 scoped conclusion records and164 paper identities plus2 software are not full-read totals. Integration adds zero primary reads.',
        'state': 'ROOT_DELEGATED_ADDITIVE_SCOPED_ACCURACY_PRIOR_MEMORY'})
    index['read_accounting'] = account

    assert index['paper_records'][:213] == previous['paper_records']
    assert index['existing_packets'][:len(previous['existing_packets'])] == previous['existing_packets']
    assert index['canonical_identifier_normalization']['groups'][:len(old_groups)] == old_groups
    for key, value in previous['canonical_identifier_normalization'].items():
        if key != 'groups': assert index['canonical_identifier_normalization'][key] == value
    allowed_changes = set(replaced) | {'paper_records', 'existing_packets', 'canonical_identifier_normalization'}
    for key, value in previous.items():
        if key not in allowed_changes: assert index[key] == value
    norms = [g['normalized_identifier'] for g in index['canonical_identifier_normalization']['groups']]
    assert len(norms) == len(set(norms))
    current = metrics(index)
    assert current['conclusion_records'] == 215
    assert current['normalized_paper_identifiers'] == 164
    assert current['software_documentation_identifiers'] == 2
    save('LITERATURE_INDEX.json', index)
    save('VERIFICATION.json', {'UTC': utc, 'predecessor': ref(PREV / 'LITERATURE_INDEX.json'),
        'predecessor_manifest_and_seal_verified': True,
        'source_manifest': ref(PACKET / 'MANIFEST.json'),
        'source_packet_manifest_entries_verified': len(source_manifest['files']),
        'all_source_payload_hashes_verified': True,
        'exact_scope_passage_and_equation_bindings_verified': True,
        'previous_records_unchanged': 213, 'previous_groups_unchanged': len(old_groups),
        'previous_catalog_prefix_unchanged': True,
        'previous_top_level_history_fields_preserved': True,
        'replaced_metadata_saved_in_predecessor_snapshot': True,
        'duplicate_canonical_groups': 0, 'duplicate_new_scope_keys': 0,
        'new_record_indices': [213, 214], 'new_paper_identities': sorted(EXPECTED_IDS),
        'metrics': current, 'integration_primary_reads': 0,
        'source_completed_new_scoped_reads': 2, 'source_full_paper_reads': 0,
        'underlying_DPOSE_primary_read_or_record_added': False,
        'scientific_gain_novelty_execution_or_canonical_status_adoption': False})
    rows = [{'path': f.name, 'bytes': f.stat().st_size, 'sha256': ref(f)['sha256']}
            for f in sorted(HERE.iterdir()) if f.is_file()]
    save('MANIFEST.json', {'schema': 'literature-memory-manifest-v53', 'created_UTC': utc, 'files': rows})
    save('SEAL.json', {'schema': 'literature-memory-seal-v53',
        'manifest_sha256': ref(HERE / 'MANIFEST.json')['sha256'], 'immutable': True})
    print(json.dumps({'metrics': current, 'index_sha256': ref(HERE / 'LITERATURE_INDEX.json')['sha256'],
        'manifest_sha256': ref(HERE / 'MANIFEST.json')['sha256'],
        'seal_sha256': ref(HERE / 'SEAL.json')['sha256']}))


if __name__ == '__main__':
    main()
