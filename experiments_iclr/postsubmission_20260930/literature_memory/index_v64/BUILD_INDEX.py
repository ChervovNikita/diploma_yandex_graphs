"""Prospective adoption of three saved graph method scopes; no searches or primary rereads.
Reuses index_v63's append, identity grouping, manifest/seal and 2MB conventions.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
P = HERE.parent.parent
VERSION = int(HERE.name.split('_v')[-1])
PREV = P / 'literature_memory/index_v63'
PREV_MANIFEST_PIN = 'eadb6036a0c6499a23c01c95ea78a8a94375c878207d83798152c4c05232dcb9'
PREV_INDEX_PIN = 'b9e7ebe7bec178b57d9fece610baa3efc37824aa72e90208e284cc400948b4c4'
PACKETS = [
    ('private_learning_credit_recent_prior_20261005_v1', 'aae8e4bd96e1457984e589ebff00d0e793ee0ccfa64b2c4b5385251b066ec596', 'a56c5af582c9aa090884debd6141cf7e3372a121401a63d6d7023c2b774f1b39'),
    ('persistent_graph_private_learning_credit_followup_20261005_v1', '25be332ad55411b0794316e10e6abd195b9b8daa2c59649a1f82a4fa3c9e3009', '9c46a6f055812b2afa58c44f01e0b946c0c8d860d7c57d2f3b4278d57ffc8ab5'),
]
MAX_INDEX_BYTES = 2000000


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(P)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def save(name, value):
    with (HERE / name).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def ids(raw):
    return {re.sub(r'v\d+$', '', x.strip().lower()) for x in raw.split(';')}


def metrics(d):
    cat = {r['path'] for r in d['existing_packets']}
    con = {r['conclusion_file'] for r in d['paper_records'] if isinstance(r.get('conclusion_file'), str)}
    scopes = {r['read_scope_reference']['path'] for r in d['paper_records'] if 'read_scope_reference' in r}
    legacy = {r['read_scope_file_reference']['path'] for r in d['paper_records'] if 'read_scope_file_reference' in r}
    groups = d['canonical_identifier_normalization']['groups']
    return dict(conclusion_records=len(d['paper_records']), normalized_paper_identifiers=sum(g['kind']=='paper' for g in groups),
        software_documentation_identifiers=sum(g['kind']!='paper' for g in groups), catalog_entries=len(d['existing_packets']),
        unique_catalog_document_paths=len(cat), unique_conclusion_source_documents=len(con),
        unique_referenced_document_paths=len(cat | con), unique_scope_reference_document_paths=len(scopes),
        unique_scope_reference_document_paths_including_legacy_field_alias=len(scopes | legacy))


def verify_manifest(folder, pin):
    manifest = ref(folder / 'MANIFEST.json')
    assert manifest['sha256'] == pin, (folder.name, manifest['sha256'], pin)
    raw = json.loads((folder / 'MANIFEST.json').read_text())
    rows = raw if isinstance(raw, list) else raw['files']
    result = [manifest]
    for row in rows:
        path = folder / row['path']
        assert path.resolve().is_relative_to(folder) and not path.is_symlink()
        got = ref(path)
        assert (got['bytes'], got['sha256']) == (row['bytes'], row['sha256'])
        result.append(got)
    return result


def main():
    now = datetime.now(timezone.utc).isoformat()
    prev_files = verify_manifest(PREV, PREV_MANIFEST_PIN)
    assert json.loads((PREV / 'SEAL.json').read_text())['manifest_sha256'] == PREV_MANIFEST_PIN
    assert ref(PREV / 'LITERATURE_INDEX.json')['sha256'] == PREV_INDEX_PIN
    prior = json.loads((PREV / 'LITERATURE_INDEX.json').read_text())
    old_metrics = metrics(prior)
    assert old_metrics['conclusion_records'] == 238 and old_metrics['normalized_paper_identifiers'] == 186
    bindings = []
    manifest_checks = []
    source_input_checks = []
    for name, ledger_pin, manifest_pin in PACKETS:
        packet = P / name
        verified = verify_manifest(packet, manifest_pin)
        assert ref(packet / 'SOURCE_LEDGER.json')['sha256'] == ledger_pin
        bindings.extend(verified)
        manifest_checks.append(dict(packet=name, manifest_sha256=manifest_pin, verified_payload_files=len(verified)-1,
            seal_present=(packet / 'SEAL.json').exists()))
        for row in json.loads((packet / 'INPUT_BINDINGS.json').read_text()):
            got = ref(P / row['path'])
            assert got['sha256'] == row['sha256']
            source_input_checks.append(dict(path=row['path'], sha256=got['sha256'], unchanged=True))
    # Decisions remain hashed analysis notes, never paper records or reading events.
    decisions = []
    for name, pin in [
        ('private_update_marginal_value_design_synthesis_20261005_v1/REPORT.md', '08c999ae400461a373113fcec873e9a4b230199d4a6bb9e8fe232b8781155b10'),
        ('shared_ensemble_method_decision_20261005_v1/REPORT.md', '1b2e1fb13fde8d5581a740d9646e2472c2ebf8e5645a996e66959cc3345c53cb'),
    ]:
        got = ref(P / name)
        assert got['sha256'] == pin
        decisions.append(dict(**got, kind='saved_mathematical_or_method_decision_not_paper', new_paper_identity=False,
            new_primary_reads=0, new_full_paper_reads=0))
    save('SOURCE_BINDINGS.json', dict(UTC=now, predecessor_files=prev_files + [ref(PREV / 'SEAL.json')],
        files=bindings, source_manifest_checks=manifest_checks, source_input_hash_checks=source_input_checks,
        optional_decision_notes=decisions, integrity_checks_only=True, integration_semantic_primary_reads=0,
        integration_primary_retrievals=0, previously_completed_scoped_primary_method_reads=3,
        source_completed_full_paper_reads=0, raw_source_text_embedded=False,
        deleted_full_HTML_hashes_are_receipt_claims_not_recomputed_raw_file_hashes=True))

    tm_packet = P / PACKETS[0][0]
    fg_packet = P / PACKETS[1][0]
    tm = json.loads((tm_packet / 'SOURCE_LEDGER.json').read_text())
    fg = json.loads((fg_packet / 'SOURCE_LEDGER.json').read_text())
    assert tm['read_accounting']['new_bounded_method_scope_reads'] == 1
    assert tm['read_accounting']['new_full_paper_reads'] == 0 and tm['read_accounting']['retained_primary_rereads'] == 0
    assert fg['reading_credit']['new_bounded_method_scope_reads'] == 2
    assert fg['reading_credit']['new_full_paper_reads'] == 0 and fg['reading_credit']['retained_primary_rereads'] == 0
    # Automated structure/hash validation of saved excerpts only; no semantic primary reread.
    tm_text = json.loads((tm_packet / 'PRIMARY_SCOPES.json').read_text())
    assert [s['element_id'] for s in tm_text] == tm['scope']['element_ids']
    fg_text = json.loads((fg_packet / 'SCOPE_EXCERPTS.json').read_text())
    assert len(fg_text) == len(fg['sources']) == 2
    for excerpt, source in zip(fg_text, fg['sources']):
        assert excerpt['key'] == source['key']
        for section, scope in zip(excerpt['selected_complete_subsections'], source['scope']):
            assert section['html_id'] == scope['html_id']
            assert hashlib.sha256(section['text'].encode()).hexdigest() == scope['selected_text_sha256']
    candidates = [dict(packet=tm_packet, canonical=tm['citation']['canonical_id'], title=tm['citation']['title'],
        scope=copy.deepcopy(tm['scope']), ledger_scope_selector='scope', conclusion_selector='conclusion',
        excerpt_file='PRIMARY_SCOPES.json', excerpt_selector='root (all four saved element entries)',
        retrieval=tm['retrieval'], retrieval_file='SOURCE_LEDGER.json', retrieval_selector='retrieval',
        saved_takeaway=' '.join(tm['conclusion'][k] for k in ['operator', 'episode_condition', 'deployment']),
        relationship=tm['conclusion']['citation_role'], equations=tm['scope']['numbered_equations'], algorithms=[])]
    for position, source in enumerate(fg['sources']):
        findings = source['findings']
        candidates.append(dict(packet=fg_packet, canonical=source['canonical_id'], title=source['title'],
            scope=dict(complete_subsections=copy.deepcopy(source['scope']), numbered_equations_read=source['numbered_equations_read'],
                algorithms_text_read=source['algorithms_text_read'], figure_scope=source['figure_scope'], figure_pixels_inspected=False),
            ledger_scope_selector=f'sources/{position}/scope', conclusion_selector=f'sources/{position}/findings',
            excerpt_file='SCOPE_EXCERPTS.json', excerpt_selector=str(position), retrieval=source['retrieval'],
            retrieval_file='PRIMARY_RETRIEVAL.json', retrieval_selector=str(position),
            saved_takeaway=' '.join(findings[k] for k in ['shared_representation_learning', 'persistent_private_learners', 'private_learning_credit', 'sharing_failure']),
            relationship=findings['successor'], equations=source['numbered_equations_read'], algorithms=source['algorithms_text_read']))
    assert [c['canonical'] for c in candidates] == ['arxiv:2506.00453v1', 'arxiv:2603.20338v1', 'arxiv:2406.11943v1']

    d = copy.deepcopy(prior)
    changed = ['schema', 'created_UTC', 'latest_adoption', 'read_accounting', 'predecessor_index', 'predecessor_index_sha256']
    d[f'integration_v{VERSION}_predecessor_v63_snapshot'] = {**{k:copy.deepcopy(prior[k]) for k in changed},
        'index_reference':ref(PREV / 'LITERATURE_INDEX.json'), 'manifest_reference':ref(PREV / 'MANIFEST.json'),
        'seal_reference':ref(PREV / 'SEAL.json')}
    d.update(schema=f'literature-memory-index-v{VERSION}', created_UTC=now,
        predecessor_index=str((PREV / 'LITERATURE_INDEX.json').relative_to(P)), predecessor_index_sha256=PREV_INDEX_PIN)
    groups = d['canonical_identifier_normalization']['groups']
    added = []
    new_keys = set()
    for c in candidates:
        canonical = c['canonical']; normalized = next(iter(ids(canonical))); packet = c['packet']
        matches = [g for g in groups if any(normalized in ids(x) for x in
            [g['normalized_identifier']] + g.get('raw_canonical_identifiers', []) + g.get('explicit_aliases', []))]
        assert len(matches) <= 1
        payload = dict(normalized_identifier=normalized, versioned_canonical_id=canonical.lower(), exact_read_scope=c['scope'])
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        pair = (normalized, key)
        assert pair not in new_keys
        assert not any(normalized in ids(r.get('canonical_id', '')) and r.get('scope_deduplication_key_sha256') == key for r in d['paper_records'])
        # All three identities are absent from v63; existing same-paper scope records would be retained, never deleted.
        assert not matches
        assert not any(c['title'].strip().lower() == r.get('conclusion', {}).get('verified_title', '').strip().lower() for r in prior['paper_records'])
        new_keys.add(pair)
        scope_ref = {**ref(packet / 'SOURCE_LEDGER.json'), 'selector':c['ledger_scope_selector']}
        excerpt_ref = {**ref(packet / c['excerpt_file']), 'selector':c['excerpt_selector']}
        retrieval = c['retrieval']
        record_index = len(d['paper_records'])
        d['paper_records'].append(dict(canonical_id=canonical, normalized_identifier=normalized,
            conclusion=dict(canonical_id=canonical, verified_title=c['title'], primary_url=retrieval['url'],
                saved_takeaway=c['saved_takeaway'], relationship_to_candidate=c['relationship'],
                read_status='Previously completed bounded primary method scope; not full paper', full_paper_read=False,
                numeric_results_adopted=False, global_novelty_clearance=False),
            conclusion_file=scope_ref['path'], conclusion_file_sha256=scope_ref['sha256'], conclusion_source_selector=c['conclusion_selector'],
            read_scope_reference=scope_ref, exact_read_scope=copy.deepcopy(c['scope']), source_packet=packet.name,
            source_packet_binding_reference=ref(HERE / 'SOURCE_BINDINGS.json'), source_extracted_text_reference=excerpt_ref,
            source_retrieval_reference={**ref(packet / c['retrieval_file']), 'selector':c['retrieval_selector']},
            source_primary_retrieval_metadata=dict(url=retrieval['url'], sha256=retrieval['sha256'], bytes=retrieval['bytes'],
                raw_HTML_retained=False, hash_status='Saved original retrieval receipt; deleted raw HTML not rehashed in integration'),
            scope_deduplication_key_sha256=key, scope_deduplication_key_schema='normalized identity + version + exact saved scope',
            scoped_method_read=True, full_paper_read=False, author_source_read=False, numeric_result_transfer=False,
            predictive_adoption=False, global_novelty_clearance=False, execution_authorized=False,
            integration_pass_new_primary_reads=0, integration_pass_primary_reread=False))
        groups.append(dict(normalized_identifier=normalized, kind='paper', raw_canonical_identifiers=[canonical],
            explicit_aliases=[], record_indices=[record_index]))
        added.append(dict(canonical_id=normalized, exact_canonical_id=canonical, record_index=record_index, new_identity=True,
            full_paper_read=False, source_packet=packet.name, scope_deduplication_key_sha256=key))
    # The same normalized-identity/scoped-key projection would be skipped on a duplicate adoption.
    assert len(new_keys) == 3
    assert all((a['canonical_id'], a['scope_deduplication_key_sha256']) in new_keys for a in added)
    catalog_files = [
        (tm_packet, ['SOURCE_LEDGER.json', 'PRIMARY_SCOPES.json', 'REPORT.md', 'MANIFEST.json']),
        (fg_packet, ['SOURCE_LEDGER.json', 'SCOPE_EXCERPTS.json', 'PRIMARY_RETRIEVAL.json', 'REPORT.md', 'MANIFEST.json']),
    ]
    for packet, names in catalog_files:
        for name in names:
            row = {**ref(packet / name), 'kind':'saved_graph_update_or_persistent_sharing_scoped_prior',
                'source_bindings_reference':ref(HERE / 'SOURCE_BINDINGS.json'),
                'scope':'Previously completed bounded method scopes; metadata/abstract/discovery custody is not reading credit. Zero integration primary/full reads or numerical, novelty, predictive or execution adoption.'}
            assert not any(r['path'] == row['path'] for r in d['existing_packets'])
            d['existing_packets'].append(row)
    groups.sort(key=lambda g:g['normalized_identifier'])
    assert d['paper_records'][:238] == prior['paper_records']
    assert d['existing_packets'][:len(prior['existing_packets'])] == prior['existing_packets']
    assert all(g in groups for g in prior['canonical_identifier_normalization']['groups'])
    assert {k:v for k,v in d['canonical_identifier_normalization'].items() if k != 'groups'} == {k:v for k,v in prior['canonical_identifier_normalization'].items() if k != 'groups'}
    for k in prior:
        if k not in changed + ['paper_records', 'existing_packets', 'canonical_identifier_normalization']:
            assert d[k] == prior[k]
    totals = metrics(d)
    assert totals['conclusion_records'] == 241 and totals['normalized_paper_identifiers'] == 189
    account = {k:copy.deepcopy(v) for k,v in prior['read_accounting'].items() if not k.startswith(('latest_', 'integration_pass_'))}
    account.update(totals, state='PROSPECTIVE_SAVED_GRAPH_METHOD_SCOPE_ADOPTION_PENDING_ROOT_REVIEW',
        historical_path_catalog_note='All 238 v63 records, groups, catalogs, events, decision linkages and history retained. Three saved scopes appended; zero integration primary/full reads. Optional hashed decisions are not papers.',
        latest_packet_new_scoped_primary_reads=3, latest_packet_full_primary_reads=0,
        latest_packet_previously_completed_scoped_read_adoptions=3, latest_packet_new_paper_identity_groups=3,
        integration_pass_new_primary_reads=0, integration_pass_full_primary_reads=0, integration_pass_primary_method_reads=0,
        integration_pass_author_source_semantic_reads=0, integration_pass_experimental_score_artifact_reads=0,
        integration_pass_retrievals=0)
    d['read_accounting'] = account
    d['latest_adoption'] = dict(UTC=now, status='PROSPECTIVE_PENDING_ROOT_REVIEW', predecessor=ref(PREV / 'LITERATURE_INDEX.json'),
        previous_records_preserved=True, added_records=added, source_completed_new_scoped_method_reads=3,
        source_full_paper_reads=0, integration_primary_reads=0, integration_retrievals=0,
        source_binding_reference=ref(HERE / 'SOURCE_BINDINGS.json'), novelty_or_numeric_or_predictive_or_execution_adoption=False)
    d[f'graph_update_and_persistent_sharing_prior_limits_v{VERSION}'] = dict(
        TMetaNet='Topology-conditioned scalar rates update full dynamic GNN weights. Inspected snapshots/rate scopes do not specify persistent private ensemble credit or current outer endpoint-union exclusion.',
        LPSFed='Repeated user-item client learning shares pooled/predictive MLP parameters by averaging and structural-similarity interpolation; no differentiated outer query through private learning in saved scope.',
        PFedEG='Repeated local KG learners share personalized supplementary entity embedding mixtures and proximity regularization; embedding-affinity recomputation is not itself a private-update meta-gradient.',
        boundary='These scopes add graph update-map and persistent client-sharing ancestry. No exact endpoint-rule match found in them, no global absence proof, no ensemble-specific sharing mechanism or supported successor. Frozen cohort/controls unchanged.',
        decision_notes='Counterfactual-credit algebra and the method decision are hashed optional analysis notes in SOURCE_BINDINGS; they add zero paper identities or primary/full-paper reads.')
    encoded = (json.dumps(d, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    save('SIZE_LIMIT_CHECK.json', dict(UTC=now, index_bytes=len(encoded), decimal_2MB_limit=MAX_INDEX_BYTES,
        within_limit=len(encoded) <= MAX_INDEX_BYTES, history_discarded=False, raw_texts_embedded=False, publisher_modified=False))
    assert len(encoded) <= MAX_INDEX_BYTES, (len(encoded), MAX_INDEX_BYTES)
    with (HERE / 'LITERATURE_INDEX.json').open('xb') as stream:
        stream.write(encoded)
    delta = dict(UTC=now, prospective=True, root_canonical_status_or_ledger_modified=False,
        predecessor_index=ref(PREV / 'LITERATURE_INDEX.json'), successor_index=ref(HERE / 'LITERATURE_INDEX.json'),
        before=old_metrics, after=totals, metric_deltas={k:totals[k]-old_metrics[k] for k in totals}, added_records=added,
        previously_completed_scoped_reads_adopted=3, integration_primary_reads=0, integration_primary_rereads=0,
        integration_retrievals=0, new_full_paper_reads=0, decision_notes_not_papers=len(decisions),
        scope_counts=dict(complete_subsections_or_sections=11, separate_overview_paragraphs=1,
            saved_paragraph_or_list_item_containers=41, counts_are_scope_containers_not_full_papers=True))
    save('DELTA.json', delta)
    save('VERIFICATION.json', dict(UTC=now, predecessor_manifest_sha256=PREV_MANIFEST_PIN,
        predecessor_index_sha256=PREV_INDEX_PIN, predecessor_records_preserved=238,
        predecessor_groups_catalog_source_events_and_linkages_preserved=True, all_other_predecessor_keys_preserved=True,
        source_packet_manifests_and_ledger_pins_verified=True, selected_text_hashes_or_manifest_custody_verified=True,
        deleted_full_HTML_not_independently_rehashed=True, source_inputs_unchanged=True,
        normalized_identity_and_scoped_key_deduplication=True, duplicate_candidate_keys_detectable=3,
        added_records=added, recomputed_metrics=totals, source_bindings_reference=ref(HERE / 'SOURCE_BINDINGS.json'),
        index_bytes=len(encoded), within_decimal_2MB=True, raw_texts_not_duplicated=True,
        integration_primary_reads=0, integration_primary_rereads=0, integration_retrievals=0, new_full_paper_reads=0,
        no_numeric_predictive_novelty_execution_adoption=True, canonical_status_or_ledger_modified=False))
    (HERE / 'ROOT_ADOPTION_NOTES.md').write_text(
        f'# Prospective literature index v{VERSION}\n\n'
        'Recommend root adoption after reviewing the sealed packet. Central canonical status and ledger remain unchanged.\n\n'
        '241 conclusion records cover 189 paper groups and two software groups. All 238 v63 records, identity groups, catalogs, source events, decision linkages and history are preserved. '
        'Exactly three previously completed bounded method scopes are appended: TMetaNet (arxiv:2506.00453v1), LPSFed (arxiv:2603.20338v1), and PFedEG (arxiv:2406.11943v1). '
        'Their versioned retrieval receipts, exact scope metadata and saved excerpt references are bound. Raw excerpts are not embedded; deleted original HTML is represented by saved receipt hashes, not a new raw-file integrity certification.\n\n'
        'The source packets recorded three bounded method reads comprising eleven sections/subsections and one overview paragraph, with 41 saved paragraph/list-item containers. '
        'This integration performs zero searches, primary retrievals, semantic primary rereads, full-paper reads or scientific execution. Paper-group counts are not full-paper read counts; prior uncertified cumulative reading flags remain unchanged.\n\n'
        'TMetaNet bounds learned graph update rates. LPSFed and PFedEG bound persistent private graph-client representation sharing through aggregation, interpolation, mixtures and local regularization. '
        'The saved scopes do not specify the exact endpoint-conditioned shared/private credit rule and do not support a successor operator or global novelty clearance. Frozen designs and controls remain unchanged.\n\n'
        'The counterfactual-credit synthesis and saved method decision are optional hashed analysis notes in SOURCE_BINDINGS. They add no paper record or reading credit. '
        'DELTA.json reports exact changes; VERIFICATION.json and the manifest/seal bind preservation and counts.\n')
    rows = [dict(path=f.name, bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest())
        for f in sorted(HERE.iterdir()) if f.is_file()]
    save('MANIFEST.json', dict(UTC=now, prospective=True, files=rows))
    save('SEAL.json', dict(manifest_sha256=ref(HERE / 'MANIFEST.json')['sha256'], payload_files=len(rows)))
    for f in HERE.iterdir():
        if f.is_file():
            f.chmod(0o444)
    print(json.dumps(dict(**totals, index_bytes=len(encoded), version=VERSION, status='PROSPECTIVE_READY_FOR_ROOT_REVIEW')))


if __name__ == '__main__':
    main()
