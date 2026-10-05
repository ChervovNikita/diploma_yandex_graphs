"""Adopt three saved scoped primary methods; no discovery or semantic reread."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
P = HERE.parent.parent
PREV = P / 'literature_memory/index_v58'
PACKET = P / 'graph_fast_factor_target_serving_discovery_20261005_v1'
PREDECESSOR_MANIFEST_PIN = 'be4398d50a0261542497cad919af91a8b2b6f3ee270410fcac070fc36c6ce39c'
SOURCE_PINS = {
    'arxiv:2501.10062': '5e84a077405fb2076a1568075af4ce38d9fdc263df40b428550e03e4fdb195ff',
    'arxiv:2202.07919': 'b9887380f8246466deff43ec6301024336a5229a7cd3314490d5166ec24e16f2',
    'arxiv:2405.08540': 'bf2a95f27b5082fbb73e1e04214baf74de34d09602431215cca574519e7dc1c2',
}


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(P)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def save(name, value):
    with (HERE / name).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def verify_predecessor():
    observed = ref(PREV / 'MANIFEST.json')['sha256']
    assert observed == PREDECESSOR_MANIFEST_PIN
    assert json.loads((PREV / 'SEAL.json').read_text())['manifest_sha256'] == observed
    for row in json.loads((PREV / 'MANIFEST.json').read_text())['files']:
        path = PREV / row['path']
        assert path.resolve().is_relative_to(PREV) and not path.is_symlink()
        got = ref(path)
        assert (got['bytes'], got['sha256']) == (row['bytes'], row['sha256'])


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


def main():
    now = datetime.now(timezone.utc).isoformat()
    verify_predecessor()
    prior = json.loads((PREV / 'LITERATURE_INDEX.json').read_text())
    scopes = json.loads((PACKET / 'READ_SCOPES.json').read_text())
    conclusions = json.loads((PACKET / 'PAPER_CONCLUSIONS.json').read_text())
    assert scopes['primary_method_scopes'] == 3 and scopes['full_papers'] == 0
    assert scopes['metadata_queries_not_reads'] is True
    assert len(scopes['scopes']) == len(conclusions) == 3
    assert metrics(prior)['conclusion_records'] == 225
    assert metrics(prior)['normalized_paper_identifiers'] == 173
    source_rows = [ref(PACKET / 'READ_SCOPES.json'), ref(PACKET / 'PAPER_CONCLUSIONS.json')]
    for scope in scopes['scopes']:
        identifier = next(iter(ids(scope['canonical_id'])))
        assert identifier in SOURCE_PINS
        source = PACKET / scope['source_path']
        got = ref(source)
        assert got['sha256'] == scope['source_sha256'] == SOURCE_PINS[identifier]
        assert got['bytes'] == scope['source_bytes']
        source_rows.extend([got, ref(PACKET / scope['blocks_path']),
                            ref(PACKET / scope['blocks_path'].replace('.blocks.json', '.read_passages.json'))])
        blocks = json.loads((PACKET / scope['blocks_path']).read_text())
        passages = json.loads((PACKET / scope['blocks_path'].replace('.blocks.json', '.read_passages.json')).read_text())
        expected = [row for row in blocks if any(lo <= row['i'] <= hi for lo, hi in scope['block_index_ranges_inclusive'])]
        assert passages == expected
        assert set(scope['exact_ids']).issubset({row['id'] for row in passages})
        assert scope['full_paper_certification'] is False and scope['author_code_read'] is False
    save('SOURCE_BINDINGS.json', dict(UTC=now, files=source_rows, integrity_checks_only=True,
        packet_has_no_manifest_or_seal=True, source_semantic_rereads=0, source_scoped_reads_previously_completed=3,
        full_paper_read_certifications=0, metadata_only_HTKGE_excluded=True))
    d = copy.deepcopy(prior)
    changed = ['schema', 'created_UTC', 'latest_adoption', 'read_accounting', 'predecessor_index', 'predecessor_index_sha256']
    d['integration_v59_predecessor_v58_snapshot'] = {
        **{k:copy.deepcopy(prior[k]) for k in changed}, 'index_reference':ref(PREV / 'LITERATURE_INDEX.json'),
        'manifest_reference':ref(PREV / 'MANIFEST.json'), 'seal_reference':ref(PREV / 'SEAL.json')}
    d.update(schema='literature-memory-index-v59', created_UTC=now,
             predecessor_index=str((PREV / 'LITERATURE_INDEX.json').relative_to(P)),
             predecessor_index_sha256=ref(PREV / 'LITERATURE_INDEX.json')['sha256'])
    added = []
    groups = d['canonical_identifier_normalization']['groups']
    for position, (exact, conclusion) in enumerate(zip(scopes['scopes'], conclusions)):
        assert conclusion['canonical_id'] == exact['canonical_id']
        identifier = next(iter(ids(exact['canonical_id'])))
        assert conclusion['global_novelty_clearance'] is False and conclusion['predictive_results_adopted'] is False
        scope_ref = {**ref(PACKET / 'READ_SCOPES.json'), 'selector':f'scopes/{position}'}
        conclusion_ref = ref(PACKET / 'PAPER_CONCLUSIONS.json')
        payload = dict(canonical_id=identifier, exact_scope=exact, scope_reference=scope_ref,
                       conclusion_reference=conclusion_ref, conclusion_selector=str(position))
        key = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert not any(r.get('scope_deduplication_key_sha256')==key for r in d['paper_records'])
        matching = [g for g in groups if any(identifier in ids(x) for x in
                    [g['normalized_identifier']] + g.get('raw_canonical_identifiers',[]) + g.get('explicit_aliases',[]))]
        assert len(matching) <= 1
        index = len(d['paper_records'])
        d['paper_records'].append(dict(canonical_id=exact['canonical_id'], normalized_identifier=identifier,
            conclusion=copy.deepcopy(conclusion),
            conclusion_file=conclusion_ref['path'], conclusion_file_sha256=conclusion_ref['sha256'],
            conclusion_source_selector=str(position), read_scope_reference=scope_ref, exact_read_scope=copy.deepcopy(exact),
            source_packet=PACKET.name, source_packet_binding_reference=ref(HERE/'SOURCE_BINDINGS.json'),
            source_primary_reference=ref(PACKET/exact['source_path']),
            source_read_passages_reference=ref(PACKET/exact['blocks_path'].replace('.blocks.json','.read_passages.json')),
            scope_deduplication_key_sha256=key, scoped_method_read=True, full_paper_read=False,
            author_source_read=False, numeric_result_transfer=False, predictive_adoption=False,
            global_novelty_clearance=False, execution_authorized=False, integration_pass_new_primary_reads=0,
            integration_pass_primary_reread=False))
        group = matching[0] if matching else dict(normalized_identifier=identifier,kind='paper',raw_canonical_identifiers=[],explicit_aliases=[],record_indices=[])
        if not matching: groups.append(group)
        if exact['canonical_id'] not in group['raw_canonical_identifiers']:group['raw_canonical_identifiers'].append(exact['canonical_id'])
        group['record_indices'].append(index)
        added.append(dict(canonical_id=identifier,versioned_canonical_id=exact['canonical_id'],record_index=index,new_identity=not matching,
                          full_paper_read=False,scope_deduplication_key_sha256=key))
    names = ['PAPER_CONCLUSIONS.json','READ_SCOPES.json'] + [s['blocks_path'].replace('.blocks.json','.read_passages.json') for s in scopes['scopes']]
    for name in names:
        row = {**ref(PACKET/name),'kind':'saved_scoped_endpoint_transform_method_memory',
               'source_bindings_reference':ref(HERE/'SOURCE_BINDINGS.json'),
               'scope':'Three previously completed bounded method scopes; zero integration primary reads or numeric/novelty/predictive/execution adoption'}
        assert not any(r['path']==row['path'] for r in d['existing_packets'])
        d['existing_packets'].append(row)
    groups.sort(key=lambda g:g['normalized_identifier'])
    assert d['paper_records'][:225]==prior['paper_records']
    assert d['existing_packets'][:len(prior['existing_packets'])]==prior['existing_packets']
    assert all(g in groups for g in prior['canonical_identifier_normalization']['groups'])
    for k in prior:
        if k not in changed+['paper_records','existing_packets','canonical_identifier_normalization']:assert d[k]==prior[k]
    totals = metrics(d)
    assert totals['conclusion_records']==228 and totals['normalized_paper_identifiers']==176
    assert all(a['new_identity'] for a in added)
    account = {k:copy.deepcopy(v) for k,v in prior['read_accounting'].items() if not k.startswith(('latest_','integration_pass_'))}
    account.update(totals, state='COMPLETED_ENDPOINT_TRANSFORM_SCOPED_METHOD_ADOPTION',
                   historical_path_catalog_note='All 225 v58 records, groups, catalogs and failure history retained. Three saved method scopes appended; no integration primary or full reads.',
                   latest_packet_new_scoped_primary_reads=3,latest_packet_full_primary_reads=0,
                   latest_packet_previously_completed_scoped_read_adoptions=3,latest_packet_new_paper_identity_groups=3,
                   integration_pass_new_primary_reads=0,integration_pass_full_primary_reads=0,
                   integration_pass_primary_method_reads=0,integration_pass_experimental_score_artifact_reads=0)
    d['read_accounting'] = account
    d['latest_adoption'] = dict(UTC=now,predecessor=ref(PREV/'LITERATURE_INDEX.json'),previous_records_preserved=True,
        added_records=added,source_completed_new_scoped_method_reads=3,source_full_paper_reads=0,integration_primary_reads=0,
        source_packet_binding_reference=ref(HERE/'SOURCE_BINDINGS.json'),source_separate_manifest_or_seal_present=False,
        metadata_only_HTKGE_excluded=True,novelty_or_numeric_or_predictive_or_execution_adoption=False)
    d['endpoint_frame_prior_limits_v59'] = dict(
        OMoE='Per-token inter-expert LoRA output orthogonalization and learned routing are prior; its displayed unnormalized Gram–Schmidt expression is not a qualified transferable implementation.',
        HousE='Graph endpoint Householder reflection products, head/tail relation projections and compact vector computation are direct prior.',
        GoldE_UOP='Generalized quadratic-metric Householder graph transforms include normalized nonunit axes and efficient vector operations; the orthogonal parameterization is prior.',
        candidate='Private versus shared interaction frames before the NCNC outer endpoint product is a restricted attributed adaptation, not a new reflection principle or proven whole-model expressivity separation.',
        unadopted='Metadata queries and the HTKGE abstract-only publisher lead are not primary method reads, numerical evidence or novelty clearance.')
    save('LITERATURE_INDEX.json',d)
    save('VERIFICATION.json',dict(UTC=now,predecessor_records_preserved=225,predecessor_groups_and_catalog_history_preserved=True,
        predecessor_manifest_sha256=PREDECESSOR_MANIFEST_PIN,predecessor_manifest_files_verified=True,
        source_bindings_reference=ref(HERE/'SOURCE_BINDINGS.json'),source_sha_and_bytes_verified=True,
        added_records=added,recomputed_metrics=totals,integration_primary_reads=0,new_full_paper_reads=0,
        scoped_not_full_flags_verified=True,metadata_only_HTKGE_excluded=True,no_numeric_predictive_novelty_execution_adoption=True))
    (HERE/'ROOT_ADOPTION_NOTES.md').write_text(
        '# Literature index v59\n\n'
        '228 scoped conclusion records cover 176 paper groups and two software groups. All 225 v58 records, identifier groups, catalog entries and prior history are preserved. '
        'Three previously completed method scopes are adopted: OMoE arXiv:2501.10062v1, HousE arXiv:2202.07919v1, and GoldE/UOP arXiv:2405.08540v1. '
        'They add three identities, not three certified full-paper reads. This integration adds zero primary reads or numerical, predictive, execution or novelty adoption.\n\n'
        'Orthogonal expert responses are established OMoE ancestry. Compact Householder graph endpoint transformations are direct HousE ancestry. '
        'Generalized quadratic-metric graph transforms and normalized nonunit-axis Euclidean reflection operations are established GoldE/UOP ancestry. '
        'A member-private frame before endpoint product compression is only a restricted attributed adaptation requiring competent single and ordinary ensemble controls. '
        'Metadata queries and the HTKGE abstract-only publisher lead remain outside primary method read accounting.\n')
    rows = [dict(path=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(HERE.iterdir()) if f.is_file()]
    save('MANIFEST.json',dict(UTC=now,files=rows))
    save('SEAL.json',dict(manifest_sha256=ref(HERE/'MANIFEST.json')['sha256'],payload_files=len(rows)))
    for f in HERE.iterdir():
        if f.is_file():f.chmod(0o444)
    print(json.dumps(totals))


if __name__=='__main__':main()
