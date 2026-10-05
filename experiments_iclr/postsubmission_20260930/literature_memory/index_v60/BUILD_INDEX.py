"""Adopt two saved methods and three associated author-code scopes; no new reads."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
P = HERE.parent.parent
PREV = P / 'literature_memory/index_v59'
PACKET = P / 'endpoint_frame_function_preserving_prior_20261005_v1'
SMALL = P / 'small_representative_link_benchmark_scout_20261005_v1'
PREV_PIN = '70d7cf92483697229de4b9950bbfb07d7e0b48745ee326e8906bd8933a7d20c8'
PACKET_PIN = 'dfce499febd0af4431c5743eeda1de92c32e03373c38c7866ca61ae44e00d360'


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(P)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def save(name, value):
    with (HERE / name).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def verify_manifest(folder, pin, seal=False):
    assert ref(folder / 'MANIFEST.json')['sha256'] == pin
    if seal:
        assert json.loads((folder/'SEAL.json').read_text())['manifest_sha256'] == pin
    for row in json.loads((folder / 'MANIFEST.json').read_text())['files']:
        path = folder / row['path']
        assert path.resolve().is_relative_to(folder) and not path.is_symlink()
        got = ref(path)
        assert (got['bytes'], got['sha256']) == (row['bytes'], row['sha256'])


def ids(raw):
    return {re.sub(r'v\d+$', '', x.strip().lower()) for x in raw.split(';')}


def find_group(groups, identifier):
    matches = [g for g in groups if any(identifier in ids(x) for x in
        [g['normalized_identifier']] + g.get('raw_canonical_identifiers',[]) + g.get('explicit_aliases',[]))]
    assert len(matches) <= 1
    return matches[0] if matches else None


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
    verify_manifest(PREV, PREV_PIN, seal=True)
    verify_manifest(PACKET, PACKET_PIN)
    prior = json.loads((PREV/'LITERATURE_INDEX.json').read_text())
    scopes = json.loads((PACKET/'READ_SCOPES.json').read_text())
    conclusions = json.loads((PACKET/'PAPER_CONCLUSIONS.json').read_text())
    assert scopes['new_primary_method_scopes'] == 2 and scopes['new_full_paper_certifications'] == 0
    assert scopes['retained_source_incremental_initializer_read'] == 1
    assert metrics(prior)['conclusion_records'] == 228
    assert metrics(prior)['normalized_paper_identifiers'] == 176
    bindings = [ref(PACKET/'MANIFEST.json'),ref(PACKET/'READ_SCOPES.json'),ref(PACKET/'PAPER_CONCLUSIONS.json')]
    for s in scopes['new_primary_methods']:
        source = ref(PACKET/s['source'])
        assert (source['sha256'],source['bytes']) == (s['source_sha256'],s['source_bytes'])
        assert ref(PACKET/s['blocks_path'])['sha256'] == s['blocks_sha256']
        blocks = json.loads((PACKET/s['blocks_path']).read_text())
        passages = json.loads((PACKET/s['read_passages_path']).read_text())
        assert passages == [b for b in blocks if b['index'] in s['actual_fully_printed_indices']]
        assert s['full_paper_read'] is False and s['author_code_read'] is False
        assert s['reported_numeric_results_adopted'] is False and s['proof_audit'] is False
        bindings.extend([source,ref(PACKET/s['blocks_path']),ref(PACKET/s['read_passages_path'])])
    lora = json.loads((PACKET/'LORA_RETAINED_SOURCE_PASSAGES.json').read_text())
    lora_source = P/lora['source_path']
    assert ref(lora_source)['sha256'] == lora['source_sha256']
    lines = lora_source.read_text().splitlines()
    for passage in lora['passages']:
        lo,hi = passage['range']
        assert passage['text'] == '\n'.join(lines[lo-1:hi])
    bindings.extend([ref(PACKET/'LORA_RETAINED_SOURCE_PASSAGES.json'),ref(lora_source)])
    small_scopes = json.loads((SMALL/'SOURCE_LOCATORS_AND_READ_SCOPES.json').read_text())
    small_reused = json.loads((SMALL/'REUSED_SOURCE_BINDINGS.json').read_text())
    small_fetched = json.loads((SMALL/'PUBLIC_SOURCE_FETCH_RECEIPTS.json').read_text())
    assert small_scopes['new_primary_paper_reads'] == 0 and small_scopes['new_author_source_scopes'] == 2
    assert small_scopes['metadata_queries'] == small_scopes['numerical_runs'] == 0
    for row in small_reused['bindings']:
        path = P.parent/row['path']
        assert path.resolve().is_relative_to(P)
        got = ref(path)
        assert (got['bytes'],got['sha256']) == (row['bytes'],row['sha256'])
        bindings.append(got)
    for row in small_fetched['files']:
        path = P.parent/row['path']
        assert path.resolve().is_relative_to(SMALL)
        got = ref(path)
        assert (got['bytes'],got['sha256']) == (row['bytes'],row['sha256'])
        raw = path.read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == row['git_blob']
        assert ref(P.parent/row['saved_tree_path'])['sha256'] == row['saved_tree_sha256']
        bindings.append(got)
    bindings.extend(ref(SMALL/name) for name in ['REPORT.md','SOURCE_LOCATORS_AND_READ_SCOPES.json',
        'REUSED_SOURCE_BINDINGS.json','PUBLIC_SOURCE_FETCH_RECEIPTS.json'])
    save('SOURCE_BINDINGS.json',dict(UTC=now,files=bindings,integrity_checks_only=True,
        integration_semantic_rereads=0,primary_method_scopes_previously_completed=2,
        author_source_scopes_previously_completed=3,small_source_git_blobs_verified=3,
        primary_reported_results_not_adopted=True))
    d = copy.deepcopy(prior)
    changed = ['schema','created_UTC','latest_adoption','read_accounting','predecessor_index','predecessor_index_sha256']
    d['integration_v60_predecessor_v59_snapshot'] = {**{k:copy.deepcopy(prior[k]) for k in changed},
        'index_reference':ref(PREV/'LITERATURE_INDEX.json'),'manifest_reference':ref(PREV/'MANIFEST.json'),
        'seal_reference':ref(PREV/'SEAL.json')}
    d.update(schema='literature-memory-index-v60',created_UTC=now,
        predecessor_index=str((PREV/'LITERATURE_INDEX.json').relative_to(P)),
        predecessor_index_sha256=ref(PREV/'LITERATURE_INDEX.json')['sha256'])
    groups = d['canonical_identifier_normalization']['groups']
    added = []
    for position,s in enumerate(scopes['new_primary_methods']):
        conclusion_position = next(i for i,c in enumerate(conclusions) if c['canonical_id']==s['canonical_id'])
        conclusion = conclusions[conclusion_position]
        identifier = next(iter(ids(s['canonical_id'])))
        group = find_group(groups,identifier)
        prior_indices = list(group['record_indices']) if group else []
        scope_ref = {**ref(PACKET/'READ_SCOPES.json'),'selector':f'new_primary_methods/{position}'}
        conclusion_ref = ref(PACKET/'PAPER_CONCLUSIONS.json')
        payload = dict(canonical_id=identifier,exact_scope=s,scope_reference=scope_ref,
            conclusion_reference=conclusion_ref,conclusion_selector=str(conclusion_position))
        key = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert not any(r.get('scope_deduplication_key_sha256')==key for r in d['paper_records'])
        index = len(d['paper_records'])
        d['paper_records'].append(dict(canonical_id=s['canonical_id'],normalized_identifier=identifier,
            conclusion=copy.deepcopy(conclusion),conclusion_file=conclusion_ref['path'],
            conclusion_file_sha256=conclusion_ref['sha256'],conclusion_source_selector=str(conclusion_position),
            read_scope_reference=scope_ref,exact_read_scope=copy.deepcopy(s),
            source_packet=PACKET.name,source_packet_manifest_reference=ref(PACKET/'MANIFEST.json'),
            source_primary_reference=ref(PACKET/s['source']),source_read_passages_reference=ref(PACKET/s['read_passages_path']),
            scope_deduplication_key_sha256=key,scoped_method_read=True,full_paper_read=False,
            author_source_read=False,numeric_result_transfer=False,predictive_adoption=False,
            global_novelty_clearance=False,execution_authorized=False,integration_pass_new_primary_reads=0,
            integration_pass_primary_reread=False))
        new_identity = group is None
        if group is None:
            group = dict(normalized_identifier=identifier,kind='paper',raw_canonical_identifiers=[],explicit_aliases=[],record_indices=[])
            groups.append(group)
        if s['canonical_id'] not in group['raw_canonical_identifiers']:group['raw_canonical_identifiers'].append(s['canonical_id'])
        group['record_indices'].append(index)
        added.append(dict(canonical_id=identifier,versioned_canonical_id=s['canonical_id'],record_index=index,
            new_identity=new_identity,prior_record_indices=prior_indices,full_paper_read=False,scope_deduplication_key_sha256=key))
    events = []
    lora_group = find_group(groups,'arxiv:2405.14438')
    assert lora_group is not None
    events.append(dict(source_event_id='author-source:prs-eth/LoRA-Ensemble@'+lora['commit']+':initializer69-155_206-244',
        event_kind='retained_identity_targeted_author_source_initializer_upgrade',
        paper_identity_association=dict(canonical_id=conclusions[0]['canonical_id'],normalized_identifier='arxiv:2405.14438',
            existing_record_indices=copy.deepcopy(lora_group['record_indices'])),
        conclusion=copy.deepcopy(conclusions[0]),conclusion_reference={**ref(PACKET/'PAPER_CONCLUSIONS.json'),'selector':'0'},
        read_scope_reference=ref(PACKET/'LORA_RETAINED_SOURCE_PASSAGES.json'),source_primary_reference=ref(lora_source),
        compact_source_scopes=dict(repository=lora['repository'],commit=lora['commit'],
            line_ranges_inclusive=lora['printed_line_ranges_inclusive'],incremental_range=lora['source_initialization_extension_beyond_retained_range']),
        source_report_reference=ref(PACKET/'REPORT.md'),new_primary_paper_method_read=False,new_paper_identity=False,
        full_paper_read=False,source_execution=False,predictive_results_adopted=False,global_novelty_clearance=False,
        integration_pass_new_semantic_read=False,dataset_or_score_or_checkpoint_bytes_read=False))
    for name,identifier,versioned,scope_positions,selector,takeaway in [
        ('HeaRT','arxiv:2306.10453','arxiv:2306.10453v3',list(range(5)),'Native NCNC operation and selector',
         'Pinned Citeseer NCN/NCNC recipe uses complete native VALID-MRR selection, fixed endpoint-personalized hard pools, TRAIN topology and current-positive-minibatch removal. Native code evaluates TEST during training; any separated heldout scoring and validation-selected checkpoint custody are disclosed adaptations. Existing source claims are not reproduced numerical results.'),
        ('PENCIL','arxiv:2602.01553','arXiv:2602.01553v4',[5,6,7],'Strong singles and a recent competitor',
         'Pinned Citeseer config and Planetoid loader specify query-subgraph transformer ranking, fixed HeaRT pool mapping and TRAIN topology. Endpoint-token concatenation differs from NCNC coordinate products. Native source evaluates TEST during training; feasibility, competitive superiority and any score are not certified.'),
    ]:
        group = find_group(groups,identifier)
        assert group is not None
        rows = [r for r in small_fetched['files'] if name.lower() in r['repo'].lower()]
        assert rows
        events.append(dict(source_event_id='author-source:'+rows[0]['repo']+'@'+rows[0]['commit']+':Citeseer-scope',
            event_kind='retained_identity_small_link_benchmark_author_source_upgrade',
            paper_identity_association=dict(canonical_id=versioned,normalized_identifier=identifier,
                existing_record_indices=copy.deepcopy(group['record_indices'])),
            conclusion=dict(saved_takeaway=takeaway,read_status='bounded_pinned_author_source_upgrade_not_new_primary_paper_read'),
            conclusion_reference={**ref(SMALL/'REPORT.md'),'selector':selector},
            read_scope_reference={**ref(SMALL/'SOURCE_LOCATORS_AND_READ_SCOPES.json'),
                'selector':'source_scopes/'+','.join(str(i) for i in scope_positions)},
            compact_source_scopes=[copy.deepcopy(small_scopes['source_scopes'][i]) for i in scope_positions],
            fetched_source_references=[copy.deepcopy(row) for row in rows],
            reused_source_binding_reference=ref(SMALL/'REUSED_SOURCE_BINDINGS.json'),
            public_source_fetch_receipt_reference=ref(SMALL/'PUBLIC_SOURCE_FETCH_RECEIPTS.json'),
            source_report_reference=ref(SMALL/'REPORT.md'),new_primary_paper_method_read=False,new_paper_identity=False,
            full_paper_read=False,source_execution=False,predictive_results_adopted=False,global_novelty_clearance=False,
            integration_pass_new_semantic_read=False,dataset_or_score_or_checkpoint_bytes_read=False))
    assert len(events)==3 and len({e['source_event_id'] for e in events})==3
    assert 'public_author_source_scope_upgrades_v60' not in prior
    d['public_author_source_scope_upgrades_v60'] = events
    save('AUTHOR_SOURCE_SCHEMA_HANDLING.json',dict(UTC=now,
        precedent='Associated public_ddi_source_recipe_events in v59: source-only events have canonical associations and do not increment primary-paper records/groups.',
        successor_collection='public_author_source_scope_upgrades_v60',author_source_events=3,
        paper_record_mutations=0,new_paper_identities_from_source_events=0,new_primary_method_records_from_source_events=0,
        identities=[e['paper_identity_association'] for e in events],
        event_conclusions_retained=True,retained_existing_records_unchanged=True,
        author_source_upgrade_is_not_whole_repository_audit=True,integration_new_semantic_reads=0))
    catalog_paths = [PACKET/name for name in ['MANIFEST.json','READ_SCOPES.json','PAPER_CONCLUSIONS.json','REPORT.md',
        'LORA_RETAINED_SOURCE_PASSAGES.json','primary/ether.read_passages.json','primary/hra.read_passages.json']]
    catalog_paths += [SMALL/name for name in ['REPORT.md','SOURCE_LOCATORS_AND_READ_SCOPES.json',
        'REUSED_SOURCE_BINDINGS.json','PUBLIC_SOURCE_FETCH_RECEIPTS.json','source/heart_citeseer.sh',
        'source/pencil_heart_citeseer_bert.yaml','source/pencil_planetoid_dataset.py']]
    for path in catalog_paths:
        row = {**ref(path),'kind':'saved_scoped_method_or_associated_author_source_memory',
            'source_bindings_reference':ref(HERE/'SOURCE_BINDINGS.json'),
            'scope':'Two previously completed primary method scopes and three retained-identity author-source upgrades. Zero integration primary reads or numerical/novelty/predictive/execution adoption.'}
        assert not any(r['path']==row['path'] for r in d['existing_packets'])
        d['existing_packets'].append(row)
    groups.sort(key=lambda g:g['normalized_identifier'])
    assert d['paper_records'][:228]==prior['paper_records']
    assert d['existing_packets'][:len(prior['existing_packets'])]==prior['existing_packets']
    assert all(g in groups for g in prior['canonical_identifier_normalization']['groups'])
    for k in prior:
        if k not in changed+['paper_records','existing_packets','canonical_identifier_normalization']:assert d[k]==prior[k]
    totals = metrics(d)
    assert totals['conclusion_records']==230 and totals['normalized_paper_identifiers']==178
    assert all(a['new_identity'] for a in added)
    account = {k:copy.deepcopy(v) for k,v in prior['read_accounting'].items() if not k.startswith(('latest_','integration_pass_'))}
    account.update(totals,state='COMPLETED_FUNCTION_PRESERVING_METHOD_AND_AUTHOR_SOURCE_ADOPTION',
        historical_path_catalog_note='All 228 v59 records, groups, catalogs and failure history retained. Two saved method scopes appended; three author-source scopes recorded as retained-identity events. No integration primary or full reads.',
        latest_packet_new_scoped_primary_reads=2,latest_packet_full_primary_reads=0,
        latest_packet_previously_completed_scoped_read_adoptions=2,latest_packet_new_paper_identity_groups=2,
        latest_packet_previously_completed_author_source_scope_adoptions=3,
        latest_packet_new_primary_paper_reads_from_author_source_events=0,
        latest_packet_new_paper_identity_groups_from_author_source_events=0,
        integration_pass_new_primary_reads=0,integration_pass_full_primary_reads=0,
        integration_pass_primary_method_reads=0,integration_pass_author_source_semantic_reads=0,
        integration_pass_experimental_score_artifact_reads=0)
    d['read_accounting'] = account
    d['latest_adoption'] = dict(UTC=now,predecessor=ref(PREV/'LITERATURE_INDEX.json'),previous_records_preserved=True,
        added_records=added,source_completed_new_scoped_method_reads=2,source_full_paper_reads=0,integration_primary_reads=0,
        retained_identity_author_source_events=3,source_packet_manifest_reference=ref(PACKET/'MANIFEST.json'),
        source_binding_reference=ref(HERE/'SOURCE_BINDINGS.json'),novelty_or_numeric_or_predictive_or_execution_adoption=False)
    d['function_preserving_endpoint_frame_prior_limits_v60'] = dict(
        LoRA_Ensemble='Separate random A_m and zero B_m implement initially equal adapted maps with member-dependent derivatives. Function-preserving branches with different Jacobians are explicit ensemble prior; complete private-head predictions need not be equal.',
        ETHER='ETHER+ supplies identity at equal unit normals and normal-dependent first-order variations. Neutral reflection adapters and differing Jacobians are prior; actual equal-normal ensemble initialization is not certified.',
        HTA='Reflection-diagonal-reflection plus low-rank update is prior. The printed normalization caveat does not negate sandwich ancestry.',
        candidate='Exact pre-product graph-ensemble placement and the utility of private versus shared interaction frames are untested attributed adaptations, not cleared methodological novelty.',
        source_controls='HeaRT and PENCIL small-benchmark author-source scopes improve protocol/comparator readiness only; no predictive result, current-SOTA certification or run release follows.',
        excluded='Metadata queries, failed retrieval events and retained Laplace-LoRA initializer locator searches do not add primary method reads.')
    save('LITERATURE_INDEX.json',d)
    save('VERIFICATION.json',dict(UTC=now,predecessor_records_preserved=228,predecessor_groups_and_catalog_history_preserved=True,
        predecessor_manifest_sha256=PREV_PIN,prior_manifest_files_verified=True,saved_prior_packet_manifest_sha256=PACKET_PIN,
        saved_prior_packet_manifest_files_verified=True,source_bindings_reference=ref(HERE/'SOURCE_BINDINGS.json'),
        source_hash_bytes_and_passage_bindings_verified=True,small_fetched_git_blob_checks=3,
        added_primary_records=added,added_retained_identity_author_source_events=3,recomputed_metrics=totals,
        integration_primary_reads=0,integration_author_source_semantic_reads=0,new_full_paper_reads=0,
        primary_reported_results_not_adopted=True,no_numeric_predictive_novelty_execution_adoption=True))
    (HERE/'ROOT_ADOPTION_NOTES.md').write_text(
        '# Literature index v60\n\n'
        '230 conclusion records cover 178 paper groups and two software groups. All 228 v59 records, identifier groups, catalog entries and prior history are preserved. '
        'Exactly two previously completed primary-method scopes are appended: ETHER arXiv:2405.20271v1 and HTA arXiv:2410.22952v1. '
        'Three retained-identity code upgrades are associated events: LoRA-Ensemble initializer, HeaRT Citeseer recipe, and PENCIL Citeseer config/loader. '
        'Following the existing DDI source-event precedent, these events add zero primary-paper records or identities and preserve the prior paper records.\n\n'
        'Neutral adapters with different member Jacobians already have LoRA-Ensemble and ETHER+ ancestry. Reflection-diagonal-reflection sandwiches have HTA ancestry; '
        'the printed HTA normalization caveat does not negate that ancestry. Exact graph pre-product ensemble placement remains an untested adaptation, with no novelty clearance. '
        'Small-benchmark source scopes improve comparator and protocol readiness, without adopting numerical results or dispatching runs. '
        'Integration adds zero primary, author-source semantic or full-paper reads and no numerical, predictive, execution or novelty adoption.\n')
    rows = [dict(path=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(HERE.iterdir()) if f.is_file()]
    save('MANIFEST.json',dict(UTC=now,files=rows))
    save('SEAL.json',dict(manifest_sha256=ref(HERE/'MANIFEST.json')['sha256'],payload_files=len(rows)))
    for f in HERE.iterdir():
        if f.is_file():f.chmod(0o444)
    print(json.dumps(totals))


if __name__=='__main__':main()
