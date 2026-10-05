"""Adopt two saved train-only transfer scopes; no discovery or primary reread."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
P = HERE.parent.parent
PREV = P / 'literature_memory/index_v62'
PACKET = P / 'shared_core_train_only_meta_prior_root_20261005_v1'
PREV_PIN = '86990b2af8de3717f5f411379a335eb6674c32a66b2b6125b97ac44a5f19aa6c'
CONCLUSIONS_PIN = '21414bd48597b06b376d3bfee961e5c86edc93e3e8f9cad14758f1736074dd60'
CANONICAL = ('NeurIPS2018:647bba344396e7c8170902bcf2e15551', 'arxiv:1710.03463v1')
MAX_INDEX_BYTES = 2000000


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(P)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def save(name,value):
    with (HERE/name).open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')


def ids(raw):
    return {re.sub(r'v\d+$','',x.strip().lower()) for x in raw.split(';')}


def metrics(d):
    cat = {r['path'] for r in d['existing_packets']}
    con = {r['conclusion_file'] for r in d['paper_records'] if isinstance(r.get('conclusion_file'),str)}
    scopes = {r['read_scope_reference']['path'] for r in d['paper_records'] if 'read_scope_reference' in r}
    legacy = {r['read_scope_file_reference']['path'] for r in d['paper_records'] if 'read_scope_file_reference' in r}
    groups = d['canonical_identifier_normalization']['groups']
    return dict(conclusion_records=len(d['paper_records']),normalized_paper_identifiers=sum(g['kind']=='paper' for g in groups),
        software_documentation_identifiers=sum(g['kind']!='paper' for g in groups),catalog_entries=len(d['existing_packets']),
        unique_catalog_document_paths=len(cat),unique_conclusion_source_documents=len(con),
        unique_referenced_document_paths=len(cat|con),unique_scope_reference_document_paths=len(scopes),
        unique_scope_reference_document_paths_including_legacy_field_alias=len(scopes|legacy))


def main():
    now = datetime.now(timezone.utc).isoformat()
    assert ref(PREV/'MANIFEST.json')['sha256']==PREV_PIN
    assert json.loads((PREV/'SEAL.json').read_text())['manifest_sha256']==PREV_PIN
    for row in json.loads((PREV/'MANIFEST.json').read_text())['files']:
        path = PREV/row['path']
        assert path.resolve().is_relative_to(PREV) and not path.is_symlink()
        got = ref(path)
        assert (got['bytes'],got['sha256'])==(row['bytes'],row['sha256'])
    prior = json.loads((PREV/'LITERATURE_INDEX.json').read_text())
    assert metrics(prior)['conclusion_records']==236 and metrics(prior)['normalized_paper_identifiers']==184
    assert ref(PACKET/'SCOPED_CONCLUSIONS.json')['sha256']==CONCLUSIONS_PIN
    scopes = json.loads((PACKET/'SCOPED_CONCLUSIONS.json').read_text())
    assert scopes['new_scoped_primary_method_reads']==2 and scopes['full_paper_reads']==0
    assert scopes['report_sha256']==ref(PACKET/'REPORT.md')['sha256']
    assert len(scopes['records'])==2
    bindings = [ref(PACKET/name) for name in ['SCOPED_CONCLUSIONS.json','REPORT.md','RETRIEVAL_RECEIPTS.json',
        'METAREG_ABSTRACT_RETRIEVAL.json','METAREG_FULLTEXT_RETRIEVAL.json','MLDG_PRIMARY_RETRIEVAL.json']]
    for scope in scopes['records']:
        primary, text = ref(PACKET/scope['source']), ref(PACKET/scope['extracted'])
        assert primary['sha256']==scope['source_sha256'] and text['sha256']==scope['extracted_sha256']
        assert scope['full_paper_read'] is False and scope['experiments_verified'] is False and scope['outcome_access'] is False
        bindings.extend([primary,text])
    for name,position in [('METAREG_FULLTEXT_RETRIEVAL.json',0),('MLDG_PRIMARY_RETRIEVAL.json',1)]:
        receipt = json.loads((PACKET/name).read_text())
        scope = scopes['records'][position]
        assert receipt['sha256']==scope['source_sha256'] and receipt['text_sha256']==scope['extracted_sha256']
        assert receipt['bytes']==ref(PACKET/scope['source'])['bytes']
    discovery = json.loads((PACKET/'RETRIEVAL_RECEIPTS.json').read_text())
    assert discovery['new_primary_method_scopes_completed']==0
    for row in discovery['requests']:
        if 'path' in row:
            got = ref(PACKET/row['path'])
            assert got['sha256']==row['sha256'] and got['bytes']==row['bytes']
            bindings.append(got)
    save('SOURCE_BINDINGS.json',dict(UTC=now,files=bindings,integrity_checks_only=True,
        integration_semantic_reads=0,previously_completed_scoped_primary_method_reads=2,
        source_packet_has_no_manifest_or_seal=True,metadata_index_abstract_and_failed_discovery_not_read_credit=True,
        full_paper_certifications=0,raw_PDF_or_extracted_text_not_embedded_in_index=True,
        inferred_MetaReg_arxiv_or_DOI_alias=False))
    d = copy.deepcopy(prior)
    changed = ['schema','created_UTC','latest_adoption','read_accounting','predecessor_index','predecessor_index_sha256']
    d['integration_v63_predecessor_v62_snapshot'] = {**{k:copy.deepcopy(prior[k]) for k in changed},
        'index_reference':ref(PREV/'LITERATURE_INDEX.json'),'manifest_reference':ref(PREV/'MANIFEST.json'),
        'seal_reference':ref(PREV/'SEAL.json')}
    d.update(schema='literature-memory-index-v63',created_UTC=now,
        predecessor_index=str((PREV/'LITERATURE_INDEX.json').relative_to(P)),
        predecessor_index_sha256=ref(PREV/'LITERATURE_INDEX.json')['sha256'])
    groups = d['canonical_identifier_normalization']['groups']
    added = []
    for position,(scope,canonical) in enumerate(zip(scopes['records'],CANONICAL)):
        identifier = next(iter(ids(canonical)))
        matching = [g for g in groups if any(identifier in ids(x) for x in
            [g['normalized_identifier']]+g.get('raw_canonical_identifiers',[])+g.get('explicit_aliases',[]))]
        assert len(matching)<=1
        # Identity/title/source checks are explicit; no cross-scheme alias is inferred.
        title_matches = [i for i,r in enumerate(prior['paper_records']) if
            r.get('conclusion',{}).get('verified_title','').strip().lower()==scope['paper'].strip().lower()]
        source_matches = [i for i,r in enumerate(prior['paper_records']) if
            r.get('source_primary_reference',{}).get('sha256')==scope['source_sha256']]
        assert not title_matches and not source_matches
        scope_ref = {**ref(PACKET/'SCOPED_CONCLUSIONS.json'),'selector':f'records/{position}'}
        payload = dict(canonical_id=identifier,exact_scope=scope,scope_reference=scope_ref)
        key = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert not any(r.get('scope_deduplication_key_sha256')==key for r in d['paper_records'])
        index = len(d['paper_records'])
        d['paper_records'].append(dict(canonical_id=canonical,normalized_identifier=identifier,
            conclusion=dict(canonical_id=canonical,verified_title=scope['paper'],saved_takeaway=scope['conclusion'],
                read_status='Previously completed scoped primary method; not full paper',full_paper_read=False,
                numeric_results_adopted=False,global_novelty_clearance=False),
            conclusion_file=scope_ref['path'],conclusion_file_sha256=scope_ref['sha256'],conclusion_source_selector=f'records/{position}',
            read_scope_reference=scope_ref,exact_read_scope=copy.deepcopy(scope),source_packet=PACKET.name,
            source_packet_binding_reference=ref(HERE/'SOURCE_BINDINGS.json'),
            source_primary_reference=ref(PACKET/scope['source']),source_extracted_text_reference=ref(PACKET/scope['extracted']),
            scope_deduplication_key_sha256=key,scoped_method_read=True,full_paper_read=False,
            author_source_read=False,numeric_result_transfer=False,predictive_adoption=False,
            global_novelty_clearance=False,execution_authorized=False,integration_pass_new_primary_reads=0,
            integration_pass_primary_reread=False))
        group = matching[0] if matching else dict(normalized_identifier=identifier,kind='paper',
            raw_canonical_identifiers=[],explicit_aliases=[],record_indices=[])
        if not matching:groups.append(group)
        if canonical not in group['raw_canonical_identifiers']:group['raw_canonical_identifiers'].append(canonical)
        group['record_indices'].append(index)
        added.append(dict(canonical_id=identifier,exact_canonical_id=canonical,record_index=index,new_identity=not matching,
            full_paper_read=False,scope_deduplication_key_sha256=key))
    for name in ['SCOPED_CONCLUSIONS.json','REPORT.md','RETRIEVAL_RECEIPTS.json','METAREG_ABSTRACT_RETRIEVAL.json',
                 'METAREG_FULLTEXT_RETRIEVAL.json','MLDG_PRIMARY_RETRIEVAL.json']:
        row = {**ref(PACKET/name),'kind':'saved_scoped_train_only_transfer_prior_and_discovery_custody',
            'source_bindings_reference':ref(HERE/'SOURCE_BINDINGS.json'),
            'scope':'Two previously completed method scopes; metadata/index/abstract/failure custody is not reading credit. Zero integration retrieval/primary/full-paper reads or numerical/novelty/predictive/execution adoption.'}
        assert not any(r['path']==row['path'] for r in d['existing_packets'])
        d['existing_packets'].append(row)
    groups.sort(key=lambda g:g['normalized_identifier'])
    assert d['paper_records'][:236]==prior['paper_records']
    assert d['existing_packets'][:len(prior['existing_packets'])]==prior['existing_packets']
    assert all(g in groups for g in prior['canonical_identifier_normalization']['groups'])
    for k in prior:
        if k not in changed+['paper_records','existing_packets','canonical_identifier_normalization']:assert d[k]==prior[k]
    totals = metrics(d)
    assert totals['conclusion_records']==238 and totals['normalized_paper_identifiers']==186
    assert all(a['new_identity'] for a in added)
    account = {k:copy.deepcopy(v) for k,v in prior['read_accounting'].items() if not k.startswith(('latest_','integration_pass_'))}
    account.update(totals,state='COMPLETED_TRAIN_ONLY_TRANSFER_PRIOR_ADOPTION',
        historical_path_catalog_note='All 236 v62 records, groups, catalogs, source events, proposal/decision linkages and history retained. Two saved scoped methods appended; no integration primary or full reads.',
        latest_packet_new_scoped_primary_reads=2,latest_packet_full_primary_reads=0,
        latest_packet_previously_completed_scoped_read_adoptions=2,latest_packet_new_paper_identity_groups=2,
        integration_pass_new_primary_reads=0,integration_pass_full_primary_reads=0,
        integration_pass_primary_method_reads=0,integration_pass_author_source_semantic_reads=0,
        integration_pass_experimental_score_artifact_reads=0,integration_pass_retrievals=0)
    d['read_accounting'] = account
    d['latest_adoption'] = dict(UTC=now,predecessor=ref(PREV/'LITERATURE_INDEX.json'),previous_records_preserved=True,
        added_records=added,source_completed_new_scoped_method_reads=2,source_full_paper_reads=0,integration_primary_reads=0,
        integration_retrievals=0,source_binding_reference=ref(HERE/'SOURCE_BINDINGS.json'),
        novelty_or_numeric_or_predictive_or_execution_adoption=False)
    d['train_only_shared_core_meta_prior_limits_v63'] = dict(
        MetaReg='Shared features, persistent source-specific heads and differentiable virtual private updates are prior. Feature freezing and final fresh single differ from the candidate; no-inference-adaptation is already established.',
        MLDG='Train-only transfer optimization, deployment without adaptation and gradient-agreement interpretation are direct prior. An extra derivative through an update is not a new learning principle.',
        graph_candidate='Endpoint-separated correlated queries, served mean-logit/member-competence objective and recomputed persistent weights are unproved recipe differences requiring adapted-single, untied-ensemble, detached-credit and matched-random-separation controls.',
        limits='No unbiased cross-fitting, independent task distribution, novelty, native-Adam, ranking/generalization or predictive-superiority guarantee. Native sparse derivative failure is an implementation limit, not disproof of the hypothesis.')
    encoded = (json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    save('SIZE_LIMIT_CHECK.json',dict(UTC=now,index_bytes=len(encoded),decimal_2MB_limit=MAX_INDEX_BYTES,
        within_limit=len(encoded)<=MAX_INDEX_BYTES,history_discarded=False,raw_texts_embedded=False,publisher_modified=False))
    if len(encoded)>MAX_INDEX_BYTES:
        print(json.dumps(dict(status='INDEX_EXCEEDS_2MB_NO_HISTORY_DISCARDED',index_bytes=len(encoded),limit=MAX_INDEX_BYTES)))
        return
    with (HERE/'LITERATURE_INDEX.json').open('xb') as stream:stream.write(encoded)
    save('VERIFICATION.json',dict(UTC=now,predecessor_records_preserved=236,
        predecessor_groups_catalog_source_events_and_linkages_preserved=True,predecessor_manifest_sha256=PREV_PIN,
        scoped_conclusions_pin_verified=True,PDF_and_text_hashes_and_retrieval_byte_counts_verified=True,
        source_bindings_reference=ref(HERE/'SOURCE_BINDINGS.json'),added_records=added,recomputed_metrics=totals,
        index_bytes=len(encoded),within_decimal_2MB=True,raw_texts_not_duplicated=True,
        integration_primary_reads=0,integration_retrievals=0,new_full_paper_reads=0,
        no_numeric_predictive_novelty_execution_adoption=True,inferred_MetaReg_arxiv_or_DOI_alias=False))
    (HERE/'ROOT_ADOPTION_NOTES.md').write_text(
        '# Literature index v63\n\n'
        '238 conclusion records cover 186 paper groups and two software groups. All 236 v62 records, identifier groups, catalog entries, source events, proposal/decision linkages and history are preserved. '
        'Exactly two previously completed scopes are appended: the exact NeurIPS 2018 MetaReg publisher identity and MLDG arxiv:1710.03463v1. '
        'PDF and extracted-text hashes are bound without embedding raw text. No inferred MetaReg arXiv/DOI alias, integration retrieval, primary/full-paper read or numerical/predictive/novelty/execution adoption follows.\n\n'
        'Train-only transfer, deployment without adaptation, persistent source heads and differentiated virtual updates have direct MetaReg/MLDG ancestry. '
        'The graph-specific endpoint construction, served ensemble objective and recomputed committed route weights remain unproved recipe differences. '
        'They require adapted-single, untied-ensemble, detached-credit and matched-random-separation controls. No graph-task independence or gradient-transfer novelty is claimed.\n')
    rows = [dict(path=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(HERE.iterdir()) if f.is_file()]
    save('MANIFEST.json',dict(UTC=now,files=rows))
    save('SEAL.json',dict(manifest_sha256=ref(HERE/'MANIFEST.json')['sha256'],payload_files=len(rows)))
    for f in HERE.iterdir():
        if f.is_file():f.chmod(0o444)
    print(json.dumps(dict(**totals,index_bytes=len(encoded))))


if __name__=='__main__':main()
