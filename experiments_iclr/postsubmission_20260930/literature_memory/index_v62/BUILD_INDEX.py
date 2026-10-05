"""Adopt three saved meta-learning scopes and a bounded proposal/decision link."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
P = HERE.parent.parent
PREV = P / 'literature_memory/index_v61'
PACKET = P / 'shared_backbone_quality_next_hypothesis_20261005_v1'
PREV_PIN = '742a9e36d1ba00469f5f4c3fc2e953ef04161a537498abfca9250eee7d9633b4'
SCOPES_PIN = 'a5fceb31b725e1ef813756b835eef338762c5252690f7f5221601bfac0236b77'
CONCLUSIONS_PIN = '4aa66dca222e1a7b804ac5c315d3f33a381118e5db936069618cf8c8d7888e40'


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
    assert metrics(prior)['conclusion_records']==233 and metrics(prior)['normalized_paper_identifiers']==181
    assert ref(PACKET/'READ_SCOPES.json')['sha256']==SCOPES_PIN
    assert ref(PACKET/'PAPER_CONCLUSIONS.json')['sha256']==CONCLUSIONS_PIN
    scopes = json.loads((PACKET/'READ_SCOPES.json').read_text())
    conclusions = json.loads((PACKET/'PAPER_CONCLUSIONS.json').read_text())
    triage = json.loads((PACKET/'ROOT_TRIAGE_DECISION.json').read_text())
    assert scopes['new_primary_method_scopes']==3 and scopes['full_paper_reads']==0
    assert conclusions['global_novelty_clearance'] is False and conclusions['predictive_utility_established'] is False
    assert conclusions['prototype_admitted'] is False and conclusions['numerical_model_execution'] is False
    assert triage['report_sha256']==ref(PACKET/'REPORT.md')['sha256']
    assert triage['outcome_or_TEST_access'] is False and triage['fit_count']==0
    assert scopes['outcome_access'] is False and scopes['server_actions'] is False
    assert len(scopes['scopes'])==len(conclusions['new_primary_method_scopes'])==3
    bindings = [ref(PACKET/name) for name in ['READ_SCOPES.json','PAPER_CONCLUSIONS.json','RETRIEVAL_RECEIPTS.json',
        'REPORT.md','ROOT_TRIAGE_DECISION.json','REUSED_CONCLUSIONS.json']]
    for scope in scopes['scopes']:
        got = ref(PACKET/scope['source'])
        assert got['sha256']==scope['sha256']
        assert scope['whole_paper_read'] is False and scope['author_code_read'] is False and scope['results_adopted'] is False
        bindings.append(got)
        stem = scope['key']
        blockpath = PACKET/'sources'/f'{stem}_blocks.json'
        excerptpath = PACKET/'sources'/f'{stem}_selected_method.json'
        blocks = json.loads(blockpath.read_text())
        excerpt = json.loads(excerptpath.read_text())
        assert excerpt['blocks']==[dict(index=i,**blocks[i]) for i in excerpt['selected_indices']]
        if stem=='bmaml':
            assert excerpt['shared_parameter_paragraphs']==[dict(index=47,**blocks[47])]
            assert excerpt['emaml_definition_paragraphs']==[dict(index=43,**blocks[43])]
        bindings.extend([ref(blockpath),ref(excerptpath)])
    for row in json.loads((PACKET/'RETRIEVAL_RECEIPTS.json').read_text()):
        got = ref(PACKET/row['path'])
        assert got['sha256']==row['sha256']
        if 'bytes' in row:assert got['bytes']==row['bytes']
        assert row.get('retrieval_only') is True or row.get('method_read') is False
        bindings.append(got)
    save('SOURCE_BINDINGS.json',dict(UTC=now,files=bindings,integrity_checks_only=True,
        integration_semantic_reads=0,previously_completed_primary_method_scopes=3,full_paper_certifications=0,
        source_packet_has_no_manifest_or_seal=True,metadata_retrieval_not_primary_read_credit=True,
        proposal_and_root_triage_linkage_is_not_paper_read=True,reported_results_not_adopted=True))
    d = copy.deepcopy(prior)
    changed = ['schema','created_UTC','latest_adoption','read_accounting','predecessor_index','predecessor_index_sha256']
    d['integration_v62_predecessor_v61_snapshot'] = {**{k:copy.deepcopy(prior[k]) for k in changed},
        'index_reference':ref(PREV/'LITERATURE_INDEX.json'),'manifest_reference':ref(PREV/'MANIFEST.json'),
        'seal_reference':ref(PREV/'SEAL.json')}
    d.update(schema='literature-memory-index-v62',created_UTC=now,
        predecessor_index=str((PREV/'LITERATURE_INDEX.json').relative_to(P)),
        predecessor_index_sha256=ref(PREV/'LITERATURE_INDEX.json')['sha256'])
    groups = d['canonical_identifier_normalization']['groups']
    added = []
    for position,scope in enumerate(scopes['scopes']):
        conclusion_position = next(i for i,c in enumerate(conclusions['new_primary_method_scopes']) if c['canonical_id']==scope['canonical_id'])
        conclusion = conclusions['new_primary_method_scopes'][conclusion_position]
        identifier = next(iter(ids(scope['canonical_id'])))
        matching = [g for g in groups if any(identifier in ids(x) for x in
            [g['normalized_identifier']]+g.get('raw_canonical_identifiers',[])+g.get('explicit_aliases',[]))]
        assert len(matching)<=1
        scope_ref = {**ref(PACKET/'READ_SCOPES.json'),'selector':f'scopes/{position}'}
        conclusion_ref = ref(PACKET/'PAPER_CONCLUSIONS.json')
        payload = dict(canonical_id=identifier,exact_scope=scope,scope_reference=scope_ref,
            conclusion_reference=conclusion_ref,conclusion_selector=f'new_primary_method_scopes/{conclusion_position}')
        key = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert not any(r.get('scope_deduplication_key_sha256')==key for r in d['paper_records'])
        index = len(d['paper_records'])
        d['paper_records'].append(dict(canonical_id=scope['canonical_id'],normalized_identifier=identifier,
            conclusion=copy.deepcopy(conclusion),conclusion_file=conclusion_ref['path'],
            conclusion_file_sha256=conclusion_ref['sha256'],conclusion_source_selector=f'new_primary_method_scopes/{conclusion_position}',
            read_scope_reference=scope_ref,exact_read_scope=copy.deepcopy(scope),source_packet=PACKET.name,
            source_packet_binding_reference=ref(HERE/'SOURCE_BINDINGS.json'),
            source_primary_reference=ref(PACKET/scope['source']),
            primary_excerpt_reference=ref(PACKET/'sources'/f"{scope['key']}_selected_method.json"),
            scope_deduplication_key_sha256=key,scoped_method_read=True,full_paper_read=False,
            author_source_read=False,numeric_result_transfer=False,predictive_adoption=False,
            global_novelty_clearance=False,execution_authorized=False,integration_pass_new_primary_reads=0,
            integration_pass_primary_reread=False))
        group = matching[0] if matching else dict(normalized_identifier=identifier,kind='paper',
            raw_canonical_identifiers=[],explicit_aliases=[],record_indices=[])
        if not matching:groups.append(group)
        if scope['canonical_id'] not in group['raw_canonical_identifiers']:group['raw_canonical_identifiers'].append(scope['canonical_id'])
        group['record_indices'].append(index)
        added.append(dict(canonical_id=identifier,versioned_canonical_id=scope['canonical_id'],record_index=index,
            new_identity=not matching,full_paper_read=False,scope_deduplication_key_sha256=key))
    proposal_keys = ['hypothesis','method','graph_task_definition','what_is_new_in_this_program','theory_limit',
        'strong_controls','falsification','cost_contract','predictive_utility_established','prototype_admitted']
    d['bounded_proposal_root_triage_linkages_v62'] = [dict(
        linkage_kind='bounded_hypothesis_and_root_triage_not_paper_read',
        name='Persistent shared NCN/private-route learning after endpoint-separated adaptation',
        proposal_reference=ref(PACKET/'PAPER_CONCLUSIONS.json'),report_reference=ref(PACKET/'REPORT.md'),
        root_triage_reference=ref(PACKET/'ROOT_TRIAGE_DECISION.json'),
        proposal={k:copy.deepcopy(conclusions[k]) for k in proposal_keys},root_triage=copy.deepcopy(triage),
        new_primary_paper_reads=0,new_paper_identity=False,numeric_results_adopted=False,
        global_novelty_clearance=False,predictive_utility_established=False,execution_authorized_by_this_index=False,
        decision_boundary='Source-feasibility hypothesis only. No scientific fit admitted; native mixed derivatives, endpoint versus random separation and full paid compute require qualification. Existing frozen cohorts remain unchanged.')]
    names = ['PAPER_CONCLUSIONS.json','READ_SCOPES.json','REPORT.md','ROOT_TRIAGE_DECISION.json',
        'RETRIEVAL_RECEIPTS.json','REUSED_CONCLUSIONS.json','sources/anil_selected_method.json',
        'sources/metagraph_selected_method.json','sources/bmaml_selected_method.json']
    for name in names:
        row = {**ref(PACKET/name),'kind':'saved_scoped_meta_learning_prior_or_nonreading_proposal_triage',
            'source_bindings_reference':ref(HERE/'SOURCE_BINDINGS.json'),
            'scope':'Three previously completed primary methods and one bounded proposal/root decision linkage; zero integration retrieval/primary/full-paper reads or predictive/novelty/execution adoption.'}
        assert not any(r['path']==row['path'] for r in d['existing_packets'])
        d['existing_packets'].append(row)
    groups.sort(key=lambda g:g['normalized_identifier'])
    assert d['paper_records'][:233]==prior['paper_records']
    assert d['existing_packets'][:len(prior['existing_packets'])]==prior['existing_packets']
    assert all(g in groups for g in prior['canonical_identifier_normalization']['groups'])
    for k in prior:
        if k not in changed+['paper_records','existing_packets','canonical_identifier_normalization']:assert d[k]==prior[k]
    totals = metrics(d)
    assert totals['conclusion_records']==236 and totals['normalized_paper_identifiers']==184
    assert all(a['new_identity'] for a in added)
    account = {k:copy.deepcopy(v) for k,v in prior['read_accounting'].items() if not k.startswith(('latest_','integration_pass_'))}
    account.update(totals,state='COMPLETED_SHARED_META_ENSEMBLE_PRIOR_AND_BOUNDED_TRIAGE_ADOPTION',
        historical_path_catalog_note='All 233 v61 records, groups, catalogs, source events, mechanism linkages and prior history retained. Three saved meta-learning method scopes appended; one bounded proposal/root decision linked without reading credit.',
        latest_packet_new_scoped_primary_reads=3,latest_packet_full_primary_reads=0,
        latest_packet_previously_completed_scoped_read_adoptions=3,latest_packet_new_paper_identity_groups=3,
        latest_packet_nonreading_proposal_root_triage_linkages=1,
        integration_pass_new_primary_reads=0,integration_pass_full_primary_reads=0,
        integration_pass_primary_method_reads=0,integration_pass_author_source_semantic_reads=0,
        integration_pass_experimental_score_artifact_reads=0,integration_pass_retrievals=0)
    d['read_accounting'] = account
    d['latest_adoption'] = dict(UTC=now,predecessor=ref(PREV/'LITERATURE_INDEX.json'),previous_records_preserved=True,
        added_records=added,source_completed_new_scoped_method_reads=3,source_full_paper_reads=0,integration_primary_reads=0,
        integration_retrievals=0,nonreading_proposal_root_triage_linkages=1,
        source_binding_reference=ref(HERE/'SOURCE_BINDINGS.json'),novelty_or_numeric_or_predictive_or_execution_adoption=False)
    d['shared_meta_ensemble_prior_limits_v62'] = dict(
        ANIL='Head-only inner adaptation and shared-feature outer meta-updates are direct prior; no new shared-body meta operator.',
        Meta_Graph='Link-prediction meta-learning over actual multiple graph tasks is prior. Endpoint blocks in one graph do not inherit independent-task or deployment-distribution guarantees.',
        BMAML_EMAML='A shared feature extractor and private ensemble classifiers with head-only inner updates and shared-feature/classifier meta-updates explicitly precede the proposal.',
        candidate='Persistent-route recomputation and endpoint-separated raw-logit graph supervision are a conditional composition hypothesis, not cleared novelty or established quality.',
        limits='No native-Adam, ranking, generalization or universal superiority theorem; NCNC detached completion is not admitted under the exact-derivative claim.',
        triage='Only source-feasibility and the method/prior/control/falsification specification retained. No fit, heldout access, frozen-cohort change or predictive adoption.')
    save('LITERATURE_INDEX.json',d)
    save('VERIFICATION.json',dict(UTC=now,predecessor_records_preserved=233,
        predecessor_groups_catalog_source_events_and_linkages_preserved=True,predecessor_manifest_sha256=PREV_PIN,
        source_scope_and_conclusion_pins_verified=True,source_hash_bytes_and_excerpt_bindings_verified=True,
        source_bindings_reference=ref(HERE/'SOURCE_BINDINGS.json'),added_records=added,recomputed_metrics=totals,
        bounded_proposal_root_triage_linkages=1,paper_read_credit_from_linkage=0,integration_primary_reads=0,
        integration_retrievals=0,new_full_paper_reads=0,no_numeric_predictive_novelty_execution_adoption=True))
    (HERE/'ROOT_ADOPTION_NOTES.md').write_text(
        '# Literature index v62\n\n'
        '236 conclusion records cover 184 paper groups and two software groups. All 233 v61 records, identifier groups, catalog entries, associated source events, mechanism linkages and history are preserved. '
        'Exactly three previously completed primary-method scopes are appended: ANIL arxiv:1909.09157v2, Meta-Graph arxiv:1912.09867v2 and BMAML arxiv:1806.03836v4. '
        'Identity and exact scope were deduplicated before append. This integration adds zero retrievals, primary or full-paper reads and no numerical, predictive, execution or novelty adoption.\n\n'
        'Head-only adaptation with shared-feature meta-updates is ANIL ancestry. Actual graph-task link-prediction meta-learning is Meta-Graph ancestry. '
        'Shared feature extractors, private ensemble classifiers and head-only inner updates are explicit BMAML/EMAML ancestry. '
        'The endpoint-separated persistent-route recomputation recipe remains a conditional composition hypothesis. '
        'The bounded proposal and ROOT_TRIAGE_DECISION are linked separately with no reading credit or fit admission; source feasibility and representative controls remain required.\n')
    rows = [dict(path=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(HERE.iterdir()) if f.is_file()]
    save('MANIFEST.json',dict(UTC=now,files=rows))
    save('SEAL.json',dict(manifest_sha256=ref(HERE/'MANIFEST.json')['sha256'],payload_files=len(rows)))
    for f in HERE.iterdir():
        if f.is_file():f.chmod(0o444)
    print(json.dumps(totals))


if __name__=='__main__':main()
