"""Adopt three saved method scopes and a non-reading mechanism protocol link."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
P = HERE.parent.parent
PREV = P / 'literature_memory/index_v60'
PACKET = P / 'endpoint_frame_prior_overlap_assessment_20261005_v1'
MECHANISM = P / 'citeseer_frame_mechanism_protocol_20261005_v1'
PREV_PIN = '85063eacf24fbea2bb5f7e7521973b183ad19deb744fe23c260e4242f57b15a0'
SCOPES_PIN = 'e9a3f66d6a2d307ef225e167ccbe704d523e3c5226f9952f91e5667ad7d58929'
CONCLUSIONS_PIN = '0496dab6fd15ec5e73a7fdb51b0ab4e2a8ec1e1b46e9fbf6cbbd6a61f47acc92'


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
    assert metrics(prior)['conclusion_records']==230 and metrics(prior)['normalized_paper_identifiers']==178
    assert ref(PACKET/'READ_SCOPES.json')['sha256']==SCOPES_PIN
    assert ref(PACKET/'PAPER_CONCLUSIONS.json')['sha256']==CONCLUSIONS_PIN
    scopes = json.loads((PACKET/'READ_SCOPES.json').read_text())
    conclusions = json.loads((PACKET/'PAPER_CONCLUSIONS.json').read_text())
    assert scopes['primary_method_identity_count']==conclusions['primary_method_identity_count']==3
    assert conclusions['whole_paper_reads']==0 and conclusions['global_novelty_clearance'] is False
    assert scopes['outcome_access'] is False and scopes['server_actions'] is False
    assert len(scopes['primary_scopes'])==len(conclusions['conclusions'])==3
    bindings = [ref(PACKET/name) for name in ['READ_SCOPES.json','PAPER_CONCLUSIONS.json','RETRIEVAL_RECEIPTS.json','REPORT.md']]
    for scope in scopes['primary_scopes']:
        got = ref(PACKET/scope['source'])
        assert got['sha256']==scope['source_sha256']
        assert scope['full_paper_read'] is False and scope['author_source_read'] is False
        assert scope['empirical_results_adopted'] is False
        bindings.append(got)
        bindings.extend(ref(PACKET/name) for name in scope['excerpts'])
    for row in json.loads((PACKET/'RETRIEVAL_RECEIPTS.json').read_text()):
        got = ref(PACKET/row['path'])
        assert got['sha256']==row['sha256']
        if 'bytes' in row:assert got['bytes']==row['bytes']
        assert row['method_read'] is False
        bindings.append(got)
    lowfer = json.loads((PACKET/'sources/lowfer_selected_method.json').read_text())
    blocks = json.loads((PACKET/'sources/lowfer_blocks.json').read_text())
    lo,hi = lowfer['block_indices_inclusive']
    assert (lo,hi)==(16,40) and lowfer['blocks']==blocks[lo:hi+1]
    bindings.append(ref(PACKET/'sources/lowfer_blocks.json'))
    for name in ['sources/mhvgae_method_page-03.png','sources/mhvgae_method_page-04.png','sources/ntn_equation_page-03.png',
                 'sources/RETAINED_V60_CONCLUSIONS.json']:
        bindings.append(ref(PACKET/name))
    mechanism = json.loads((MECHANISM/'PROTOCOL.json').read_text())
    plan = P/'citeseer_endpoint_frame_paired_development_20261005_v1/PLAN.json'
    head = P/'citeseer_heart_ncn_trainval_runner_source_20261005_v1/heads.py'
    assert mechanism['plan_sha256']==ref(plan)['sha256']
    assert mechanism['head_source_sha256']==ref(head)['sha256']
    assert mechanism['fits']==mechanism['optimizer_updates']==mechanism['new_primary_paper_reads']==0
    assert mechanism['current_access']['comparative_scores_read'] is False
    bindings.extend([ref(MECHANISM/'PROTOCOL.json'),ref(MECHANISM/'README.md'),ref(plan),ref(head)])
    save('SOURCE_BINDINGS.json',dict(UTC=now,files=bindings,integrity_checks_only=True,
        integration_semantic_reads=0,previously_completed_primary_method_scopes=3,full_paper_certifications=0,
        source_packet_has_no_manifest_or_seal=True,metadata_receipts_not_primary_read_credit=True,
        mechanism_protocol_is_not_paper_read=True,visual_page_bytes_hashed_not_viewed_in_this_integration=True,
        ANIL_materials_not_adopted=True))
    d = copy.deepcopy(prior)
    changed = ['schema','created_UTC','latest_adoption','read_accounting','predecessor_index','predecessor_index_sha256']
    d['integration_v61_predecessor_v60_snapshot'] = {**{k:copy.deepcopy(prior[k]) for k in changed},
        'index_reference':ref(PREV/'LITERATURE_INDEX.json'),'manifest_reference':ref(PREV/'MANIFEST.json'),
        'seal_reference':ref(PREV/'SEAL.json')}
    d.update(schema='literature-memory-index-v61',created_UTC=now,
        predecessor_index=str((PREV/'LITERATURE_INDEX.json').relative_to(P)),
        predecessor_index_sha256=ref(PREV/'LITERATURE_INDEX.json')['sha256'])
    groups = d['canonical_identifier_normalization']['groups']
    added = []
    for position,scope in enumerate(scopes['primary_scopes']):
        conclusion_position = next(i for i,c in enumerate(conclusions['conclusions']) if c['canonical_id']==scope['canonical_id'])
        conclusion = conclusions['conclusions'][conclusion_position]
        assert conclusion['empirical_results_adopted'] is False
        identifier = next(iter(ids(scope['canonical_id'])))
        matching = [g for g in groups if any(identifier in ids(x) for x in
            [g['normalized_identifier']]+g.get('raw_canonical_identifiers',[])+g.get('explicit_aliases',[]))]
        assert len(matching)<=1
        scope_ref = {**ref(PACKET/'READ_SCOPES.json'),'selector':f'primary_scopes/{position}'}
        conclusion_ref = ref(PACKET/'PAPER_CONCLUSIONS.json')
        payload = dict(canonical_id=identifier,exact_scope=scope,scope_reference=scope_ref,
            conclusion_reference=conclusion_ref,conclusion_selector=f'conclusions/{conclusion_position}')
        key = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert not any(r.get('scope_deduplication_key_sha256')==key for r in d['paper_records'])
        index = len(d['paper_records'])
        d['paper_records'].append(dict(canonical_id=scope['canonical_id'],normalized_identifier=identifier,
            conclusion=copy.deepcopy(conclusion),conclusion_file=conclusion_ref['path'],
            conclusion_file_sha256=conclusion_ref['sha256'],conclusion_source_selector=f'conclusions/{conclusion_position}',
            read_scope_reference=scope_ref,exact_read_scope=copy.deepcopy(scope),source_packet=PACKET.name,
            source_packet_binding_reference=ref(HERE/'SOURCE_BINDINGS.json'),
            source_primary_reference=ref(PACKET/scope['source']),
            primary_excerpt_references=[ref(PACKET/name) for name in scope['excerpts']],
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
    d['bounded_mechanism_protocol_linkages_v61'] = [dict(
        linkage_kind='prospective_hypothesis_and_decision_linkage_not_paper_read',
        name='Citeseer endpoint-frame structural-regime and operator-movement description',
        protocol_reference=ref(MECHANISM/'PROTOCOL.json'),readme_reference=ref(MECHANISM/'README.md'),
        source_plan_reference=ref(plan),head_source_reference=ref(head),
        protocol=copy.deepcopy(mechanism),new_primary_paper_reads=0,new_paper_identity=False,
        numeric_results_adopted=False,global_novelty_clearance=False,predictive_utility_established=False,
        execution_authorized_by_this_index=False,
        decision_boundary='Describe every fixed stratum after complete audited cohort; subgroup outcomes cannot rescue aggregate quality. Frame-map diversity is not prediction diversity. No TEST, fit, retuning, reselection or unsaved member-logit analysis is admitted.')]
    catalog_paths = [PACKET/name for name in ['PAPER_CONCLUSIONS.json','READ_SCOPES.json','REPORT.md','RETRIEVAL_RECEIPTS.json',
        'sources/lowfer_selected_method.json','sources/ntn_selected_pages.json','sources/mhvgae_selected_pages.json',
        'sources/mhvgae_page5_before_results.txt']]+[MECHANISM/'PROTOCOL.json',MECHANISM/'README.md']
    for path in catalog_paths:
        row = {**ref(path),'kind':'saved_scoped_prior_or_nonreading_mechanism_linkage',
            'source_bindings_reference':ref(HERE/'SOURCE_BINDINGS.json'),
            'scope':'Three previously completed method scopes and one non-reading prospective mechanism protocol; zero integration primary reads, full-paper reads or predictive/novelty/execution adoption.'}
        assert not any(r['path']==row['path'] for r in d['existing_packets'])
        d['existing_packets'].append(row)
    groups.sort(key=lambda g:g['normalized_identifier'])
    assert d['paper_records'][:230]==prior['paper_records']
    assert d['existing_packets'][:len(prior['existing_packets'])]==prior['existing_packets']
    assert all(g in groups for g in prior['canonical_identifier_normalization']['groups'])
    for k in prior:
        if k not in changed+['paper_records','existing_packets','canonical_identifier_normalization']:assert d[k]==prior[k]
    totals = metrics(d)
    assert totals['conclusion_records']==233 and totals['normalized_paper_identifiers']==181
    assert all(a['new_identity'] for a in added)
    account = {k:copy.deepcopy(v) for k,v in prior['read_accounting'].items() if not k.startswith(('latest_','integration_pass_'))}
    account.update(totals,state='COMPLETED_ENDPOINT_INTERACTION_PRIOR_AND_MECHANISM_LINKAGE_ADOPTION',
        historical_path_catalog_note='All 230 v60 records, groups, catalogs, author-source events and prior history retained. Three saved method scopes appended; one prospective mechanism protocol linked without reading credit.',
        latest_packet_new_scoped_primary_reads=3,latest_packet_full_primary_reads=0,
        latest_packet_previously_completed_scoped_read_adoptions=3,latest_packet_new_paper_identity_groups=3,
        latest_packet_nonreading_hypothesis_decision_linkages=1,
        integration_pass_new_primary_reads=0,integration_pass_full_primary_reads=0,
        integration_pass_primary_method_reads=0,integration_pass_author_source_semantic_reads=0,
        integration_pass_experimental_score_artifact_reads=0)
    d['read_accounting'] = account
    d['latest_adoption'] = dict(UTC=now,predecessor=ref(PREV/'LITERATURE_INDEX.json'),previous_records_preserved=True,
        added_records=added,source_completed_new_scoped_method_reads=3,source_full_paper_reads=0,integration_primary_reads=0,
        nonreading_hypothesis_decision_linkages=1,source_binding_reference=ref(HERE/'SOURCE_BINDINGS.json'),
        novelty_or_numeric_or_predictive_or_execution_adoption=False,ANIL_materials_not_adopted=True)
    d['endpoint_interaction_architecture_prior_limits_v61'] = dict(
        LowFER='Learned projected-input Hadamard fusion and factorized bilinear pooling are established link-prediction operations. Subject/relation inputs differ from two graph endpoints; native loss sign caveat and unverified venue retained.',
        NTN2013='Multiple full bilinear tensor slices plus affine endpoint terms before nonlinear link scoring are established prior. Relation channels are not ensemble members; no whole-network containment asserted.',
        Multi_HeadVGAE2025='Graph multi-head embeddings fused before latent synthesis and a single pair decoder are prior. Inspected sections lack separately supervised NCN routes/private reflections; scoped difference is not a literature absence claim.',
        candidate='Reflected linear endpoint channels are bilinear interfaces; ordinary independently parameterized framed ensembles contain the candidate through tying. Any gain must concern estimation or optimization, not universal expressivity.',
        mechanism='Fixed TRAIN topology strata and frame movement are descriptive hypotheses before scoring, not causal proof or a way to select winning subgroups.',
        exclusions='No numerical result, author-code reproduction, whole-paper read, ANIL material or global novelty clearance adopted.')
    save('LITERATURE_INDEX.json',d)
    save('VERIFICATION.json',dict(UTC=now,predecessor_records_preserved=230,
        predecessor_groups_catalog_and_author_source_history_preserved=True,predecessor_manifest_sha256=PREV_PIN,
        source_scope_and_conclusion_pins_verified=True,source_hash_bytes_and_LowFER_excerpt_bindings_verified=True,
        source_bindings_reference=ref(HERE/'SOURCE_BINDINGS.json'),added_records=added,recomputed_metrics=totals,
        hypothesis_decision_linkages=1,paper_read_credit_from_linkage=0,integration_primary_reads=0,new_full_paper_reads=0,
        no_numeric_predictive_novelty_execution_adoption=True,ANIL_materials_not_adopted=True))
    (HERE/'ROOT_ADOPTION_NOTES.md').write_text(
        '# Literature index v61\n\n'
        '233 conclusion records cover 181 paper groups and two software groups. All 230 v60 records, identifier groups, catalog entries, associated author-source events and history are preserved. '
        'Exactly three previously completed primary-method scopes are appended: LowFER arxiv:2008.10858v1, NTN2013 using its exact saved author-paper identity, and Multi-HeadVGAE doi:10.30919/es1406. '
        'No unverified cross-scheme NTN alias or venue update is inferred. This integration adds zero primary or full-paper reads and no numerical, predictive, execution or novelty adoption.\n\n'
        'Projected-input Hadamard fusion, factorized bilinear pooling, multiple nonlinear tensor interaction channels and graph multi-head fusion are established ancestry. '
        'The exact reflected graph-ensemble operation remains an attributed adaptation with unresolved estimation/optimization utility, not universal expressivity or cleared novelty. '
        'The Citeseer structural-stratum/frame-movement protocol is separately linked as a prospective hypothesis/decision, with no reading or identity credit. '
        'Its descriptive subgroups cannot rescue an unsuccessful aggregate result. ANIL materials are not adopted.\n')
    rows = [dict(path=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(HERE.iterdir()) if f.is_file()]
    save('MANIFEST.json',dict(UTC=now,files=rows))
    save('SEAL.json',dict(manifest_sha256=ref(HERE/'MANIFEST.json')['sha256'],payload_files=len(rows)))
    for f in HERE.iterdir():
        if f.is_file():f.chmod(0o444)
    print(json.dumps(totals))


if __name__=='__main__':main()
