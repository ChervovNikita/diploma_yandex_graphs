"""Append authenticated saved conclusions/scope metadata; never reopen primary text."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
PREV = BASE / 'literature_memory/index_v66'
SOURCE = BASE / 'training_only_shared_ensemble_recent_prior_synthesis_20261005_v1'
PREV_INDEX_PIN = '6d0a53f16002aa811b5b18c8514d9cc4e44ee9f997af981bd46df4b01da9cdbd'
PREV_MANIFEST_PIN = 'd4539ba802693840591dceadfd0eea8a2292ee21cba893f02b880e7d8d958545'
SOURCE_MANIFEST_PIN = 'd6468beb8f2b6467a2890703726f9d7d1bb7859fd4812f2a69067ce8b943b4c8'
SOURCE_SEAL_PIN = 'c69f56cb93320e4d90a21c31007510b6474bc99ccbf5f227b5f8210b0eaca00d'
SAFE_SOURCE_METADATA = (
    'PAPER_CONCLUSIONS.json', 'READ_ACCOUNTING.json', 'DEDUPLICATION.json',
    'PRIMARY_RETRIEVAL.json', 'SOURCE_BINDINGS.json', 'VERIFICATION.json',
    'HYPOTHESIS_AND_FALSIFICATION.json', 'REPORT.md')
MAX_BYTES = 2_000_000


def ref(path):
    path = Path(path)
    assert path.resolve().is_relative_to(BASE) and not path.is_symlink()
    # References to selected primary text are inherited from the sealed manifest
    # below. This integration has no byte or semantic access to those payloads.
    assert not path.name.endswith('_PRIMARY_SCOPES.json')
    assert path.suffix not in ('.pdf', '.html', '.xml', '.pt', '.npy', '.jsonl')
    raw = path.read_bytes()
    return {'path':str(path.relative_to(BASE)), 'bytes':len(raw),
            'sha256':hashlib.sha256(raw).hexdigest()}


def save(name, value):
    with (HERE/name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def compact(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def normalized(value):
    result = set()
    for item in value.split(';'):
        item = item.strip().lower()
        if item.startswith('arxiv:'):
            item = re.sub(r'v\d+$','',item)
        if item:
            result.add(item)
    return result


def metrics(index):
    catalog = {row['path'] for row in index['existing_packets']}
    conclusions = {row['conclusion_file'] for row in index['paper_records']
                   if isinstance(row.get('conclusion_file'),str)}
    scopes = {row['read_scope_reference']['path'] for row in index['paper_records']
              if 'read_scope_reference' in row}
    legacy = {row['read_scope_file_reference']['path'] for row in index['paper_records']
              if 'read_scope_file_reference' in row}
    groups = index['canonical_identifier_normalization']['groups']
    return {'conclusion_records':len(index['paper_records']),
            'normalized_paper_identifiers':sum(row['kind']=='paper' for row in groups),
            'software_documentation_identifiers':sum(row['kind']!='paper' for row in groups),
            'catalog_entries':len(index['existing_packets']),
            'unique_catalog_document_paths':len(catalog),
            'unique_conclusion_source_documents':len(conclusions),
            'unique_referenced_document_paths':len(catalog|conclusions),
            'unique_scope_reference_document_paths':len(scopes),
            'unique_scope_reference_document_paths_including_legacy_field_alias':len(scopes|legacy)}


def verify_predecessor():
    manifest = ref(PREV/'MANIFEST.json')
    assert manifest['sha256']==PREV_MANIFEST_PIN
    rows = json.loads((PREV/'MANIFEST.json').read_text())['files']
    verified = [manifest]
    for row in rows:
        member = Path(row['path'])
        assert not member.is_absolute() and '..' not in member.parts
        got = ref(PREV/member)
        assert (got['bytes'],got['sha256'])==(row['bytes'],row['sha256'])
        verified.append(got)
    seal = json.loads((PREV/'SEAL.json').read_text())
    assert seal['manifest_sha256']==PREV_MANIFEST_PIN and seal['payload_files']==len(rows)
    assert ref(PREV/'LITERATURE_INDEX.json')['sha256']==PREV_INDEX_PIN
    return verified+[ref(PREV/'SEAL.json')]


def source_metadata():
    manifest = ref(SOURCE/'MANIFEST.json')
    seal_ref = ref(SOURCE/'SEAL.json')
    assert manifest['sha256']==SOURCE_MANIFEST_PIN and seal_ref['sha256']==SOURCE_SEAL_PIN
    rows = json.loads((SOURCE/'MANIFEST.json').read_text())['files']
    assert len({row['path'] for row in rows})==len(rows)
    declared = {row['path']:row for row in rows}
    seal = json.loads((SOURCE/'SEAL.json').read_text())
    assert seal['manifest_sha256']==SOURCE_MANIFEST_PIN
    verified = [manifest,seal_ref]
    for name in SAFE_SOURCE_METADATA:
        got = ref(SOURCE/name);row = declared[name]
        assert (got['bytes'],got['sha256'])==(row['bytes'],row['sha256'])
        verified.append(got)
    assert seal['paper_conclusions_sha256']==ref(SOURCE/'PAPER_CONCLUSIONS.json')['sha256']
    assert seal['report_sha256']==ref(SOURCE/'REPORT.md')['sha256']
    assert seal['new_primary_paper_identities']==2 and seal['new_full_paper_reads']==0
    assert seal['global_novelty_clearance'] is False and seal['scientific_or_model_execution'] is False
    return declared,verified


def declared_scope(declared, name):
    row = declared[name]
    return {'path':str((SOURCE/name).relative_to(BASE)), 'bytes':row['bytes'], 'sha256':row['sha256'],
            'binding_status':'Inherited authenticated sealed-manifest declaration; selected primary text payload not reopened or independently rehashed in integration',
            'payload_accessed_in_integration':False, 'full_paper_read':False}


def main():
    assert {path.name for path in HERE.iterdir()}=={'BUILD_INDEX.py'}
    now = datetime.now(timezone.utc).isoformat()
    predecessor_refs = verify_predecessor()
    declared,source_refs = source_metadata()
    previous = json.loads((PREV/'LITERATURE_INDEX.json').read_text())
    before = metrics(previous)
    assert before=={key:previous['read_accounting'][key] for key in before}
    assert (before['conclusion_records'],before['normalized_paper_identifiers'],before['software_documentation_identifiers'])==(245,193,2)
    conclusions = json.loads((SOURCE/'PAPER_CONCLUSIONS.json').read_text())
    accounting = json.loads((SOURCE/'READ_ACCOUNTING.json').read_text())
    dedup = json.loads((SOURCE/'DEDUPLICATION.json').read_text())
    receipts = json.loads((SOURCE/'PRIMARY_RETRIEVAL.json').read_text())
    source_check = json.loads((SOURCE/'VERIFICATION.json').read_text())
    source_local_disclosure = json.loads((SOURCE/'SOURCE_BINDINGS.json').read_text())
    hypothesis = json.loads((SOURCE/'HYPOTHESIS_AND_FALSIFICATION.json').read_text())
    candidates = conclusions['primary_method_scopes']
    assert [row['canonical_id'] for row in candidates]==['arxiv:2508.14285','doi:10.3390/s25144434']
    assert conclusions['new_paper_identities']==accounting['new_distinct_primary_paper_identities']==2
    assert accounting['new_primary_version_documents_read']==3 and accounting['selected_method_containers']==10
    assert accounting['new_bounded_primary_method_papers']==2
    assert accounting['new_full_paper_reads']==accounting['previously_saved_primary_method_rereads']==0
    assert accounting['reused_primary_paper_new_read_credit']==0
    assert conclusions['global_novelty_clearance'] is False and conclusions['publication_or_execution_admitted'] is False
    assert source_check['new_direct_full_operator_matches']==source_check['new_method_successors_proposed']==0
    assert hypothesis['original30_gate_unchanged'] is True and hypothesis['no_companion_threshold_added'] is True
    assert hypothesis['complete39_analysis_descriptive_only'] is True
    assert len(dedup['new_primary_paper_identities'])==2
    assert all(row['absent_from_index66'] for row in dedup['new_primary_paper_identities'])
    assert candidates[0]['earlier_version_read']['id']=='arxiv:2508.14285v1'
    assert candidates[0]['exact_primary_id']=='arxiv:2508.14285v3'
    assert normalized(candidates[0]['earlier_version_read']['id'])==normalized(candidates[0]['exact_primary_id'])

    version_docs = [
        {'canonical_id':'arxiv:2508.14285','exact_primary_id':'arxiv:2508.14285v1',
         'scope_locators':['S3.SS1','S3.SS2','S4'],
         'selected_container_count':3, 'scope_locator_source':'PAPER_CONCLUSIONS.json:primary_method_scopes/0/earlier_version_read',
         'scope_payload_reference':declared_scope(declared,'ABMLL_PRIMARY_SCOPES.json'),'retrieval_key':'abmll'},
        {'canonical_id':'arxiv:2508.14285','exact_primary_id':'arxiv:2508.14285v3',
         'scope_locators':copy.deepcopy(candidates[0]['scope_locators']),
         'selected_container_count':5, 'scope_locator_source':'PAPER_CONCLUSIONS.json:primary_method_scopes/0/scope_locators',
         'scope_payload_reference':declared_scope(declared,'ABMLL_V3_PRIMARY_SCOPES.json'),'retrieval_key':'abmll_v3'},
        {'canonical_id':'doi:10.3390/s25144434','exact_primary_id':'PMC12298271.1',
         'scope_locators':copy.deepcopy(candidates[1]['scope_locators']),
         'selected_containers':['sec3-sensors-25-04434','sec5dot3-sensors-25-04434'],
         'selected_container_count':2, 'scope_locator_source':'PAPER_CONCLUSIONS.json:primary_method_scopes/1/scope_locators',
         'scope_payload_reference':declared_scope(declared,'SENSOR_PRIMARY_SCOPES.json'),'retrieval_key':'sensor_task_relations'}]
    document_keys = set()
    for doc in version_docs:
        payload = {key:doc[key] for key in ('canonical_id','exact_primary_id','scope_locators','scope_payload_reference')}
        doc['scope_deduplication_key_sha256']=hashlib.sha256(compact(payload)).hexdigest()
        assert doc['scope_deduplication_key_sha256'] not in document_keys
        document_keys.add(doc['scope_deduplication_key_sha256'])
        receipt = next(row for row in receipts if row.get('key')==doc['retrieval_key'] and row.get('status')==200)
        assert Path(doc['scope_payload_reference']['path']).name in receipt['retained_selected_scope_paths']
        assert receipt['full_raw_deleted_after_scoping'] is True
        doc['saved_retrieval_metadata']={key:receipt[key] for key in ('url','final_url','retrieval_UTC','bytes','sha256','status')}
        doc['saved_retrieval_metadata']['hash_status']='Original deleted-raw-body receipt claim, not independently retrieved or rehashed in integration'
    assert sum(row['selected_container_count'] for row in version_docs)==10

    safe_bindings = {'UTC':now,'predecessor_files':predecessor_refs,'files':source_refs,
        'source_manifest_sha256':SOURCE_MANIFEST_PIN,'source_seal_sha256':SOURCE_SEAL_PIN,
        'safe_source_metadata_members_authenticated':len(SAFE_SOURCE_METADATA),
        'source_scope_payload_declarations':[doc['scope_payload_reference'] for doc in version_docs],
        'source_manifest_authenticated_all_payloads_not_rehashed':True,
        'selected_primary_scope_payloads_opened':False,'selected_primary_text_hashes_recomputed':0,
        'raw_primary_bodies_opened':False,'integration_searches':0,'integration_retrievals':0,
        'integration_primary_reads':0,'integration_primary_rereads':0,'integration_full_paper_reads':0,
        'source_completed_bounded_primary_method_papers':2,'source_completed_version_documents':3,
        'source_selected_containers':10,'source_full_paper_reads':0,'reused_primary_new_credit':0,
        'source_accounting_reference':ref(SOURCE/'READ_ACCOUNTING.json'),
        'source_prior_selected_text_verification_claim':source_check['selected_text_hashes_verified'],
        'source_local_read_disclosure_reference':ref(SOURCE/'SOURCE_BINDINGS.json'),
        'source_incidental_saved_summary_disclosure':source_local_disclosure['incidental_saved_summary_disclosure'],
        'source_older_compact_development_result_statements_not_reused_by_integration':True,
        'candidate_source_fixed_plan_analytic_report_or_outcome_bindings_not_dereferenced_in_integration':True,
        'integrity_checks_of_saved_metadata_only':True}
    save('SOURCE_BINDINGS.json',safe_bindings)

    index = copy.deepcopy(previous)
    current = ['schema','created_UTC','latest_adoption','read_accounting','predecessor_index','predecessor_index_sha256']
    index['integration_v67_predecessor_v66_snapshot']={
        **{key:copy.deepcopy(previous[key]) for key in current},
        'index_reference':ref(PREV/'LITERATURE_INDEX.json'),
        'manifest_reference':ref(PREV/'MANIFEST.json'),'seal_reference':ref(PREV/'SEAL.json')}
    index.update(schema='literature-memory-index-v67',created_UTC=now,
        predecessor_index='literature_memory/index_v66/LITERATURE_INDEX.json',predecessor_index_sha256=PREV_INDEX_PIN)
    groups = index['canonical_identifier_normalization']['groups']
    added = []
    record_keys = set()
    for pos,row in enumerate(candidates):
        identity = next(iter(normalized(row['canonical_id'])))
        assert all(identity not in normalized(value) for group in groups for value in
                   [group['normalized_identifier']]+group.get('raw_canonical_identifiers',[])+group.get('explicit_aliases',[]))
        assert all(identity not in normalized(record.get('canonical_id','')) for record in previous['paper_records'])
        docs = [copy.deepcopy(doc) for doc in version_docs if doc['canonical_id']==identity]
        key = hashlib.sha256(compact({'normalized_identifier':identity,'version_document_scopes':docs})).hexdigest()
        assert (identity,key) not in record_keys
        record_keys.add((identity,key))
        position = len(index['paper_records'])
        canonical = row['exact_primary_id'] if identity.startswith('arxiv:') else row['canonical_id']
        index['paper_records'].append({
            'canonical_id':canonical,'normalized_identifier':identity,'conclusion':copy.deepcopy(row),
            'conclusion_file':str((SOURCE/'PAPER_CONCLUSIONS.json').relative_to(BASE)),
            'conclusion_file_sha256':ref(SOURCE/'PAPER_CONCLUSIONS.json')['sha256'],
            'conclusion_source_selector':f'primary_method_scopes/{pos}',
            'read_scope_reference':{**ref(SOURCE/'PAPER_CONCLUSIONS.json'),'selector':f'primary_method_scopes/{pos}/scope_locators',
                'scope_kind':'Authenticated saved locator metadata; selected primary text not reopened'},
            'exact_read_scope':docs,'source_packet':SOURCE.name,
            'source_packet_binding_reference':ref(HERE/'SOURCE_BINDINGS.json'),
            'source_retrieval_reference':ref(SOURCE/'PRIMARY_RETRIEVAL.json'),
            'scope_deduplication_key_sha256':key,
            'scope_deduplication_key_schema':'normalized paper identity + separately versioned saved scope metadata and declared scope payload bindings',
            'read_status':'Previously completed bounded source method scope(s); zero integration searches, retrievals or primary reads; not a full-paper read',
            'scoped_method_read':True,'full_paper_read':False,'author_source_read':False,
            'numeric_result_transfer':False,'predictive_adoption':False,'global_novelty_clearance':False,
            'execution_authorized':False,'integration_pass_new_primary_reads':0,'integration_pass_primary_reread':False,
            'version_followup_is_additional_paper_identity':False})
        raw_ids = [canonical]
        if identity.startswith('arxiv:'):
            raw_ids = [identity,*[doc['exact_primary_id'] for doc in docs]]
        groups.append({'normalized_identifier':identity,'kind':'paper','raw_canonical_identifiers':raw_ids,
                       'explicit_aliases':[],'record_indices':[position]})
        added.append({'canonical_id':identity,'exact_canonical_id':canonical,'record_index':position,
                      'new_identity':True,'bounded_version_documents':len(docs),'full_paper_read':False,
                      'source_packet':SOURCE.name,'scope_deduplication_key_sha256':key})

    # Catalog only authenticated safe metadata and declared unread scope payloads.
    for row in source_refs:
        assert all(old['path']!=row['path'] for old in index['existing_packets'])
        index['existing_packets'].append({**row,'kind':'saved_recent_training_only_component_scope_metadata',
            'source_bindings_reference':ref(HERE/'SOURCE_BINDINGS.json'),
            'scope':'Two saved paper identities, three bounded version documents; no integration primary reading or novelty/predictive/execution adoption.'})
    for doc in version_docs:
        row = doc['scope_payload_reference']
        assert all(old['path']!=row['path'] for old in index['existing_packets'])
        index['existing_packets'].append({**row,'kind':'sealed_manifest_declared_selected_scope_payload_unopened_in_integration',
            'source_bindings_reference':ref(HERE/'SOURCE_BINDINGS.json')})

    assert index['paper_records'][:245]==previous['paper_records']
    assert compact(index['paper_records'][:245])==compact(previous['paper_records'])
    assert index['existing_packets'][:len(previous['existing_packets'])]==previous['existing_packets']
    assert groups[:len(previous['canonical_identifier_normalization']['groups'])]==previous['canonical_identifier_normalization']['groups']
    assert {key:value for key,value in index['canonical_identifier_normalization'].items() if key!='groups'}=={
        key:value for key,value in previous['canonical_identifier_normalization'].items() if key!='groups'}
    for key in previous:
        if key not in current+['paper_records','existing_packets','canonical_identifier_normalization']:
            assert index[key]==previous[key],key
    assert set(number for group in groups for number in group['record_indices'])==set(range(247))
    assert len([group for group in groups if group['normalized_identifier']=='arxiv:2508.14285'])==1
    assert len([group for group in groups if group['normalized_identifier']=='doi:10.3390/s25144434'])==1
    totals = metrics(index)
    assert (totals['conclusion_records'],totals['normalized_paper_identifiers'],totals['software_documentation_identifiers'])==(247,195,2)
    account = copy.deepcopy(previous['read_accounting'])
    account.update(totals,state='PROSPECTIVE_SAVED_RECENT_COMPONENT_SCOPE_ADOPTION_PENDING_ROOT_REVIEW',
        historical_path_catalog_note='All 245 v66 records, 193 paper and 2 software groups, catalog prefixes, source events, decision linkages and history/summary fields retained. Append ABMLL one identity with v1 to v3 follow-up and METDG one identity. Zero integration searches/retrievals/primary reads. Six changed current fields snapshotted exactly.',
        latest_packet_new_scoped_primary_reads=2,latest_packet_full_primary_reads=0,
        latest_packet_previously_completed_scoped_read_adoptions=2,latest_packet_new_paper_identity_groups=2,
        latest_packet_source_primary_version_documents=3,latest_packet_version_followup_additional_paper_identities=0,
        integration_pass_searches=0,integration_pass_new_primary_reads=0,integration_pass_full_primary_reads=0,
        integration_pass_primary_method_reads=0,integration_pass_author_source_semantic_reads=0,
        integration_pass_experimental_score_artifact_reads=0,integration_pass_retrievals=0,
        integration_pass_raw_primary_reads=0,integration_pass_selected_primary_text_scope_reads=0)
    index['read_accounting']=account
    index['latest_adoption']={'UTC':now,'status':'PROSPECTIVE_PENDING_ROOT_REVIEW',
        'predecessor':ref(PREV/'LITERATURE_INDEX.json'),'previous_records_preserved':True,
        'added_records':added,'source_completed_new_scoped_method_papers':2,'source_bounded_version_documents':3,
        'source_full_paper_reads':0,'integration_searches':0,'integration_primary_reads':0,
        'integration_primary_rereads':0,'integration_raw_primary_reads':0,'integration_retrievals':0,
        'reused_primary_new_read_credit':0,'version_followup_extra_paper_identities':0,
        'source_binding_reference':ref(HERE/'SOURCE_BINDINGS.json'),
        'novelty_or_numeric_or_predictive_or_execution_adoption':False}
    index['recent_training_only_shared_ensemble_component_limits_v67']={
        'source_conclusions_reference':ref(SOURCE/'PAPER_CONCLUSIONS.json'),
        'source_report_reference':ref(SOURCE/'REPORT.md'),'source_verification_reference':ref(SOURCE/'VERIFICATION.json'),
        'ABMLL':copy.deepcopy(candidates[0]),'METDG':copy.deepcopy(candidates[1]),
        'saved_hypothesis_and_falsification_metadata':copy.deepcopy(hypothesis),
        'hypothesis_reference':ref(SOURCE/'HYPOTHESIS_AND_FALSIFICATION.json'),
        'source_conclusions_preserved_without_rewrite':True,'new_direct_full_operator_matches':0,
        'new_supported_method_successors':0,'new_arm_grid_gate_threshold_or_fixed_study_change':False,
        'metadata_only_missing_older_leads_not_added_as_paper_records':True,
        'allowed_statement':'Saved bounded component-prior evidence: efficient global/local factor meta-learning and shared graph-conditioned generated predictors. No additional direct full-operator match, supported successor, global novelty clearance or predictive utility established.',
        'scope_limits':'Two identities, three version documents, ten selected source containers, zero source full-paper reads; zero integration searches/retrievals/primary reads. ABMLL version equivalence not assumed. Selected-text hashes are inherited authenticated manifest declarations, not rehashed here.'}

    encoded = compact(index)+b'\n'
    assert json.loads(encoded)==index and len(encoded)<=MAX_BYTES
    save('SIZE_LIMIT_CHECK.json',{'UTC':now,'index_bytes':len(encoded),'decimal_2MB_limit':MAX_BYTES,
        'within_limit':True,'serialization':'Full schema-compatible compact UTF-8 JSON, sorted object keys',
        'JSON_roundtrip_equal':True,'measured_before_index_write':True,'history_discarded':False,
        'new_raw_texts_embedded':False,'predecessor_content_including_legacy_embedded_fields_preserved':True,
        'publisher_modified':False})
    with (HERE/'LITERATURE_INDEX.json').open('xb') as stream:
        stream.write(encoded)
    delta={'UTC':now,'prospective':True,'predecessor_index':ref(PREV/'LITERATURE_INDEX.json'),
        'successor_index':ref(HERE/'LITERATURE_INDEX.json'),'before':before,'after':totals,
        'metric_deltas':{key:totals[key]-before[key] for key in totals},'added_records':added,
        'previously_completed_bounded_method_papers_adopted':2,'previously_completed_version_documents':3,
        'source_selected_method_containers':10,'version_followup_extra_paper_identity_credit':0,
        'integration_searches':0,'integration_primary_reads':0,'integration_primary_rereads':0,
        'integration_raw_primary_reads':0,'integration_selected_primary_scope_payload_reads':0,
        'integration_retrievals':0,'new_full_paper_reads':0,'reused_primary_read_credit':0,
        'source_claims_rewritten':False,'canonical_status_or_ledger_modified':False}
    save('DELTA.json',delta)
    save('VERIFICATION.json',{'UTC':now,'status':'PASS_METADATA_ONLY_PROSPECTIVE_PENDING_ROOT_REVIEW',
        'predecessor_index_sha256':PREV_INDEX_PIN,'predecessor_manifest_sha256':PREV_MANIFEST_PIN,
        'source_manifest_sha256':SOURCE_MANIFEST_PIN,'source_seal_sha256':SOURCE_SEAL_PIN,
        'predecessor_records_preserved':245,'predecessor_record_prefix_canonical_bytes_sha256':hashlib.sha256(compact(previous['paper_records'])).hexdigest(),
        'all245_predecessor_record_canonical_bytes_equal':True,'predecessor_group_order_and_values_preserved':True,
        'predecessor_catalog_prefix_preserved':True,'all_other_predecessor_fields_preserved':True,
        'changed_current_metadata_snapshotted_exactly':current,'history_source_events_and_decision_linkages_preserved':True,
        'safe_source_metadata_members_hashes_verified':len(SAFE_SOURCE_METADATA),'source_manifest_and_seal_authenticated':True,
        'source_selected_primary_text_payloads_not_reopened':True,'source_selected_text_hashes_not_recomputed':True,
        'saved_scope_locator_and_retrieval_metadata_authenticated':True,'two_new_normalized_paper_identity_groups':True,
        'ABMLL_v1_and_v3_one_identity_version_equivalence_not_assumed':True,
        'unverified_cross_scheme_aliases_or_title_only_merges':0,'group_indices_cover_all247_records':True,
        'new_scope_document_deduplication_keys':sorted(document_keys),'added_records':added,
        'recomputed_metrics':totals,'source_bindings_reference':ref(HERE/'SOURCE_BINDINGS.json'),
        'index_bytes':len(encoded),'within_decimal_2MB':True,'full_JSON_roundtrip_equal':True,
        'new_raw_texts_not_embedded':True,'prior_uncertified_cumulative_read_flags_preserved':True,
        'integration_searches':0,'integration_primary_reads':0,'integration_primary_rereads':0,
        'integration_raw_primary_reads':0,'integration_retrievals':0,'new_full_paper_reads':0,
        'no_numeric_predictive_novelty_execution_adoption':True,'no_new_arm_gate_or_fixed_study_change':True,
        'canonical_status_or_ledger_modified':False})
    notes=f'''# Prospective literature index v67

Prepared for independent root verification and adoption. This successor contains 247 conclusion records, 195 normalized paper groups and 2 software groups. All 245 v66 records and all 193 prior paper/2 software groups, their order and values, catalog prefixes, source events, decision linkages and historical fields are preserved. The six changed current metadata fields are retained exactly in `integration_v67_predecessor_v66_snapshot`.

Only two authenticated saved `PAPER_CONCLUSIONS.json` rows from `training_only_shared_ensemble_recent_prior_synthesis_20261005_v1` are appended. ABMLL (`arxiv:2508.14285v3`) retains its earlier v1 scope and explicit non-equivalence boundary within one paper identity. METDG uses DOI `10.3390/s25144434`, with article-version locator `PMC12298271.1`; no unverified cross-scheme alias is introduced. The two identities are absent from all v66 record IDs, group identifiers, raw IDs and explicit aliases. Metadata-only leads and reused ancestry add no records or reading credit.

The source author completed 2 bounded paper method reads across 3 version documents and 10 selected containers, with 0 full-paper reads. This integration performs 0 searches, 0 retrievals, 0 primary rereads, 0 raw primary reads and 0 full-paper reads. Eight safe saved metadata/conclusion/report members, source manifest/seal and all predecessor packet members were authenticated. Selected primary text scope files were not reopened or rehashed; their bindings are explicit inherited sealed-manifest declarations. Deleted full body retrieval hashes remain saved receipt claims. The source's incidental older saved development summary disclosure is retained, without dereferencing those summaries or outcomes here.

ABMLL supplies efficient shared/global and private/local factor ancestry, with local resetting and explicit test adaptation. METDG supplies a shared graph encoder and support-conditioned generated predictor; its test algorithm lists no gradient update and does not prescribe a fixed mean-logit committee. The unchanged saved roles and unresolved implementation details are retained. The saved verdict establishes no new direct complete-operator match, global novelty clearance, predictive benefit or supported successor. Its optimizer-history falsifier remains conceptual: no new arm, grid, run, threshold, gate or fixed-study change is adopted.

The complete JSON is {len(encoded):,} bytes, below the decimal 2,000,000-byte limit, with parsed roundtrip equality and exact canonical-byte preservation of the 245-record prefix. Paper counts remain distinct from full-reading totals; the prior uncertified cumulative-read flags remain false. Canonical index66/status/ledger and publisher are untouched. SOURCE_BINDINGS, DELTA, VERIFICATION and SIZE_LIMIT_CHECK document the precise metadata append and its limits.
'''
    with (HERE/'ROOT_ADOPTION_NOTES.md').open('x',encoding='utf-8') as stream:
        stream.write(notes)
    # Recheck only saved safe metadata after serialization; no primary bodies.
    for row in predecessor_refs+source_refs:
        actual = ref(BASE/row['path'])
        assert (actual['bytes'],actual['sha256'])==(row['bytes'],row['sha256'])
    rows=[{'path':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
          for path in sorted(HERE.iterdir()) if path.is_file()]
    save('MANIFEST.json',{'UTC':now,'prospective':True,'files':rows})
    save('SEAL.json',{'manifest_sha256':ref(HERE/'MANIFEST.json')['sha256'],'payload_files':len(rows)})
    print(json.dumps({**totals,'index_bytes':len(encoded),
        'index_sha256':ref(HERE/'LITERATURE_INDEX.json')['sha256'],
        'manifest_sha256':ref(HERE/'MANIFEST.json')['sha256'],'seal_sha256':ref(HERE/'SEAL.json')['sha256'],
        'status':'PROSPECTIVE_READY_FOR_ROOT_REVIEW'}))


if __name__=='__main__':
    main()
