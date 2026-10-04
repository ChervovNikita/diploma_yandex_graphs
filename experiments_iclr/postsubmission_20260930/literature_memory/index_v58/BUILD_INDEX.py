"""Adopt two completed operation scopes from saved notes; no primary semantic reads."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
P = HERE.parent.parent
PREV = P / 'literature_memory/index_v57'
PACKET = P / 'informative_structural_exposure_gap_20261005_v1'
PIN = '8ba3181fe91fc0503c09673871b5b0bb304466d9f16e62a3fcede9a9e7fb19fe'


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(P)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def save(name, value):
    with (HERE / name).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def verify(folder, pin=None):
    observed = ref(folder / 'MANIFEST.json')['sha256']
    assert pin is None or observed == pin
    seal = folder / 'SEAL.json'
    if seal.exists():
        assert json.loads(seal.read_text())['manifest_sha256'] == observed
    else:
        assert pin is not None
    for r in json.loads((folder / 'MANIFEST.json').read_text())['files']:
        f = folder / r['path']
        assert f.resolve().is_relative_to(folder) and not f.is_symlink()
        x = ref(f)
        assert (x['bytes'], x['sha256']) == (r['bytes'], r['sha256'])


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
    verify(PREV)
    verify(PACKET, PIN)
    prior = json.loads((PREV / 'LITERATURE_INDEX.json').read_text())
    scopes = json.loads((PACKET / 'READ_SCOPES.json').read_text())
    assert scopes['new_primary_method_scopes'] == 2 and scopes['full_paper_reads'] == 0
    assert metrics(prior)['conclusion_records'] == 223
    d = copy.deepcopy(prior)
    changed = ['schema', 'created_UTC', 'latest_adoption', 'read_accounting', 'predecessor_index', 'predecessor_index_sha256']
    d['integration_v58_predecessor_v57_snapshot'] = {
        **{k:copy.deepcopy(prior[k]) for k in changed}, 'index_reference':ref(PREV / 'LITERATURE_INDEX.json'),
        'manifest_reference':ref(PREV / 'MANIFEST.json'), 'seal_reference':ref(PREV / 'SEAL.json')}
    d.update(schema='literature-memory-index-v58', created_UTC=now,
             predecessor_index=str((PREV / 'LITERATURE_INDEX.json').relative_to(P)),
             predecessor_index_sha256=ref(PREV / 'LITERATURE_INDEX.json')['sha256'])
    descriptions = [
        ('arxiv:cond-mat/0205380', 'cond-mat/0205380v1', 'Specificity and stability in topology of protein networks',
         'Maslov–Sneppen table row in Closest operations and attribution',
         'Directed two-edge endpoint switches preserve in/out degrees and reject existing replacement edges. This supplies degree-switch/null-model ancestry, not native link-score matching-likelihood training.'),
        ('arxiv:2306.10453', '2306.10453v3', 'Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking',
         'HeaRT table row in Closest operations and attribution',
         'Endpoint-personalized heuristic hard evaluation alternatives, RA/PPR/feature ranking and filtering with a temporal-label caveat. The scoped HeaRT recipe retains training; it is not this matched TRAIN corruption or complete-assignment likelihood.'),
    ]
    added = []
    groups = d['canonical_identifier_normalization']['groups']
    for position, (identifier, version, title, selector, takeaway) in enumerate(descriptions):
        exact = scopes['scopes'][position]
        assert exact['canonical_id'] == version
        scope_ref = {**ref(PACKET / 'READ_SCOPES.json'), 'selector':f'scopes/{position}'}
        conclusion_ref = ref(PACKET / 'ASSESSMENT.md')
        payload = dict(canonical_id=identifier, exact_scope=exact, scope_reference=scope_ref,
                       conclusion_reference=conclusion_ref, conclusion_selector=selector)
        key = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert not any(r.get('scope_deduplication_key_sha256')==key for r in d['paper_records'])
        matching = [g for g in groups if any(identifier in ids(x) for x in
                    [g['normalized_identifier']] + g.get('raw_canonical_identifiers',[]) + g.get('explicit_aliases',[]))]
        assert len(matching) <= 1
        index = len(d['paper_records'])
        d['paper_records'].append(dict(canonical_id='arxiv:'+version, normalized_identifier=identifier,
            conclusion=dict(canonical_id='arxiv:'+version, verified_title=title, saved_takeaway=takeaway,
                            read_status='Previously completed scoped primary operation; not full paper',
                            full_paper_read=False, numeric_results_adopted=False, global_novelty_clearance=False),
            conclusion_file=conclusion_ref['path'], conclusion_file_sha256=conclusion_ref['sha256'],
            conclusion_source_selector=selector, read_scope_reference=scope_ref, exact_read_scope=copy.deepcopy(exact),
            source_packet=PACKET.name, source_packet_manifest_reference=ref(PACKET/'MANIFEST.json'),
            scope_deduplication_key_sha256=key, scoped_method_read=True, full_paper_read=False,
            author_source_read=False, numeric_result_transfer=False, predictive_adoption=False,
            global_novelty_clearance=False, execution_authorized=False, integration_pass_new_primary_reads=0,
            integration_pass_primary_reread=False))
        group = matching[0] if matching else dict(normalized_identifier=identifier,kind='paper',raw_canonical_identifiers=[],explicit_aliases=[],record_indices=[])
        if not matching: groups.append(group)
        if 'arxiv:'+version not in group['raw_canonical_identifiers']:group['raw_canonical_identifiers'].append('arxiv:'+version)
        group['record_indices'].append(index)
        added.append(dict(canonical_id=identifier,record_index=index,new_identity=not matching,
                          full_paper_read=False,scope_deduplication_key_sha256=key))
    for name in ['PAPER_CONCLUSIONS.json','READ_SCOPES.json','READ_BINDINGS.json','ASSESSMENT.md','MANIFEST.json']:
        row = {**ref(PACKET/name),'kind':'saved_scoped_structural_operation_memory', 'source_manifest_sha256':PIN,
               'scope':'Two completed bounded operation scopes; zero integration primary reads or numeric/novelty/predictive/execution adoption'}
        assert not any(r['path']==row['path'] for r in d['existing_packets'])
        d['existing_packets'].append(row)
    groups.sort(key=lambda g:g['normalized_identifier'])
    assert d['paper_records'][:223]==prior['paper_records']
    assert d['existing_packets'][:len(prior['existing_packets'])]==prior['existing_packets']
    assert all(g in groups for g in prior['canonical_identifier_normalization']['groups'])
    for k in prior:
        if k not in changed+['paper_records','existing_packets','canonical_identifier_normalization']:assert d[k]==prior[k]
    totals = metrics(d)
    assert totals['conclusion_records']==225 and totals['normalized_paper_identifiers']==173
    account = {k:copy.deepcopy(v) for k,v in prior['read_accounting'].items() if not k.startswith(('latest_','integration_pass_'))}
    account.update(totals, state='ROOT_COMPLETED_STRUCTURAL_OPERATION_SCOPE_ADOPTION',
                   historical_path_catalog_note='All 223 v57 records, groups, catalogs and failure history retained. Two saved operation scopes appended; no integration primary or full reads.',
                   latest_packet_new_scoped_primary_reads=2,latest_packet_full_primary_reads=0,
                   latest_packet_previously_completed_scoped_read_adoptions=2,latest_packet_new_paper_identity_groups=2,
                   integration_pass_new_primary_reads=0,integration_pass_full_primary_reads=0,
                   integration_pass_primary_method_reads=0,integration_pass_experimental_score_artifact_reads=0)
    d['read_accounting'] = account
    d['latest_adoption'] = dict(UTC=now,predecessor=ref(PREV/'LITERATURE_INDEX.json'),previous_records_preserved=True,
        added_records=added,source_completed_new_scoped_method_reads=2,source_full_paper_reads=0,integration_primary_reads=0,
        source_packet_manifest_reference=ref(PACKET/'MANIFEST.json'),source_separate_seal_present=False,
        novelty_or_numeric_or_predictive_or_execution_adoption=False)
    d['structured_matching_limits_v58'] = dict(degree_preserving_switch='Established switch/null-model ancestry.',
        HeaRT='Established endpoint-personalized hard evaluation alternatives; unchanged training and temporal filtering caveat retained.',
        candidate='One attributed untested implementation adaptation. Matching likelihood/disagreement alone cannot establish improved unchanged native serving rankings, sharing benefit or novelty.')
    save('LITERATURE_INDEX.json',d)
    save('VERIFICATION.json',dict(UTC=now,predecessor_records_preserved=223,predecessor_groups_and_catalog_history_preserved=True,
        saved_packet_manifest_sha256=PIN,saved_packet_manifest_files_verified=True,added_records=added,recomputed_metrics=totals,
        integration_primary_reads=0,new_full_paper_reads=0,scoped_not_full_flags_verified=True,no_numeric_predictive_novelty_execution_adoption=True))
    (HERE/'ROOT_ADOPTION_NOTES.md').write_text('# Literature index v58\n\n225 scoped conclusion records cover 173 paper groups and two software groups. All 223 v57 records and history are preserved. Two completed operation scopes are adopted: Maslov–Sneppen cond-mat/0205380v1 degree-preserving switches and HeaRT 2306.10453v3 endpoint-personalized evaluation alternatives. These scopes are not full reads; integration adds zero primary reads, numerical results, novelty clearance or launch. The matching adaptation remains untested for improved native serving.\n')
    rows = [dict(path=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(HERE.iterdir()) if f.is_file()]
    save('MANIFEST.json',dict(UTC=now,files=rows))
    save('SEAL.json',dict(manifest_sha256=ref(HERE/'MANIFEST.json')['sha256'],payload_files=len(rows)))
    for f in HERE.iterdir():
        if f.is_file():f.chmod(0o444)
    print(json.dumps(totals))


if __name__=='__main__':main()
