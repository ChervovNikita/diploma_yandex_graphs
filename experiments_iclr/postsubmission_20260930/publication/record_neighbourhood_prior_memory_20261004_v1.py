"""Append two scoped method conclusions, retaining the preceding index exactly."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json

P = Path(__file__).resolve().parents[1]
OLD = P / 'literature_memory/index_v45/LITERATURE_INDEX.json'
OUT = P / 'literature_memory/index_v46'
PACKET = P / 'conditional_neighbourhood_generation_prior_check_20261004_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(path.relative_to(P)), bytes=path.stat().st_size, sha256=sha(path))


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    prior = json.loads(OLD.read_text())
    current = copy.deepcopy(prior)
    scopes = json.loads((PACKET / 'READ_SCOPES.json').read_text())
    notes = ref(PACKET / 'CONCLUSIONS.md')
    scope_ref = ref(PACKET / 'READ_SCOPES.json')
    additions = []
    for row in scopes:
        identity = row['canonical_id']
        assert not any(g['normalized_identifier'] == identity for g in current['canonical_identifier_normalization']['groups'])
        assert row['full_paper_read'] is False
        record = dict(canonical_id=identity, conclusion_file=notes['path'],
                      conclusion_file_sha256=notes['sha256'],
                      citation_metadata=dict(title=row['title'], version=row['version'],
                                             canonical_id=identity, primary_PDF_sha256=row['primary_sha256']),
                      read_scope_reference=scope_ref, source_report_reference=notes,
                      exact_read_scope=row['scope'],
                      read_status='new_scoped_primary_method_conclusion',
                      full_paper_read=False, proof_audit=False, author_code_read=False,
                      reproduction=False, global_novelty_or_absence_certificate=False,
                      scientific_results_or_runtime_adopted=False,
                      conclusion=dict(summary='See the bounded method comparison and claim limits in CONCLUSIONS.md.',
                                      generic_neighbourhood_reconstruction_is_prior=True,
                                      our_predictive_transfer_established=False))
        index = len(current['paper_records'])
        current['paper_records'].append(record)
        current['canonical_identifier_normalization']['groups'].append(
            dict(kind='paper', normalized_identifier=identity, raw_canonical_identifiers=[identity],
                 record_indices=[index], explicit_aliases=[]))
        additions.append(dict(canonical_id=identity, record_index=index, scope=row['scope'], full_paper_read=False))
    current['canonical_identifier_normalization']['groups'].sort(key=lambda g: g['normalized_identifier'])
    for document in (PACKET / 'CONCLUSIONS.md', PACKET / 'READ_SCOPES.json'):
        current['existing_packets'].append(dict(ref(document), kind='scoped_primary_method_document',
            scope='Two scoped method reads; no full-paper, experimental reproduction or novelty certification.'))
    math_ref = ref(P / 'pattern_responsibility_overlap_analysis_20261004_v1/ANALYSIS.md')
    current['existing_packets'].append(dict(math_ref, kind='source_math_interpretation',
        scope='Finite-mixture responsibility-overlap identity; no diversity guarantee, novelty theorem or predictive result.'))
    now = datetime.now(timezone.utc).isoformat()
    event = dict(UTC=now, predecessor=ref(OLD), added_records=additions,
                 prior_records_preserved_without_modification=True,
                 new_scoped_method_conclusions=2, new_paper_identity_groups=2,
                 full_paper_read_certifications=0, scientific_execution=False,
                 mathematical_interpretation=math_ref)
    current['predecessor_v46_latest_adoption_snapshot'] = copy.deepcopy(prior['latest_adoption'])
    current['predecessor_v46_read_accounting_snapshot'] = copy.deepcopy(prior['read_accounting'])
    current['post_v45_append'] = event
    current['latest_adoption'] = event
    current['predecessor_index'] = str(OLD.relative_to(P))
    current['predecessor_index_sha256'] = sha(OLD)
    a = current['read_accounting']
    a['conclusion_records'] = len(current['paper_records'])
    a['normalized_paper_identifiers'] = sum(g['kind'] == 'paper' for g in current['canonical_identifier_normalization']['groups'])
    a['catalog_entries'] = len(current['existing_packets'])
    a['unique_catalog_document_paths'] = len({r['path'] for r in current['existing_packets']})
    a['unique_conclusion_source_documents'] = len({r['conclusion_file'] for r in current['paper_records'] if isinstance(r.get('conclusion_file'), str)})
    a['unique_referenced_document_paths'] = len({r['path'] for r in current['existing_packets']} | {r['conclusion_file'] for r in current['paper_records'] if isinstance(r.get('conclusion_file'), str)})
    a['latest_index_growth'] = 2
    a['latest_packet_new_scoped_primary_reads'] = 2
    a['latest_packet_new_paper_identity_groups'] = 2
    a['latest_packet_full_primary_reads'] = 0
    a['latest_packet_bounded_author_source_scope_events'] = 0
    a['latest_packet_author_code_semantic_file_scopes'] = 0
    a['full_paper_read_total_certified'] = False
    a['historical_path_catalog_note'] = 'All preceding records retained. Two scoped method conclusions and one mathematical interpretation appended; these are not full-paper read totals.'
    assert current['paper_records'][:len(prior['paper_records'])] == prior['paper_records']
    assert current['existing_packets'][:len(prior['existing_packets'])] == prior['existing_packets']
    OUT.mkdir()
    save(OUT / 'LITERATURE_INDEX.json', current)
    save(OUT / 'ADOPTION.json', event)
    files = [dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(OUT.iterdir())]
    save(OUT / 'MANIFEST.json', dict(files=files))
    save(OUT / 'SEAL.json', dict(manifest_sha256=sha(OUT / 'MANIFEST.json'), predecessor_sha256=sha(OLD)))
    for path in OUT.iterdir():
        path.chmod(0o444)
    OUT.chmod(0o555)
    print(json.dumps(dict(index=str(OUT.relative_to(P)), conclusions=a['conclusion_records'], identities=a['normalized_paper_identifiers'], full_paper_read_total_certified=False)))


if __name__ == '__main__':
    main()
