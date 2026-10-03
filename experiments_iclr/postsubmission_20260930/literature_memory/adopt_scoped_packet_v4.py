"""Index a sealed scoped-method packet without rereading its primary text."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

PHASE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confined(relative):
    path = PHASE / relative
    assert path.resolve().is_relative_to(PHASE) and not path.is_symlink()
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', required=True)
    parser.add_argument('--packet', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prior_path = confined(args.prior) / 'LITERATURE_INDEX.json'
    packet, output = confined(args.packet), confined(args.output)
    prior = json.loads(prior_path.read_text())
    index = copy.deepcopy(prior)
    manifest = packet / ('MANIFEST.sha256' if (packet/'MANIFEST.sha256').exists() else 'MANIFEST.json')
    seal = json.loads((packet / 'SEAL.json').read_text())
    assert sha(manifest) == seal['manifest_sha256']
    payload = []
    if manifest.suffix == '.json':
        manifest_rows = json.loads(manifest.read_text())['files']
        for row in manifest_rows:
            path = packet / row['path']
            assert path.resolve().is_relative_to(packet) and sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
            payload.append(row['path'])
        assert len(payload) == seal['payload_files']
    else:
        for line in manifest.read_text().splitlines():
            expected, relative = line.split('  ', 1)
            path = packet / relative
            assert path.resolve().is_relative_to(packet) and sha(path) == expected
            payload.append(relative)
        assert len(payload) == seal['files_sealed']
    assert set(payload) == {str(p.relative_to(packet)) for p in packet.rglob('*')
                            if p.is_file() and p.name not in ('MANIFEST.sha256', 'MANIFEST.json', 'SEAL.json')}
    conclusions = json.loads((packet / 'PAPER_CONCLUSIONS.json').read_text())
    scope = json.loads((packet / 'READ_SCOPES.json').read_text())
    papers = conclusions.get('primary_methods', conclusions.get('records'))
    declared = scope.get('accounting', dict(primary_method_scopes_this_packet=scope.get('new_unique_primary_method_papers'), new_full_primary_reads=scope.get('new_full_paper_reads'), new_method_driver_or_launch_adoptions=scope.get('execution_counts', {}).get('adoptions')))
    assert declared['primary_method_scopes_this_packet'] == len(papers)
    assert declared['new_full_primary_reads'] == 0
    assert declared['new_method_driver_or_launch_adoptions'] == 0
    groups = index['canonical_identifier_normalization']['groups']
    for paper in papers:
        position = len(index['paper_records'])
        index['paper_records'].append(dict(
            canonical_id=paper['canonical_id'], conclusion=paper,
            conclusion_file=str((packet / 'PAPER_CONCLUSIONS.json').relative_to(PHASE)),
            conclusion_file_sha256=sha(packet / 'PAPER_CONCLUSIONS.json'),
            read_scope_reference=dict(path=str((packet / 'READ_SCOPES.json').relative_to(PHASE)),
                                      sha256=sha(packet / 'READ_SCOPES.json'),
                                      paper_canonical_id=paper['canonical_id'])))
        key = re.sub(r'v\d+$', '', paper['canonical_id'].split(';', 1)[0].lower())
        group = next((g for g in groups if g['normalized_identifier'] == key), None)
        if group is None:
            group = dict(normalized_identifier=key, kind='paper',
                         raw_canonical_identifiers=[], explicit_aliases=[], record_indices=[])
            groups.append(group)
        if paper['canonical_id'] not in group['raw_canonical_identifiers']:
            group['raw_canonical_identifiers'].append(paper['canonical_id'])
        group['record_indices'].append(position)
    for name, kind in [('PAPER_CONCLUSIONS.json', 'structured_paper_conclusions'),
                       ('READ_SCOPES.json', 'bounded_primary_read_scopes'),
                       ('REPORT.md', 'conditional_composition_assessment'),
                       ('OPERATOR_AND_GAP.json', 'source_math_assessment'),
                       ('ONE_OPTIONAL_VARIANT.json', 'unadopted_prospective_candidate_specification'),
                       ('QUALIFICATION_LIMITS.json', 'qualification_limits')]:
        path = packet / name
        index['existing_packets'].append(dict(
            path=str(path.relative_to(PHASE)), sha256=sha(path), kind=kind,
            sealed_packet_manifest_sha256=sha(manifest),
            scope='Saved scoped conclusions or conditional proposal. No predictive gain or execution admission.'))
    now = datetime.now(timezone.utc).isoformat()
    index.update(created_UTC=now, predecessor_index=str(prior_path.relative_to(PHASE)),
                 predecessor_index_sha256=sha(prior_path))
    account = index['read_accounting']
    cats = {r['path'] for r in index['existing_packets']}
    cons = {r['conclusion_file'] for r in index['paper_records']}
    refs = {r['read_scope_reference']['path'] for r in index['paper_records'] if 'read_scope_reference' in r}
    account.update(conclusion_records=len(index['paper_records']),
                   normalized_paper_identifiers=sum(g['kind'] == 'paper' for g in groups),
                   software_documentation_identifiers=sum(g['kind'] != 'paper' for g in groups),
                   unique_conclusion_source_documents=len(cons), catalog_entries=len(index['existing_packets']),
                   unique_catalog_document_paths=len(cats), unique_referenced_document_paths=len(cats | cons),
                   unique_scope_reference_document_paths=len(refs),
                   cataloged_source_math_conclusions_not_paper_records=
                       account['cataloged_source_math_conclusions_not_paper_records'] + 1,
                   latest_adoption_packets=1, latest_packet_new_scoped_primary_reads=len(papers),
                   latest_packet_full_primary_reads=0, latest_packet_retained_primary_revisits=0,
                   latest_packet_first_scoped_method_identity=declared.get('first_scoped_method_identity_with_no_retained_primary_citation'),
                   latest_packet_retained_abstract_only_scope_upgrades=declared.get('retained_abstract_only_citation_method_scope_upgrades'),
                   integration_pass_new_primary_reads=0, integration_pass_full_primary_reads=0,
                   integration_pass_retained_primary_revisits=0)
    index['latest_adoption'] = dict(
        adopted_UTC=now, previous_adoption_reference=str((prior_path.parent / 'ADOPTION_RECEIPT.json').relative_to(PHASE)),
        packet_bindings=[dict(packet=packet.name, manifest_sha256=sha(manifest),
                             seal_sha256=sha(packet / 'SEAL.json'), payload_files_verified=len(payload),
                             state='ADOPTED_CONCLUSIONS_ONLY')], novelty_or_execution_authorized=False)
    assert index['paper_records'][:len(prior['paper_records'])] == prior['paper_records']
    assert index['existing_packets'][:len(prior['existing_packets'])] == prior['existing_packets']
    output.mkdir(exist_ok=False)
    (output / 'LITERATURE_INDEX.json').write_text(json.dumps(index, indent=2) + '\n')
    names = ', '.join(p.get('title', p['canonical_id']) for p in papers)
    (output / 'README.md').write_text(
        '# Literature memory\n\n'
        f"{account['conclusion_records']} saved conclusion records cover {account['normalized_paper_identifiers']} "
        f"normalized paper identifiers and {account['software_documentation_identifiers']} software identifiers. "
        f"This packet adds {len(papers)} primary method scopes: {names}. "
        'Neither scoped method reads nor revisits count as full-paper reads. '
        'The cumulative full-paper-read total remains uncertified. '
        'Integration opened no primary text.\n\n'
        f'New saved conclusions: {args.packet}/REPORT.md and PAPER_CONCLUSIONS.json. '
        'Previously indexed conclusions and their hashes are preserved. '
        'Composition assessments are conditional proposals, not predictive results or execution admissions. '
        'Known ingredients and current resource limits do not by themselves reject scientific merit.\n')
    receipt = dict(schema='scoped-literature-packet-adoption-v1', UTC=now,
                   predecessor_sha256=sha(prior_path), packet_manifest_sha256=sha(manifest),
                   payload_files_verified=len(payload), declared_packet_accounting=declared,
                   preserved_prior_records=len(prior['paper_records']), read_accounting=account,
                   integration_primary_text_read=False, scientific_arrays_read=False,
                   novelty_or_execution_authorized=False,
                   outputs=[dict(path=str(p.relative_to(PHASE)), sha256=sha(p), bytes=p.stat().st_size)
                            for p in (output / 'LITERATURE_INDEX.json', output / 'README.md')])
    (output / 'ADOPTION_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(index_sha256=sha(output / 'LITERATURE_INDEX.json'),
                         conclusion_records=account['conclusion_records'],
                         normalized_papers=account['normalized_paper_identifiers'],
                         full_paper_read_total_certified=False, execution_authorized=False)))


if __name__ == '__main__':
    main()
