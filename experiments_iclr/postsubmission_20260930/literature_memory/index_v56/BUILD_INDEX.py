"""Preserve v55 and add two completed, precisely bounded method scopes."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
P = HERE.parent.parent
PREV = P / 'literature_memory/index_v55'
PACKETS = [
    ('graph_conditioned_prediction_disagreement_primary_scout_20261005_v1',
     'arxiv:1909.02811',
     'd824d4b5b112cbe3579f93abefa4adea084b2c769476b39d511d84c931ad17bc'),
    ('emr_gnn_scoped_method_assessment_root_20261005_v1',
     'arxiv:2205.12076', None),
]


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(P)), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


def normalized(identifier):
    return re.sub(r'v\d+$', '', identifier.lower().strip())


def sealed(folder, expected=None):
    manifest = json.loads((folder / 'MANIFEST.json').read_text())
    pin = ref(folder / 'MANIFEST.json')['sha256']
    if (folder / 'SEAL.json').exists():
        seal = json.loads((folder / 'SEAL.json').read_text())
        assert pin == seal['manifest_sha256']
    else:
        # The scout uses a pinned manifest and read-only payloads, without a
        # separate seal document. Do not modify its immutable packet.
        assert expected is not None and pin == expected
        assert all(not (f.stat().st_mode & 0o222) for f in folder.rglob('*')
                   if f.is_file())
    assert expected is None or pin == expected
    for row in manifest['files']:
        path = folder / row['path']
        assert path.resolve().is_relative_to(folder) and not path.is_symlink()
        actual = ref(path)
        assert actual['bytes'] == row['bytes'] and actual['sha256'] == row['sha256']
    return pin


def main():
    now = datetime.now(timezone.utc).isoformat()
    sealed(PREV)
    prior = json.loads((PREV / 'LITERATURE_INDEX.json').read_text())
    result = copy.deepcopy(prior)
    groups = result['canonical_identifier_normalization']['groups']
    added = []
    bindings = []
    for relative, identifier, expected in PACKETS:
        folder = P / relative
        manifest_pin = sealed(folder, expected)
        paper = json.loads((folder / 'PAPER_CONCLUSIONS.json').read_text())
        scope = json.loads((folder / 'READ_SCOPES.json').read_text())
        assert normalized(paper['canonical_id']) == identifier
        key_material = dict(canonical_id=identifier,
                            conclusion=ref(folder / 'PAPER_CONCLUSIONS.json'),
                            exact_scope=ref(folder / 'READ_SCOPES.json'))
        key = hashlib.sha256(json.dumps(key_material, sort_keys=True).encode()).hexdigest()
        assert not any(r.get('scope_deduplication_key_sha256') == key
                       for r in result['paper_records'])
        matching = [g for g in groups if identifier in {
            normalized(x) for x in [g['normalized_identifier']] +
            g.get('raw_canonical_identifiers', []) + g.get('explicit_aliases', [])}]
        assert len(matching) <= 1
        position = len(result['paper_records'])
        result['paper_records'].append(dict(
            canonical_id=paper['canonical_id'], conclusion=paper,
            conclusion_file=str((folder / 'PAPER_CONCLUSIONS.json').relative_to(P)),
            conclusion_file_sha256=ref(folder / 'PAPER_CONCLUSIONS.json')['sha256'],
            read_scope_reference=ref(folder / 'READ_SCOPES.json'),
            scope_deduplication_key_sha256=key, full_paper_read=False,
            scoped_method_read=True, integration_new_primary_reads=0))
        if matching:
            group = matching[0]
        else:
            group = dict(normalized_identifier=identifier, kind='paper',
                         raw_canonical_identifiers=[], explicit_aliases=[], record_indices=[])
            groups.append(group)
        if paper['canonical_id'] not in group['raw_canonical_identifiers']:
            group['raw_canonical_identifiers'].append(paper['canonical_id'])
        group['record_indices'].append(position)
        added.append(dict(canonical_id=identifier, record_index=position,
                          new_identity=not matching, full_paper_read=False,
                          scope_deduplication_key_sha256=key))
        for name in ['PAPER_CONCLUSIONS.json', 'READ_SCOPES.json', 'REPORT.md']:
            row = ref(folder / name)
            assert not any(x['path'] == row['path'] for x in result['existing_packets'])
            row.update(kind='bounded_scoped_method_memory',
                       sealed_packet_manifest_sha256=manifest_pin,
                       scope='Saved exact scopes and limits; no numerical/novelty/launch adoption')
            result['existing_packets'].append(row)
        bindings.append(dict(packet=relative, manifest_sha256=manifest_pin,
                             seal_sha256=(ref(folder / 'SEAL.json')['sha256']
                                          if (folder / 'SEAL.json').exists() else None)))
    assert result['paper_records'][:len(prior['paper_records'])] == prior['paper_records']
    assert result['existing_packets'][:len(prior['existing_packets'])] == prior['existing_packets']
    groups.sort(key=lambda g: g['normalized_identifier'])
    cat = {r['path'] for r in result['existing_packets']}
    cons = {r['conclusion_file'] for r in result['paper_records']
            if isinstance(r.get('conclusion_file'), str)}
    scopes = {r['read_scope_reference']['path'] for r in result['paper_records']
              if 'read_scope_reference' in r}
    legacy_scopes = {r['read_scope_file_reference']['path'] for r in result['paper_records']
                     if 'read_scope_file_reference' in r}
    account = result['read_accounting']
    result['predecessor_v56_read_accounting_snapshot'] = copy.deepcopy(account)
    result['predecessor_v56_latest_adoption_snapshot'] = copy.deepcopy(result['latest_adoption'])
    account.update(conclusion_records=len(result['paper_records']),
                   normalized_paper_identifiers=sum(g['kind'] == 'paper' for g in groups),
                   software_documentation_identifiers=sum(g['kind'] != 'paper' for g in groups),
                   catalog_entries=len(result['existing_packets']),
                   unique_catalog_document_paths=len(cat),
                   unique_conclusion_source_documents=len(cons),
                   unique_referenced_document_paths=len(cat | cons),
                   unique_scope_reference_document_paths=len(scopes),
                   unique_scope_reference_document_paths_including_legacy_field_alias=len(scopes | legacy_scopes),
                   latest_packet_new_scoped_primary_reads=2,
                   latest_packet_full_primary_reads=0,
                   integration_pass_new_primary_reads=0,
                   integration_pass_full_primary_reads=0)
    result['latest_adoption'] = dict(
        UTC=now, predecessor=ref(PREV / 'LITERATURE_INDEX.json'),
        previous_records_preserved=True, added_records=added,
        source_completed_new_scoped_method_reads=2, source_full_paper_reads=0,
        integration_primary_reads=0, packet_bindings=bindings,
        novelty_or_numeric_or_predictive_or_execution_adoption=False)
    result['graph_predictive_diversity_limits_v56'] = dict(
        GRE='Greedy validation-selected embedding concatenation, not shared member-error training.',
        EMR_GNN='Ensemble relation operators in one predictor, not a classifier ensemble.',
        GEENI='Error-node message-suppression abstract exposed; exact primary method unresolved. Reopen only with a new specific locator.',
        DIVE='Abstract/locator only at this adoption; pending next distinct method scope.',
        graph_kernel_NCL='Attributed adaptation; no novel training principle or promoted pilot established.',
        embedding_repulsion='Representation spread can leave logits unchanged or harm competence; no quality guarantee.')
    result['created_UTC'] = now
    result['predecessor_index'] = str((PREV / 'LITERATURE_INDEX.json').relative_to(P))
    result['predecessor_index_sha256'] = ref(PREV / 'LITERATURE_INDEX.json')['sha256']
    assert len(result['paper_records']) == 220
    assert account['normalized_paper_identifiers'] == 168
    with (HERE / 'LITERATURE_INDEX.json').open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    verification = dict(UTC=now, predecessor_records_preserved=True,
                        exact_sources_verified=True, added_records=added,
                        read_accounting=account, new_full_paper_reads=0,
                        no_predictive_or_novelty_or_execution_claim=True)
    with (HERE / 'VERIFICATION.json').open('x') as handle:
        json.dump(verification, handle, indent=2, sort_keys=True)
        handle.write('\n')
    (HERE / 'ROOT_ADOPTION_NOTES.md').write_text(
        '# Literature index v56\n\n'
        '220 scoped conclusion records cover168 paper identities plus two software identities. '
        'This adds GRE and EMR-GNN method scopes; neither is a full-paper audit. '
        'GEENI remains unresolved and DIVE remains abstract-only at this adoption. '
        'The integration adds no primary reads, predictive result or novelty claim. '
        'All earlier records and conclusions are preserved.\n')
    rows = [dict(path=f.name, bytes=f.stat().st_size,
                 sha256=hashlib.sha256(f.read_bytes()).hexdigest())
            for f in sorted(HERE.iterdir()) if f.is_file()]
    with (HERE / 'MANIFEST.json').open('x') as handle:
        json.dump(dict(UTC=now, files=rows), handle, indent=2)
        handle.write('\n')
    with (HERE / 'SEAL.json').open('x') as handle:
        json.dump(dict(manifest_sha256=ref(HERE / 'MANIFEST.json')['sha256'],
                       payload_files=len(rows)), handle, indent=2)
        handle.write('\n')
    for f in HERE.iterdir():
        if f.is_file():
            f.chmod(0o444)
    print(json.dumps(dict(records=220, paper_identities=168,
                         new_full_paper_reads=0, scientific_launch=False)))


if __name__ == '__main__':
    main()
