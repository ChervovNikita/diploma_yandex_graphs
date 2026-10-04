"""Append six scoped conclusions; preserve prior memory and verify source bytes."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

PHASE = Path(__file__).resolve().parents[1]
DEST = PHASE / 'literature_memory/index_v44'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def descriptor(p):
    return dict(path=str(p.relative_to(PHASE)), bytes=p.stat().st_size, sha256=sha(p))


def write(name, value):
    with (DEST / name).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def verify_packet(relative, expected):
    folder = PHASE / relative
    manifest = folder / 'MANIFEST.json'
    assert sha(manifest) == expected
    value = json.loads(manifest.read_text())
    checked = []
    for row in value['files']:
        file = folder / row['path']
        assert file.resolve().is_relative_to(folder.resolve())
        assert file.stat().st_size == row['bytes'] and sha(file) == row['sha256']
        checked.append(descriptor(file))
    return folder, checked


def normalize(identifier):
    return re.sub(r'v\d+$', '', identifier.lower()) if identifier.lower().startswith('arxiv:') else identifier.lower()


assert not DEST.exists(), 'Immutable destination already exists'
previous, previous_checked = verify_packet('literature_memory/index_v43', '802f3f128aa17950c03304b5b77b789029e80683c4b3ddc7afbc28b9b589ff51')
new, new_checked = verify_packet('graph_count_conditioned_masked_link_auxiliary_prior_assessment_preparation_20261004_v1', 'f2ddd630207538dbe297fb6a4b5a354f142892446b2bb694db786d3279f250f5')
earlier, earlier_checked = verify_packet('graph_count_conditioned_pattern_literature_assessment_preparation_20261004_v1', 'bf13098c0b973423fdf732e7d352baca59c79ec3e61f81e3e382167f70217bea')
old = json.loads((previous / 'LITERATURE_INDEX.json').read_text())
index = copy.deepcopy(old)
utc = datetime.now(timezone.utc).isoformat()
growth = []
for folder in [new, earlier]:
    conclusions = json.loads((folder / 'PAPER_CONCLUSIONS.json').read_text())['new_primary_conclusions']
    scopes = json.loads((folder / 'READ_SCOPES.json').read_text())['new_primary']
    for conclusion in conclusions:
        citation = conclusion.get('citation', conclusion)
        identity = citation['canonical_id']
        normalized = normalize(identity)
        assert not any(g['normalized_identifier'] == normalized for g in index['canonical_identifier_normalization']['groups'])
        scope = next(s for s in scopes if s.get('citation', s)['canonical_id'] == identity)
        record_index = len(index['paper_records'])
        index['paper_records'].append(dict(
            canonical_id=identity, conclusion_file=str((folder / 'PAPER_CONCLUSIONS.json').relative_to(PHASE)),
            conclusion_file_sha256=sha(folder / 'PAPER_CONCLUSIONS.json'), conclusion=copy.deepcopy(conclusion),
            citation_metadata=copy.deepcopy(citation), exact_read_scope=copy.deepcopy(scope),
            scope_path_base=str(folder.relative_to(PHASE)), read_scope_reference=descriptor(folder / 'READ_SCOPES.json'),
            source_packet_manifest_reference=descriptor(folder / 'MANIFEST.json'),
            read_status='scoped_primary_method_read', primary_method_read_in_source_packet=True,
            full_paper_read=False, proof_audit=False, author_code_read=False, author_source_executed=False,
            scientific_results_or_runtime_adopted=False, integration_primary_method_read=False,
            global_novelty_or_absence_certificate=False))
        index['canonical_identifier_normalization']['groups'].append(dict(
            kind='paper', normalized_identifier=normalized, raw_canonical_identifiers=[identity],
            record_indices=[record_index], explicit_aliases=[]))
        growth.append(dict(canonical_id=identity, normalized_identifier=normalized, record_index=record_index,
                           source_packet=str(folder.relative_to(PHASE)), scoped_primary_read=True, full_paper_read=False))
    catalog_paths = {row['path'] for row in index['existing_packets']}
    for name in ['PAPER_CONCLUSIONS.json', 'READ_SCOPES.json', 'MANIFEST.json']:
        row = descriptor(folder / name)
        if row['path'] not in catalog_paths:
            row.update(kind='scoped_count_auxiliary_prior_document', scope='Stored conclusions and exact scopes; no scientific or global novelty verdict.')
            index['existing_packets'].append(row)

assert index['paper_records'][:len(old['paper_records'])] == old['paper_records']
assert index['canonical_identifier_normalization']['groups'][:len(old['canonical_identifier_normalization']['groups'])] == old['canonical_identifier_normalization']['groups']
assert len(growth) == 6
index['schema'] = 'literature-memory-index-v44'
index['created_UTC'] = utc
index['predecessor_index'] = str((previous / 'LITERATURE_INDEX.json').relative_to(PHASE))
index['predecessor_index_sha256'] = sha(previous / 'LITERATURE_INDEX.json')
index['predecessor_v44_latest_adoption_snapshot'] = copy.deepcopy(old['latest_adoption'])
index['predecessor_v44_read_accounting_snapshot'] = copy.deepcopy(old['read_accounting'])
paper_groups = sum(g['kind'] == 'paper' for g in index['canonical_identifier_normalization']['groups'])
software_groups = sum(g['kind'] != 'paper' for g in index['canonical_identifier_normalization']['groups'])
catalog = {r['path'] for r in index['existing_packets']}
conclusion_paths = {r['conclusion_file'] for r in index['paper_records'] if 'conclusion_file' in r}
scope_paths = {r['read_scope_reference']['path'] for r in index['paper_records'] if isinstance(r.get('read_scope_reference'), dict)}
adoption = dict(UTC=utc, predecessor=descriptor(previous / 'LITERATURE_INDEX.json'), source_manifests=[descriptor(new / 'MANIFEST.json'), descriptor(earlier / 'MANIFEST.json')],
    added_records=growth, new_scoped_primary_method_reads_in_recent_packet=5, previously_completed_chen_scope_adopted=1,
    integration_primary_reads=0, full_paper_certifications=0, preserved_old_records=True, preserved_old_groups=True,
    preserved_old_unresolved_leads=True, scientific_execution=False, methodological_novelty_established=False,
    interpretation='Six indexed scoped conclusions, not six new full reads. Classical fixed-count likelihood, categorical/InfoNCE and DPP equivalences are prior. Joint association and predictive utility remain unconfirmed.')
index['latest_adoption'] = adoption
index['post_v43_append'] = adoption
index['read_accounting'].update(conclusion_records=len(index['paper_records']), normalized_paper_identifiers=paper_groups,
    software_documentation_identifiers=software_groups, catalog_entries=len(index['existing_packets']),
    latest_adoption_packets=2, latest_index_growth=6, latest_packet_first_scoped_method_identity=growth,
    latest_packet_new_scoped_primary_reads=5, latest_packet_scoped_primary_method_events=5,
    latest_packet_previously_completed_scoped_read_adoptions=1, latest_packet_full_primary_reads=0,
    latest_packet_bounded_author_source_scope_events=0, latest_packet_bounded_author_source_repositories=0,
    latest_packet_author_source_retrieved_only_files=0, integration_pass_new_primary_reads=0,
    integration_pass_primary_method_reads=0, integration_pass_full_primary_reads=0,
    integration_pass_retained_primary_revisits=0, integration_pass_metadata_identity_checks=6,
    unique_catalog_document_paths=len(catalog), unique_conclusion_source_documents=len(conclusion_paths),
    unique_referenced_document_paths=len(catalog | conclusion_paths), unique_scope_reference_document_paths=len(scope_paths),
    latest_accounting_correction_reference='literature_memory/index_v44/ADOPTION_AND_VERIFICATION.json',
    historical_path_catalog_note='All predecessor records, groups, scopes, aliases and unresolved leads preserved. Five recent scoped conclusions and one earlier Chen–Liu scope appended; zero full-paper certifications.')
DEST.mkdir()
write('LITERATURE_INDEX.json', index)
write('ADOPTION_AND_VERIFICATION.json', adoption)
write('SOURCE_PAYLOAD_VERIFICATION.json', dict(predecessor=previous_checked, recent_packet=new_checked, earlier_packet=earlier_checked))
(DEST / 'README.md').write_text('# Additive literature memory v44\n\n196 scoped conclusion records across 147 paper identities and two software identities. These counts are not full-paper read totals. Prior conclusions and unresolved leads remain intact. No methodological or predictive claim is established.\n')
rows = [dict(path=f.name, bytes=f.stat().st_size, sha256=sha(f)) for f in sorted(DEST.iterdir())]
write('MANIFEST.json', dict(schema='literature-memory-successor-manifest-v44', files=rows))
write('SEAL.json', dict(manifest_sha256=sha(DEST / 'MANIFEST.json'), record_count=len(index['paper_records']), normalized_paper_count=paper_groups,
                      software_count=software_groups, preserved_old_records_and_groups=True, scientific_execution=False))
print(json.dumps(dict(manifest_sha256=sha(DEST / 'MANIFEST.json'), records=len(index['paper_records']), papers=paper_groups, software=software_groups)))
