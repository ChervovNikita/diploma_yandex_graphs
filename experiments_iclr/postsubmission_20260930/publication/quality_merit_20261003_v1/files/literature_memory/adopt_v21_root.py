"""Retain four scoped conclusions and source-bound research notes, not a new method."""
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import re

phase = Path(__file__).resolve().parents[1]
prior_path = phase / 'literature_memory/index_v20/LITERATURE_INDEX.json'
packet = phase / 'graph_extension_distinct_ideas_20261002_v1'
out = phase / 'literature_memory/index_v21'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value):
    first = value.split(';', 1)[0].lower()
    return re.sub(r'v\d+$', '', first) if first.startswith('arxiv:') else first


prior = json.loads(prior_path.read_text())
index = copy.deepcopy(prior)
manifest = packet / 'MANIFEST.sha256'
seal = json.loads((packet / 'SEAL.json').read_text())
assert seal['manifest_sha256'] == sha(manifest)
payload = []
for line in manifest.read_text().splitlines():
    expected, relative = line.split('  ', 1)
    path = packet / relative
    assert path.resolve().is_relative_to(packet) and sha(path) == expected
    payload.append(relative)
assert len(payload) == seal['files_sealed'] == 67
assert set(payload) == {
    str(p.relative_to(packet)) for p in packet.rglob('*')
    if p.is_file() and p.name not in ('MANIFEST.sha256', 'SEAL.json')
}
scope = json.loads((packet / 'READ_SCOPES.json').read_text())
assert scope['accounting']['new_scoped_primary_methods'] == 4
assert scope['accounting']['new_full_primary_reads'] == 0
assert scope['accounting']['retained_primary_rereads'] == 0
papers = json.loads((packet / 'PAPER_CONCLUSIONS.json').read_text())['papers']
assert len(papers) == 4
conclusion_rel = str((packet / 'PAPER_CONCLUSIONS.json').relative_to(phase))
scope_rel = str((packet / 'READ_SCOPES.json').relative_to(phase))
groups = index['canonical_identifier_normalization']['groups']
for paper in papers:
    record = dict(canonical_id=paper['canonical_id'], conclusion_file=conclusion_rel,
                  conclusion_file_sha256=sha(packet / 'PAPER_CONCLUSIONS.json'),
                  conclusion=paper, read_scope_reference=dict(
                      path=scope_rel, sha256=sha(packet / 'READ_SCOPES.json'),
                      paper_canonical_id=paper['canonical_id']))
    position = len(index['paper_records'])
    index['paper_records'].append(record)
    ident = normalized(paper['canonical_id'])
    group = next((g for g in groups if g['normalized_identifier'] == ident), None)
    if group is None:
        group = dict(normalized_identifier=ident, kind='paper',
                     raw_canonical_identifiers=[], explicit_aliases=[], record_indices=[])
        groups.append(group)
    if paper['canonical_id'] not in group['raw_canonical_identifiers']:
        group['raw_canonical_identifiers'].append(paper['canonical_id'])
    group['record_indices'].append(position)

documents = {
    'PAPER_CONCLUSIONS.json': 'structured_paper_conclusions',
    'READ_SCOPES.json': 'bounded_primary_read_scopes',
    'REPORT.md': 'existing_packet_report',
    'FULL_NODE_COTANGENT_LIFT.md': 'conditional_support_amendment_and_falsifier',
    'DERIVATION.md': 'source_math_assessment',
    'SHARED_UPDATE_RANK_ASSESSMENT.md': 'source_math_assessment',
    'QUALITY_LANE_ASSESSMENT.md': 'quality_prior_equivalence_assessment',
    'QUALITY_RESOURCE_PLAN.json': 'prospective_resource_plan',
    'LARGE_DATASET_SIZE_BINDINGS.json': 'official_documentation_resource_binding',
    'CANDIDATE_SPEC.json': 'prospective_candidate_specification',
}
for relative, kind in documents.items():
    path = packet / relative
    row = dict(path=str(path.relative_to(phase)), sha256=sha(path), kind=kind,
               sealed_packet_manifest_sha256=sha(manifest),
               scope='Saved conclusions or prospective assessment only. No method, fit, quality or global novelty certified.')
    if relative == 'PAPER_CONCLUSIONS.json':
        row['records'] = 4
    index['existing_packets'].append(row)

assert index['paper_records'][:len(prior['paper_records'])] == prior['paper_records']
assert index['existing_packets'][:len(prior['existing_packets'])] == prior['existing_packets']
now = datetime.now(timezone.utc).isoformat()
index['created_UTC'] = now
index['predecessor_index'] = str(prior_path.relative_to(phase))
index['predecessor_index_sha256'] = sha(prior_path)
catalog_paths = {r['path'] for r in index['existing_packets']}
conclusion_paths = {r['conclusion_file'] for r in index['paper_records']}
scope_paths = {r['read_scope_reference']['path'] for r in index['paper_records'] if 'read_scope_reference' in r}
account = index['read_accounting']
account.update(conclusion_records=len(index['paper_records']),
               normalized_paper_identifiers=sum(g['kind'] == 'paper' for g in groups),
               software_documentation_identifiers=sum(g['kind'] != 'paper' for g in groups),
               unique_conclusion_source_documents=len(conclusion_paths),
               catalog_entries=len(index['existing_packets']),
               unique_catalog_document_paths=len(catalog_paths),
               unique_referenced_document_paths=len(catalog_paths | conclusion_paths),
               unique_scope_reference_document_paths=len(scope_paths),
               cataloged_source_math_conclusions_not_paper_records=account['cataloged_source_math_conclusions_not_paper_records'] + 2,
               latest_adoption_packets=1, latest_packet_new_scoped_primary_reads=4,
               latest_packet_full_primary_reads=0, latest_packet_retained_primary_revisits=0,
               integration_pass_new_primary_reads=0, integration_pass_full_primary_reads=0,
               integration_pass_retained_primary_revisits=0)
index['latest_adoption'] = dict(adopted_UTC=now, previous_adoption_reference='literature_memory/index_v20/ADOPTION_RECEIPT.json',
                               packet_bindings=[dict(packet=packet.name, manifest_sha256=sha(manifest),
                                                    seal_sha256=sha(packet / 'SEAL.json'), payload_files_verified=len(payload),
                                                    state='ADOPTED_CONCLUSIONS_ONLY')],
                               novelty_or_execution_authorized=False)
out.mkdir(exist_ok=False)
(out / 'LITERATURE_INDEX.json').write_text(json.dumps(index, indent=2) + '\n')
(out / 'README.md').write_text(
    '# Literature memory v21\n\n'
    f"{account['conclusion_records']} saved conclusion records cover {account['normalized_paper_identifiers']} normalized paper identifiers and two software identifiers. "
    'Four added scoped method reads concern CoRe-GNN, nonlinear dynamic graph propagation, GraphMix and node MoE. None is certified as a full-paper read. Integration opened no primary-paper text.\n\n'
    'The full-node cotangent lift remains a conditional support amendment. Its corrected-target distillation interpretation, unchanged-mean squared-loss limit and representative quality falsifier are retained. Generic covariance pooling reduces to stacking or MoE. The general maintenance and compression candidates remain parked on scientific grounds. Present GPU availability is not a scientific rejection criterion.\n\n'
    'Prior conclusions, source hashes, failure notes and exact reading scopes remain preserved. This index authorizes no experiment and asserts no novelty or predictive gain.\n')
receipt = dict(schema='literature-index-v21-adoption-v1', UTC=now, predecessor_sha256=sha(prior_path),
               checked_packet_files=len(payload), preserved_prior_records=len(prior['paper_records']),
               preserved_prior_catalog_entries=len(prior['existing_packets']), read_accounting=account,
               integration_primary_text_read=False, scientific_arrays_read=False,
               novelty_or_execution_authorized=False,
               outputs=[dict(path=str(p.relative_to(phase)), bytes=p.stat().st_size, sha256=sha(p))
                        for p in (out / 'LITERATURE_INDEX.json', out / 'README.md')])
(out / 'ADOPTION_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(index_sha256=sha(out / 'LITERATURE_INDEX.json'),
                      conclusion_records=account['conclusion_records'],
                      normalized_papers=account['normalized_paper_identifiers'],
                      checked_packet_files=len(payload), execution_authorized=False)))
