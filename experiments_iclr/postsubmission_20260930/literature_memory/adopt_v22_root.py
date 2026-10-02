"""Index two scoped primary conclusions and the conditional curvature proposal."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

phase = Path(__file__).resolve().parents[1]
prior_path = phase / 'literature_memory/index_v21/LITERATURE_INDEX.json'
packet = phase / 'graph_pooled_curvature_intervention_20261003_v1'
output = phase / 'literature_memory/index_v22'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(identifier):
    value = identifier.split(';', 1)[0].lower()
    return re.sub(r'v\d+$', '', value) if value.startswith('arxiv:') else value


prior = json.loads(prior_path.read_text())
index = copy.deepcopy(prior)
manifest = packet / 'MANIFEST.sha256'
seal = json.loads((packet / 'SEAL.json').read_text())
assert sha(manifest) == seal['manifest_sha256']
paths = []
for line in manifest.read_text().splitlines():
    expected, relative = line.split('  ', 1)
    path = packet / relative
    assert path.resolve().is_relative_to(packet) and sha(path) == expected
    paths.append(relative)
assert len(paths) == seal['files_sealed'] == 31
assert set(paths) == {str(p.relative_to(packet)) for p in packet.rglob('*')
                      if p.is_file() and p.name not in ('MANIFEST.sha256', 'SEAL.json')}
conclusions = json.loads((packet / 'PAPER_CONCLUSIONS.json').read_text())
scopes = json.loads((packet / 'READ_SCOPES.json').read_text())
assert scopes['accounting']['new_scoped_primary_methods'] == 2
assert scopes['accounting']['new_full_primary_reads'] == 0
assert len(conclusions['methods']) == 2
assert not conclusions['read_accounting']['model_dataset_checkpoint_or_logit_access']
assert not conclusions['composition_assessment']['adoption_or_execution']
groups = index['canonical_identifier_normalization']['groups']
for paper in conclusions['methods']:
    position = len(index['paper_records'])
    index['paper_records'].append(dict(
        canonical_id=paper['canonical_id'],
        conclusion_file=str((packet / 'PAPER_CONCLUSIONS.json').relative_to(phase)),
        conclusion_file_sha256=sha(packet / 'PAPER_CONCLUSIONS.json'),
        conclusion=paper,
        read_scope_reference=dict(path=str((packet / 'READ_SCOPES.json').relative_to(phase)),
                                  sha256=sha(packet / 'READ_SCOPES.json'),
                                  paper_canonical_id=paper['canonical_id'])))
    key = normalized(paper['canonical_id'])
    group = next((g for g in groups if g['normalized_identifier'] == key), None)
    if group is None:
        group = dict(normalized_identifier=key, kind='paper',
                     raw_canonical_identifiers=[], explicit_aliases=[], record_indices=[])
        groups.append(group)
    if paper['canonical_id'] not in group['raw_canonical_identifiers']:
        group['raw_canonical_identifiers'].append(paper['canonical_id'])
    group['record_indices'].append(position)

documents = {
    'PAPER_CONCLUSIONS.json': 'structured_paper_conclusions',
    'READ_SCOPES.json': 'bounded_primary_read_scopes',
    'REPORT.md': 'conditional_composition_assessment',
    'CE_CURVATURE_DERIVATION.md': 'source_math_assessment',
    'CANDIDATE_SPEC.json': 'unadopted_prospective_candidate_specification',
    'RESOURCE_ESTIMATE.json': 'prospective_resource_plan',
}
for relative, kind in documents.items():
    path = packet / relative
    assert path.is_file()
    row = dict(path=str(path.relative_to(phase)), sha256=sha(path), kind=kind,
               sealed_packet_manifest_sha256=sha(manifest),
               scope='Saved scoped conclusions or conditional proposal. No measured gain or execution admission.')
    if relative == 'PAPER_CONCLUSIONS.json':
        row['records'] = 2
    index['existing_packets'].append(row)

now = datetime.now(timezone.utc).isoformat()
index['created_UTC'] = now
index['predecessor_index'] = str(prior_path.relative_to(phase))
index['predecessor_index_sha256'] = sha(prior_path)
account = index['read_accounting']
catalog_paths = {r['path'] for r in index['existing_packets']}
conclusion_paths = {r['conclusion_file'] for r in index['paper_records']}
scope_paths = {r['read_scope_reference']['path'] for r in index['paper_records']
               if 'read_scope_reference' in r}
account.update(conclusion_records=len(index['paper_records']),
               normalized_paper_identifiers=sum(g['kind'] == 'paper' for g in groups),
               software_documentation_identifiers=sum(g['kind'] != 'paper' for g in groups),
               unique_conclusion_source_documents=len(conclusion_paths),
               catalog_entries=len(index['existing_packets']),
               unique_catalog_document_paths=len(catalog_paths),
               unique_referenced_document_paths=len(catalog_paths | conclusion_paths),
               unique_scope_reference_document_paths=len(scope_paths),
               cataloged_source_math_conclusions_not_paper_records=
                   account['cataloged_source_math_conclusions_not_paper_records'] + 1,
               latest_adoption_packets=1, latest_packet_new_scoped_primary_reads=2,
               latest_packet_full_primary_reads=0, latest_packet_retained_primary_revisits=0,
               integration_pass_new_primary_reads=0, integration_pass_full_primary_reads=0,
               integration_pass_retained_primary_revisits=0)
index['latest_adoption'] = dict(
    adopted_UTC=now,
    previous_adoption_reference='literature_memory/index_v21/ADOPTION_RECEIPT.json',
    packet_bindings=[dict(packet=packet.name, manifest_sha256=sha(manifest),
                         seal_sha256=sha(packet / 'SEAL.json'), payload_files_verified=len(paths),
                         state='ADOPTED_CONCLUSIONS_ONLY')],
    novelty_or_execution_authorized=False)
assert index['paper_records'][:len(prior['paper_records'])] == prior['paper_records']
assert index['existing_packets'][:len(prior['existing_packets'])] == prior['existing_packets']
output.mkdir(exist_ok=False)
(output / 'LITERATURE_INDEX.json').write_text(json.dumps(index, indent=2) + '\n')
(output / 'README.md').write_text(
    '# Literature memory v22\n\n'
    f"{account['conclusion_records']} saved conclusion records cover "
    f"{account['normalized_paper_identifiers']} normalized paper identifiers and two software identifiers. "
    'The two added scoped method reads are SAM v1 and MAML v1. Neither is counted as a full-paper read. '
    'Gradient Starvation was retrieved but its method remains unread and supplies no substantive claim. '
    'This integration opened no primary text.\n\n'
    'The conditional graph-head covariance selector is retained as a mechanism question. '
    'Balanced affine contrasts can leave initial mean logits unchanged while changing later cross-entropy updates. '
    'Useful graph restrictions require equally selected random and altered-topology controls. '
    'No source driver, training launch, predictive gain or novelty claim is adopted. '
    'Current resources do not decide scientific merit.\n')
receipt = dict(schema='literature-index-v22-adoption-v1', UTC=now,
               predecessor_sha256=sha(prior_path), checked_packet_files=len(paths),
               preserved_prior_records=len(prior['paper_records']),
               preserved_prior_catalog_entries=len(prior['existing_packets']),
               read_accounting=account, integration_primary_text_read=False,
               scientific_arrays_read=False, novelty_or_execution_authorized=False,
               outputs=[dict(path=str(p.relative_to(phase)), bytes=p.stat().st_size, sha256=sha(p))
                        for p in (output / 'LITERATURE_INDEX.json', output / 'README.md')])
(output / 'ADOPTION_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(index_sha256=sha(output / 'LITERATURE_INDEX.json'),
                     conclusion_records=account['conclusion_records'],
                     normalized_papers=account['normalized_paper_identifiers'],
                     packet_files_verified=len(paths), execution_authorized=False)))
