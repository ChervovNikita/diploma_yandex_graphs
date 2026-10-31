"""Adopt already read conclusions after checking their immutable byte bindings."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

phase = Path(__file__).resolve().parents[1]
prepared = phase / 'literature_memory/index_v20_preparation_v1'
out = phase / 'literature_memory/index_v20'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(base, binding):
    path = base / binding['path']
    if path.stat().st_size != binding['bytes'] or digest(path) != binding['sha256']:
        raise ValueError(f'Changed input: {path}')


manifest = json.loads((prepared / 'MANIFEST.json').read_text())
for binding in manifest['payload']:
    check(prepared, binding)
bindings = json.loads((prepared / 'INPUT_BINDINGS.json').read_text())
verified = {}
for name, value in bindings.items():
    if isinstance(value, list):
        for binding in value:
            check(phase, binding)
        verified[name] = len(value)
prior = json.loads((phase / 'literature_memory/index_v19/LITERATURE_INDEX.json').read_text())
index = json.loads((prepared / 'DRAFT_LITERATURE_INDEX.json').read_text())
assert index['paper_records'][:92] == prior['paper_records']
assert index['existing_packets'][:103] == prior['existing_packets']
assert len(index['paper_records']) == 98
assert index['canonical_identifier_normalization']['groups'][:50] == prior['canonical_identifier_normalization']['groups']
assert len(index['canonical_identifier_normalization']['groups']) == 56
now = datetime.now(timezone.utc).isoformat()
pending = index.pop('pending_integration')
index.pop('preparation_status')
index['created_UTC'] = now
index['latest_adoption'] = {
    'adopted_UTC': now,
    'packet_bindings': pending['packet_bindings'],
    'previous_adoption_reference': 'literature_memory/index_v19/ADOPTION_RECEIPT.json',
    'preparation_manifest_sha256': digest(prepared / 'MANIFEST.json'),
    'novelty_or_execution_authorized': False,
}
for binding in index['latest_adoption']['packet_bindings']:
    binding['state'] = 'ADOPTED_CONCLUSIONS_ONLY'
account = index['read_accounting']
for name in list(account):
    if name.startswith('pending_') or name.startswith('preparation_'):
        account.pop(name)
account.update(state='ADOPTED', latest_adoption_packets=2,
               latest_packet_new_scoped_primary_reads=6,
               latest_packet_full_primary_reads=0,
               latest_packet_retained_primary_revisits=0,
               integration_pass_new_primary_reads=0,
               integration_pass_full_primary_reads=0,
               integration_pass_retained_primary_revisits=0)
out.mkdir(exist_ok=False)
(out / 'LITERATURE_INDEX.json').write_text(json.dumps(index, indent=2) + '\n')
(out / 'README.md').write_text(
    '# Literature memory v20\n\n'
    '98 retained conclusion records cover 54 normalized paper identifiers and two software identifiers. '
    'The six added method conclusions concern SEA, normalization ensembles, AdaGCN, GraphMerge, GRAND and link stacking. '
    'They are bounded method reads, not six full-paper reads. The integration performed no new primary reads.\n\n'
    'Broad diversity, warm starts, graph filtering and shared caches have precedents. '
    'A graph-error private-factor pulse remains a conditional mechanism hypothesis. '
    'Its acquisition, finite realization, topology comparison and full costs remain unresolved. '
    'This adoption creates no experiment, performance or novelty claim.\n\n'
    'Prior indexes, source packets, failed leads and read limits remain unchanged. '
    'The adoption receipt records the checked input hashes.\n')
receipt = dict(schema='literature-index-v20-adoption-receipt-v1', adopted_UTC=now,
               predecessor_sha256=digest(phase / 'literature_memory/index_v19/LITERATURE_INDEX.json'),
               preparation_manifest_sha256=digest(prepared / 'MANIFEST.json'),
               checked_binding_counts=verified,
               old_records_preserved=92, old_catalog_entries_preserved=103,
               adopted_conclusion_records=98, normalized_papers=54, software_ids=2,
               newly_adopted_scoped_method_reads=6, integration_new_primary_reads=0,
               newly_adopted_full_paper_reads=0, cumulative_full_read_total_certified=False,
               papers_or_scientific_arrays_read_in_integration=False,
               experiment_or_novelty_authorized=False,
               root_decision='Retain attributed ingredients and conditional pulse falsifiers, pending completed current cohort.',
               outputs=[dict(path=str(p.relative_to(phase)), sha256=digest(p), bytes=p.stat().st_size)
                        for p in [out / 'LITERATURE_INDEX.json', out / 'README.md']])
(out / 'ADOPTION_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(index_sha256=digest(out / 'LITERATURE_INDEX.json'), checked_binding_counts=verified,
                     conclusion_records=98, normalized_papers=54, software_ids=2)))
