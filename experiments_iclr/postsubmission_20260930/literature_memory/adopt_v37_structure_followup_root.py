"""Append one scoped prior conclusion. Preserve every previous index record."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT / 'literature_memory/index_v36'
PACKET = ROOT / 'graph_structure_conditioned_diversity_primary_followup_20261003_v1'
OUT = ROOT / 'literature_memory/index_v37'


def load(p):
    return json.loads(p.read_text())


def binding(p):
    b = p.read_bytes()
    return dict(path=str(p.relative_to(ROOT)), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def save(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


OUT.mkdir(exist_ok=False)
utc = datetime.now(timezone.utc).isoformat()
manifest = load(PACKET / 'MANIFEST.json')
assert binding(PACKET / 'MANIFEST.json')['sha256'] == '5239814fdfede414f83e7dd146e1fce5d3c668127a132615eacc2fd6eda138c1'
assert manifest['payload_count'] == len(manifest['payload']) == 24
for row in manifest['payload']:
    observed = binding(PACKET / row['path'])
    assert observed['sha256'] == row['sha256'] and observed['bytes'] == row['bytes']
old = load(PREV / 'LITERATURE_INDEX.json')
new = copy.deepcopy(old)
conclusions = load(PACKET / 'PAPER_CONCLUSIONS.json')
scopes = load(PACKET / 'READ_SCOPES.json')
assert len(old['paper_records']) == 156 and len(conclusions['new_primary_conclusions']) == 1
paper = conclusions['new_primary_conclusions'][0]
cid = paper['canonical_id'].casefold()
assert not any(r.get('canonical_id', '').casefold() == cid for r in old['paper_records'])
assert not any(g['normalized_identifier'].casefold() == cid for g in old['canonical_identifier_normalization']['groups'])
assert scopes['new_primary_read_count'] == 1 and scopes['new_full_paper_certifications'] == 0
assert scopes['retained_primary_method_rereads'] == 0
assert conclusions['disposition'] == 'NO_SUPPORTED_DISTINCT_GAP_ZERO_PROMOTED_PILOTS'
new['paper_records'].append(dict(canonical_id=cid,
    conclusion_file=str((PACKET/'PAPER_CONCLUSIONS.json').relative_to(ROOT)),
    conclusion_file_sha256=binding(PACKET/'PAPER_CONCLUSIONS.json')['sha256'],
    conclusion=paper, exact_read_scope=scopes['new_primary_scoped_method_reads'][0],
    source_report_reference=binding(PACKET/'REPORT.md'),
    source_packet_manifest_reference=binding(PACKET/'MANIFEST.json'),
    integration_primary_method_read=False))
new['canonical_identifier_normalization']['groups'].append(dict(kind='paper',
    normalized_identifier=cid, raw_canonical_identifiers=[cid], record_indices=[156], explicit_aliases=[]))
new['existing_packets'].append(dict(**binding(PACKET/'REPORT.md'),
    kind='bounded_graph_structure_diversity_primary_followup',
    scope='One scoped dynamic-NCL method. Fixed graph-error transforms reduce to graph-kernel NCL. Zero promoted pilots.'))
event = dict(UTC=utc, predecessor=binding(PREV/'LITERATURE_INDEX.json'),
    source_manifest=binding(PACKET/'MANIFEST.json'), added_conclusion_records=1,
    new_normalized_paper_identities=[cid], source_packet_first_scoped_primary_method_reads=1,
    integration_primary_method_reads=0, source_packet_full_paper_certifications=0,
    promoted_pilots=0, methodological_novelty_established=False,
    metadata_only_unread=['doi:10.1109/icassp55912.2026.11460998'],
    blocked_GENN_routes_retried=0,
    no_global_novelty_absence_claim=True)
new['predecessor_index'] = binding(PREV/'LITERATURE_INDEX.json')['path']
new['predecessor_index_sha256'] = binding(PREV/'LITERATURE_INDEX.json')['sha256']
new['created_UTC'] = utc
new['latest_adoption'] = event
new['post_v36_append'] = event
new['read_accounting']['conclusion_records'] = 157
new['read_accounting']['normalized_paper_identifiers'] = 108
new['read_accounting']['software_documentation_identifiers'] = 2
new['read_accounting']['latest_index_growth'] = event
new['read_accounting']['full_paper_read_total_certified'] = False
new['read_accounting']['cumulative_scoped_or_full_read_totals_certified'] = False
new['read_accounting']['latest_packet_new_scoped_primary_reads'] = 1
new['read_accounting']['latest_packet_retained_primary_revisits'] = 0
new['read_accounting']['integration_pass_primary_method_reads'] = 0
assert new['paper_records'][:156] == old['paper_records']
assert new['canonical_identifier_normalization']['groups'][:-1] == old['canonical_identifier_normalization']['groups']
save('LITERATURE_INDEX.json', new)
save('INTEGRATION.json', event)
save('MANIFEST.json', dict(schema='append_only_literature_metadata_manifest_v37', UTC=utc,
    files=[{**binding(OUT/n), 'path':n} for n in ('LITERATURE_INDEX.json','INTEGRATION.json')]))
print(json.dumps(dict(index=binding(OUT/'LITERATURE_INDEX.json'), conclusion_records=157,
    paper_identities=108, software_identities=2, promoted_pilots=0)))
