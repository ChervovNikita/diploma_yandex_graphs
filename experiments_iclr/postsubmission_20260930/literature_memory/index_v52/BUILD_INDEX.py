"""Adopt one completed AM-GCN scope without rereading primary methods."""
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json

HERE=Path(__file__).resolve().parent
P=HERE.parent.parent
PREV=P/'literature_memory/index_v51'
PACKET=P/'shared_graph_view_quality_prior_update_20261004_v1'


def ref(path):
    raw=path.read_bytes()
    return dict(path=str(path.relative_to(P)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def save(name,value):
    with (HERE/name).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def metrics(index):
    catalog={r['path'] for r in index['existing_packets']}
    conclusions={r['conclusion_file'] for r in index['paper_records'] if 'conclusion_file' in r}
    scopes={r['read_scope_reference']['path'] for r in index['paper_records'] if 'read_scope_reference' in r}
    legacy={r['read_scope_file_reference']['path'] for r in index['paper_records'] if 'read_scope_file_reference' in r}
    groups=index['canonical_identifier_normalization']['groups']
    return dict(conclusion_records=len(index['paper_records']),normalized_paper_identifiers=sum(g['kind']=='paper' for g in groups),
        software_documentation_identifiers=sum(g['kind']!='paper' for g in groups),catalog_entries=len(index['existing_packets']),
        unique_catalog_document_paths=len(catalog),unique_conclusion_source_documents=len(conclusions),
        unique_referenced_document_paths=len(catalog|conclusions),unique_scope_reference_document_paths=len(scopes),
        unique_scope_reference_document_paths_including_legacy_field_alias=len(scopes|legacy))


def main():
    utc=datetime.now(timezone.utc).isoformat()
    previous=json.loads((PREV/'LITERATURE_INDEX.json').read_text())
    previous_manifest=json.loads((PREV/'MANIFEST.json').read_text())
    old_row=next(r for r in previous_manifest['files'] if r['path']=='LITERATURE_INDEX.json')
    assert ref(PREV/'LITERATURE_INDEX.json')['sha256']==old_row['sha256']
    manifest=json.loads((PACKET/'MANIFEST.json').read_text())
    for r in manifest['files']:
        actual=ref(PACKET/r['path'])
        assert actual['bytes']==r['bytes'] and actual['sha256']==r['sha256']
    scopes=json.loads((PACKET/'READ_SCOPES.json').read_text())
    conclusions=json.loads((PACKET/'PAPER_CONCLUSIONS.json').read_text())
    assert scopes['new_scoped_primary_method_identities']==1 and scopes['new_full_paper_reads']==0
    assert len(scopes['papers'])==len(conclusions['paper_records'])==1
    scope=scopes['papers'][0];paper=conclusions['paper_records'][0];canonical=paper['canonical_id']
    assert canonical==scope['canonical_id']=='arxiv:2007.02265'
    assert not any(g['normalized_identifier']==canonical or canonical in g['raw_canonical_identifiers'] for g in previous['canonical_identifier_normalization']['groups'])
    index=copy.deepcopy(previous)
    index['schema']='literature-memory-index-v52'
    index['created_UTC']=utc
    index['integration_v52_predecessor_v51_snapshot']={k:copy.deepcopy(previous.get(k)) for k in ('latest_adoption','read_accounting','predecessor_index','predecessor_index_sha256')}
    index['predecessor_index']=str((PREV/'LITERATURE_INDEX.json').relative_to(P))
    index['predecessor_index_sha256']=ref(PREV/'LITERATURE_INDEX.json')['sha256']
    position=len(index['paper_records'])
    key=hashlib.sha256(json.dumps(dict(canonical=canonical,version=scope['versioned_id'],ranges=scope['zero_based_block_ranges_inclusive']),sort_keys=True).encode()).hexdigest()
    record=dict(canonical_id=canonical,normalized_identifier=canonical,conclusion=paper,compact_primary_scope=scope,
        conclusion_file=str((PACKET/'PAPER_CONCLUSIONS.json').relative_to(P)),conclusion_file_sha256=ref(PACKET/'PAPER_CONCLUSIONS.json')['sha256'],
        read_scope_reference=ref(PACKET/'READ_SCOPES.json'),exact_read_scope=scope,
        source_packet_manifest_reference=ref(PACKET/'MANIFEST.json'),source_report_reference=ref(PACKET/'REPORT.md'),
        primary_payload_reference=ref(PACKET/scope['source_path']),scope_deduplication_key_sha256=key,
        full_paper_read=False,global_novelty_clearance=False,predictive_adoption=False,execution_authorized=False,
        integration_pass_primary_reread=False,integration_pass_new_semantic_read=False,
        integration_read_status='previously completed scoped read; integration adds no primary read',
        genuinely_new_scoped_paper_identity_in_source_packet=True)
    index['paper_records'].append(record)
    index['canonical_identifier_normalization']['groups'].append(dict(normalized_identifier=canonical,kind='paper',
        raw_canonical_identifiers=[canonical,'arxiv:'+scope['versioned_id'],'doi:'+paper['doi']],
        record_indices=[position],explicit_aliases=['doi:'+paper['doi']]))
    for name in ('REPORT.md','PAPER_CONCLUSIONS.json','READ_SCOPES.json','PRIMARY_PASSAGES.json','MEMORY_REUSE.json','RETRIEVAL.json','MANIFEST.json'):
        index['existing_packets'].append(dict(ref(PACKET/name),kind='stored_scoped_graph_evidence_sharing_prior',
            scope='One completed AM-GCN method scope; DGCN metadata-only. No primary rereading or scientific adoption in integration.'))
    index['latest_adoption']=dict(UTC=utc,source=ref(PACKET/'MANIFEST.json'),added_records=[dict(canonical_id=canonical,
        record_index=position,scope_deduplication_key_sha256=key,full_paper_read=False,integration_new_primary_reads=0)],
        metadata_only_unread_lead=scopes['metadata_only_unread_lead'],scientific_or_novelty_adoption=False)
    index['post_v52_append']=copy.deepcopy(index['latest_adoption'])
    account=copy.deepcopy(previous['read_accounting']);account.update(metrics(index))
    account.update(latest_packet_genuinely_new_scoped_paper_identities=1,latest_packet_new_scoped_primary_reads=1,
        latest_packet_full_primary_reads=0,integration_pass_primary_method_reads=0,integration_pass_new_primary_reads=0,
        integration_pass_full_primary_reads=0,integration_pass_retained_primary_revisits=0,
        full_paper_read_total_certified=False,cumulative_scoped_or_full_read_totals_certified=False,
        historical_path_catalog_note='All v51 records preserved; AM-GCN sharing/ablation scope added. DGCN remains unread. Integration adds zero primary reads.')
    index['read_accounting']=account
    assert index['paper_records'][:-1]==previous['paper_records']
    assert metrics(index)['normalized_paper_identifiers']==162 and len(index['paper_records'])==213
    save('LITERATURE_INDEX.json',index)
    save('VERIFICATION.json',dict(UTC=utc,predecessor=ref(PREV/'LITERATURE_INDEX.json'),source_manifest=ref(PACKET/'MANIFEST.json'),
        all_source_payload_hashes_verified=True,previous_records_unchanged=True,metrics=metrics(index),
        integration_primary_reads=0,source_completed_new_scoped_reads=1,source_full_paper_reads=0,
        DGCN_method_read=False,scientific_gain_or_novelty_claim=False))
    rows=[dict(path=str(f.relative_to(HERE)),bytes=f.stat().st_size,sha256=ref(f)['sha256']) for f in sorted(HERE.iterdir()) if f.is_file()]
    save('MANIFEST.json',dict(schema='literature-memory-manifest-v52',created_UTC=utc,files=rows))
    save('SEAL.json',dict(schema='literature-memory-seal-v52',manifest_sha256=ref(HERE/'MANIFEST.json')['sha256'],immutable=True))
    print(json.dumps(metrics(index)))


if __name__=='__main__':main()
