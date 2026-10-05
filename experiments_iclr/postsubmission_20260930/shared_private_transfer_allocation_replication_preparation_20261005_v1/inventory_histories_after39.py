#!/usr/bin/env python3
"""Separately disabled history byte custody after reauthentication of all fixed39."""
import argparse
from pathlib import Path
import protocol as p
import collect39_metadata as m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args();phase = m.PHASE
    release_path = p.file_in(phase,str(args.release.resolve(strict=True).relative_to(phase)))
    release = p.read(release_path)
    p.require(release['root_history_inventory_approved'] is True and release['all39_required'] is True
              and release['retrospective_custody_only'] is True
              and all(release[k] is False for k in ('fits_authorized','TEST_access','scores_read','pre_fit_authority')),
              'Disabled until a separate root retrospective history inventory release')
    p.verify_packet(release['inventory_manifest_sha256'])
    p.require(release['source_review_evidence'],'Reviewed inventory authority required')
    for ref in release['source_review_evidence']:p.binding(phase,ref)
    registry_path = p.binding(phase,release['complete39_registry']);registry = p.read(registry_path)
    p.require(registry['schema'] == 'authenticated_complete_prospective_allocation_replication39_v1'
              and registry['complete'] is True and registry['full39_terminal_source_artifact_custody_passed'] is True,
              'Actual complete39 amended custody registry required')
    collection_ref = {'path':str((registry_path.parent/'COLLECTION_RELEASE.json').relative_to(phase)),
                      'sha256':registry['collection_release_sha256']}
    collection = p.read(p.binding(phase,collection_ref))
    p.require(collection['root_collection_approved'] is True and collection['fits_authorized'] is False
              and collection['scores_read'] is False and collection['TEST_access'] is False
              and collection['collector_manifest_sha256'] == release['inventory_manifest_sha256'], 'Actual approved collection required')
    p.require(p.sha(p.HERE/'PROSPECTIVE_AMENDMENT.json') == registry['amendment_sha256'] == collection['amendment_sha256']
              and p.sha(p.HERE/'ATTEMPT_HISTORY.json') == collection['attempt_history_sha256'], 'Amended prospective history/donors differ')
    c,i = m.helpers();a = p.read(p.HERE/'PROSPECTIVE_AMENDMENT.json')
    records,inputs = m.authenticate39(c,i,collection,a)
    p.require(registry['input_identities'] == inputs
              and [r['cell_id'] for r in registry['completed']] == a['selected39_order'], 'Exact complete39 registry differs')
    for row,record in zip(registry['completed'],records):
        p.require(row['job_sha256'] == record['receipt']['job_sha256']
                  and row['freeze_sha256'] == record['receipt']['freeze_sha256']
                  and row['checkpoint_sha256'] == record['checkpoint']['sha256']
                  and row['VALID_logits_sha256'] == record['logits']['sha256'], 'Actual complete39 artifact identities changed')
    # All39 reauthentication and registry artifact crosschecks finish before the first history byte read.
    rows = []
    for record in records:
        path = c.phase_file(str((record['output']/'VALID_HISTORY.jsonl').relative_to(phase)))
        rows.append({'cell_id':record['cell']['cell_id'],'history_binding':c.reference(path),
                     'freeze_binding':c.reference(record['output']/'FREEZE.json')})
    output = c.fresh_output(args.output);output.mkdir()
    (output/'HISTORY_INVENTORY_RELEASE.json').write_bytes(release_path.read_bytes())
    c.write(output/'VALID_HISTORY_INVENTORY.json',{'schema':'retrospective_allocation_replication_full39_history_inventory_v1',
        'complete':True,'selected_logical_scientific_fits':39,'new_physical_fits':26,
        'created_after_all39_terminal_source_artifact_checks':True,'history_or_FREEZE_JSON_parsed':False,
        'scores_read':False,'pre_fit_authority':False,'complete39_registry':release['complete39_registry'],
        'records':rows,'release_binding':c.reference(release_path),'inventory_manifest_sha256':release['inventory_manifest_sha256'],
        'source_FREEZE_history_semantic_crosscheck_deferred_to_root_reviewed_D2_amended_registry_adapter':True})
    print('Observed all39 history byte custody after full39 reauthentication; no history values parsed.')


if __name__ == '__main__':
    main()
