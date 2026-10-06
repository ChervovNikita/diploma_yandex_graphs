"""Disabled allocation-only completion metadata/source-copy binding builder."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys

SOURCE_RELEASED = False
HERE = Path(__file__).resolve().parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
COLLECTOR = 'shared_private_transfer_allocation_replication_preparation_20261005_v2'
COLLECTOR_SHA = 'c29ccad053cfcaa4c11499854fdaca962ade48a910efa82435241d7d2f3e2ec6'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    return {'path': str(path.relative_to(PHASE)), 'sha256': sha(path)}


def exact_queue_handle_present(owner):
    path = Path('/proc') / str(owner['PID']) / 'stat'
    if not path.exists():
        return False
    value = path.read_text()
    return int(value[value.rfind(')') + 2:].split()[19]) == owner['start_ticks']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-authorized', action='store_true')
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(SOURCE_RELEASED and args.execute_authorized, 'Metadata builder remains disabled')
    require(sys.platform.startswith('linux') and socket.gethostname() == 'anogena-2-0'
            and Path.cwd().resolve() == REPO and PHASE.resolve() == PHASE,
            'Exact authorized allocation repository only')
    release_path = args.release.absolute()
    require(release_path.is_relative_to(PHASE) and not release_path.is_symlink()
            and release_path.stat().st_mode & 0o222 == 0 and sha(release_path) == args.release_sha256,
            'Exact immutable root metadata-only release required')
    authority = read(release_path)
    require(authority['schema'] == 'root_complete39_binding_builder_metadata_release_v1'
            and authority['root_metadata_builder_approved'] is True
            and authority['only_after_all_six_queues_terminal_cleanup'] is True
            and authority['source_sha256'] == sha(__file__)
            and authority['source_manifest_sha256'] == sha(HERE / 'MANIFEST.json')
            and authority['plan_sha256'] == sha(HERE / 'BINDING_PLAN.json'), 'Exact reviewed builder authority required')
    for key in ('fits_authorized', 'scores_read', 'history_bytes_observed', 'FREEZE_CONFIG_JSON_parsed',
                'tensor_deserialization', 'scientific_model_execution', 'historical_donor_change'):
        require(authority[key] is False, 'Builder admits no ' + key)
    for row in read(HERE / 'MANIFEST.json')['files']:
        pth = HERE / row['path']
        require(pth.resolve(strict=True).is_relative_to(HERE) and not pth.is_symlink()
                and pth.stat().st_size == row['bytes'] and sha(pth) == row['sha256'], 'Builder packet changed')
    require(authority['builder_review_evidence'] and authority['source_copy_review_evidence'],
            'Actual builder/source-copy reviews required')
    # Import only the exact reviewed stdlib metadata helpers after authority.
    sys.path.insert(0, str(PHASE / COLLECTOR))
    import protocol as p
    import collect39_metadata as m
    require(Path(p.__file__).resolve() == PHASE / COLLECTOR / 'protocol.py'
            and Path(m.__file__).resolve() == PHASE / COLLECTOR / 'collect39_metadata.py', 'Wrong metadata helpers')
    p.verify_packet(COLLECTOR_SHA)
    for row in authority['builder_review_evidence'] + authority['source_copy_review_evidence']:
        p.binding(PHASE, row)
    c, i = m.helpers()
    plan = read(HERE / 'BINDING_PLAN.json')
    collection = c.read_metadata(p.binding(PHASE, plan['collection_template']))
    require([(d['family'], d['block'], d['donor_directory_relative']) for d in collection['donors']]
            == [tuple(x) for x in plan['fixed_donors']], 'Exactly the original13 plus fixednew26 donors required')
    # All terminal/cleanup checks finish before any source-copy/output mutation.
    starts = {}
    for donor in collection['donors']:
        prefix = donor['donor_directory_relative']
        donor['replica_directory_relative'] = prefix  # Same-host contract; no tensor copy.
        q = c.read_metadata(c.phase_file(prefix + '/QUEUE.json'))
        block = c.read_metadata(c.phase_file(prefix + '/BLOCK_FREEZE.json'))
        start = c.read_metadata(c.phase_file(prefix + '/QUEUE_START.json'))
        expected = 10 if donor['family'] == 'original30' else 3
        require(block['selected_blocks_complete'] is True and block['physical_fits'] == expected
                and len(block['completed']) == expected and not c.phase_file(prefix + '/QUEUE.json').parent.joinpath('QUEUE_FAILURE.json').exists()
                and [x['cell_id'] for x in q['entries']] == [x['cell_id'] for x in block['completed']],
                'Predetermined complete whole block required')
        for receipt in block['completed']:
            c.terminal(receipt)
        require(not exact_queue_handle_present(start['supervisor_identity']), 'Exact registered queue is not yet closed')
        starts[(donor['family'], donor['block'])] = start
    output = args.output.absolute()
    require(output.is_relative_to(PHASE) and output.parent.is_dir() and not output.exists()
            and output.parent.resolve().is_relative_to(PHASE), 'Fresh source-metadata output only')
    output.mkdir()
    copy_root = output / 'source_copies'
    copied = {}

    def copy_one(original, expected_sha, expected_bytes=None):
        require(Path(original).suffix in ('.py', '.json', '.md', '.txt', '.patch')
                and Path(original).name not in ('FREEZE.json', 'CONFIG.json', 'VALID_HISTORY.jsonl'),
                'Only reviewed source/metadata bytes may be copied')
        source = c.phase_file(original)
        require(sha(source) == expected_sha and (expected_bytes is None or source.stat().st_size == expected_bytes),
                'Genuine original source bytes differ; no substitute snapshot')
        if original not in copied:
            target = copy_root / c.relative(original)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as src, target.open('xb') as dst:
                for chunk in iter(lambda: src.read(1024 * 1024), b''):
                    dst.write(chunk)
            target.chmod(0o444)
            require(sha(target) == expected_sha and sha(source) == expected_sha, 'Source copy changed')
            copied[original] = {'original_path': original, 'local_path': str(target.relative_to(PHASE)),
                                'sha256': expected_sha, 'bytes': target.stat().st_size}
        require(copied[original]['sha256'] == expected_sha, 'Duplicate original source has differing pins')

    for entry in plan['source_roots']:
        original, expected = entry['path'], entry['sha256']
        copy_one(original, expected)
        if entry['manifest']:
            for row in c.read_metadata(c.phase_file(original))['files']:
                copy_one(str(c.relative(original).parent / c.relative(row['path'])), row['sha256'], row['bytes'])
    inventory_path = output / 'SOURCE_COPY_INVENTORY.json'
    inventory = {'schema': 'authenticated_retrospective_source_copy_inventory_v1',
                 'UTC': datetime.now(timezone.utc).isoformat(), 'retrospective_source_copy_only': True,
                 'pre_fit_authority': False, 'provider': p.PROVIDER, 'hostname': 'anogena-2-0',
                 'repository': str(REPO), 'source_copy_review_evidence': authority['source_copy_review_evidence'],
                 'files': [copied[k] for k in sorted(copied)]}
    c.write(inventory_path, inventory)
    inventory_path.chmod(0o444)
    inventory_ref = ref(inventory_path)

    def source_ref(original):
        row = copied[original]
        return {'path': row['local_path'], 'sha256': row['sha256'], 'original_path': original,
                'copy_inventory': inventory_ref}

    for donor in collection['donors']:
        family, block, prefix = donor['family'], donor['block'], donor['donor_directory_relative']
        spec = plan['by_donor'][family + ':' + block]
        for key, name in (('queue_binding', 'QUEUE.json'), ('root_release_binding', 'ROOT_RELEASE.json'),
                          ('block_freeze_binding', 'BLOCK_FREEZE.json'), ('cohort_plan_binding', 'COHORT_PLAN.json')):
            donor[key] = ref(c.phase_file(prefix + '/' + name))
        donor['launch_receipt'] = spec['launch_receipt'] if block == 'b0' else ref(c.phase_file(prefix + '/LAUNCH_RECEIPT.json'))
        c.binding(donor['launch_receipt'])
        custody = {'training_manifest': source_ref(spec['training_manifest']),
                   'operational_manifest': source_ref(spec['operational_manifest']),
                   'external_dependencies': {name: source_ref(name) for name in c.EXTERNAL_TRAINING},
                   'supervisor': source_ref(c.SUPERVISOR[p.PROVIDER]['path']),
                   'operational_review_evidence': spec['operational_review_evidence']}
        donor['source_custody'] = custody
        if family == 'original30' and block == 'b0':
            donor['queue_sha256'] = donor['queue_binding']['sha256']
            donor['root_release_sha256'] = donor['root_release_binding']['sha256']
            donor['provider_admission'] = spec['provider_admission']
            donor['original_operational_custody'] = {
                'launch_receipt': donor['launch_receipt'], 'queue_start_binding': ref(c.phase_file(prefix + '/QUEUE_START.json')),
                'source_custody': custody, 'actual_launch_custody_review_evidence': spec['actual_launch_custody_review_evidence']}
        if family == 'companion9' and block == 'b0':
            donor.update(spec['genuine_b0_prefit_bindings'])
            q = c.read_metadata(c.phase_file(prefix + '/QUEUE.json'))
            job = c.read_metadata(c.phase_file(q['entries'][0]['job_relative']))
            donor['training_step_gate'] = job['training_step_gate']
    collection['collector_manifest_sha256'] = COLLECTOR_SHA
    collection['amendment_sha256'] = sha(PHASE / COLLECTOR / 'PROSPECTIVE_AMENDMENT.json')
    collection['attempt_history_sha256'] = sha(PHASE / COLLECTOR / 'ATTEMPT_HISTORY.json')
    collection['source_review_evidence'] = plan['collector_review_evidence']
    require(collection['root_collection_approved'] is False and collection['fits_authorized'] is False
            and collection['scores_read'] is False, 'This builder does not issue collection/scoring authority')
    # Reuse the reviewed complete39 authenticator; no history/FREEZE/CONFIG parse.
    amendment = c.read_metadata(p.binding(PHASE, plan['amendment']))
    records, _ = m.authenticate39(c, i, collection, amendment)
    require(len(records) == 39 and len({r['cell']['cell_id'] for r in records}) == 39, 'Full39 required')
    candidate = output / 'COLLECTION_RELEASE_CANDIDATE.json'
    c.write(candidate, collection)
    candidate.chmod(0o444)
    receipt = {'schema': 'complete39_genuine_binding_builder_metadata_receipt_v1',
               'complete39_authenticated': True, 'root_collection_approved': False,
               'collection_candidate': ref(candidate), 'source_copy_inventory': inventory_ref,
               'source_files_copied': len(copied), 'replica_directories_are_actual_donor_directories': True,
               'checkpoint_logit_dataset_copies': 0, 'history_or_FREEZE_CONFIG_semantics_read': False,
               'tensor_deserialization_or_scores_read': False, 'new_fits_or_donor_changes': False}
    c.write(output / 'BUILDER_RECEIPT.json', receipt)
    (output / 'BUILDER_RECEIPT.json').chmod(0o444)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
