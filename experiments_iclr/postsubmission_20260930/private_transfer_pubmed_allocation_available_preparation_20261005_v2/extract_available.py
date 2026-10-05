"""Extract exactly four authenticated TRAIN/VALID members; no model imports."""
import argparse
import json
from pathlib import Path
import tarfile
import time

from available_custody import ARCHIVE, ARCHIVE_BYTES, ARCHIVE_SHA, EXECUTION, EXPECTED, authorize, confined, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    recipe = authorize(Path(__file__), args.recipe, 'extract_available')
    output = confined(args.output, exists=False)
    if output != EXECUTION / 'available' or output.exists() or not output.parent.is_dir():
        raise ValueError('Exact fresh available directory required')
    archive_path = confined(ARCHIVE, exists=True)
    if not archive_path.is_file() or archive_path.stat().st_size != ARCHIVE_BYTES or sha(archive_path) != ARCHIVE_SHA:
        raise ValueError('Full resident public archive authentication failed')
    if time.monotonic() - started > recipe['bounds']['soft_seconds']:
        raise TimeoutError('Extraction soft bound reached')
    output.mkdir()
    folder = output / 'pubmed'
    folder.mkdir()
    records = {}
    try:
        # Stream through archive headers, reading only the four named contents.
        # No extractall, TEST member extraction, or unrelated content decoding.
        with tarfile.open(archive_path, 'r|gz') as archive:
            for member in archive:
                if member.name not in recipe['members']:
                    continue
                name = member.name[len('dataset/pubmed/'):]
                if name in records or not member.isfile() or not 0 < member.size < 67108864:
                    raise ValueError('Duplicate, nonregular or oversized required member')
                if time.monotonic() - started > recipe['bounds']['soft_seconds']:
                    raise TimeoutError('Extraction soft bound reached')
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError('Required member is unreadable')
                path = folder / name
                with stream, path.open('xb') as target:
                    while part := stream.read(1024 * 1024):
                        target.write(part)
                if path.stat().st_size != member.size or sha(path) != EXPECTED[name]:
                    raise ValueError('Exact original available member bytes differ')
                records[name] = {'relative_path': 'available/pubmed/' + name,
                    'sha256': EXPECTED[name], 'bytes': member.size, 'archive_member': member.name}
        if set(records) != set(EXPECTED):
            raise ValueError('Exactly four original available members required')
        records['train_pos.txt']['counts'] = {'raw_rows': 37676, 'self_loops': 0, 'native_nonself_rows': 37676}
        records['valid_pos.txt']['counts'] = {'raw_rows': 2216, 'self_loops': 0, 'native_nonself_rows': 2216}
        # Counts above are the pinned contract, to be checked by the separate
        # inspector; extraction alone does not establish their semantics.
        manifest = {'schema': 'allocation_Pubmed_original_available_extraction_v1',
            'files': records, 'archive_sha256': ARCHIVE_SHA,
            'TEST_available_to_loader': False, 'model_access': False,
            'fit_admission': False, 'input_semantics_inspected': False,
            'counts_are_pinned_contract_not_this_stage_measurements': True,
            'source_manifest_sha256': recipe['source_manifest_sha256'],
            'recipe_sha256': sha(args.recipe), 'elapsed_seconds': time.monotonic() - started}
        if manifest['elapsed_seconds'] > recipe['bounds']['soft_seconds']:
            raise TimeoutError('Extraction soft bound reached')
        with (EXECUTION / 'ACQUISITION_MANIFEST.json').open('x') as target:
            json.dump(manifest, target, indent=2, sort_keys=True)
            target.write('\n')
        print(json.dumps({'status': 'FOUR_MEMBERS_AUTHENTICATED_NOT_FIT_ADMITTED',
            'files': records, 'TEST_access': False, 'fit_admission': False}))
    except BaseException as error:
        with (output / 'FAILURE.json').open('x') as target:
            json.dump({'error': type(error).__name__ + ': ' + str(error),
                'partial_files_preserved': True, 'retry': False,
                'fit_admission': False}, target, indent=2)
            target.write('\n')
        raise


if __name__ == '__main__':
    main()
