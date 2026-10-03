"""Finish the preserved metadata preparation after a literal-header false positive."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib
import json
import runpy

PHASE = Path(__file__).resolve().parents[2]
PACKET = Path(__file__).resolve().parent
PUB = str(PACKET.relative_to(PHASE))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(name, value):
    with (PACKET / name).open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def validate_text(path):
    data = path.read_bytes()
    assert len(data) < 2_000_000 and b'\0' not in data, str(path)
    text = data.decode('utf-8')
    private_headers = {'-----BEGIN PRIVATE KEY-----', '-----BEGIN OPENSSH PRIVATE KEY-----',
                       '-----BEGIN RSA PRIVATE KEY-----', '-----BEGIN EC PRIVATE KEY-----'}
    assert not private_headers.intersection(line.strip() for line in text.splitlines()), str(path)
    if path.suffix == '.json':
        json.loads(text)
    elif path.suffix == '.jsonl':
        for line in text.splitlines():
            if line.strip():
                json.loads(line)


def main():
    prepare_name = 'publication/prepare_v6_v4_new_records_inventory_20261004_v1.py'
    prepare = runpy.run_path(str(PHASE / prepare_name), run_name='preserved_preparation_definitions_only')
    assert prepare['PUB'] == PUB
    local, row = prepare['local'], prepare['row']
    original = local('publication/publish_exact_inventory_v9.py').read_text()
    successor = local(prepare['HELPER']).read_text()
    assert successor == original.replace("source.suffix in ('.py', '.json', '.md',",
                                         "source.suffix in ('.py', '.json', '.jsonl', '.md',")
    old_defs = runpy.run_path(str(local('publication/publish_exact_inventory_v9.py')), run_name='v9_definitions_only')
    new_defs = runpy.run_path(str(local(prepare['HELPER'])), run_name='v10_definitions_only')
    assert old_defs['REMOTE'] == new_defs['REMOTE']
    baseline_read = json.loads((PACKET / 'BASELINE_GIT_READ_RECEIPT.json').read_text())
    assert baseline_read['exit_code'] == 0 and baseline_read['operation'] == 'read_only_baseline_git_metadata'
    baseline_value = json.loads(baseline_read['stdout'])
    assert baseline_value['head'] == prepare['BASE'] and baseline_value['no_staged_edits']
    baseline = {item['target']: item for item in baseline_value['files']}
    exclusions = json.loads((PACKET / 'EXCLUSIONS.json').read_text())
    excluded = {item['path'] for item in exclusions['nontext_or_large']}
    sources = set(prepare['EXPLICIT'])
    for name in prepare['ROOTS']:
        root = local(name)
        assert root.is_dir(), name
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            relative = str(path.relative_to(PHASE))
            if relative in excluded:
                continue
            assert '__pycache__' not in path.parts and path.suffix in prepare['SUFFIXES']
            sources.add(relative)
    candidates = [row(name) for name in sorted(sources)]
    unchanged, selected = [], []
    for item in candidates:
        prior = baseline.get(item['target'])
        if prior and prior['sha256'] == item['sha256']:
            unchanged.append(item)
        else:
            selected.append(item)
    assert {item['source'] for item in unchanged} == {item['source'] for item in exclusions['already_published_unchanged']}
    note = dict(schema='metadata_preparation_adapter_false_positive_receipt_v1',
                UTC=datetime.now(timezone.utc).isoformat(), status='PRESERVED_AND_FINALIZED_WITHOUT_SOURCE_OVERWRITE',
                failed_source=row(prepare_name), failed_exit_code=1,
                cause='The substring guard matched its own private-key-header string literals; no credential was found.',
                correction='New finalizer checks for actual standalone private-key-header lines.',
                baseline_read_reused=True, additional_remote_baseline_reads=False,
                scientific_source_overwrite=False, numerical_execution=False, process_monitoring=False,
                commit=False, push=False)
    write_json('METADATA_PREPARATION_ADAPTER_NOTE.json', note)
    extra = {prepare_name, prepare['HELPER'], PUB + '/FINALIZE_INVENTORY.py'}
    extra |= {PUB + '/' + name for name in ('BASELINE_GIT_READ_RECEIPT.json', 'BASELINE_TARGET_PINS.json',
              'README_MAIN_BEFORE.md', 'README_AMENDMENT.md', 'README_AMENDMENT.diff', 'HELPER_ALLOWLIST.diff',
              'EXCLUSIONS.json', 'METADATA_PREPARATION_ADAPTER_NOTE.json')}
    files = [row(PUB + '/README_MAIN.md', 'README.md')] + selected + [row(name) for name in sorted(extra)]
    assert len(files) == len({item['target'] for item in files})
    for item in files:
        validate_text(local(item['source']))
    manifests = [prepare['verify_manifest'](local(name)) for name in sorted(sources)
                 if PurePosixPath(name).name == 'MANIFEST.json']
    state_pins = [item for item in files if item['source'] in {'PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json'}]
    assert len(state_pins) == 3
    assert 'fresh V4 small-input numerical pair passed once' in local('PUBLIC_STATUS.md').read_text()
    assert 'V4 small-input value/gradient/RNG/update parity passed once' in local('RESEARCH_STATE.md').read_text()
    validation = dict(schema='narrow_new_records_publication_validation_v1', UTC=datetime.now(timezone.utc).isoformat(),
                      expected_head=prepare['BASE'], selected_records_before_validation=len(files),
                      selected_bytes_before_validation=sum(item['bytes'] for item in files),
                      already_published_unchanged=len(unchanged), excluded_nontext_or_large=len(excluded),
                      UTF8_text_and_JSON_JSONL_parsing=True, no_duplicate_targets=True, path_confinement=True,
                      maximum_file_bytes=max(item['bytes'] for item in files),
                      helper=dict(original_sha256=sha(original.encode()), successor_sha256=sha(successor.encode()),
                                  change='Add .jsonl to the local text suffix allowlist only',
                                  remote_body_byte_identical=True, original_source_preserved=True),
                      payload_manifest_checks=manifests, state_pins=state_pins,
                      scope=dict(roots=prepare['ROOTS'], explicit=sorted(prepare['EXPLICIT'])),
                      actions=dict(remote_read_only=True, numerical_work=False, process_monitoring=False,
                                   commit=False, push=False, source_overwrite=False, checkpoint_or_data_access=False))
    write_json('VALIDATION.json', validation)
    files.append(row(PUB + '/VALIDATION.json'))
    inventory = dict(schema='exact_research_publication_inventory_v1', expected_head=prepare['BASE'],
                     branch=old_defs['BRANCH'], message='Record Amazon original15 launch, NCNC V4 numerical PASS and literature integration',
                     files=files, remove=[], excludes=exclusions['nontext_or_large'],
                     already_published_unchanged=len(unchanged), preparation_only=True)
    write_json('INVENTORY.json', inventory)
    print(json.dumps(dict(inventory=PUB + '/INVENTORY.json', expected_head=prepare['BASE'], files=len(files),
                         bytes=sum(item['bytes'] for item in files), already_published_unchanged=len(unchanged),
                         nontext_or_large_excluded=len(excluded), manifests_checked=len(manifests))))


if __name__ == '__main__':
    main()
