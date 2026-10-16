"""Freeze rebinding-only downstream packages after root confirms port4's SHA."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]
SPECS = (
    ('qualification/all_fit_smoke_20261016_v3', 'qualification/all_fit_smoke_20261016_v4', 'test_source', 4),
    ('campaign_execution_20261016_v4', 'campaign_execution_20261016_v5', 'test_execution', 9),
    ('server51_campaign_projection_cache_v3', 'server51_campaign_projection_cache_v4', 'test_cache', 1),
)
OLD_PORT_SHA = 'f8f5a2092b8b8658f7a35e043598e5a1375f287ea9f07494c3f8b391503861b5'
REPLACEMENTS = (
    ('server51_campaign_port_v3', 'server51_campaign_port_v4'),
    ('qualification/all_fit_smoke_20261016_v3', 'qualification/all_fit_smoke_20261016_v4'),
    ('campaign_execution_20261016_v4', 'campaign_execution_20261016_v5'),
    ('server51_campaign_projection_cache_v3', 'server51_campaign_projection_cache_v4'),
    (OLD_PORT_SHA, 'PORT4_MANIFEST_SHA_PENDING'), ('port3', 'port4'),
)


def pin(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def rebound(text):
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port-sha256', required=True)
    args = parser.parse_args()
    assert re.fullmatch('[0-9a-f]{64}', args.port_sha256)
    port = E / 'server51_campaign_port_v4'
    assert pin(port / 'MANIFEST.json')['sha256'] == args.port_sha256, 'Confirmed port4 identity required'
    for row in json.loads((port / 'MANIFEST.json').read_text())['files']:
        assert pin(port / row['path']) == {k: row[k] for k in ('bytes', 'sha256')}, row['path']
    output = HERE / 'PORT4_DOWNSTREAM_SOURCE_FREEZE.json'
    assert not output.exists(), 'Preserve previous freeze receipt'
    parents = {}
    for old, new, _, _ in SPECS:
        parent, folder = E / old, E / new
        assert not (folder / 'MANIFEST.json').exists(), new
        assert (folder / 'DRAFT_NOT_ADMITTED.json').exists(), new
        manifest = json.loads((parent / 'MANIFEST.json').read_text())
        parents[new] = copy.deepcopy(manifest)
        for row in manifest['files']:
            assert pin(parent / row['path']) == {k: row[k] for k in ('bytes', 'sha256')}, row['path']
            expected = rebound((parent / row['path']).read_text())
            actual = (folder / row['path']).read_text()
            # Allows an interrupted pre-freeze retry after filling the same pin.
            assert actual in (expected, expected.replace('PORT4_MANIFEST_SHA_PENDING', args.port_sha256)), row['path']
            final = expected.replace('PORT4_MANIFEST_SHA_PENDING', args.port_sha256)
            if (folder / row['path']).suffix == '.py':
                ast.parse(final)
            (folder / row['path']).write_text(final)
    checks = []
    temp = HERE / '.port4_focused_test_tmp'
    temp.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(temp))
    try:
        for _, new, test, expected_count in SPECS:
            command = [sys.executable, '-B', '-m', 'unittest', test, '-v']
            child = subprocess.run(command, cwd=E / new, env=env, capture_output=True, text=True)
            count = re.search(r'Ran (\d+) tests?', child.stderr)
            checks.append({'package': new, 'command': command, 'returncode': child.returncode,
                           'tests': int(count.group(1)) if count else None,
                           'stdout': child.stdout, 'stderr': child.stderr})
            assert child.returncode == 0 and count and int(count.group(1)) == expected_count, child.stderr
    finally:
        temp.rmdir()
    assert sum(row['tests'] for row in checks) == 14
    assert pin(port / 'MANIFEST.json')['sha256'] == args.port_sha256
    for old, new, _, _ in SPECS:
        folder, parent = E / new, E / old
        manifest = parents[new]
        if (folder / 'SOURCE_CHECKS.json').exists():
            check = next(row for row in checks if row['package'] == new)
            write(folder / 'SOURCE_CHECKS.json', {'status': 'PASS', 'source_only': True,
                'native_executions': 0, 'remote_calls': 0, 'upstream_manifest_sha256': args.port_sha256, **check})
        manifest.update(predecessor_manifest_sha256=pin(parent / 'MANIFEST.json')['sha256'],
                        source_rebinding_only=True, upstream_port_manifest_sha256=args.port_sha256)
        if 'frozen_parent_manifest_sha256' in manifest:
            manifest['frozen_parent_manifest_sha256'] = args.port_sha256
        manifest['files'] = [{'path': row['path'], **pin(folder / row['path'])} for row in manifest['files']]
        write(folder / 'MANIFEST.json', manifest)
        (folder / 'DRAFT_NOT_ADMITTED.json').rename(folder / 'PREPARATION_DRAFT_HISTORY.json')
    wrapper = E / 'server51_campaign_projection_cache_v4/prepare_cached.py'
    preparation = HERE / 'prepare_port4.py'
    text = preparation.read_text()
    assert 'PORT4_MANIFEST_SHA_PENDING' in text and 'CACHE4_WRAPPER_SHA_PENDING' in text
    text = text.replace('PORT4_MANIFEST_SHA_PENDING', args.port_sha256).replace('CACHE4_WRAPPER_SHA_PENDING', pin(wrapper)['sha256'])
    ast.parse(text); preparation.write_text(text)
    helper = HERE / 'collect_global_smoke_barrier_port4.py'
    previous = (HERE / 'collect_global_smoke_barrier_port3.py').read_text()
    expected = rebound(previous).replace('execution-v3 worker plan', 'execution-v5 worker plan')
    assert helper.read_text() == expected
    ast.parse(helper.read_text())
    report = {'port_manifest': pin(port / 'MANIFEST.json'), 'cache_wrapper': pin(wrapper),
        'sources': {new: pin(E / new / 'MANIFEST.json') for _, new, _, _ in SPECS},
        'coordinator': {'path': helper.name, **pin(helper)},
        'prepare_port4_helper': pin(preparation), 'checks': checks,
        'focused_tests_passed': 14, 'native_executions': 0, 'remote_calls': 0,
        'settings_resources_scopes_IDs_changed': False, 'source_rebinding_only': True}
    write(output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
