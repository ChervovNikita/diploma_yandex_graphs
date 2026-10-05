#!/usr/bin/env python3
"""Validate future pip dry-run report and freeze only missing wheel bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--before-metadata', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    before = json.loads(args.before_metadata.read_text())
    env = report['environment']
    if env['python_version'] != '3.11' or env['sys_platform'] != 'linux':
        raise ValueError('Resolve native wheels for actual Python3.11/Linux')
    rows, names, lock = [], set(), []
    for item in report['install']:
        native = item['metadata']
        name = canonical(native['name'])
        version = native['version']
        if name in names or name in before['installed']:
            raise ValueError('Resolver proposes duplicate/replacement of an existing package: ' + name)
        if name in ('numpy', 'torch', 'torch-geometric', 'torch-sparse', 'torch-scatter'):
            raise ValueError('Protected numerical core may not be installed into this overlay')
        names.add(name)
        download = item['download_info']
        url = download['url']
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.hostname != 'files.pythonhosted.org' or not parsed.path.endswith('.whl'):
            raise ValueError('Expected ordinary PyPI binary wheel, no source build or unreviewed host')
        checksum = download['archive_info']['hashes']['sha256']
        if not re.fullmatch('[0-9a-f]{64}', checksum):
            raise ValueError('Exact wheel SHA256 absent')
        if '-cp312-' in parsed.path or '-cp310-cp310-' in parsed.path:
            raise ValueError('An interpreter-specific wheel from another host was reused')
        rows.append(dict(name=name, version=version, wheel=url, sha256=checksum,
                         Requires_Python=native.get('requires_python'),
                         Requires_Dist=native.get('requires_dist', [])))
        lock.append(native['name'] + ' @ ' + url + ' --hash=sha256:' + checksum + '\n')
    versions = {r['name']: r['version'] for r in rows}
    if any(versions.get(name) != version for name, version in before['primary'].items()):
        raise ValueError('All seven exact primary ancillary pins must resolve')
    if args.lock.exists() or args.summary.exists():
        raise ValueError('Preserve prior resolver result; use fresh output names')
    args.lock.write_text(''.join(lock))
    summary = dict(status='MISSING_WHEEL_SET_FROZEN_NOT_INSTALLED',
                   packages=rows, package_count=len(rows),
                   report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
                   before_metadata_sha256=hashlib.sha256(args.before_metadata.read_bytes()).hexdigest(),
                   lock_sha256=hashlib.sha256(args.lock.read_bytes()).hexdigest(),
                   core_or_existing_replacements=False, numerical_imports=False,
                   import_or_operator_readiness=False)
    args.summary.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(status=summary['status'], package_count=len(rows)), sort_keys=True))


if __name__ == '__main__':
    main()
