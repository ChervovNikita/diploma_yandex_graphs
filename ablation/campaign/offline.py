"""Build portable input hashes and check the prepared campaign without networking.

The build subcommand writes one manifest. The check subcommand reads source,
installed packages and prepared data, prints a JSON report and performs no
training. Python networking is rejected during the entire check, including
package imports. Run check with -B after sourcing environment.sh.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import sys

from .schema import canonical_hash, load_plan, relative_path
from .store import atomic_json

REPOSITORY = Path(__file__).resolve().parents[2]
DISTRIBUTIONS = (
    'torch', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'dgl',
    'torchdata', 'ogb', 'datasketch', 'numpy', 'scipy', 'pandas', 'scikit-learn',
    'packaging', 'pyyaml', 'pydantic', 'psutil',
)
IMPORTS = (
    'torch', 'torch_geometric', 'torch_scatter', 'torch_sparse', 'dgl',
    'torchdata', 'ogb', 'datasketch', 'numpy', 'scipy', 'pandas', 'sklearn',
    'models', 'datasets',
    'ablation.p09_molecular.models', 'ablation.p11_collab_buddy.models',
    'ablation.p13_node_contrast.models',
    'ablation.p12_collab_ncn.models',
)
NODE_DATASETS = ('roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions')
OGB_DATASETS = ('ogbg-molbace', 'ogbg-molhiv', 'ogbl-collab')
SOURCE_SUFFIXES = {'.py', '.sh', '.sbatch', '.md', '.yml', '.yaml', '.toml', '.json', '.csv', '.txt'}


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    return {'bytes': path.stat().st_size, 'sha256': digest_file(path)}


def inside_repository(path):
    path = Path(path).resolve()
    if not path.is_relative_to(REPOSITORY):
        raise ValueError(f'Campaign files must remain inside {REPOSITORY}')
    return path


def source_files():
    files = {path for path in REPOSITORY.iterdir()
             if path.is_file() and path.suffix in SOURCE_SUFFIXES}
    for name in ('ablation', 'external_sources'):
        for path in (REPOSITORY / name).rglob('*'):
            if not path.is_file() or {'__pycache__', '.git', 'runs', 'test_runs', 'evidence', 'cache'} & set(path.parts):
                continue
            if path.suffix in SOURCE_SUFFIXES or path.name in {'LICENSE', 'NOTICE'}:
                files.add(path)
    return sorted(files)


def package_source_files():
    # OGB dataset metadata and atom/bond encoders are runtime inputs as well as
    # the vendored reference models. Bind every installed OGB text source.
    origin = importlib.util.find_spec('ogb').origin
    package = Path(origin).resolve().parent
    files = {}
    for path in sorted(package.rglob('*')):
        if path.is_file() and path.suffix in {'.py', '.csv', '.json', '.txt'}:
            files[str(path.relative_to(package))] = record(path)
    return files


def versions():
    # Required packages must exist. Record the complete environment so indirect
    # numerical/preprocessing dependencies are bound as well.
    for name in DISTRIBUTIONS:
        importlib.metadata.version(name)
    return dict(sorted((package.metadata['Name'].lower().replace('_', '-'), package.version)
                       for package in importlib.metadata.distributions()
                       if package.metadata['Name']))


def build(args):
    destination = inside_repository(args.manifest)
    plan_path = inside_repository(args.plan) if args.plan else None
    sources = [path for path in source_files() if path.resolve() != destination]
    data = [path for path in sorted((REPOSITORY / 'data').rglob('*'))
            if path.is_file() and path.suffix not in {'.lock', '.part'}]
    caches = [path for path in sorted((REPOSITORY / 'ablation/p11_collab_buddy/cache').rglob('*'))
              if path.is_file() and path.suffix not in {'.lock', '.part'}]
    files = {str(path.relative_to(REPOSITORY)): record(path) for path in sources + data + caches}
    manifest = {
        'schema': 'gnnm-campaign-offline-v1',
        'python': platform.python_version(), 'platform': platform.system(),
        'machine': platform.machine(), 'versions': versions(),
        'files': files, 'ogb_package_sources': package_source_files(),
        'data_bytes': sum(path.stat().st_size for path in data),
        'source_files': len(sources), 'data_files': len(data),
        'cache_files': len(caches), 'cache_bytes': sum(path.stat().st_size for path in caches),
        'source_paths': [str(path.relative_to(REPOSITORY)) for path in sources],
    }
    if plan_path:
        plan = load_plan(plan_path)
        manifest['plan'] = {'path': str(plan_path.relative_to(REPOSITORY)),
                            'sha256': canonical_hash(plan), 'tasks': len(plan['tasks'])}
    atomic_json(destination, manifest)
    print(json.dumps({'manifest': str(destination), 'files': len(files),
                      'data_bytes': manifest['data_bytes'],
                      'manifest_sha256': digest_file(destination)}, indent=2))
    return 0


def install_network_audit():
    attempts = []

    def audit(event, arguments):
        blocked = event in {'socket.getaddrinfo', 'socket.gethostbyname',
                            'socket.gethostbyaddr', 'urllib.Request'}
        if event == 'socket.connect':
            blocked = arguments[0].family in (socket.AF_INET, socket.AF_INET6)
        if blocked:
            attempts.append(event)
            raise RuntimeError(f'Offline qualification rejected networking event {event}')

    sys.addaudithook(audit)
    return attempts


def prepared_data_summary():
    import numpy as np
    from ogb.graphproppred import PygGraphPropPredDataset
    from ogb.linkproppred import PygLinkPropPredDataset

    summaries = {}
    for name in NODE_DATASETS:
        path = REPOSITORY / 'data' / (name.replace('-', '_') + '.npz')
        with np.load(path) as data:
            summaries[name] = {'nodes': len(data['node_features']), 'edges': len(data['edges']),
                               'mask_shapes': {role: list(data[role + '_masks'].shape)
                                               for role in ('train', 'val', 'test')}}
    for name in OGB_DATASETS:
        directory = REPOSITORY / 'data' / name.replace('-', '_')
        if not (directory / 'processed/geometric_data_processed.pt').is_file():
            raise FileNotFoundError(f'Preprocessed OGB input is missing for {name}')
        if not list(directory.glob('RELEASE_v*.txt')):
            raise FileNotFoundError(f'OGB release marker is missing for {name}')
        constructor = PygLinkPropPredDataset if name.startswith('ogbl') else PygGraphPropPredDataset
        dataset = constructor(name=name, root=str(REPOSITORY / 'data'))
        splits = dataset.get_edge_split() if name.startswith('ogbl') else dataset.get_idx_split()
        summaries[name] = {'graphs': len(dataset), 'metric': dataset.eval_metric,
                           'split_roles': sorted(splits)}
        if name.startswith('ogbl'):
            summaries[name]['nodes'] = dataset[0].num_nodes
    return summaries


def check(args):
    sys.dont_write_bytecode = True
    os.environ['OGB_SKIP_UPDATE_CHECK'] = '1'
    attempts = install_network_audit()
    manifest_path = inside_repository(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'gnnm-campaign-offline-v1':
        raise ValueError('Unsupported offline manifest schema')
    failures = []
    for name, expected in manifest['files'].items():
        path = inside_repository(REPOSITORY / relative_path(name))
        if not path.is_file():
            failures.append({'file': name, 'reason': 'missing'})
        elif record(path) != expected:
            failures.append({'file': name, 'reason': 'content_changed'})
    current_sources = [str(path.relative_to(REPOSITORY)) for path in source_files()
                       if path.resolve() != manifest_path]
    if current_sources != manifest['source_paths']:
        failures.append({'reason': 'source_inventory_changed'})
    current_versions = versions()
    if current_versions != manifest['versions']:
        failures.append({'reason': 'package_versions_changed', 'actual': current_versions})
    if platform.python_version() != manifest['python']:
        failures.append({'reason': 'python_version_changed', 'actual': platform.python_version()})
    if platform.machine() != manifest['machine'] or platform.system() != manifest['platform']:
        failures.append({'reason': 'platform_changed'})
    if package_source_files() != manifest['ogb_package_sources']:
        failures.append({'reason': 'installed_ogb_sources_changed'})
    if manifest.get('plan'):
        plan_path = inside_repository(REPOSITORY / relative_path(manifest['plan']['path']))
        if canonical_hash(load_plan(plan_path)) != manifest['plan']['sha256']:
            failures.append({'reason': 'campaign_plan_changed'})
    report = {'schema': 'gnnm-campaign-offline-check-v1', 'checked_files': len(manifest['files']),
              'manifest_sha256': digest_file(manifest_path), 'failures': failures,
              'network_attempts': attempts, 'dataset_loading': not args.skip_dataset_load}
    if failures:
        report['passed'] = False
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    imported = []
    for name in IMPORTS:
        importlib.import_module(name)
        imported.append(name)
    from ablation.p09_molecular.protocol import provenance
    from ablation.p11_collab_buddy.protocol import source_identity as buddy_identity
    from ablation.p12_collab_ncn.protocol import source_identity as ncn_identity
    provenance()
    buddy_identity()
    ncn_identity()
    report['imports'] = imported
    if not args.skip_dataset_load:
        report['datasets'] = prepared_data_summary()
    report['passed'] = not attempts
    report['target_gpu_driver_check'] = 'pending until destination CUDA smoke'
    report['target_cross_host_locks'] = 'pending until destination multi-host qualification'
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['passed'] else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    build_parser = commands.add_parser('build')
    build_parser.add_argument('--manifest', type=Path, required=True)
    build_parser.add_argument('--plan', type=Path)
    check_parser = commands.add_parser('check')
    check_parser.add_argument('--manifest', type=Path, required=True)
    check_parser.add_argument('--skip-dataset-load', action='store_true')
    return result


if __name__ == '__main__':
    arguments = parser().parse_args()
    raise SystemExit(build(arguments) if arguments.command == 'build' else check(arguments))
