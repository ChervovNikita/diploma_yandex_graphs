"""Readable counts and exact declared domains from one disabled final master."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import socket
import subprocess

RECIPE_OMISSIONS = {'estimated_seconds', 'reservation_mib', 'wall_seconds', 'resource_evidence',
    'stage', 'initialization_seed', 'factor_seed', 'batch_stream_seed', 'episode_seed',
    'structural_seed', 'seed', 'paired_seed_block', 'cell_id'}


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(v):
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def safe(root, name):
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Canonical project-relative path required')
    result = root / rel
    if not result.resolve().is_relative_to(root.resolve()) or result.is_symlink():
        raise ValueError('Project path escapes or is a symlink')
    return result


def receipt(root, path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(root)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.name.endswith('.gz') else raw)


def recipe_config(v):
    if isinstance(v, dict):
        return {k: recipe_config(x) for k, x in v.items() if k not in RECIPE_OMISSIONS}
    if isinstance(v, list):
        return [recipe_config(x) for x in v]
    return v


def leaves(v, prefix=''):
    if isinstance(v, dict) and v and not {'path', 'bytes', 'sha256'} <= v.keys():
        for key, value in sorted(v.items()):
            yield from leaves(value, prefix + '/' + key)
    else:
        yield prefix, v


def cli_recipe(run):
    cli = run.get('cli', {})
    if not cli:
        return None
    argv = list(cli.get('argv', []))
    for index, token in enumerate(argv[:-1]):
        if token in {'--declared-seed', '--seed'}:
            argv[index + 1] = 'DECLARED_SEED'
    return {'argv': argv, 'environment': cli.get('environment', {}), 'protocol': cli.get('protocol', {}),
            'python_executable': cli.get('python_executable'), 'execution_repository': cli.get('execution_repository')}


def backbone(run, roster, visiting=None):
    cfg = run['config']
    name = cfg.get('backbone') or cfg.get('source_model')
    if name:
        return str(name)
    visiting = set() if visiting is None else visiting
    if run['run_id'] in visiting:
        raise ValueError('Cyclic recipe ancestry')
    visiting = visiting | {run['run_id']}
    parents = {backbone(roster[p], roster, visiting) for p in run['parents']}
    return '+'.join(sorted(parents)) if parents else 'unspecified'


def build(root, plan, summary):
    placement = {}
    for group in plan['groups']:
        for key in group['run_keys']:
            if key in placement:
                raise ValueError('Repeated assigned record')
            placement[key] = group['gpu_assignment']
    census = Counter()
    worker_counts = Counter()
    domains = {}
    contracts = []
    keys = set()
    for row in plan['studies']:
        path = safe(root, row['path'])
        study = read(path)
        if fingerprint(study) != row['spec_hash']:
            raise ValueError('Frozen study contract changed')
        contracts.append({'study_id': study['study_id'], 'spec_hash': row['spec_hash'], **receipt(root, path)})
        roster = {run['run_id']: run for run in study['runs']}
        for run in study['runs']:
            key = study['study_id'] + '/' + run['run_id']
            if key in keys or key not in placement:
                raise ValueError('Missing or repeated assigned record')
            keys.add(key)
            dataset = str(run['config'].get('dataset', study['dataset']))
            stage = run['config'].get('stage', 'unspecified')
            arch = backbone(run, roster)
            adapter = run.get('adapter') or run.get('cli', {}).get('protocol', {}).get('kind') or 'author_cli'
            group = (dataset, arch, stage, run['method'], run['kind'], adapter)
            census[group] += 1
            worker_counts[(placement[key], stage, run['kind'])] += 1
            entry = domains.setdefault(group, {'records': 0, 'studies': set(), 'seeds': Counter(), 'splits': Counter(), 'bodies': Counter(),
                'leaf_values': defaultdict(Counter), 'recipes': {}})
            entry['records'] += 1
            entry['studies'].add(study['study_id'])
            for name, value in [('seeds', run.get('seed')), ('splits', run.get('split')), ('bodies', run.get('body'))]:
                entry[name][canonical(value)] += 1
            declared = {'config': run['config'], 'budget': run['budget'], 'selection': run['selection']}
            if run.get('cli'):
                declared['cli_argv'] = run['cli']['argv']
                declared['cli_protocol'] = run['cli'].get('protocol', {})
            for name, value in leaves(declared):
                entry['leaf_values'][name][canonical(value)] += 1
            recipe = {'config': recipe_config(run['config']), 'budget': run['budget'], 'selection': run['selection'], 'cli': cli_recipe(run)}
            rh = fingerprint(recipe)
            slot = entry['recipes'].setdefault(rh, {'recipe_hash': rh, 'recipe': recipe, 'records': 0, 'study_ids': set()})
            slot['records'] += 1
            slot['study_ids'].add(study['study_id'])
    if keys != set(placement):
        raise ValueError('Assignment and study record census differ')
    totals = summary['counts']['totals']
    fits = sum(v for k, v in census.items() if k[4] == 'model_fit')
    if totals['run_records'] != len(keys) or totals['atomic_fit_run_groups'] != fits:
        raise ValueError('Final authoritative typed counts differ')
    return census, worker_counts, domains, contracts


def csv_file(path, header, rows):
    with path.open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def json_file(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(gzip.compress(raw, mtime=0) if path.name.endswith('.gz') else raw)


def output(root, folder, plan, summary, binding):
    if folder.exists():
        raise FileExistsError('Fresh census directory required; prior outputs retained')
    census, workers, domains, contracts = build(root, plan, summary)
    folder.mkdir(parents=True)
    header = ['dataset', 'backbone', 'stage', 'method_or_control_id', 'record_kind', 'source_adapter', 'records', 'distinct_source_recipes']
    csv_file(folder / 'METHOD_STAGE_COUNTS.csv', header, [list(k) + [n, len(domains[k]['recipes'])] for k, n in sorted(census.items())])
    ds = Counter()
    for key, count in census.items():
        ds[key[:3] + (key[4],)] += count
    csv_file(folder / 'DATASET_BACKBONE_STAGE_COUNTS.csv', ['dataset', 'backbone', 'stage', 'record_kind', 'records'], [list(k) + [v] for k, v in sorted(ds.items())])
    csv_file(folder / 'WORKER_STAGE_COUNTS.csv', ['worker', 'stage', 'record_kind', 'records'], [list(k) + [v] for k, v in sorted(workers.items())])
    domain_rows, recipes, recipe_counts = [], [], []
    for group, entry in sorted(domains.items()):
        identity = dict(zip(header[:6], group))
        row = {**identity, 'records': entry['records'], 'study_ids': sorted(entry['studies']), 'distinct_source_recipes': len(entry['recipes']),
               'declared_identity_domains': {name: [{'value': json.loads(x), 'records': n} for x, n in sorted(entry[name].items())] for name in ('seeds', 'splits', 'bodies')},
               'exact_declared_leaf_domains': {name: {'present_records': sum(values.values()), 'absent_records': entry['records'] - sum(values.values()),
                   'values': [{'value': json.loads(x), 'records': n} for x, n in sorted(values.items())]} for name, values in sorted(entry['leaf_values'].items())}}
        domain_rows.append(row)
        for value in entry['recipes'].values():
            recipes.append({**identity, **value, 'study_ids': sorted(value['study_ids'])})
        recipe_counts.append(list(group) + [entry['records'], len(entry['recipes'])])
    csv_file(folder / 'SOURCE_RECIPE_COUNTS.csv', header, recipe_counts)
    json_file(folder / 'HYPERPARAMETER_DOMAINS.json.gz', {'binding': binding, 'scope': 'Every exact declared config/budget/selection leaf; opaque source/input receipts retained as metadata, never decoded. Author argv/protocol domains included.', 'groups': domain_rows})
    json_file(folder / 'SOURCE_RECIPES.json.gz', {'binding': binding, 'definition': 'Exact source adapter/method/dataset/backbone/stage/kind plus config without the enumerated seed/identity/scheduling fields, exact budget/selection and normalized author CLI.', 'config_fields_omitted_recursively': sorted(RECIPE_OMISSIONS), 'CLI_seed_value_replacement': ['--declared-seed', '--seed'], 'recipes': recipes})
    records = sum(census.values())
    counts = summary['counts']['totals']
    bystage = summary['counts']['by_stage']
    lines = ['# Final registered run census', '', 'This is a disabled preparation roster. New scientific fits require explicit user GO.', '',
        '- Plan canonical SHA256: `' + binding['plan_hash'] + '`', '- Exact SUMMARY SHA256: `' + binding['summary']['sha256'] + '`',
        '- Studies: ' + str(len(contracts)), '- Registered records: ' + str(records), '- Atomic model fit groups: ' + str(counts['atomic_fit_run_groups']),
        '- Author CLI groups: ' + str(sum(v for k, v in census.items() if k[4] == 'author_cli_group')), '- Served banks: ' + str(counts.get('served_banks', 0)), '',
        '## Per-stage registered work', '', '| Stage | Records | Model fit groups | Author CLI groups | Bank exports | CAL jobs | Evaluation jobs |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for stage, count in sorted(bystage.items()):
        author = sum(v for k, v in census.items() if k[2] == stage and k[4] == 'author_cli_group')
        lines.append('| ' + stage + ' | ' + ' | '.join(str(x) for x in [count.get('run_records', 0), count.get('atomic_fit_run_groups', 0), author, count.get('bank_exports', 0), count.get('scalar_CAL_fit_jobs', 0), count.get('evaluation_jobs', 0)]) + ' |')
    lines += ['', '## Dataset, backbone, stage and kind', '', '| Dataset | Backbone | Stage | Kind | Records |', '|---|---|---|---|---:|']
    for key, count in sorted(ds.items()):
        lines.append('| ' + ' | '.join(str(x).replace('|', '\\|') for x in [*key, count]) + ' |')
    lines += ['', '## Exact methods, controls and recipes', '',
        'METHOD_STAGE_COUNTS.csv lists every method/control ID by dataset, backbone, stage, record kind and source adapter. SOURCE_RECIPE_COUNTS.csv gives exact recipe counts under the definition recorded in SOURCE_RECIPES.json.gz. Seed, split and body domains remain explicit.', '',
        'HYPERPARAMETER_DOMAINS.json.gz retains every declared config, budget and selector value per source group, including absent-field counts. Lists remain exact JSON lists. No grid cross-product is inferred from marginal domains. Author CLI argument arrays and protocol declarations are retained.', '',
        'CENSUS.json carries the authoritative typed count object, resource gate and study contract receipts. Unknown or bounded optimizer counts remain unknown or bounded; none are replaced by zero.', '']
    with (folder / 'RUN_CENSUS.md').open('x') as stream:
        stream.write('\n'.join(lines))
    info = {'schema': 'exact-final-disabled-master-readable-census-v1', 'binding': binding, 'study_contracts': contracts,
        'counts': summary['counts'], 'resource_gate': summary.get('resource_gate'), 'records': records, 'domain_groups': len(domains),
        'distinct_source_recipes': len(recipes), 'source_arrays_decoded': 0, 'models': 0, 'scientific_fits': 0, 'GO': False,
        'files': [receipt(root, q) for q in sorted(folder.iterdir()) if q.is_file()]}
    json_file(folder / 'CENSUS.json', info)
    for q in folder.iterdir():
        q.chmod(0o444)
    return {k: info[k] for k in ('schema', 'records', 'domain_groups', 'distinct_source_recipes', 'scientific_fits', 'GO')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--master', required=True)
    parser.add_argument('--master-hash', required=True)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--summary-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'anogena-2-0' or subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines() != ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Scientific allocation route verification failed')
    root = Path(args.project_root).resolve()
    pp, sp = safe(root, args.master), safe(root, args.summary)
    plan, summary = read(pp), read(sp)
    if fingerprint(plan) != args.master_hash or receipt(root, sp)['sha256'] != args.summary_sha256 or summary['plan_hash'] != args.master_hash:
        raise ValueError('Exact final master/SUMMARY binding differs')
    if plan.get('scientific_launch_enabled') is not False or plan.get('human_GO_required') is not True or plan.get('slots_per_gpu') != 2:
        raise ValueError('Disabled two-slot master required')
    print(canonical(output(root, safe(root, args.output), plan, summary, {'plan_hash': args.master_hash, 'plan': receipt(root, pp), 'summary': receipt(root, sp)})))


if __name__ == '__main__':
    main()
