"""Tiny fabricated metadata census; no source/data execution."""
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile

OUT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('census', OUT / 'runtime/build_run_census.py')
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)
with tempfile.TemporaryDirectory(dir=OUT / 'tests') as tmp:
    root = Path(tmp)
    runs = []
    for seed, rate in [(0, .001), (1, .001), (2, .01)]:
        runs.append({'run_id': 'fit' + str(seed), 'kind': 'model_fit', 'method': 'native_M1', 'parents': [], 'seed': seed, 'split': 0, 'body': 0,
            'config': {'backbone': 'fabricated_SAGE', 'stage': 'search', 'learning_rate': rate, 'initialization_seed': seed, 'reservation_mib': 1},
            'budget': {'policy': 'full_budget', 'maximum': 2}, 'selection': {'ties': 'first'}})
    runs.append({'run_id': 'bank', 'kind': 'ensemble_assembly', 'method': 'native_I4', 'parents': ['fit0', 'fit1', 'fit2'], 'seed': 0, 'split': 0, 'body': 0,
        'config': {'stage': 'bank_export'}, 'budget': {}, 'selection': {}})
    study = {'study_id': 'synthetic', 'dataset': 'fabricated', 'runs': runs}
    path = root / 'study.json.gz'
    path.write_bytes(gzip.compress(census.canonical(study).encode(), mtime=0))
    plan = {'studies': [{'path': path.name, 'spec_hash': census.fingerprint(study)}], 'groups': [{'gpu_assignment': 'allocation/gpu0', 'run_keys': ['synthetic/' + run['run_id'] for run in runs]}]}
    summary = {'counts': {'totals': {'run_records': 4, 'atomic_fit_run_groups': 3, 'served_banks': 1}, 'by_stage': {'search': {'run_records': 3, 'atomic_fit_run_groups': 3}, 'bank_export': {'run_records': 1, 'bank_exports': 1}}, 'model_optimizer_step_call_total': None}}
    binding = {'plan_hash': 'fabricated', 'summary': {'sha256': 'fabricated'}}
    value = census.output(root, root / 'output', plan, summary, binding)
    assert value['records'] == 4 and value['distinct_source_recipes'] == 3
    exact = census.read(root / 'output/HYPERPARAMETER_DOMAINS.json.gz')
    fits = next(row for row in exact['groups'] if row['record_kind'] == 'model_fit')
    assert fits['distinct_source_recipes'] == 2
    assert fits['exact_declared_leaf_domains']['/config/learning_rate']['values'] == [{'value': .001, 'records': 2}, {'value': .01, 'records': 1}]
    assert {row['value'] for row in fits['declared_identity_domains']['seeds']} == {0, 1, 2}
    bank = next(row for row in exact['groups'] if row['record_kind'] == 'ensemble_assembly')
    assert bank['backbone'] == 'fabricated_SAGE'
    assert census.read(root / 'output/CENSUS.json')['counts']['model_optimizer_step_call_total'] is None
    try:
        census.output(root, root / 'output', plan, summary, binding)
    except FileExistsError:
        pass
    else:
        raise AssertionError('Prior output was overwritten')
print(json.dumps({'synthetic_studies': 1, 'synthetic_records': 4, 'exact_rate_and_seed_domains': True,
    'parent_backbone_join': True, 'recipe_identity_omissions_explicit': True, 'unknown_optimizer_count_preserved': True,
    'existing_output_rejected': True, 'scientific_fits': 0, 'GO': False}, sort_keys=True))
