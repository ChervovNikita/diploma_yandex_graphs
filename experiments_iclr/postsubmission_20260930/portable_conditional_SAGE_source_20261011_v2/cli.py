"""Public fixed Physics prepare/acquire/export/score commands; lazy numerics."""
import argparse
import importlib.util
from pathlib import Path
import time
from urllib.request import urlopen
from common import HERE, SEEDS, ARMS, bundle, canonical, checked, closed_states, load, local_validate, read, receipt, require, runtime, sha, write


def configs(work, prepared, limits):
    import engine
    directory = Path(work) / 'prepared'
    files = prepared['files']; get = lambda name: files[name]
    dependency_paths = {
        'training_family': 'deps/e_owner/run_family.py', 'value_installer': 'deps/e_owner/value_installer.py',
        'value_helper': 'deps/e_owner/value_transport.py', 'contracts': 'deps/e_owner/contracts.py',
        'sequential_port': 'deps/sequential/port.py', 'own_ce': 'deps/sequential/own_ce.py',
        'common_routes': 'deps/routes/common_routes.py', 'factors': 'deps/factors/factors.py',
        'models': 'deps/native/models.py', 'run_common': 'deps/native/run_common.py'}
    sources = {key: receipt(HERE / path) for key, path in dependency_paths.items()}
    pyg = Path(importlib.util.find_spec('torch_geometric').origin).parent
    sources.update({key: receipt(pyg / ('nn/conv/' + name)) for key, name in
                    (('sage_conv', 'sage_conv.py'), ('message_passing', 'message_passing.py'))})
    require(all(sources[k]['sha256'] == engine.PINNED[k] for k in engine.PINNED),
            'Canonical native/PyG source differs; alternate release is unverified')
    blocks = {str(s): {'train_npz': get(f'block_{s}/train.npz')['path'],
                      'select_npz': get(f'block_{s}/select.npz')['path']} for s in SEEDS}
    roles = {str(s): {'train': get(f'block_{s}/train.npz'), 'select': get(f'block_{s}/select.npz')} for s in SEEDS}
    bindings = {f'{s}:{role}': item for s, row in roles.items() for role, item in row.items()}
    bindings.update({f'{s}:{role}_ids': get(f'block_{s}/{role}_ids.npy') for s in SEEDS for role in ('cal', 'confirm')})
    preparation = receipt(directory / 'CUSTODIAL_EXPORT_RECEIPT.json')
    cfg = dict(engine.FIXED, permission='LOCAL_RUN_PERMITTED', seeds=list(SEEDS),
               native_repo=str(HERE / 'deps/native'), common_routes=str(HERE / 'deps/routes/common_routes.py'),
               factors=str(HERE / 'deps/factors/factors.py'), source_receipts=sources,
               profile=prepared['profile'], role_blocks=blocks, role_receipts=roles,
               active_role_seed=SEEDS[0], train_npz=blocks[str(SEEDS[0])]['train_npz'],
               valid_npz=blocks[str(SEEDS[0])]['select_npz'],
               safe_roles={'train': roles[str(SEEDS[0])]['train'], 'valid': roles[str(SEEDS[0])]['select']},
               family_limits=limits, device='cuda:0', protocol=read(HERE / 'PROTOCOL.json'),
               stage='F', K=None, k_selection_receipt=None, file_bindings=bindings, preparation_receipt=preparation)
    recipe = load('portable_author_recipe', HERE / 'deps/author/author_model.py')
    author = {'permission': 'LOCAL_RUN_PERMITTED', 'execution_enabled': True, 'seeds': list(SEEDS),
        'point': recipe.POINT, 'device': 'cuda:0', 'limits': limits,
        'profile': dict(prepared['profile'], cal_rows=150, confirm_rows=34093),
        'role_blocks': {str(s): {'train': blocks[str(s)]['train_npz'], 'select': blocks[str(s)]['select_npz'],
            'cal_ids': get(f'block_{s}/cal_ids.npy')['path'], 'confirm_ids': get(f'block_{s}/confirm_ids.npy')['path']} for s in SEEDS},
        'file_bindings': bindings, 'preparation_receipt': preparation}
    write(Path(work) / 'CORE_CONFIG.json', cfg); write(Path(work) / 'AUTHOR_CONFIG.json', author)


def prepare(args):
    work = args.work.resolve(); work.mkdir(parents=True, exist_ok=False)
    source = bundle()
    provider = load('portable_canonical_provider', HERE / 'prepare_data.py')
    raw = args.raw.resolve() if args.raw else work / 'canonical_physics.npz'
    if args.download:
        require(not raw.exists(), 'No raw archive overwrite')
        with urlopen(provider.DOWNLOAD_URL, timeout=60) as response, raw.open('xb') as target:
            for block in iter(lambda: response.read(1048576), b''): target.write(block)
    require(raw.is_file(), 'Supply --raw or --download for the pinned public archive')
    result = provider.prepare(raw, work / 'prepared')
    limits = {'wall_seconds': args.wall_seconds, 'rss_bytes': 68719476736, 'cuda_bytes': 51539607552}
    configs(work, result, limits)
    write(work / 'LOCAL_PREPARATION.json', {'source_bundle': source, 'complete': True,
          'preparation': receipt(work / 'prepared/CUSTODIAL_EXPORT_RECEIPT.json'),
          'runtime': runtime(), 'resource_or_portability_verification': False})


def acquire(args):
    import engine
    work = args.work.resolve(); started = time.monotonic(); bundle()
    core_cfg, author_cfg = read(work / 'CORE_CONFIG.json'), read(work / 'AUTHOR_CONFIG.json')
    engine.acquire(core_cfg, work / 'core_acquisition_v1', local_validate)
    consumer = load('portable_author_consumer', HERE / 'deps/author/consumer.py')
    consumer.run(author_cfg, work / 'author_acquisition_v1', local_validate)
    _, _, states, closures = closed_states(work)
    write(work / 'COHORT_COMPLETE.json', {'complete': True, 'banks': 60, 'fits': 90,
          'selected_states': states, 'closures': closures, 'seconds': time.monotonic() - started,
          'CAL_TEST_truth_access': False, 'source_bundle': bundle(), 'runtime': runtime()})


def export(args):
    import engine
    work = args.work.resolve(); bundle()
    core, author, states, closures = closed_states(work)
    cohort = read(work / 'COHORT_COMPLETE.json')
    require(cohort.get('complete') is True and cohort.get('fits') == 90
            and cohort['selected_states'] == states and cohort['closures'] == closures, 'Exact local90-state closure')
    cfg = read(work / 'CORE_CONFIG.json'); prepared = read(checked(cfg['preparation_receipt']))
    blocks = {str(s): {role + '_ids': prepared['files'][f'block_{s}/{role}_ids.npy']
                      for role in ('cal', 'confirm')} for s in SEEDS}
    family = engine.construct(cfg, work / 'selected_restore_v1', local_validate)
    exporter = load('portable_selected_exporter', HERE / 'deps/export/selected_export.py')
    def validate(actual, **context):
        require(actual == cfg and closed_states(work)[2] == states, 'Config and all90 state hashes remain fixed')
        return {'status': 'LOCAL_EXPORT_PERMITTED', 'cohort': receipt(work / 'COHORT_COMPLETE.json'),
                'runtime': runtime(), 'source_bundle': bundle(), 'selected_states': states}
    release = {'status': 'LOCAL_EXPORT_PERMITTED', 'joined_selected_states90': 'LOCAL_COMPLETE_90_STATE_HASHES',
               'limits': cfg['family_limits']}
    value = exporter.export_all(family, closures['core']['path'], blocks, work / 'core_predictions_v1', validate, release)
    rows = []
    for original in value['records'] + author['records']:
        row = dict(original); row['source_arm'] = row['arm']; row['source_status'] = row['status']
        if row['arm'] == 'author_meanSAGE_M1': row['arm'] = ARMS[-1]
        require(row['status'] == 'SELECTED_COMPLETE', 'Every bank selected; no survivor subset')
        row['status'] = 'selected_finite'
        row['fits'] = [dict(f, status='selected_finite', source_status=f['status']) for f in row['fits']]
        rows.append(row)
    closure = {'schema': 'Physics_selected_predictions_complete_v1', 'terminal': True,
        'blocks': list(SEEDS), 'arms': list(ARMS), 'banks': 60, 'body_fit_units': 90,
        'protocol_sha256': sha(HERE / 'PROTOCOL.json'), 'CAL_truth_access': False, 'TEST_truth_access': False,
        'local_acquisition_and_export_complete': True, 'records': rows,
        'source_bindings': {'bundle': bundle()}, 'cost_records': [read(work / 'LOCAL_PREPARATION.json'), cohort, value],
        'source_closures': closures}
    contracts = load('contracts', HERE / 'deps/readout/contracts.py')
    contracts.validate_predictions(closure, closure['protocol_sha256'])
    write(work / 'COMPLETE_SELECTED_PREDICTIONS.json', closure)


def score(args):
    work = args.work.resolve(); source = bundle()
    contracts = load('contracts', HERE / 'deps/readout/contracts.py')
    inference = load('inference', HERE / 'deps/readout/inference.py')
    math_reader = load('portable_complete_reader', HERE / 'deps/readout/readout.py')
    prediction = receipt(work / 'COMPLETE_SELECTED_PREDICTIONS.json')
    predictions = contracts.close_predictions(prediction, sha(HERE / 'PROTOCOL.json'))
    require(all(r['status'] == 'selected_finite' for r in predictions['records']), 'All60 prediction banks must be finite')
    require(not (work / 'score_v1').exists(), 'One score attempt only; prior TEST release retained')
    core_cfg = read(work / 'CORE_CONFIG.json'); local_validate(core_cfg, action='complete_score')
    prepared = read(checked(core_cfg['preparation_receipt']))
    import importlib
    modules = {name: receipt(importlib.import_module(name).__file__) for name in
        ('numpy', 'torch', 'scipy', 'scipy.stats._continuous_distns', 'scipy.stats._distn_infrastructure', 'scipy.special._ufuncs')}
    cfg = {'prediction_closure': prediction, 'provider_modules': modules,
        'roles': {str(s): {role: prepared['files'][path] for role, path in (
            ('cal_ids', f'block_{s}/cal_ids.npy'), ('confirm_ids', f'block_{s}/confirm_ids.npy'),
            ('cal_truth', f'calibration_only/block_{s}_cal_targets.npz'),
            ('confirm_truth', f'scorer_only/block_{s}_confirm_targets.npz'))} for s in SEEDS},
        'stage_seconds': {'calibrate': 3600, 'test': 3600}}
    custody = {'config_sha256': canonical(cfg), 'source_seal_sha256': source['sha256'],
               'protocol_sha256': sha(HERE / 'PROTOCOL.json'), 'local_run_permission': True, 'runtime': runtime()}
    for row in cfg['roles'].values(): checked(row['cal_truth'])
    output = work / 'score_v1'; output.mkdir(); cal = output / 'cal'; cal.mkdir()
    result = math_reader.calibrate(cfg, {}, custody, predictions, cal)
    require(result['terminal'] is True and result['scalar_endpoints'] == 60 and result['all_finite'] is True
            and all(r['updates'] == 500 for r in result['records']), 'All60 fixed500 CAL endpoints before TEST; retain failure')
    for row in cfg['roles'].values(): checked(row['confirm_truth'])
    write(output / 'TEST_RELEASE.json', {'one_complete_release': True,
          'calibration': receipt(cal / 'COMPLETE_CAL.json'), 'prediction_closure': prediction,
          'local_permission': True, 'historical_author_admission_reused': False})
    test = output / 'test'; test.mkdir()
    math_reader.test(cfg, {'calibration_terminal': receipt(cal / 'COMPLETE_CAL.json'),
                         'local_calibration_returned_complete': True}, custody, predictions, test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'acquire', 'export', 'score'):
        sub = commands.add_parser(name); sub.add_argument('--work', type=Path, required=True)
        if name == 'prepare':
            source = sub.add_mutually_exclusive_group(required=True)
            source.add_argument('--raw', type=Path); source.add_argument('--download', action='store_true')
            sub.add_argument('--wall-seconds', type=int, default=172800)
    args = parser.parse_args()
    if args.command == 'prepare': require(args.wall_seconds > 0, 'Finite positive wall cap')
    try: globals()[args.command](args)
    except BaseException as error:
        if args.work.exists():
            path = args.work / (args.command.upper() + '_FAILED_RETAINED.json')
            if not path.exists(): write(path, {'complete': False, 'type': type(error).__name__,
                'message': str(error), 'retry': False, 'automatic_later_stage': False})
        raise


if __name__ == '__main__': main()
