"""One reviewed command per physical GPU; allocation path uses the durable v2 engine."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from host_binding import config_path, confined, guard, load_config, pin, require, verify


def command(config, run_spec):
    """Pure construction; no resolved-Python substitution and no GPU probe."""
    require(config['worker'] == 'allocation/gpu0',
            '77 scheduler migration pending: shared host reservations, global ownership and STOP custody')
    root = Path(config['project_root'])
    require(run_spec['schema'] == 'portable-reviewed-allocation-run-v1' and run_spec['worker'] == config['worker'], 'Wrong run-spec scope')
    env = run_spec['python_environment']
    require(env['lexical_executable'] == str(root / '.venv/bin/python'), 'Preserve allocation lexical venv entrypoint')
    require(env['prefix'] == str(root / '.venv') and env['prefix'] != env['base_prefix'], 'Own lexical venv required')
    phase = root / config['phase_relative']
    dispatcher = phase / 'benchmark_expansion_20261015_v1/expanded_dispatcher_v2/dispatcher.py'
    require(run_spec['dispatcher']['path'] == str(dispatcher), 'Fixed reviewed durable allocation dispatcher required')
    argv = [env['lexical_executable'], '-B', '-u', str(dispatcher), 'run', '--worker', config['worker']]
    for name in ('release', 'go', 'readiness'):
        require(run_spec.get(name), 'Exact reviewed ' + name + ' pin required')
        argv.extend(['--' + name, run_spec[name]['path']])
    for name in ('resumes', 'continuation', 'reconciliations'):
        if run_spec.get(name):
            argv.extend(['--' + name, run_spec[name]['path']])
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('allocation/gpu0', 'server77/gpu0', 'server77/gpu1'), required=True)
    parser.add_argument('--run-spec', required=True)
    args = parser.parse_args()
    config, config_pin = load_config(config_path(args.worker))
    require(config['host_enabled'], 'Host disabled before probes; explicit user reauthorization and actual host receipts required')
    # No77 route is sent to the allocation-only engine even if a config is edited.
    require(config['worker'] == 'allocation/gpu0', '77 shared-host scheduler migration is not yet admitted')
    os.environ['BENCHMARK_HOST_CONFIG'] = config_pin['path']
    os.environ['BENCHMARK_HOST_CONFIG_SHA256'] = config_pin['sha256']
    os.environ['CUDA_VISIBLE_DEVICES'] = config['physical_gpu_uuid']
    guard(config)
    root = Path(config['project_root'])
    run_path = confined(args.run_spec, root)
    spec = json.loads(run_path.read_text())
    argv = command(config, spec)
    require(spec['host_config'] == config_pin, 'Run specification must bind this exact host configuration')
    for key in ('dispatcher', 'release', 'go', 'readiness', 'resumes', 'continuation', 'reconciliations'):
        if spec.get(key):
            verify(spec[key], root, parse_json=False)
    environment = verify(spec['python_environment_receipt'], root)
    require(environment == spec['python_environment'], 'Interpreter specification mismatch')
    # Guard against replacing a source tree without updating the compiled release.
    release = json.loads(Path(spec['release']['path']).read_text())
    expected_output = root / config['phase_relative'] / 'benchmark_expansion_20261015_v1/campaign_outputs/portable_allocation_v1_REQUIRES_REVIEW'
    require(release['output_root'] == str(expected_output), 'Keep the one permanent allocation output/claim namespace')
    require(all(t['worker'] == 'allocation/gpu0' for t in release['tasks']), 'This first allocation path never repartitions the campaign')
    closure = {p['path']: p['sha256'] for p in release['source_files']}
    require(closure.get(config_pin['path']) == config_pin['sha256'], 'Host configuration must be in the reviewed release closure')
    require(closure.get(str(Path(__file__).resolve())) == pin(__file__)['sha256'], 'Launcher must be in reviewed source closure')
    os.chdir(root)
    os.execv(argv[0], argv)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({'status': 'BLOCKED', 'reason': str(error), 'children_started': 0}))
        raise SystemExit(2)
