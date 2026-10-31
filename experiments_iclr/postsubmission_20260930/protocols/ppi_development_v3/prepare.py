"""Verify a prospective packet and print commands; never launch full fits.

No torch or provider tensors are loaded by check. Four actual runner freezes
must execute in the coordinator's existing repository runtime. Hardware
capture queries nvidia-smi only; finalize writes new, immutable joint freezes.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
PACKET = Path(__file__).resolve().parent
PHASE = PACKET.parents[1]
REMOTE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
MANIFEST = 'new_tasks/data/ppi/frozen_shareable_manifest_v2.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def inside(path):
    path = Path(path).resolve()
    if PHASE not in path.parents:
        raise ValueError('Output/evidence must remain inside the authorized phase')
    return path


def locks():
    # Execute the actual pure-stdlib lock function, with its dependencies
    # supplied explicitly. Do not import the torch-dependent runner locally.
    path = PHASE / 'protocols/ppi_study.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'selector_locks']
    if len(tree.body) != 1:
        raise ValueError('Missing unique runner lock function')
    namespace = dict(PHASE=PHASE, CODE=PHASE, AUTHORIZED_PHASES=[
        Path('/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930'), REMOTE],
        Path=Path, json=json, sha=sha)
    exec(compile(tree, str(path), 'exec'), namespace)
    return {str(b): namespace['selector_locks'](b / 100, PHASE / MANIFEST) for b in (10, 100)}


def verify_comparison():
    # Recompute the report using its exact comparison code, replacing only
    # the exclusive writer with a return. Existing report bytes are preserved.
    path = PHASE / 'sharing_policy/compare_repoenv_reproductions.py'
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    cut = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'destination' for t in n.targets))
    main.body = main.body[:cut] + [ast.Return(value=ast.Name(id='report', ctx=ast.Load()))]
    namespace = {'__file__': str(path), '__name__': 'prospective_identity_check'}
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    if namespace['main']() != read(PHASE / 'sharing_policy/PPI_REPOENV_REPRO_COMPARISON_v1.json'):
        raise ValueError('Six-run reproduction report no longer matches its sources')


def check():
    coverage = {str(p.relative_to(PACKET)): sha(p) for p in PACKET.rglob('*')
                if p.is_file() and p.name != 'artifact_manifest.json'}
    if coverage != read(PACKET / 'artifact_manifest.json'):
        raise ValueError('Prospective packet coverage or bytes changed')
    plan = read(PACKET / 'JOINT_DESIGN.json')
    for collection in ('source_sha256', 'execution_source_sha256', 'evidence_sha256'):
        for name, digest in plan[collection].items():
            if sha(PHASE / name) != digest:
                raise ValueError('Prospective identity changed: ' + name)
    verify_comparison()
    if locks() != plan['selector_locks']:
        raise ValueError('Prospective selector evidence changed')
    with (PACKET / 'QUEUE.csv').open() as stream:
        queue = list(csv.DictReader(stream))
    keys = {(r['arm'], r['seed'], r['label_budget']) for r in queue}
    if len(queue) != 96 or len(keys) != 96 or {r['arm'] for r in queue} != set(plan['union_arms']):
        raise ValueError('Queue differs from the 96-fit ceiling')
    return plan, queue


def shell(argv):
    return shlex.join([str(x) for x in argv])


def print_freezes(args):
    check()
    runtime = REMOTE / 'protocols/ppi_development_v3_runtime_v1'
    script = REMOTE / 'protocols/ppi_development_v3/prepare.py'
    print(shell(['source', REMOTE / 'protocols/repo_env.sh']))
    print('export CUDA_VISIBLE_DEVICES=' + shlex.quote(str(args.physical_gpu)))
    print(shell([args.python, script, 'check']))
    print(shell([args.python, script, 'capture-hardware', '--physical-gpu', args.physical_gpu,
                 '--authorization-file', args.authorization_file, '--output', runtime / 'hardware.json']))
    for budget in (10, 100):
        for kind in ('sharing', 'architecture'):
            print(shell([args.python, REMOTE / 'protocols/run_logged.py', '--output',
                         REMOTE / f'protocols/logged_runs/ppi_development_v3_freeze_{kind}_label{budget}', '--',
                         args.python, REMOTE / 'protocols/ppi_study.py', 'freeze', '--kind', kind,
                         '--label-fraction', str(budget / 100), '--data-manifest', REMOTE / MANIFEST,
                         '--output', runtime / f'raw_{kind}_label{budget}.json']))
    print(shell([args.python, REMOTE / 'protocols/run_logged.py', '--output',
                 REMOTE / 'protocols/logged_runs/ppi_development_v3_selector_path_gate', '--',
                 args.python, REMOTE / 'protocols/verify_ppi_repoenv_selector_v3.py', '--output',
                 runtime / 'SELECTOR_PATH_GATE.json']))
    print(shell([args.python, script, 'finalize', '--runtime-directory', runtime]))


def capture_hardware(args):
    check()
    if PHASE != REMOTE or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.physical_gpu):
        raise ValueError('Use the existing authorized remote runtime and one visible physical GPU')
    authorization = inside(args.authorization_file)
    if not authorization.read_text().strip():
        raise ValueError('Bind existing human GPU authorization evidence')
    output = inside(args.output)
    if output.exists():
        raise ValueError('Refuse to overwrite hardware evidence')
    fields = ['index', 'uuid', 'name', 'driver_version', 'memory.total', 'memory.used', 'utilization.gpu']
    query = ['nvidia-smi', '-i', str(args.physical_gpu), '--query-gpu=' + ','.join(fields),
             '--format=csv,noheader,nounits']
    observed = subprocess.run(query, check=True, capture_output=True, text=True).stdout.strip()
    values = next(csv.reader([observed], skipinitialspace=True))
    if len(values) != len(fields) or int(values[0]) != args.physical_gpu:
        raise ValueError('Unexpected authorized GPU hardware record')
    gpu = dict(zip(fields, values))
    if int(gpu['memory.used']) > 100 or int(gpu['utilization.gpu']) != 0:
        raise ValueError('Authorized GPU is occupied')
    write(output, {'schema': 'gnnm-ppi-development-hardware-v3', 'physical_gpu': args.physical_gpu,
                   'gpu': gpu, 'capture_UTC': datetime.now(timezone.utc).isoformat(),
                   'platform': platform.platform(), 'machine': platform.machine(),
                   'logical_cpu_count': os.cpu_count(), 'interpreter': sys.executable,
                   'authorization_path': str(authorization.relative_to(PHASE)),
                   'authorization_sha256': sha(authorization), 'query_argv': query,
                   'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
                   'no_training_or_cuda_context_requested': True})


def finalize(args):
    plan, _ = check()
    runtime = inside(args.runtime_directory)
    if PHASE != REMOTE:
        raise ValueError('Actual freezes must be finalized in the existing remote runtime')
    hardware = read(runtime / 'hardware.json')
    if sha(PHASE / hardware['authorization_path']) != hardware['authorization_sha256']:
        raise ValueError('GPU authorization evidence changed')
    if hardware['schema'] != 'gnnm-ppi-development-hardware-v3':
        raise ValueError('Missing actual hardware capture')
    gate_path = runtime / 'SELECTOR_PATH_GATE.json'
    gate = read(gate_path)
    if (gate['status'] != 'PASS' or gate['source_sha256'] != plan['source_sha256']
            or gate['gate_source_sha256'] != sha(PHASE / 'protocols/verify_ppi_repoenv_selector_v3.py')
            or gate['shareable_manifest_sha256'] != plan['data_manifest_sha256']
            or gate['dataset_loader_constructed'] or gate['optimizer_steps'] != 0
            or gate['GPU_context'] or gate['validation_or_test_scored'] or gate['evidence_bytes_mutated']
            or len(gate['checks']) != 3 or any(c['status'] != 'PASS' for c in gate['checks'])):
        raise ValueError('Require the bounded actual-runtime selector path gate')
    protocols, environments = {}, []
    for budget in (10, 100):
        for kind in ('sharing', 'architecture'):
            raw_path = runtime / f'raw_{kind}_label{budget}.json'
            raw = read(raw_path)
            if (raw['schema'] != 'gnnm-ppi-development-v1' or raw['status'] != 'prospective_not_fit'
                    or raw['kind'] != kind or raw['arms'] != plan['families'][kind]
                    or raw['seeds'] != [0, 1, 2] or raw['label_fraction'] != budget / 100
                    or raw['recipe'] != plan['recipe'] or raw['source_sha256'] != plan['source_sha256']
                    or raw['data_manifest_sha256'] != plan['data_manifest_sha256']
                    or len(raw['mask_contract']) != 20 or raw['test_loader_accessed']
                    or raw['selector_locks'] != (plan['selector_locks'][str(budget)] if kind == 'sharing' else None)):
                raise ValueError('Raw runner freeze differs from the registered joint design')
            environments.append(raw['environment'])
            supervisor = PHASE / f'protocols/logged_runs/ppi_development_v3_freeze_{kind}_label{budget}'
            completed = read(supervisor / 'completion.json')
            if completed['exit_code'] != 0:
                raise ValueError('Supervised runtime freeze failed')
            for key, name in [('environment_sha256', 'environment.json'),
                              ('command_sha256', 'command.json'), ('log_sha256', 'stdout_stderr.log')]:
                if completed[key] != sha(supervisor / name):
                    raise ValueError('Runtime freeze supervision changed')
            supervision_env = read(supervisor / 'environment.json')
            roots = supervision_env['cache_and_temp']
            if (supervision_env['script_sha256'] != plan['source_sha256']['protocols/ppi_study.py']
                    or supervision_env['wrapper_sha256'] != plan['execution_source_sha256']['protocols/repo_env.sh']
                    or len(roots) != 17 or any(not v or not Path(v).is_relative_to(PHASE) for v in roots.values())):
                raise ValueError('Runtime freeze source/hardware/confinement supervision differs')
            receipts = []
            for line in (supervisor / 'stdout_stderr.log').read_text().splitlines():
                try:
                    receipt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(receipt, dict) and receipt.get('status') == 'prospective_protocol_saved':
                    receipts.append(receipt)
            if len(receipts) != 1 or receipts[0]['sha256'] != sha(raw_path) or receipts[0]['training']:
                raise ValueError('Raw freeze differs from the supervised saved-protocol receipt')
            final = dict(raw, study_id=plan['study_id'], joint_design_sha256=sha(PACKET / 'JOINT_DESIGN.json'),
                         queue_sha256=sha(PACKET / 'QUEUE.csv'), raw_runner_freeze_sha256=sha(raw_path),
                         hardware_manifest_sha256=sha(runtime / 'hardware.json'),
                         selector_path_gate_sha256=sha(gate_path),
                         shared_control_reuse=plan['shared_control_reuse'],
                         staged_continue_stop=plan['staged_continue_stop'],
                         predeclared_analysis=plan['analysis'],
                         runtime_freeze_supervisor_completion_sha256=sha(PHASE /
                             f'protocols/logged_runs/ppi_development_v3_freeze_{kind}_label{budget}/completion.json'))
            path = runtime / f'{kind}_label{budget}.json'
            write(path, final)
            protocols[f'{kind}{budget}'] = {'path': str(path.relative_to(PHASE)), 'sha256': sha(path)}
    if any(x != environments[0] for x in environments):
        raise ValueError('The four runtime freezes have different environments')
    write(runtime / 'FROZEN_BUNDLE.json', {'status': 'prospective_frozen_not_fit',
          'study_id': plan['study_id'], 'joint_design_sha256': sha(PACKET / 'JOINT_DESIGN.json'),
          'queue_sha256': sha(PACKET / 'QUEUE.csv'), 'protocols': protocols,
          'hardware_path': str((runtime / 'hardware.json').relative_to(PHASE)),
          'hardware_sha256': sha(runtime / 'hardware.json'),
          'selector_path_gate_path': str(gate_path.relative_to(PHASE)),
          'selector_path_gate_sha256': sha(gate_path), 'fits_launched': 0, 'test_scored': False})


def print_fit(args):
    plan, queue = check()
    row = next(r for r in queue if int(r['ordinal']) == args.ordinal)
    bundle = read(inside(args.runtime_bundle))
    if (bundle['status'] != 'prospective_frozen_not_fit' or bundle['joint_design_sha256'] != sha(PACKET / 'JOINT_DESIGN.json')
            or bundle['queue_sha256'] != sha(PACKET / 'QUEUE.csv')):
        raise ValueError('Require finalized runtime bundle')
    hardware_path = PHASE / bundle['hardware_path']
    hardware = read(hardware_path)
    if sha(hardware_path) != bundle['hardware_sha256'] or hardware['physical_gpu'] != args.physical_gpu:
        raise ValueError('Physical GPU differs from the frozen hardware')
    if sha(PHASE / hardware['authorization_path']) != hardware['authorization_sha256']:
        raise ValueError('Authorization evidence changed')
    if sha(PHASE / bundle['selector_path_gate_path']) != bundle['selector_path_gate_sha256']:
        raise ValueError('Runtime selector path gate changed')
    if row['condition'] != 'initial_10':
        if args.gate_receipt is None:
            raise ValueError('Conditional row requires all-three-seed stage decision evidence')
        gates = read(inside(args.gate_receipt))
        if (gates['joint_design_sha256'] != bundle['joint_design_sha256'] or not gates['all_three_seeds_replayed']
                or gates['test_scored']):
            raise ValueError('Stage gate evidence differs from prospective plan')
        terms = row['condition'].split('|')
        if not any(gates['decisions'].get(t, False) for t in terms):
            raise ValueError('Predeclared continue rule stopped this row')
    key = row['owner_kind'] + row['label_budget']
    protocol = PHASE / bundle['protocols'][key]['path']
    if sha(protocol) != bundle['protocols'][key]['sha256']:
        raise ValueError('Runtime protocol changed')
    stem = f"{row['arm']}_seed{row['seed']}_label{row['label_budget']}_attempt{args.attempt:02d}"
    output = PHASE / 'results/ppi_development_v3' / stem
    log = PHASE / 'protocols/logged_runs' / ('ppi_development_v3_' + stem)
    replay = PHASE / 'protocols/ppi_development_v3_replays' / (stem + '.json')
    if output.exists() or log.exists() or replay.exists():
        raise ValueError('Attempt or replay already exists; preserve it and use a new registered attempt')
    print('# Queue row ' + row['ordinal'] + '; condition: ' + row['condition'])
    print(shell(['source', PHASE / 'protocols/repo_env.sh']))
    print('export CUDA_VISIBLE_DEVICES=' + shlex.quote(str(args.physical_gpu)))
    print(shell([args.python, PHASE / 'protocols/run_logged.py', '--output', log, '--', args.python,
          PHASE / 'protocols/ppi_study.py', 'train', '--arm', row['arm'], '--seed', row['seed'],
          '--label-fraction', str(int(row['label_budget']) / 100), '--device', 'cuda:0',
          '--physical-gpu', args.physical_gpu, '--protocol', protocol, '--data-manifest', PHASE / MANIFEST,
          '--output', output]))
    print('# After exit 0, replay all retained epochs and the selected checkpoint before the next fit.')
    print(shell([args.python, PHASE / 'protocols/run_logged.py', '--output', str(log) + '_cpu_replay', '--',
          args.python, output / 'source/protocols/ppi_study.py', 'replay', '--run', output,
          '--data-manifest', PHASE / MANIFEST, '--output', replay]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'print-freezes', 'capture-hardware', 'finalize', 'print-fit'])
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--physical-gpu', type=int)
    parser.add_argument('--authorization-file', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--runtime-directory', type=Path)
    parser.add_argument('--runtime-bundle', type=Path)
    parser.add_argument('--ordinal', type=int, choices=range(1, 97))
    parser.add_argument('--attempt', type=int, choices=[1, 2], default=1)
    parser.add_argument('--gate-receipt', type=Path)
    args = parser.parse_args()
    if args.command in ('print-freezes', 'capture-hardware', 'print-fit') and args.physical_gpu is None:
        parser.error('An already authorized physical GPU is required')
    if args.command in ('print-freezes', 'capture-hardware') and args.authorization_file is None:
        parser.error('Bind existing GPU authorization evidence; do not request duplicate authorization')
    if args.command == 'capture-hardware' and args.output is None:
        parser.error('Hardware capture requires a new output path')
    if args.command == 'finalize' and args.runtime_directory is None:
        parser.error('Finalize requires the directory containing four completed runtime freezes')
    if args.command == 'print-fit' and (args.runtime_bundle is None or args.ordinal is None):
        parser.error('Print-fit requires a finalized runtime bundle and one queue ordinal')
    if args.command == 'check':
        check(); print(json.dumps({'status': 'PASS', 'scope': 'stdlib source/evidence/queue check',
                                  'torch_or_provider_loaded': False, 'training_launched': False}))
    elif args.command == 'print-freezes': print_freezes(args)
    elif args.command == 'capture-hardware': capture_hardware(args)
    elif args.command == 'finalize': finalize(args)
    elif args.command == 'print-fit': print_fit(args)


if __name__ == '__main__':
    main()
