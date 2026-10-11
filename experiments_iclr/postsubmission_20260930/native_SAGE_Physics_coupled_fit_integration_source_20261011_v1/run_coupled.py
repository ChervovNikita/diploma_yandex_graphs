"""Inactive finite two-child Physics owner; root supplies frozen admission hook.

No recipe changes, retries, survivor subset or readout. Import is stdlib-only.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

GPU_UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SEEDS = [8101, 8219, 8363, 8501, 8663, 8819, 9001, 9199, 9403, 9623]
CORE_ARMS = ['BASE', 'E4', 'N4', 'equipped_M1', 'equipped_I4']


def route_guard():
    if socket.gethostname() != 'anogena-2-0': raise ValueError('Excluded route: wrong scientific host')
    uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
    if [u.strip() for u in uuids] != [GPU_UUID]: raise ValueError('Excluded route: exact sole GPU UUID required')


def receipt(path):
    path = Path(path).resolve(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def checked(item):
    if not isinstance(item, dict) or set(item) != {'path', 'bytes', 'sha256'} or receipt(item['path']) != item:
        raise ValueError('Exact frozen file receipt differs')
    return Path(item['path'])


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def admitted(spec):
    """Root hook is pinned, stdlib-only and validates every scientific receipt."""
    route_guard()
    if spec.get('status') != 'ROOT_COUPLED_FIT_ADMITTED' or spec.get('parent_wall_seconds') != 172800:
        raise ValueError('Inactive integration: root whole60/90 admission and finite48h owner required')
    hook = load('Physics_root_frozen_admission', checked(spec['root_hook']))
    custody = hook.validate_parent(spec, source_receipt=receipt(__file__))
    if not isinstance(custody, dict) or custody.get('status') != 'ROOT_COUPLED_FIT_ADMITTED':
        raise ValueError('Root must bind exact configs, all receipts, both actual PASS and whole60/90 protocol')
    core_path, author_path = checked(spec['core_config']), checked(spec['author_config'])
    core, author = json.loads(core_path.read_text()), json.loads(author_path.read_text())
    if core.get('seeds') != SEEDS or author.get('seeds') != SEEDS or core.get('device') != 'cuda:0' or author.get('device') != 'cuda:0':
        raise ValueError('Whole paired ten-mask roster on same sole GPU required')
    return hook, custody, core, author


def construct_core(spec, output):
    """Root/watcher helper: admitted V5 Family constructor, no run/update.

    spec fields documented in README; output must be a fresh directory inside H.
    Root validate_core returns original E learning-lock metadata required by V5.
    """
    hook, custody, cfg, author_cfg = admitted(spec)
    output = Path(output).resolve(); output.relative_to(Path(spec['H']).resolve())
    output.mkdir(parents=True, exist_ok=False)
    port = load('Physics_core_V5_port', Path(spec['core_port_directory']) / 'port.py')
    def validate(actual, **kwargs):
        if actual != cfg: raise ValueError('Core config differs from exact root-frozen bytes')
        return hook.validate_core(actual, **kwargs)
    Family, port_receipt = port.bind(spec['e_directory'], cfg, validate, spec['core_release'])
    return Family(cfg, output, spec['core_config']['sha256']), port_receipt


def child(spec, role):
    H = Path(spec['H']).resolve()
    if role == 'core':
        output = H / 'core_acquisition_v1'
        family, port_receipt = construct_core(spec, output)
        try: family.run()
        except BaseException as error:
            write(output / 'FAILURE.json', {'status': 'FAILED_RETAINED', 'complete': False,
                'type': type(error).__name__, 'message': str(error), 'retry': False,
                'current': family.current, 'completed_units': family.completed_units})
            raise
    elif role == 'author':
        hook, custody, core_cfg, cfg = admitted(spec)
        consumer = load('Physics_author_sealed_consumer', Path(spec['author_context_directory']) / 'consumer.py')
        def validate(actual, **kwargs):
            if actual != cfg: raise ValueError('Author config differs from exact root-frozen bytes')
            return hook.validate_author(actual, **kwargs)
        consumer.run(cfg, H / 'author_acquisition_v1', validate)
    else: raise ValueError('Only fixed core or author child')


def parent(spec, spec_path, spec_sha256):
    hook, custody, core_cfg, author_cfg = admitted(spec)
    H = Path(spec['H']).resolve(); owner = H / 'cohort_owner_v1'; owner.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); deadline = started + 172800
    result = {'status': 'RUNNING', 'complete': False, 'banks': 0, 'fits': 0,
              'custody': custody, 'children': {}, 'parent_wall_limit_seconds': 172800,
              'same_sole_GPU_UUID': GPU_UUID, 'retry': False, 'survivor_subset': False,
              'CAL_CONFIRM_truth_access': False, 'automatic_readout': False,
              'source_author_arm': 'author_meanSAGE_M1', 'protocol_author_arm': 'published_mean_SAGE_source_port_M1'}
    children = {}; logs = []
    try:
        for role in ('core', 'author'):
            logfile = (owner / (role + '.log')).open('xb'); logs.append(logfile)
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                '--spec', str(Path(spec_path).resolve()), '--spec-sha256', spec_sha256, '--child', role],
                stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
            children[role] = process; result['children'][role] = {'pid': process.pid, 'exit_code': None}
        write(owner / 'OWNER_START.json', result)
        while any(process.poll() is None for process in children.values()):
            if time.monotonic() >= deadline: raise TimeoutError('Finite48h whole-cohort owner deadline exceeded')
            if any(process.poll() not in (None, 0) for process in children.values()):
                raise RuntimeError('A fixed child failed; retain both outcomes, no survivor readout')
            time.sleep(.5)
        for role, process in children.items():
            result['children'][role]['exit_code'] = process.wait()
        if any(r['exit_code'] != 0 for r in result['children'].values()): raise RuntimeError('Both direct zero exits required')
        core_path = H / 'core_acquisition_v1' / 'COMPLETE_FAMILY.json'
        author_path = H / 'author_acquisition_v1' / 'COMPLETE_AUTHOR_CONTEXT.json'
        core, author = json.loads(core_path.read_text()), json.loads(author_path.read_text())
        if (core.get('complete') is not True or core.get('full_same_runtime_stage_complete') is not True
                or core.get('groups') != 50 or core.get('fit_units') != 80
                or core.get('fixed_seed_roster') != SEEDS or core.get('fixed_arm_roster') != CORE_ARMS
                or len(core.get('results', [])) != 50
                or {(r['seed'], r['arm']) for r in core['results']} != {(s, a) for s in SEEDS for a in CORE_ARMS}
                or sum(len(r['fits']) for r in core['results']) != 80
                or author.get('complete') is not True or author.get('banks') != 10 or author.get('fits') != 10
                or len(author.get('records', [])) != 10
                or {r['seed'] for r in author['records']} != set(SEEDS)
                or any(r['status'] != 'SELECTED_COMPLETE' or r['arm'] != 'author_meanSAGE_M1' for r in author['records'])):
            raise ValueError('Exact full50/80 core and10/10 author closure required')
        result.update(status='BOTH_COMPLETE_ZERO_EXIT', complete=True, banks=60, fits=90,
                      core_complete_receipt=receipt(core_path), author_complete_receipt=receipt(author_path))
    except BaseException as error:
        result.update(status='FAILED_RETAINED', failure={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        for role, process in children.items():
            if process.poll() is None:
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    process.wait(timeout=5)
            result['children'][role]['exit_code'] = process.returncode
        for stream in logs: stream.close()
        result['owner_seconds'] = time.monotonic() - started
        write(owner / 'OWNER_END.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True); parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--child', choices=('core', 'author'))
    args = parser.parse_args(); route_guard()  # Before any project metadata read.
    path = Path(args.spec); raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.spec_sha256: raise ValueError('Exact root-frozen integration config bytes differ')
    spec = json.loads(raw)
    if args.child: child(spec, args.child)
    else: parent(spec, path, args.spec_sha256)


if __name__ == '__main__': main()
