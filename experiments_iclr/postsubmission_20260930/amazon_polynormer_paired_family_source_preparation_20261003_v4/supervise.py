"""Ordinary CPU process supervisor: deadlines, descendant RSS, exits, atomic freeze."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
sys.dont_write_bytecode = True
import common as c


def descendants(pid):
    children = Path('/proc') / str(pid) / 'task' / str(pid) / 'children'
    if not children.exists():
        return []
    try:
        direct = [int(v) for v in children.read_text().split()]
    except FileNotFoundError:
        return []
    return direct + [child for parent in direct for child in descendants(parent)]


def process_rss(pid):
    status = Path('/proc') / str(pid) / 'status'
    if not status.exists():
        return 0
    try:
        lines = status.read_text().splitlines()
    except FileNotFoundError:
        return 0
    for line in lines:
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024
    return 0


def process_identity(pid):
    try:
        # comm can contain spaces/parentheses; the last ')' closes it.
        text = (Path('/proc') / str(pid) / 'stat').read_text()
        return int(text.rsplit(')', 1)[1].split()[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def terminate_tree(process, known=None):
    # Descendants can own different sessions (the fifteen-fit supervisor nesting).
    members = dict(known or {})
    for pid in [*descendants(process.pid), process.pid]:
        identity = process_identity(pid)
        if identity is not None:
            members.setdefault(pid, identity)
    for pid in reversed(members):
        try:
            if process_identity(pid) == members[pid]:
                os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # Root exit alone is insufficient: descendants may have distinct sessions.
    for pid in reversed(members):
        try:
            if process_identity(pid) == members[pid]:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def failure(output, error, code, started, peak):
    if not (output / 'FAILURE.json').exists():
        c.write(output / 'FAILURE.json', {'schema': 'amazon_polynormer_outer_failure_v2', 'UTC': c.utc(),
                'status': 'failed', 'error': str(error), 'physical_exit_code': code,
                'wall_seconds': time.perf_counter() - started, 'aggregate_peak_observed_rss_bytes': peak,
                'automatic_retry_authorized': False})
    if not (output / 'TERMINAL.json').exists():
        c.write(output / 'TERMINAL.json', {'schema': 'amazon_polynormer_physical_terminal_v2', 'UTC': c.utc(),
                'status': 'failed', 'physical_exit_code': code})
    return {'status': 'failed', 'failure': c.record(output / 'FAILURE.json')}


def run_one(kind, release_path, output):
    started = time.perf_counter()
    peak = 0
    process = None
    known = {}
    output = c.confined(output)
    c.require(output != c.PHASE and not output.is_relative_to(c.PACKET), 'Execution output outside source required')
    output.mkdir(parents=True, exist_ok=False)
    try:
        a, admission_record = c.gate(release_path, kind, output)
        c.require(sys.platform.startswith('linux') and Path('/proc/self/status').exists(),
                  'Reviewed Linux /proc whole-process supervisor required')
        if kind == 'fit':
            import study
            registry, row = study.validate_registry(a)
            study.science_gate(a)
            claim = next(v['claim_path'] for v in registry['physical_fits'] if v['id'] == a['fit_id'])
            c.write(claim, {'schema': 'amazon_polynormer_once_only_claim_v2', 'UTC': c.utc(),
                    'fit_id': a['fit_id'], 'registry': a['registry'], 'release': admission_record,
                    'output': a['output'], 'no_retry_or_replacement': True})
        if kind == 'evaluate':
            import evaluate
            evaluate.cohort(a)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        command = [a['interpreter']['path'], '-B', str(c.PACKET / 'worker.py'), '--kind', kind,
                   '--release', str(c.confined(release_path)), '--output', str(output)]
        with (output / 'stdout.txt').open('x') as stdout, (output / 'stderr.txt').open('x') as stderr:
            process = subprocess.Popen(command, cwd=c.PHASE, env=env, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            c.write(output / 'LAUNCH.json', {'schema': 'amazon_polynormer_launch_v2', 'UTC': c.utc(),
                    'command': command, 'worker_pid': process.pid, 'admission': admission_record,
                    'caps': a['caps'], 'no_retry': True})
            while process.poll() is None:
                members = [process.pid, *descendants(process.pid)]
                for pid in members:
                    identity = process_identity(pid)
                    if identity is not None:
                        known.setdefault(pid, identity)
                total = sum(process_rss(pid) for pid in members)
                peak = max(peak, total)
                if time.perf_counter() - started > a['caps']['wall_seconds']:
                    terminate_tree(process, known)
                    raise RuntimeError('Whole-process wall cap exceeded; worker/descendants terminated')
                if total > a['caps']['rss_bytes']:
                    terminate_tree(process, known)
                    raise RuntimeError('Aggregate worker/descendant RSS cap exceeded')
                time.sleep(0.5)
            code = process.wait()
        c.require(code == 0, 'Worker physical exit is unsuccessful: ' + str(code))
        c.require(not any(process_identity(pid) == identity for pid, identity in known.items()
                          if pid != process.pid), 'Worker descendants remain after exit')
        ready = c.read(output / 'READY.json')
        c.require(ready['schema'] == 'amazon_polynormer_body_ready_v2' and ready['kind'] == kind and
                  ready['source'] == a['source'] and ready['admission'] == admission_record and
                  ready['result'] == c.record(output / 'RESULT.json'), 'Missing/coherently bound body result')
        c.preserve(a, admission_record)
        c.write(output / 'TERMINAL.json', {'schema': 'amazon_polynormer_physical_terminal_v2', 'UTC': c.utc(),
                'status': 'success', 'physical_exit_code': code, 'whole_process_wall_seconds': time.perf_counter() - started,
                'aggregate_peak_observed_rss_bytes': peak, 'admission': admission_record})
        files = [{'relative': str(p.relative_to(output)), 'descriptor': c.record(p)}
                 for p in sorted(output.rglob('*')) if p.is_file()]
        c.preserve(a, admission_record)
        for row in files:
            c.require(c.record(output / row['relative']) == row['descriptor'], 'Output mutated before success freeze')
        value = {'schema': 'amazon_polynormer_success_freeze_v2', 'UTC': c.utc(), 'status': 'success',
                 'source': a['source'], 'admission': admission_record, 'files': files}
        temporary = output / 'FREEZE.json.tmp'
        c.write(temporary, value)
        os.link(temporary, output / 'FREEZE.json')
        temporary.unlink()
        return {'status': 'success', 'freeze': c.record(output / 'FREEZE.json')}
    except BaseException as error:
        if process is not None:
            terminate_tree(process, known)
        if not (output / 'OUTER_TRACEBACK.txt').exists():
            (output / 'OUTER_TRACEBACK.txt').write_text(traceback.format_exc())
        return failure(output, error, None if process is None else process.returncode, started, peak)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', required=True, choices=('register', 'runtime_capture', 'consumer_prepare', 'qualify', 'fit', 'close', 'all', 'evaluate'))
    parser.add_argument('--release', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    receipt = run_one(args.kind, args.release, args.output)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt['status'] == 'success' else 1


if __name__ == '__main__':
    raise SystemExit(main())
