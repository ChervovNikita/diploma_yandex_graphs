"""Run only the three frozen complete scientific fits, stopping on failure."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'pencil_collab_paired_predictive_preparation_20261004_v1'
PIN = '1a22966029e9a1b4b4affad7e5cc24f6b3d795245a8d3fc8d0ac12af4b2983c2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def update(value):
    p = ROOT / 'QUEUE_PROGRESS.json'
    temporary = ROOT / 'QUEUE_PROGRESS.tmp'
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary, p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    assert os.uname().nodename == 'peptide' and sha(SOURCE / 'MANIFEST.json') == PIN
    assert sha(ROOT / 'ROOT_RELEASE.json') == args.release_sha256
    proposed = json.loads((SOURCE / 'PROPOSED_COMMAND.json').read_text())
    assert proposed['execution_order'] == [0,1,2] and len(proposed['commands']) == 3
    completed = []
    for row in proposed['commands']:
        seed = row['seed']
        command = [args.release_sha256 if value == 'ROOT_BIND_RELEASE_DIGEST' else value for value in row['argv']]
        assert command[0] == sys.executable and not (ROOT / ('seed%d' % seed)).exists()
        update(dict(UTC=datetime.now(timezone.utc).isoformat(), current_seed=seed,
                    completed_seeds=completed, status='RUNNING', predictive_values_read=False))
        with (ROOT / ('seed%d_SUPERVISOR_STDOUT.txt' % seed)).open('xb') as out, (ROOT / ('seed%d_SUPERVISOR_STDERR.txt' % seed)).open('xb') as err:
            child = subprocess.Popen(command, cwd=proposed['cwd'], env=dict(os.environ,**proposed['environment']),
                                     stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
            raw = (Path('/proc') / str(child.pid) / 'stat').read_text()
            fields = raw[raw.rfind(')')+2:].split()
            assert int(fields[2]) == int(fields[3]) == child.pid
            save('seed%d_SUPERVISOR_LAUNCH.json' % seed, dict(PID=child.pid, start_ticks=int(fields[19]),
                 group=child.pid, session=child.pid, command=command, seed=seed, release_sha256=args.release_sha256))
            exit_code = child.wait()
        folder = ROOT / ('seed%d' % seed) / 'supervision/run01'
        terminal_path = folder / 'TERMINAL.json'
        terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else None
        valid = (exit_code == 0 and terminal is not None and terminal['status'] == 'COMPLETE_PREDICTIVE_FIT'
                 and terminal['physical_session_closed'] is True and terminal['direct_child_reaped'] is True
                 and terminal['physical_exit_code'] == 0 and terminal['stop'] is None
                 and terminal['source_manifest_sha256'] == PIN and terminal['release_sha256'] == args.release_sha256
                 and terminal['completed_scientific_fits'] == 1 and not (ROOT / 'ACTIVE_FIT.json').exists())
        if not valid:
            save('QUEUE_FAILURE.json', dict(UTC=datetime.now(timezone.utc).isoformat(), failed_seed=seed,
                 supervisor_exit_code=exit_code, terminal_status=terminal.get('status') if terminal else None,
                 completed_seeds=completed, unattempted_seeds=[n for n in (0,1,2) if n > seed],
                 automatic_retry=False, predictive_values_read=False))
            return 1
        custody = json.loads((folder / 'SUPERVISOR_CUSTODY.json').read_text())
        assert custody['terminal_sha256'] == sha(terminal_path) and custody['status'] == terminal['status']
        assert custody['final_parent_budget_check'] == terminal['final_parent_budget_check']
        assert terminal['final_parent_budget_check']['status'] == 'PASS'
        completed.append(seed)
    save('QUEUE_COMPLETE.json', dict(UTC=datetime.now(timezone.utc).isoformat(), completed_seeds=completed,
         source_manifest_sha256=PIN, release_sha256=args.release_sha256, complete_scientific_fits=3,
         TEST_reads=False, predictive_values_read=False, automatic_retry=False))
    update(dict(UTC=datetime.now(timezone.utc).isoformat(), completed_seeds=completed,
                status='COMPLETE_THREE_FITS', predictive_values_read=False))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except BaseException as error:
        if not isinstance(error, SystemExit):
            path = ROOT / 'QUEUE_EXCEPTION.json'
            if not path.exists():
                save(path.name, dict(type=type(error).__name__, condition=str(error), automatic_retry=False))
        raise
