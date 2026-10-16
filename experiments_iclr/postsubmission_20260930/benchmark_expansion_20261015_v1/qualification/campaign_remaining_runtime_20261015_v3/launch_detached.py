"""Explicit detached launch of one already admitted supervisor; no retry or waiting."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from witness_support import ROOT, HERE, guard, inside, own_source, read, receipt, require


def main():
    guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', required=True)
    args = parser.parse_args()
    admission_path = inside(args.admission)
    admission = read(admission_path)
    require(admission.get('schema') == 'remaining-runtime-root-admission-v1'
            and admission.get('status') == 'ADMITTED' and admission.get('root_reviewed_source') is True,
            'Root must explicitly review and admit before detached launch')
    own_source(admission['source_manifest_sha256'])
    require(sys.executable == str(ROOT/'.venv/bin/python'), 'Original qualified lexical interpreter required')
    require(not inside(admission['output']).exists(), 'Fresh output required; no duplicate detached launch')
    claim = admission_path.with_suffix('.LAUNCH.json')
    log_path = admission_path.with_suffix('.supervisor.log')
    command = [sys.executable, '-B', '-u', str(HERE/'supervise.py'), '--admission', str(admission_path)]
    with claim.open('x') as stream:
        json.dump({'status': 'launching', 'admission': receipt(admission_path), 'command': command}, stream)
    try:
        with log_path.open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        stat = (Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')', 1)[1].split()
        result = {'status': 'supervisor_started', 'admission': receipt(admission_path), 'command': command,
                  'pid': child.pid, 'start_ticks': stat[19], 'log': str(log_path), 'automatic_retries': 0}
    except BaseException as error:
        result = {'status': 'launch_failed_or_outcome_unknown', 'failure': str(error), 'automatic_retries': 0,
                  'admission': receipt(admission_path), 'command': command}
    claim.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result))
    return 0 if result['status'] == 'supervisor_started' else 1


if __name__ == '__main__':
    raise SystemExit(main())
