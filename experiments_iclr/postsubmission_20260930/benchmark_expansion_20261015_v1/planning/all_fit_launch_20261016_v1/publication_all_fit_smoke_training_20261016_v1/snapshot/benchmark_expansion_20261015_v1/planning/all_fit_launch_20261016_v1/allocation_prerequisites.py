"""Run the two already reviewed lifecycle checks after the existing pilot exits."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
E = ROOT/'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
HERE = E/'planning/all_fit_launch_20261016_v1'
PILOT = E/'results/pubmed_three_gpu_v1/allocation_gpu0'
PACKAGE = E/'qualification/campaign_remaining_runtime_20261015_v2'
SHA = '521dbe3fa315e8aaa5efae3500ff1b05604bcad1992724e782177fc0012b0dca'


def born(pid):
    try:
        values = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return None if values[0] == 'Z' else values[19]
    except FileNotFoundError:
        return None


def write(value):
    path = HERE/'ALLOCATION_PREREQUISITES.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert sys.executable == str(ROOT/'.venv/bin/python')
    report = {'status': 'waiting_existing_pilot', 'pid': os.getpid(), 'cases': {}}
    write(report)
    while not (PILOT/'END.json').exists() or born(855944) == '6090581051':
        time.sleep(15)
    terminal = json.loads((PILOT/'END.json').read_text())
    assert terminal['status'] == 'complete', 'Preserve and investigate prior pilot failure'
    for case in ('bound_observer_molbace', 'buddy_selected_VALID'):
        output = E/'qualification'/('allfit_prerequisite_'+case+'_20261016_attempt01')
        admission = output.with_suffix('.admission.json')
        assert not output.exists() and not admission.exists()
        peer = born(640428)
        assert peer in (None, '6055039165')
        command = [sys.executable, '-B', str(PACKAGE/'root_admit.py'), '--case', case,
                   '--reviewed-source-sha256', SHA, '--gru-peer', 'present' if peer else 'absent',
                   '--output', str(output), '--admission', str(admission)]
        subprocess.run(command, check=True, cwd=ROOT)
        report.update(status='running', active_case=case)
        write(report)
        result = subprocess.run([sys.executable, '-B', '-u', str(PACKAGE/'supervise.py'),
                                 '--admission', str(admission)], cwd=ROOT)
        report['cases'][case] = {'exit_code': result.returncode, 'output': str(output)}
        write(report)
        if result.returncode:
            report['status'] = 'failed'
            write(report)
            return result.returncode
    report.update(status='PASS', active_case=None)
    write(report)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        write({'status': 'failed', 'error': str(exc), 'pid': os.getpid()})
        raise
