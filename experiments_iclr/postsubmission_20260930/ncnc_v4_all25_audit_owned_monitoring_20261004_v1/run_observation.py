"""Read owned audit metadata once through the established credential-free relay API."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RELAY = HERE.parent / 'gpu77_connection_recovery_v1'
WRAPPER = RELAY / 'run_gpu77_v3.py'
EXPECTED_WRAPPER = '035b740ceb50eefcfa3cec2aef6dbd1294c769d133e2bc4cf88fb52b088421ff'
RELEASE_SHA = '817300a18bcf2427f136108393ef4e83a68e89aa1699b03bf735937b813a7356'
REMOTE_ROOT = '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930/ncnc_v4_scientific_audit_release_preparation_20261004_v1/'

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)

def save_json(path, value):
    save(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=int, required=True)
    args = parser.parse_args()
    assert 0 < args.sequence < 10000
    sequence = f'{args.sequence:04d}'
    directory = HERE / f'observation_{sequence}'
    directory.mkdir(mode=0o700, exist_ok=False)
    assert hashlib.sha256(WRAPPER.read_bytes()).hexdigest() == EXPECTED_WRAPPER
    code = (HERE / 'monitor_remote.py.txt').read_text()
    compile(code, 'monitor_remote.py.txt', 'exec')
    command = '/usr/bin/python3 -I -S -B -c ' + shlex.quote(code) + '\n'
    command_file = RELAY / f'ncnc_all25_owned_monitor_20261004_{sequence}_command.txt'
    save(command_file, command.encode())
    save(directory / 'REMOTE_CODE.py.txt', code.encode())
    identifier = f'ncnc_all25_owned_monitor_20261004_{sequence}'
    argv = [sys.executable, '-B', str(WRAPPER), '--id', identifier, '--command-file', str(command_file)]
    transport = dict(UTC=datetime.now(timezone.utc).isoformat(), id=identifier, command_sha256=hashlib.sha256(command.encode()).hexdigest(), monitor_source_sha256=hashlib.sha256(code.encode()).hexdigest(), wrapper_sha256=EXPECTED_WRAPPER, timeout_seconds=170, credential_value_recorded=False)
    try:
        completed = subprocess.run(argv, capture_output=True, timeout=170)
        stdout, stderr = completed.stdout, completed.stderr
        transport.update(exit_code=completed.returncode, local_timeout=False)
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or b'', error.stderr or b''
        transport.update(exit_code=None, local_timeout=True)
    save(directory / 'TRANSPORT.stdout.txt', stdout)
    save(directory / 'TRANSPORT.stderr.txt', stderr)
    save_json(directory / 'TRANSPORT.json', transport)
    receipt_path = RELAY / 'commands' / identifier / 'RECEIPT.json'
    if receipt_path.exists():
        receipt_bytes = receipt_path.read_bytes()
        save(directory / 'RELAY_RECEIPT.json', receipt_bytes)
    else:
        print(json.dumps(dict(status='TRANSPORT_NO_RECEIPT', directory=str(directory), transport=transport)))
        return 1
    relay = json.loads(receipt_bytes)
    assert relay['id'] == identifier and relay['target'] == 'shmelev@192.168.18.77'
    assert relay['command_sha256'] == transport['command_sha256'] and relay['wrapper_sha256'] == EXPECTED_WRAPPER
    assert relay['credential_value_recorded'] is False
    if relay['exit_code'] != 0:
        print(json.dumps(dict(status='TRANSPORT_FAILED', directory=str(directory), exit_code=relay['exit_code'])))
        return 1
    lines = [line for line in relay['stdout'].splitlines() if line.startswith('{')]
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result['schema'] == 'ncnc-v4-all25-owned-audit-observation-v1' and result['release_sha256'] == RELEASE_SHA
    save_json(directory / 'REMOTE_RESULT.json', result)
    authenticated = []
    decoded = {}
    allowed = {'ROOT_ORDINARY_AUDIT_RELEASE.json', 'supervision_run01/CHILD_STARTED.json', 'supervision_run01/RESOURCE_DISPATCH_RECHECK.json', 'supervision_run01/PHYSICAL_TERMINAL.json', 'audit/run01/STATUS.json', 'audit/run01/AUDIT_RESULT.json', 'supervision_run01/CHILD.stdout.log', 'supervision_run01/CHILD.stderr.log', 'supervision_run01/SUPERVISOR.stdout.log', 'supervision_run01/SUPERVISOR.stderr.log'}
    for name, row in result['receipts'].items():
        assert name in allowed and row['path'] == REMOTE_ROOT + name
        data = base64.b64decode(row['base64'], validate=True)
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
        save(directory / 'receipts' / name, data)
        decoded[name] = data
        authenticated.append({k:row[k] for k in ('path', 'bytes', 'sha256')})
    assert hashlib.sha256(decoded['ROOT_ORDINARY_AUDIT_RELEASE.json']).hexdigest() == RELEASE_SHA
    audit_data = decoded.get('audit/run01/AUDIT_RESULT.json')
    audit = json.loads(audit_data) if audit_data else None
    if audit is not None:
        mirror = json.loads(decoded['audit/run01/STATUS.json'])
        pin = mirror['authoritative_result_receipt']
        assert pin['sha256'] == hashlib.sha256(audit_data).hexdigest() and pin['bytes'] == len(audit_data)
        assert audit['root_release_sha256'] == RELEASE_SHA and audit['status'] == mirror['status'] and audit['work'] == mirror['work']
    summary = {k: result[k] for k in ('UTC', 'release_sha256', 'source_sha256', 'supervisor', 'child', 'status', 'physical_status', 'physical_exit_code', 'authoritative_result_status', 'no_process_changes', 'no_TEST_access', 'no_checkpoint_tensor_loading')}
    summary.update(receipts_authenticated=len(authenticated), receipt_authentication='PASS', relay_receipt_path=str(receipt_path), directory=str(directory))
    if audit is not None:
        summary.update(denominator=audit['denominator'], failures=audit['failures'], final_input_custody=audit.get('final_input_custody'), old_exact_replay_qualified=audit['old_exact_replay_qualified'], TEST_opened=audit['TEST_opened'], training_updates=audit['training_updates'], served_cell_status_counts={state: sum(c['status']==state for c in audit['cells']) for state in sorted(set(c['status'] for c in audit['cells']))})
    save_json(directory / 'AUTHENTICATION.json', dict(status='PASS', receipts=authenticated, root_release_sha256=RELEASE_SHA, terminal_result_mirror_verified=audit is not None))
    save_json(directory / 'SUMMARY.json', summary)
    print(json.dumps(summary, allow_nan=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
