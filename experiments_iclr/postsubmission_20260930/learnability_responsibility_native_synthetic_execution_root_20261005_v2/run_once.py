"""Run the fixed local-derivative successor, preserving the original failure."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import importlib.util
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
PREP = PHASE / 'learnability_responsibility_native_numerical_worker_preparation_20261005_v2'
HELPER = PHASE / 'learnability_responsibility_native_synthetic_execution_root_20261005_v1/run_once.py'
MANIFEST_SHA = '84737ced1f6056744f70900af39fb466a4e92418d30cf876fa8c663145c06be0'
WORKER_SHA = '9247f4d3e405c58bd128ecc5d91ae95d661fc98d2fe0d5a4d409240708409562'


def main():
    assert not (HERE / 'EXECUTION_RELEASE.json').exists()
    assert hashlib.sha256((PREP / 'MANIFEST.json').read_bytes()).hexdigest() == MANIFEST_SHA
    manifest = json.loads((PREP / 'MANIFEST.json').read_text())
    for row in manifest['files']:
        data = (PREP / row['path']).read_bytes()
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    spec = importlib.util.spec_from_file_location('local_native_transport', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    original = "worker=phase/'learnability_responsibility_native_numerical_worker_preparation_20261005_v1/qualify.py'"
    successor = "worker=phase/'learnability_responsibility_native_numerical_worker_preparation_20261005_v2/qualify.py'"
    assert helper.REMOTE.count(original) == 1
    remote = helper.REMOTE.replace(original, successor)
    release = {
        'UTC': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Fixed same-state local derivative and complete episode engineering',
        'worker_sha256': WORKER_SHA,
        'manifest_sha256': MANIFEST_SHA,
        'transport_helper_sha256': hashlib.sha256(HELPER.read_bytes()).hexdigest(),
        'remote_wrapper_sha256': hashlib.sha256(remote.encode()).hexdigest(),
        'original_coarse_qualification_remains_failed': True,
        'tolerances_unchanged': True,
        'full_graph_or_predictive_fit_authorized': False,
    }
    (HERE / 'EXECUTION_RELEASE.json').write_text(json.dumps(release, indent=2) + '\n')
    files = []
    for path in sorted(PREP.iterdir()):
        assert path.is_file() and not path.is_symlink()
        data = path.read_bytes()
        files.append({'path': str(path.relative_to(PHASE)), 'bytes': len(data),
                      'sha256': hashlib.sha256(data).hexdigest(),
                      'data': base64.b64encode(data).decode()})
    request = {'files': files, 'packet_manifests': {PREP.name: MANIFEST_SHA},
               'worker_sha256': WORKER_SHA, 'run_name': HERE.name, 'release': release}
    command = ['ssh', '-T', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
               '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
               '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=20', helper.LOGIN,
               'cd ' + shlex.quote(helper.REPO) + ' && exec /usr/bin/python3 -I -S -B -c ' + shlex.quote(remote)]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    child.stdin.write(json.dumps(request))
    child.stdin.close()
    transcript = []
    for line in child.stdout:
        transcript.append(line)
        item = json.loads(line)
        if item['stage'] == 'launched':
            (HERE / 'LAUNCH_RECEIPT.json').write_text(json.dumps(item['owned_process'], indent=2) + '\n')
            print(json.dumps({'stage': 'launched', 'PID': item['owned_process']['PID'],
                              'start_ticks': item['owned_process']['start_ticks']}), flush=True)
        else:
            (HERE / 'TERMINAL.json').write_text(json.dumps(item['terminal'], indent=2) + '\n')
            if item['result'] is not None:
                (HERE / 'RESULT.json').write_text(json.dumps(item['result'], indent=2) + '\n')
            print(json.dumps({'stage': 'terminal', 'exit_code': item['terminal']['exit_code'],
                              'status': item['result'].get('status') if item['result'] else None,
                              'error': item['result'].get('error') if item['result'] else None}), flush=True)
    stderr = child.stderr.read()
    exit_code = child.wait()
    (HERE / 'TRANSPORT_RECEIPT.json').write_text(json.dumps({
        'UTC': datetime.now(timezone.utc).isoformat(), 'exit_code': exit_code,
        'stdout_bytes': len(''.join(transcript).encode()),
        'stdout_sha256': hashlib.sha256(''.join(transcript).encode()).hexdigest(),
        'stderr': stderr}, indent=2) + '\n')
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
