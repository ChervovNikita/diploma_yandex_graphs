"""One source-reviewed CPU synthetic qualification; no training or labels."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import importlib.util
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
P = HERE.parent
PACKET = 'matched_first_order_private_gradient_utility_numerical_qualification_preparation_20261006_v1'
WORKER_SHA = 'cc332ca9efe43d755bae884e9b202f103aa515f2850d68b9817105582883910f'
REVIEW = 'matched_first_order_private_gradient_utility_numerical_qualifier_independent_source_review_20261006_v1'
REVIEW_SHA = 'd9cdcdd7356003423763086e8d031c2abce860fb295b9a033e5c7afd825dc838'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert not (HERE / 'EXECUTION_RELEASE.json').exists()
    assert sha((P / PACKET / 'qualify.py').read_bytes()) == WORKER_SHA
    assert sha((P / REVIEW / 'REPORT.md').read_bytes()) == REVIEW_SHA
    names = set()
    for directory in (PACKET, REVIEW):
        for path in (P / directory).iterdir():
            if path.is_file():
                names.add(str(path.relative_to(P)))
    binding = json.loads((P / PACKET / 'SOURCE_BINDINGS.json').read_text())
    for row in binding['files']:
        data = (P / row['path']).read_bytes()
        assert len(data) == row['bytes'] and sha(data) == row['sha256']
        names.add(row['path'])
    payload = []
    for relative in sorted(names):
        path = P / relative
        assert path.resolve().is_relative_to(P) and not path.is_symlink()
        data = path.read_bytes()
        assert len(data) < 2_000_000
        payload.append({'path': relative, 'bytes': len(data), 'sha256': sha(data), 'data': base64.b64encode(data).decode()})
    helper_path = P / 'learnability_responsibility_native_synthetic_execution_root_20261005_v1/run_once.py'
    spec = importlib.util.spec_from_file_location('utility_synthetic_transport_helper', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    remote = helper.REMOTE
    old = "worker=phase/'learnability_responsibility_native_numerical_worker_preparation_20261005_v1/qualify.py'"
    assert remote.count(old) == 1
    remote = remote.replace(old, 'worker=phase/' + repr(PACKET + '/qualify.py'))
    old = "argv=[str(runtime),'-B',str(worker),'--execute-authorized','--mode','synthetic','--source-root',str(phase),'--output',str(out/'output')]"
    new = "argv=[str(runtime),'-B',str(worker),'--execute-authorized','--source-root',str(phase),'--site-packages',str(repo/'.venv/lib/python3.11/site-packages'),'--output',str(out/'output'),'--max-elapsed-seconds','600','--max-rss-bytes','4294967296']"
    assert remote.count(old) == 1
    remote = remote.replace(old, new)
    old = 'exit_code=child.wait()'
    assert remote.count(old) == 1
    remote = remote.replace(old, """
 watchdog_fired=False
 try:exit_code=child.wait(timeout=630)
 except subprocess.TimeoutExpired:
  watchdog_fired=True;child.terminate()
  try:exit_code=child.wait(timeout=10)
  except subprocess.TimeoutExpired:child.kill();exit_code=child.wait()
 (out/'WATCHDOG_RECEIPT.json').write_text(json.dumps({'fixed_deadline_seconds':630,'fired':watchdog_fired,'owned_PID':child.pid})+'\\n')
""")
    compile(remote, 'utility_synthetic_remote', 'exec')
    release = {'UTC': datetime.now(timezone.utc).isoformat(), 'root_engineering_authorized': True,
               'scope': 'CPU float64 smooth synthetic utility parity only, not native architecture/fit/predictive evidence',
               'worker_sha256': WORKER_SHA, 'independent_review': {'path': REVIEW + '/REPORT.md', 'sha256': REVIEW_SHA},
               'max_elapsed_seconds': 600, 'max_rss_bytes': 4294967296, 'external_watchdog_seconds': 630,
               'dataset_label_access': False, 'model_fits': 0, 'persistent_updates': 0, 'A_scoring': False,
               'original_six_arm_pilot_and_5740_bill_unchanged': True, 'automatic_retry': False,
               'remote_source_sha256': sha(remote.encode()), 'transport_source_sha256': sha(Path(__file__).read_bytes())}
    (HERE / 'EXECUTION_RELEASE.json').write_text(json.dumps(release, indent=2) + '\n')
    request = {'files': payload, 'packet_manifests': {PACKET: sha((P / PACKET / 'MANIFEST.json').read_bytes())},
               'worker_sha256': WORKER_SHA, 'run_name': HERE.name, 'release': release}
    command = ['ssh', '-T', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
        '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=20', helper.LOGIN,
        'cd ' + shlex.quote(helper.REPO) + ' && exec /usr/bin/python3 -I -S -B -c ' + shlex.quote(remote)]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child.stdin.write(json.dumps(request)); child.stdin.close()
    transcript = []
    for line in child.stdout:
        transcript.append(line)
        row = json.loads(line)
        if row['stage'] == 'launched':
            (HERE / 'LAUNCH_RECEIPT.json').write_text(json.dumps(row['owned_process'], indent=2) + '\n')
            print(json.dumps({'stage': 'launched', 'PID': row['owned_process']['PID']}), flush=True)
        else:
            (HERE / 'TERMINAL.json').write_text(json.dumps(row['terminal'], indent=2) + '\n')
            if row['result'] is not None:
                data = (json.dumps(row['result'], indent=2) + '\n').encode()
                receipt = next(r for r in row['terminal']['artifacts'] if r['path'] == 'output/RESULT.json')
                assert len(data) == receipt['bytes'] and sha(data) == receipt['sha256']
                (HERE / 'RESULT.json').write_bytes(data)
                (HERE / 'RESULT.json').chmod(0o444)
            print(json.dumps({'stage': 'terminal', 'exit_code': row['terminal']['exit_code'],
                'status': row['result']['status'] if row['result'] else None,
                'error': row['result'].get('error') if row['result'] else None}), flush=True)
    stderr = child.stderr.read(); code = child.wait()
    (HERE / 'TRANSPORT_RECEIPT.json').write_text(json.dumps({'UTC': datetime.now(timezone.utc).isoformat(),
        'exit_code': code, 'stderr': stderr, 'stdout_sha256': sha(''.join(transcript).encode()),
        'observation_failure_never_authorizes_restart': True}, indent=2) + '\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
