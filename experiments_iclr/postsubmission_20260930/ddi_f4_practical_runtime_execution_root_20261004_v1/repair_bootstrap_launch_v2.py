"""Recover the pre-launch bootstrap failure under the pinned Python 3.12.

The first system-Python bootstrap lacked hashlib.file_digest and stopped
before writing the remote release/launch marker or starting a child. Reuse the
same exact root release and reviewed scientific/physical source; preserve the
failed transport. This is a single explicit recovery, never an automatic retry.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLIENT = HERE / 'prepare_runtime.py'
CLIENT_SHA = '55000638d5bd509552f25f0b61a4d55c309848b4923486a2947b4f08c125f841'
SUPERVISOR_SHA = '2e134b43075cffaf4e6de91775f319556b3b8c0637badb35b882e6cc57b44777'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha(CLIENT) == CLIENT_SHA and sha(HERE / 'supervise_runtime.py') == SUPERVISOR_SHA
    failed = HERE / 'ddi_f4_practical_launch_20261004_LOCAL_TRANSPORT.json'
    error = json.loads(failed.read_text())
    assert error['exit_code'] != 0 and 'hashlib' in error['stdout'] and 'file_digest' in error['stdout']
    assert not (HERE / 'DETACHED_LAUNCH.json').exists()
    release = json.loads((HERE / 'ROOT_RELEASE.json').read_text())
    assert release['candidate_manifest_sha256'] == '817af133e35aefbe02b938acac9b6b50df898d0668c73abef46c83a4bb9d7a4a'
    spec = importlib.util.spec_from_file_location('exact_reviewed_ddi_runtime_root_client', CLIENT)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    original_run, original_save = client.run, client.save

    def same_release_save(name, value):
        if name == 'ROOT_RELEASE.json':
            assert set(value) == set(release)
            assert {k: v for k, v in value.items() if k != 'UTC'} == {k: v for k, v in release.items() if k != 'UTC'}
            return
        original_save(name, value)

    def pinned_run(identity, code):
        # No scientific code changes: the same bootstrap runs under the pinned
        # interpreter rather than the relay's older /usr/bin/python3.
        wrapper = 'from pathlib import Path\nimport hashlib,subprocess,json\n'
        wrapper += 'p=Path(' + repr(client.PYTHON) + ');assert hashlib.sha256(p.read_bytes()).hexdigest()==' + repr(client.PYTHON_SHA) + '\n'
        wrapper += 'r=subprocess.run(' + repr([client.PYTHON, '-B', '-c', code]) + ',capture_output=True,text=True,timeout=60)\n'
        wrapper += 'assert r.returncode==0,dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)\nprint(r.stdout.strip())\n'
        return original_run(identity + '_pinned_bootstrap_v2', wrapper)

    client.save = same_release_save
    client.run = pinned_run
    original_save('BOOTSTRAP_RECOVERY_ADMISSION.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
         status='EXPLICIT_ONE_PRENUMERICAL_BOOTSTRAP_RECOVERY', failed_transport_sha256=sha(failed),
         original_client_sha256=CLIENT_SHA, original_supervisor_sha256=SUPERVISOR_SHA,
         same_root_release_sha256=sha(HERE / 'ROOT_RELEASE.json'), repair_source_sha256=sha(Path(__file__)),
         changed_scope='Bootstrap interpreter only; exact reviewed launch code/release/workload unchanged.',
         prior_numerical_execution=False, automatic_retry=False))
    import argparse
    client.launch(argparse.Namespace(manifest=release['candidate_manifest_sha256'], seal=release['candidate_seal_sha256']))


if __name__ == '__main__':
    main()
