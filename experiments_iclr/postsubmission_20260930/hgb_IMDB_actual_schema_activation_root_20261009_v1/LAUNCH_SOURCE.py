"""Run the published IMDB input/role qualifier once under the existing runtime."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import socket
import subprocess
import time

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
A = P / 'hgb_IMDB_actual_schema_activation_root_20261009_v1'
Q = P / 'imdb_role_isolated_public_schema_backbone_preparation_20261009_v1'
COMMIT = 'c9632322396f7c6ea2cde6c494555d1a10702d80'
assert socket.gethostname() == 'anogena-2-0'
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert subprocess.check_output(['git', '-C', str(R), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
assert not (A / 'LAUNCH.json').exists()
assert hashlib.sha256((Q / 'MANIFEST.json').read_bytes()).hexdigest() == '4313648fb8ba836ef1564fbdea579aaf09b50f8346210991f54576b14b022cd8'
release = json.loads((A / 'SCHEMA_RELEASE.json').read_text())
assert not Path(release['output_directory']).exists()
runtime = json.loads((P / 'staged_label_posterior_native_metadata_root_20261008_v1/RUNTIME.json').read_text())
env = dict(os.environ, PYTHONPATH=os.pathsep.join(runtime['PYTHONPATH']),
           PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
argv = [runtime['python'], '-B', str(Q / 'qualify_schema.py'), '--qualify',
        '--release', str(A / 'SCHEMA_RELEASE.json'), '--input-root', release['input_root'],
        '--output', release['output_directory']]
started = time.perf_counter()
child = subprocess.Popen(argv, cwd=R, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
raw = Path('/proc', str(child.pid), 'stat').read_text()
fields = raw[raw.rfind(')') + 2:].split()
launch = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=child.pid,
              start_ticks=int(fields[19]), group=int(fields[2]), session=int(fields[3]),
              boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), argv=argv,
              source_commit=COMMIT, launch_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              TEST_truth_opened=False, model_or_GPU_work=False)
with (A / 'LAUNCH.json').open('x') as output:
    json.dump(launch, output, indent=2)
    output.write(chr(10))
print(json.dumps(dict(launched=launch)), flush=True)
stdout, stderr = child.communicate(timeout=100)
terminal = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), exit_code=child.returncode,
                seconds=time.perf_counter() - started, stdout=stdout, stderr=stderr,
                original_pid_absent=not Path('/proc', str(child.pid)).exists(),
                TEST_truth_opened=False, model_or_GPU_work=False)
with (A / 'TERMINAL.json').open('x') as output:
    json.dump(terminal, output, indent=2)
    output.write(chr(10))
report_path = Path(release['output_directory']) / 'SCHEMA_REPORT.json'
report = json.loads(report_path.read_text()) if report_path.exists() else None
print(json.dumps(dict(terminal=terminal, schema_report=report)), flush=True)
raise SystemExit(child.returncode)
