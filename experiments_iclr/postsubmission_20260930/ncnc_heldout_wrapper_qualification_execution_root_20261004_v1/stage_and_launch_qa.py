"""Stage reviewed sources and release one fabricated QA, without TEST access."""
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
import json
import shlex
import subprocess
import sys
import zlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
RELAY = PHASE / 'gpu77_connection_recovery_v1'
WRAPPER = RELAY / 'run_gpu77_v3.py'
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
REMOTE = REMOTE_PHASE / HERE.name
SOURCE = PHASE / 'ncnc_frozen_all25_heldout_source_preparation_20261004_v2'
PREP = PHASE / 'ncnc_frozen_all25_heldout_release_preparation_20261004_v2'
REVIEW = PHASE / 'ncnc_all25_heldout_fresh_source_review_20261004_v2'
PYTHON = '/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def run(identity, code):
    command = 'cd ' + shlex.quote(str(REPO)) + " && /usr/bin/python3 -I -S -B - <<'QAROOTPY'\n" + code + '\nQAROOTPY\n'
    assert len(command.encode()) < 100000
    path = RELAY / (identity + '.txt')
    with path.open('x') as stream:
        stream.write(command)
    result = subprocess.run([sys.executable, '-B', str(WRAPPER), '--id', identity,
                             '--command-file', str(path)], capture_output=True, text=True)
    save(HERE / (identity + '_LOCAL_TRANSPORT.json'),
         dict(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
              command_sha256=hashlib.sha256(command.encode()).hexdigest()))
    assert result.returncode == 0, result.stderr + result.stdout[:1500]
    outer = json.loads(result.stdout)
    assert outer['exit_code'] == 0, outer
    return json.loads(outer['stdout'])


def stage():
    review_path = REVIEW / 'SOURCE_REVIEW.json'
    review = json.loads(review_path.read_text())
    assert review['status'] == 'PASS' and review['heldout_source_manifest_sha256'] == sha(SOURCE / 'MANIFEST.json')
    payload = []
    for folder in (SOURCE, PREP, REVIEW):
        manifest = folder / 'MANIFEST.json'
        rows = json.loads(manifest.read_text())['files']
        for row in rows:
            local = folder / row['path']
            assert sha(local) == row['sha256'] and local.stat().st_size == row.get('bytes', row.get('size'))
        # The execution release directory owns future outputs and must remain
        # unsealed remotely. Its local source manifest authenticates payloads;
        # the production output guard rejects a MANIFEST in any output parent.
        files = [folder / row['path'] for row in rows]
        if folder != PREP:
            files.append(manifest)
        seal = folder / 'SEAL.json'
        if seal.exists() and folder != PREP:
            files.append(seal)
        for local in files:
            raw = local.read_bytes()
            payload.append(dict(path=str(local.relative_to(PHASE)), bytes=len(raw), sha256=sha(local),
                                data=base64.b64encode(raw).decode()))
    supervisor = HERE / 'physical_qa_supervisor.py'
    raw = supervisor.read_bytes()
    payload.append(dict(path=str(supervisor.relative_to(PHASE)), bytes=len(raw), sha256=sha(supervisor),
                        data=base64.b64encode(raw).decode()))
    save(HERE / 'SOURCE_STAGE_INVENTORY.json', [{k: v for k, v in row.items() if k != 'data'} for row in payload])
    compressed = zlib.compress(json.dumps(payload).encode(), 9)
    encoded = base64.b64encode(compressed).decode()
    chunks = [encoded[start:start + 40000] for start in range(0, len(encoded), 40000)]
    for number, chunk in enumerate(chunks, 1):
        code = 'from pathlib import Path\nimport hashlib,json,os\n'
        code += 'repo=Path(' + repr(str(REPO)) + ');root=Path(' + repr(str(REMOTE / 'source_chunks')) + ')\n'
        code += 'assert Path.cwd()==repo and os.uname().nodename=="peptide";root.mkdir(parents=True,exist_ok=True)\n'
        code += 'raw=' + repr(chunk) + '.encode();path=root/' + repr('part%03d.b64' % number) + '\n'
        code += 'with path.open("xb") as stream:stream.write(raw)\nprint(json.dumps(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))\n'
        staged = run('ncnc_heldout_v2_qa_source_chunk%03d_20261004' % number, code)
        assert staged['sha256'] == hashlib.sha256(chunk.encode()).hexdigest()
    code = 'from pathlib import Path\nfrom datetime import datetime,timezone\nimport base64,hashlib,json,os,zlib\n'
    code += 'repo=Path(' + repr(str(REPO)) + ');phase=Path(' + repr(str(REMOTE_PHASE)) + ');root=Path(' + repr(str(REMOTE / 'source_chunks')) + ')\n'
    code += 'assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
    code += 'compressed=base64.b64decode(b"".join((root/("part%03d.b64"%number)).read_bytes() for number in range(1,' + str(len(chunks) + 1) + ')),validate=True)\n'
    code += 'assert hashlib.sha256(compressed).hexdigest()==' + repr(hashlib.sha256(compressed).hexdigest()) + '\nrows=json.loads(zlib.decompress(compressed))\n'
    code += 'for row in rows:\n path=phase/row["path"];assert path.resolve().is_relative_to(phase);raw=base64.b64decode(row["data"],validate=True);assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"]\n if path.exists():assert not path.is_symlink() and path.read_bytes()==raw\n else:\n  path.parent.mkdir(parents=True,exist_ok=True)\n  with path.open("xb") as stream:stream.write(raw)\n'
    code += 'print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status="SEALED_REVIEWED_SOURCES_STAGED",files=len(rows),bytes=sum(r["bytes"] for r in rows),numerical_execution=False,TEST_access=False)))\n'
    staged = run('ncnc_heldout_v2_qa_source_join_20261004', code)
    save(HERE / 'SOURCE_STAGE_RECEIPT.json', staged)


def launch():
    review_path = REVIEW / 'SOURCE_REVIEW.json'
    review = json.loads(review_path.read_text())
    assert review['status'] == 'PASS'
    release = json.loads((PREP / 'ROOT_FABRICATED_QUALIFICATION_DISABLED_CANDIDATE.json').read_text())
    release.update(execution_enabled=True, root_authorization_reference=str(REMOTE / 'ROOT_QA_ADMISSION.json'),
                   independent_source_review=dict(path=str(REMOTE_PHASE / review_path.relative_to(PHASE)),
                                                  bytes=review_path.stat().st_size, sha256=sha(review_path)))
    admission = dict(UTC=datetime.now(timezone.utc).isoformat(), status='APPROVED_FABRICATED_ONLY',
                     heldout_source_manifest_sha256=sha(SOURCE / 'MANIFEST.json'),
                     independent_review_sha256=sha(review_path),
                     physical_supervisor_sha256=sha(HERE / 'physical_qa_supervisor.py'),
                     scope='One 18-case fabricated adapter/accounting QA; 40 original scorer calls with sentinel models.',
                     root_source_review='Reviewed loader, accounting, scorer/metric failure handling and root physical wrapper.',
                     successful_matching_physical_QA_terminal_required_before_TEST=True,
                     scientific_TEST_release_enabled=False, scientific_checkpoint_or_TEST_access=False,
                     study_training_updates=0, automatic_retry=False)
    save(HERE / 'ROOT_QA_ADMISSION.json', admission)
    save(HERE / 'ROOT_QA_RELEASE.json', release)
    payload = []
    for local in (HERE / 'ROOT_QA_ADMISSION.json', HERE / 'ROOT_QA_RELEASE.json'):
        payload.append(dict(path=str(REMOTE / local.name), sha256=sha(local), data=base64.b64encode(local.read_bytes()).decode()))
    environment = dict(CUDA_VISIBLE_DEVICES=release['cuda_visible_devices'], OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1',
                       PYTHONPATH=str(REPO / '.gnnm_runtime/buddy_extra_v1/site'))
    cmd = [PYTHON, '-B', str(REMOTE / 'physical_qa_supervisor.py'), '--release', str(REMOTE / 'ROOT_QA_RELEASE.json'),
           '--release-sha256', sha(HERE / 'ROOT_QA_RELEASE.json')]
    code = 'from pathlib import Path\nfrom datetime import datetime,timezone\nimport base64,hashlib,json,os,subprocess\n'
    code += 'repo=Path(' + repr(str(REPO)) + ');phase=Path(' + repr(str(REMOTE_PHASE)) + ');root=Path(' + repr(str(REMOTE)) + ')\n'
    code += 'assert Path.cwd()==repo and os.uname().nodename=="peptide"\nrows=' + repr(payload) + '\n'
    code += 'for row in rows:\n path=Path(row["path"]);assert path.resolve().is_relative_to(phase);raw=base64.b64decode(row["data"],validate=True);assert hashlib.sha256(raw).hexdigest()==row["sha256"]\n with path.open("xb") as stream:stream.write(raw)\n'
    code += 'assert not (root/"physical/run01").exists() and not Path(' + repr(release['output_directory']) + ').exists()\n'
    code += 'environment=' + repr(environment) + ';command=' + repr(cmd) + '\n'
    code += 'with (root/"DETACHED.stdout").open("xb") as out,(root/"DETACHED.stderr").open("xb") as err:\n child=subprocess.Popen(command,cwd=repo,env=dict(os.environ,**environment),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)\n'
    code += 'raw=(Path("/proc")/str(child.pid)/"stat").read_text();fields=raw[raw.rfind(")")+2:].split()\n'
    code += 'result=dict(UTC=datetime.now(timezone.utc).isoformat(),status="FABRICATED_QA_SUPERVISOR_LAUNCHED",supervisor_PID=child.pid,supervisor_start_ticks=int(fields[19]),command=command,environment=environment,TEST_access=False,study_checkpoint_access=False,study_training_updates=0)\n'
    code += 'with (root/"DETACHED_LAUNCH.json").open("x") as stream:json.dump(result,stream,indent=2);stream.write("\\n")\nprint(json.dumps(result))\n'
    result = run('ncnc_heldout_v2_fabricated_qa_launch_20261004', code)
    save(HERE / 'DETACHED_LAUNCH.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    stage()
    launch()
