"""Stage and launch one exact TRAIN-only census; preserve every attempt."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import zlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / 'ddi_native_train_support_census_preparation_20261004_v1'
PIN = '989f10f25f6b1fd2affa58609c24bd957f2ac1755c6040aa0fdf139aa5546758'
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
REMOTE = REMOTE_PHASE / HERE.name
PYTHON = '/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12'
PYTHON_SHA = '14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'
GPU = 'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def transport(identity, code):
    path = PHASE / 'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py'
    assert sha(path) == 'cfb554db15ebdb7e5f82284bc73f0c112c74e1b9bd7dc238b41ba1dce1933276'
    spec = importlib.util.spec_from_file_location('ddi_census_transport', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    return module.run('ddi_native_census_' + identity + '_20261004', code)


def context():
    return ('from pathlib import Path\nimport base64,hashlib,json,os,subprocess,zlib\n'
            'from datetime import datetime,timezone\n'
            'repo=Path(' + repr(str(REPO)) + ');phase=Path(' + repr(str(REMOTE_PHASE))
            + ');root=Path(' + repr(str(REMOTE)) + ')\n'
            'assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
            'def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()\n'
            'def save(p,v):\n with p.open("x") as s:json.dump(v,s,indent=2,sort_keys=True);s.write("\\n");s.flush();os.fsync(s.fileno())\n')


def stage():
    assert sha(SOURCE / 'MANIFEST.json') == PIN
    manifest = json.loads((SOURCE / 'MANIFEST.json').read_text())
    rows = []
    for row in manifest['files']:
        p = SOURCE / row['path']
        assert p.resolve().is_relative_to(SOURCE) and not p.is_symlink()
        assert sha(p) == row['sha256'] and p.stat().st_size == row['bytes']
    files = [SOURCE / row['path'] for row in manifest['files']]
    files += [SOURCE / 'MANIFEST.json', SOURCE / 'SEAL.json', HERE / 'supervise_owned.py', HERE / 'ROOT_AUTHORIZATION.md']
    for p in files:
        raw = p.read_bytes()
        assert len(raw) < 2_000_000 and p.resolve().is_relative_to(PHASE)
        rows.append(dict(path=str(p.relative_to(PHASE)), bytes=len(raw), sha256=sha(p), data=base64.b64encode(raw).decode()))
    save('STAGE_INVENTORY.json', [{k: v for k, v in row.items() if k != 'data'} for row in rows])
    packed = zlib.compress(json.dumps(rows).encode(), 9)
    encoded = base64.b64encode(packed).decode()
    chunks = [encoded[i:i+40000] for i in range(0, len(encoded), 40000)]
    for number, chunk in enumerate(chunks, 1):
        code = context() + 'd=root/"source_chunks";d.mkdir(parents=True,exist_ok=True)\n'
        code += 'b=' + repr(chunk) + '.encode()\nwith (d/' + repr('part%03d.b64' % number) + ').open("xb") as s:s.write(b)\n'
        code += 'print(json.dumps(dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())))\n'
        result = transport('chunk%03d' % number, code)
        assert result['sha256'] == hashlib.sha256(chunk.encode()).hexdigest()
    code = context() + 'packed=base64.b64decode(b"".join((root/"source_chunks"/("part%03d.b64"%n)).read_bytes() for n in range(1,' + str(len(chunks)+1) + ')),validate=True)\n'
    code += 'assert hashlib.sha256(packed).hexdigest()==' + repr(hashlib.sha256(packed).hexdigest()) + '\nrows=json.loads(zlib.decompress(packed))\n'
    code += 'for row in rows:\n p=phase/row["path"];assert p.resolve().is_relative_to(phase) and not p.is_symlink();b=base64.b64decode(row["data"],validate=True);assert len(b)==row["bytes"] and hashlib.sha256(b).hexdigest()==row["sha256"]\n if p.exists():assert p.read_bytes()==b\n else:\n  p.parent.mkdir(parents=True,exist_ok=True)\n  with p.open("xb") as s:s.write(b)\n'
    code += 'print(json.dumps(dict(status="STAGED",files=len(rows),numerical_execution=False)))\n'
    save('STAGE_RECEIPT.json', transport('join', code))


def launch():
    assert (HERE / 'STAGE_RECEIPT.json').exists() and not (HERE / 'ROOT_RELEASE.json').exists()
    release = json.loads((SOURCE / 'ROOT_RELEASE_TEMPLATE.json').read_text())
    release.update(status='APPROVED', root_authorization_reference=str(REMOTE / 'ROOT_AUTHORIZATION.md'), source_manifest_sha256=PIN)
    save('ROOT_RELEASE.json', release)
    code = context() + 'assert sha(Path(' + repr(PYTHON) + '))==' + repr(PYTHON_SHA) + '\n'
    code += 'source=phase/' + repr(SOURCE.name) + '\nassert sha(source/"MANIFEST.json")==' + repr(PIN) + '\n'
    code += 'plan=json.loads((source/"PLAN.json").read_text());assert sha(phase/plan["train_relative_path"])==plan["TRAIN_ONLY_sha256"]\n'
    code += 'q=subprocess.run(["nvidia-smi","--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=15)\n'
    code += 'gpu=[r.split(",") for r in q.stdout.splitlines() if r.split(",")[0].strip()==' + repr(GPU) + '];assert len(gpu)==1 and int(gpu[0][1])>=8*1024\n'
    code += 'assert not (root/"run01").exists() and not (root/"ATTEMPT_SPENT.json").exists()\n'
    raw = (HERE / 'ROOT_RELEASE.json').read_bytes()
    code += 'raw=base64.b64decode(' + repr(base64.b64encode(raw).decode()) + ',validate=True)\nassert hashlib.sha256(raw).hexdigest()==' + repr(sha(HERE / 'ROOT_RELEASE.json')) + '\n'
    code += 'with (root/"ROOT_RELEASE.json").open("xb") as s:s.write(raw)\n'
    code += 'save(root/"ATTEMPT_SPENT.json",dict(UTC=datetime.now(timezone.utc).isoformat(),source_manifest_sha256=' + repr(PIN) + ',automatic_retry=False))\n'
    command = [PYTHON, '-B', str(REMOTE / 'supervise_owned.py'), '--release-sha256', sha(HERE / 'ROOT_RELEASE.json')]
    env = dict(CUDA_VISIBLE_DEVICES=GPU, PYTHONPATH=str(REPO / '.gnnm_runtime/buddy_extra_v1/site'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
    code += 'env=os.environ.copy();env.update(' + repr(env) + ')\n'
    code += 'with (root/"SUPERVISOR_STDOUT.txt").open("xb") as out,(root/"SUPERVISOR_STDERR.txt").open("xb") as err:\n p=subprocess.Popen(' + repr(command) + ',cwd=repo,env=env,stdout=out,stderr=err,start_new_session=True)\n'
    code += 'raw=(Path("/proc")/str(p.pid)/"stat").read_text();f=raw[raw.rfind(")")+2:].split();assert int(f[2])==int(f[3])==p.pid\n'
    code += 'v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,start_ticks=int(f[19]),group=int(f[2]),session=int(f[3]),command=' + repr(command) + ',source_manifest_sha256=' + repr(PIN) + ',release_sha256=' + repr(sha(HERE / 'ROOT_RELEASE.json')) + ');save(root/"DETACHED_LAUNCH.json",v);print(json.dumps(v))\n'
    save('DETACHED_LAUNCH.json', transport('launch', code))


def monitor(sequence):
    out = HERE / ('owned_monitor%02d' % sequence)
    out.mkdir()
    expected = json.loads((HERE / 'DETACHED_LAUNCH.json').read_text())
    code = context() + 'expected=' + repr(expected) + '\nassert json.loads((root/"DETACHED_LAUNCH.json").read_text())==expected\n'
    code += 'p=Path("/proc")/str(expected["PID"]);identity=None\nif p.exists():\n raw=(p/"stat").read_text();f=raw[raw.rfind(")")+2:].split();assert int(f[19])==expected["start_ticks"];identity=dict(PID=expected["PID"],state=f[0],start_ticks=int(f[19]))\n'
    code += 'files={}\nfor name in ("OWNED_CHILD.json","TERMINAL.json","SUPERVISOR_STDERR.txt","WORKER_STDERR.txt","run01/PROGRESS.json","run01/CENSUS.json","run01/FAILURE.json","run01/DRAW.json","run01/BATCHES.jsonl"):\n p=root/name\n if p.exists():\n  assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root) and p.stat().st_size<2_000_000;b=p.read_bytes();files[name]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode())\n'
    code += 'print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),supervisor_identity=identity,files=files,signals_sent=False,automatic_retry=False,VALID_TEST_reads=False)))\n'
    value = transport('monitor%02d' % sequence, code)
    for name, row in value['files'].items():
        raw = base64.b64decode(row.pop('data'), validate=True)
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        p = out / name
        assert p.resolve().is_relative_to(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('xb') as stream:
            stream.write(raw)
    with (out / 'OBSERVATION.json').open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('stage', 'launch', 'monitor'))
    parser.add_argument('--sequence', type=int, default=1)
    args = parser.parse_args()
    {'stage': stage, 'launch': launch, 'monitor': lambda: monitor(args.sequence)}[args.operation]()
