"""Stage the fixed DDI runtime source and launch its three owned TRAIN epochs.

Source seals and root release are explicit arguments/receipts. No scientific
500-epoch gate, quality selection or TEST permission is changed by this client.
"""
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
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
REMOTE = REMOTE_PHASE / HERE.name
SOURCE = PHASE / 'hlgnn_ddi_f4_exact_cb_integration_preparation_20261004_v2'
F4 = PHASE / 'hlgnn_ddi_f4_private_hop_target_only_preparation_20261004_v1'
REVIEW = PHASE / 'ddi_f4_exact_cb_independent_source_review_20261004_v1'
FRESH_REVIEW = PHASE / 'ddi_f4_runtime_and_receipt_repair_independent_source_review_20261004_v2'
OWNED_SOURCE = PHASE / 'pencil_collab_paired_predictive_preparation_20261004_v2/supervise.py'
TRANSPORT = PHASE / 'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py'
PYTHON = '/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12'
PYTHON_SHA = '14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'
UUID = 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def run(identity, code):
    spec = importlib.util.spec_from_file_location('ddi_practical_runtime_transport', TRANSPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    return module.run('ddi_f4_practical_' + identity + '_20261004', code)


def packet(folder, manifest_pin=None, seal_pin=None):
    assert folder.is_dir() and not folder.is_symlink()
    if manifest_pin is not None:
        assert sha(folder / 'MANIFEST.json') == manifest_pin
    if seal_pin is not None:
        assert sha(folder / 'SEAL.json') == seal_pin
    manifest = json.loads((folder / 'MANIFEST.json').read_text())
    assert ('files' in manifest) != ('payloads' in manifest)
    rows = manifest['files'] if 'files' in manifest else manifest['payloads']
    paths = []
    for row in rows:
        path = folder / row['path']
        assert path.resolve().is_relative_to(folder) and not path.is_symlink()
        assert path.stat().st_size == row.get('bytes', row.get('size')) and sha(path) == row['sha256']
        paths.append(path)
    for name in ('MANIFEST.json', 'SEAL.json'):
        if (folder / name).is_file():
            paths.append(folder / name)
    return paths


def context():
    return ('from pathlib import Path\nfrom datetime import datetime,timezone\n'
            'import base64,hashlib,json,os,subprocess,zlib\n'
            'repo=Path(' + repr(str(REPO)) + ');phase=Path(' + repr(str(REMOTE_PHASE)) + ');root=Path(' + repr(str(REMOTE)) + ')\n'
            'assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
            'def sha(p):\n with Path(p).open("rb") as s:return hashlib.file_digest(s,"sha256").hexdigest()\n')


def stage(args):
    files = (packet(SOURCE, args.manifest, args.seal) + packet(F4) + packet(REVIEW)
             + packet(FRESH_REVIEW))
    files += [OWNED_SOURCE, OWNED_SOURCE.with_name('common.py'), HERE / 'supervise_runtime.py']
    binding = json.loads((SOURCE / 'INPUT_BINDINGS.json').read_text())
    for row in binding['runtime_source_pins']:
        path = PHASE / row['path']
        assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
    config = json.loads((SOURCE / 'config.json').read_text())
    assert config['release_enabled'] is False and config['recipe']['epochs'] == 500
    assert config['arms'] == ['target_only', 'joint', 'separate'] and config['seeds'] == [0, 1, 2]
    files = sorted(set(files))
    payload = []
    for path in files:
        assert not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(PHASE)
        raw = path.read_bytes()
        assert len(raw) < 2_000_000
        payload.append(dict(path=str(path.relative_to(PHASE)), bytes=len(raw), sha256=sha(path),
                            data=base64.b64encode(raw).decode()))
    inventory = [{k: row[k] for k in ('path', 'bytes', 'sha256')} for row in payload]
    save('STAGE_INVENTORY.json', inventory)
    compressed = zlib.compress(json.dumps(payload).encode(), 9)
    encoded = base64.b64encode(compressed).decode()
    chunks = [encoded[start:start + 40000] for start in range(0, len(encoded), 40000)]
    for number, chunk in enumerate(chunks, 1):
        code = context() + 'root.mkdir(parents=True,exist_ok=True);p=root/' + repr('source_part%03d.b64' % number) + '\n'
        code += 'raw=' + repr(chunk) + '.encode()\nwith p.open("xb") as s:s.write(raw)\nprint(json.dumps(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))\n'
        value = run('stage_part%03d' % number, code)
        assert value['sha256'] == hashlib.sha256(chunk.encode()).hexdigest()
    code = context()
    code += 'compressed=base64.b64decode(b"".join((root/("source_part%03d.b64"%n)).read_bytes() for n in range(1,' + str(len(chunks) + 1) + ')),validate=True)\n'
    code += 'assert hashlib.sha256(compressed).hexdigest()==' + repr(hashlib.sha256(compressed).hexdigest()) + '\nrows=json.loads(zlib.decompress(compressed))\n'
    code += 'for row in rows:\n p=phase/row["path"];assert p.resolve().is_relative_to(phase)\n for ancestor in (p,*p.parents):\n  if ancestor==phase.parent:break\n  assert not ancestor.is_symlink()\n raw=base64.b64decode(row["data"],validate=True);assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"]\n if p.exists():assert p.is_file() and p.read_bytes()==raw\n else:\n  p.parent.mkdir(parents=True,exist_ok=True)\n  with p.open("xb") as s:s.write(raw)\n'
    code += 'print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status="EXACT_DDI_RUNTIME_SOURCE_STAGED",files=len(rows),numerical_execution=False,TEST_access=False)))\n'
    save('STAGE_RECEIPT.json', run('stage_join', code))
    print('Exact DDI runtime source and dependencies staged; no numerical execution.')


def launch(args):
    packet(SOURCE, args.manifest, args.seal)
    inventory = json.loads((HERE / 'STAGE_INVENTORY.json').read_text())
    for row in inventory:
        path = PHASE / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    assert (HERE / 'STAGE_RECEIPT.json').is_file() and (HERE / 'ROOT_AUTHORIZATION.md').is_file()
    fresh = json.loads((FRESH_REVIEW / 'REVIEW.json').read_text())
    assert fresh['status'] == 'PASS' and fresh['blocking_findings'] == []
    assert fresh['candidate_manifest_sha256'] == args.manifest and fresh['execution_authorized'] is False
    assert fresh['root_client_sha256'] == sha(Path(__file__))
    assert fresh['root_supervisor_sha256'] == sha(HERE / 'supervise_runtime.py')
    release = dict(schema='ddi-f4-complete-real-TRAIN-epoch-runtime-root-v1',
          UTC=datetime.now(timezone.utc).isoformat(), execution_authorized=True,
          repository=str(REPO), GPU_UUID=UUID, candidate_manifest_sha256=args.manifest,
          candidate_seal_sha256=args.seal, source_inventory=inventory,
          source_review_v1_manifest_sha256=sha(REVIEW / 'MANIFEST.json'),
          fresh_technical_review_sha256=sha(FRESH_REVIEW / 'REVIEW.json'),
          fresh_technical_review_manifest_sha256=sha(FRESH_REVIEW / 'MANIFEST.json'),
          root_authorization_sha256=sha(HERE / 'ROOT_AUTHORIZATION.md'),
          root_authorization_reference=str(REMOTE / 'ROOT_AUTHORIZATION.md'),
          reviewer_v1_does_not_cover_new_runtime_entrypoint=True,
          owned_supervisor_source=str(OWNED_SOURCE.relative_to(PHASE)),
          owned_supervisor_sha256=sha(OWNED_SOURCE), supervisor_sha256=sha(HERE / 'supervise_runtime.py'),
          runtime_entrypoint=str((SOURCE / 'runtime_train_epoch.py').relative_to(PHASE)),
          interpreter=PYTHON, interpreter_sha256=PYTHON_SHA,
          arms=['target_only', 'joint', 'separate'], TRAIN_epochs_per_arm=1, fresh_seed=0,
          cuda_memory_fraction=0.30, minimum_free_MiB_before_arm=34 * 1024,
          minimum_global_headroom_MiB=10 * 1024,
          caps=dict(wall_seconds_per_arm=3600, host_RSS_bytes=32 * 1024**3),
          VALID_scoring=False, TEST_access=False, checkpoint_donor=False,
          co_resident_timing=True, scientific_family_release=False, automatic_retry=False,
          limits='Runtime capacity window only, no prediction or full-fit feasibility claim.')
    save('ROOT_RELEASE.json', release)
    rows = []
    for name in ('ROOT_RELEASE.json', 'ROOT_AUTHORIZATION.md'):
        p = HERE / name
        rows.append(dict(name=name, sha256=sha(p), data=base64.b64encode(p.read_bytes()).decode()))
    environment = dict(CUDA_VISIBLE_DEVICES=UUID, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1',
                       PYTHONPATH=str(REPO / '.gnnm_runtime/buddy_extra_v1/site'))
    command = [PYTHON, '-B', str(REMOTE / 'supervise_runtime.py'), '--release-sha256', sha(HERE / 'ROOT_RELEASE.json')]
    code = context() + 'rows=' + repr(rows) + '\n'
    code += 'assert sha(Path(' + repr(PYTHON) + '))==' + repr(PYTHON_SHA) + '\n'
    code += 'assert not any((root/x).exists() for x in ("ROOT_LAUNCH_SPENT.json","DETACHED_LAUNCH.json","target_only","joint","separate"))\n'
    code += 'for row in rows:\n raw=base64.b64decode(row["data"],validate=True);assert hashlib.sha256(raw).hexdigest()==row["sha256"]\n with (root/row["name"]).open("xb") as s:s.write(raw)\n'
    code += 'with (root/"ROOT_LAUNCH_SPENT.json").open("x") as s:json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),root_release_sha256=' + repr(sha(HERE / 'ROOT_RELEASE.json')) + ',automatic_retry=False),s)\n'
    code += 'command=' + repr(command) + ';environment=' + repr(environment) + '\n'
    code += 'with (root/"DETACHED.stdout").open("xb") as out,(root/"DETACHED.stderr").open("xb") as err:\n child=subprocess.Popen(command,cwd=repo,env=dict(os.environ,**environment),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)\n'
    code += 'raw=(Path("/proc")/str(child.pid)/"stat").read_text();f=raw[raw.rfind(")")+2:].split();assert int(f[2])==int(f[3])==child.pid\n'
    code += 'value=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,start_ticks=int(f[19]),group=int(f[2]),session=int(f[3]),command=command,root_release_sha256=' + repr(sha(HERE / 'ROOT_RELEASE.json')) + ',VALID_scoring=False,TEST_access=False)\n'
    code += 'with (root/"DETACHED_LAUNCH.json").open("x") as s:json.dump(value,s,indent=2);s.write("\\n")\nprint(json.dumps(value))\n'
    value = run('launch', code)
    save('DETACHED_LAUNCH.json', value)
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('stage', 'launch'))
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--seal', required=True)
    args = parser.parse_args()
    {'stage': stage, 'launch': launch}[args.operation](args)
