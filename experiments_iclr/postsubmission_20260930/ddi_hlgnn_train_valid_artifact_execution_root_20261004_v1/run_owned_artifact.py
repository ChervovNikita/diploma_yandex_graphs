"""Root client: no action on import; require exact root-reviewed hashes for staging/launch."""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import zlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def controls(args):
    plan = json.loads((HERE / 'CLIENT_PLAN.json').read_text())
    assert sha(HERE / 'EXACT_REMOTE_COMMAND.txt') == plan['exact_remote_command_sha256']
    assert sha(HERE / 'ROOT_RELEASE_PROPOSED.json') == plan['proposed_release_sha256']
    assert sha(HERE / 'REMOTE_CONTEXT.py.txt') == plan['remote_context_sha256']
    assert sha(Path(__file__)) == plan['client_sha256']
    if args.operation in ('stage', 'launch'):
        assert args.root_reviewed_command_sha256 == plan['exact_remote_command_sha256']
        assert args.root_reviewed_release_sha256 == plan['proposed_release_sha256']
    source = PHASE / plan['source_packet']
    assert sha(source / 'MANIFEST.json') == plan['source_manifest_sha256']
    assert sha(source / 'SEAL.json') == plan['source_seal_sha256']
    for pin in json.loads((source / 'MANIFEST.json').read_text())['files']:
        path = source / pin['path']
        assert path.resolve().is_relative_to(source) and not path.is_symlink()
        assert path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256']
    for pin in plan['transport_pins']:
        path = PHASE / pin['path']
        assert not path.is_symlink() and path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256']
    assert sha(HERE / 'supervise_owned_artifact.py') == plan['supervisor_sha256']
    return plan, source


def transport(identity, code, plan):
    spec = importlib.util.spec_from_file_location('ddi_artifact_confined_transport', PHASE / plan['transport_pins'][0]['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    return module.run('ddi_hlgnn_artifact_' + identity + '_20261004', code)


def stage(args, plan, source):
    assert not (HERE / 'STAGE_INVENTORY.json').exists()
    paths = [source / row['path'] for row in json.loads((source / 'MANIFEST.json').read_text())['files']]
    paths += [source / 'MANIFEST.json', source / 'SEAL.json', HERE / 'supervise_owned_artifact.py']
    rows = []
    for path in paths:
        raw = path.read_bytes()
        assert len(raw) < 2000000 and path.resolve().is_relative_to(PHASE)
        rows.append(dict(path=str(path.relative_to(PHASE)), bytes=len(raw), sha256=sha(path), data=base64.b64encode(raw).decode()))
    save('STAGE_INVENTORY.json', [{k: v for k, v in row.items() if k != 'data'} for row in rows])
    packed = zlib.compress(json.dumps(rows).encode(), 9)
    encoded = base64.b64encode(packed).decode()
    chunks = [encoded[i:i + 40000] for i in range(0, len(encoded), 40000)]
    prefix = (HERE / 'REMOTE_CONTEXT.py.txt').read_text()
    for number, chunk in enumerate(chunks, 1):
        code = prefix + 'd=root/"source_chunks";d.mkdir(parents=True,exist_ok=True)\n'
        code += 'b=' + repr(chunk) + '.encode()\nwith (d/' + repr('part%03d.b64' % number) + ').open("xb") as s:s.write(b)\n'
        code += 'print(json.dumps(dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())))\n'
        result = transport('chunk%03d' % number, code, plan)
        assert result['sha256'] == hashlib.sha256(chunk.encode()).hexdigest()
    code = prefix + 'import zlib\npacked=base64.b64decode(b"".join((root/"source_chunks"/("part%03d.b64"%n)).read_bytes() for n in range(1,' + str(len(chunks) + 1) + ')),validate=True)\n'
    code += 'assert hashlib.sha256(packed).hexdigest()==' + repr(hashlib.sha256(packed).hexdigest()) + '\nrows=json.loads(zlib.decompress(packed))\n'
    code += 'for row in rows:\n p=phase/row["path"];assert p.resolve().is_relative_to(phase) and not p.is_symlink();b=base64.b64decode(row["data"],validate=True);assert len(b)==row["bytes"] and hashlib.sha256(b).hexdigest()==row["sha256"]\n if p.exists():assert p.is_file() and p.read_bytes()==b\n else:\n  p.parent.mkdir(parents=True,exist_ok=True)\n  with p.open("xb") as s:s.write(b)\n  p.chmod(0o444)\n'
    code += 'save(root/"STAGE_RECEIPT.json",dict(status="EXACT_SOURCE_STAGED",files=len(rows),source_manifest_sha256=' + repr(plan['source_manifest_sha256']) + ',numerical_execution=False))\nprint(json.dumps(dict(status="EXACT_SOURCE_STAGED",files=len(rows))))\n'
    save('STAGE_RECEIPT.json', transport('join', code, plan))


def launch(args, plan):
    assert (HERE / 'STAGE_RECEIPT.json').is_file() and not (HERE / 'ROOT_LAUNCH_SPENT.json').exists()
    save('ROOT_LAUNCH_SPENT.json', dict(command_sha256=plan['exact_remote_command_sha256'],
         release_sha256=plan['proposed_release_sha256'], automatic_retry=False))
    command = (HERE / 'EXACT_REMOTE_COMMAND.txt').read_text()
    begin = command.index("\n", command.index("<<'QAROOTPY'")) + 1
    code = command[begin:command.rfind('\nQAROOTPY')]
    wrapper = 'cd ' + plan['remote_repository'] + " && /usr/bin/python3 -I -S -B - <<'QAROOTPY'\n"
    assert hashlib.sha256((wrapper + code + '\nQAROOTPY\n').encode()).hexdigest() == plan['exact_remote_command_sha256']
    save('DETACHED_LAUNCH.json', transport('launch', code, plan))


def monitor(args, plan):
    assert args.sequence > 0
    expected = json.loads((HERE / 'DETACHED_LAUNCH.json').read_text())
    out = HERE / ('owned_monitor%02d' % args.sequence)
    out.mkdir()
    code = (HERE / 'REMOTE_CONTEXT.py.txt').read_text() + 'expected=' + repr(expected) + '\n'
    code += 'assert json.loads((root/"DETACHED_LAUNCH.json").read_text())==expected\np=Path("/proc")/str(expected["PID"]);identity=None\nif p.exists():\n raw=(p/"stat").read_text();f=raw[raw.rfind(")")+2:].split();assert int(f[19])==expected["start_ticks"];identity=dict(PID=expected["PID"],state=f[0],start_ticks=int(f[19]))\n'
    code += 'files={}\nfor name in ("OWNED_CHILD.json","TERMINAL.json","SUPERVISOR_STDERR.txt","WORKER_STDERR.txt","run01/QUALIFICATION.json","run01/QUALIFICATION_FAILURE.json"):\n p=root/name\n if p.exists():\n  assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root) and p.stat().st_size<2000000;b=p.read_bytes();files[name]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode())\nprint(json.dumps(dict(supervisor_identity=identity,files=files,signals_sent=False,automatic_retry=False,TEST_payload_decoded=False)))\n'
    value = transport('monitor%02d' % args.sequence, code, plan)
    for name, row in value['files'].items():
        raw = base64.b64decode(row.pop('data'), validate=True)
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        path = out / name
        assert path.resolve().is_relative_to(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    with (out / 'OBSERVATION.json').open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('stage', 'launch', 'monitor'))
    parser.add_argument('--root-reviewed-command-sha256')
    parser.add_argument('--root-reviewed-release-sha256')
    parser.add_argument('--sequence', type=int, default=1)
    args = parser.parse_args()
    plan, source = controls(args)
    if args.operation == 'stage':
        stage(args, plan, source)
    elif args.operation == 'launch':
        launch(args, plan)
    else:
        monitor(args, plan)
