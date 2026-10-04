"""Stage exact reviewed sources and admit one physically supervised CPU oracle."""
from datetime import datetime, timezone
import argparse
import base64
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
SUPERVISOR = PHASE / 'graph_count_conditioned_pattern_cpu_supervisor_preparation_20261004_v3'
CORE = PHASE / 'graph_count_conditioned_pattern_loss_prototype_preparation_20261004_v1'
REVIEW = PHASE / 'graph_count_conditioned_pattern_cpu_supervisor_fresh_review_20261004_v3'
SUPERVISOR_PIN = 'e71e869ac04bd898e7b02bb8c767cee5f61422be2759865574f8e4a06d1f982a'
CORE_PIN = '9198a01dcc11c635bc47d57551b271ad0515009c41adc6b5fc7757e38aae657e'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def transport(identity, code):
    path = PHASE / 'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py'
    spec = importlib.util.spec_from_file_location('cpu_safe_project_transport', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    return module.run('count_cpu_v1_' + identity + '_20261004', code)


def context():
    return 'from pathlib import Path\nimport base64,hashlib,json,os,subprocess\nfrom datetime import datetime,timezone\nrepo=Path(' + repr(str(REPO)) + ');phase=Path(' + repr(str(REMOTE_PHASE)) + ');root=Path(' + repr(str(REMOTE)) + ')\nassert Path.cwd()==repo and os.uname().nodename=="peptide"\n'


def packet(folder, expected=None):
    if expected:
        assert sha(folder / 'MANIFEST.json') == expected
    rows = json.loads((folder / 'MANIFEST.json').read_text())['files']
    files = []
    for row in rows:
        path = folder / row['path']
        assert path.resolve().is_relative_to(folder) and not path.is_symlink()
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        files.append(path)
    return files + [folder / 'MANIFEST.json', folder / 'SEAL.json']


def stage():
    files = packet(SUPERVISOR, SUPERVISOR_PIN) + packet(CORE, CORE_PIN) + packet(REVIEW, '3c1c29a6da429667deae6daea15dbfbba6b26d84cb40bf4714759339690a00b9')
    for folder, key in [(SUPERVISOR, 'external_input_pins'), (CORE, 'external_source_pins')]:
        for row in json.loads((folder / 'SOURCE_BINDING.json').read_text())[key]:
            path = PHASE / row['path']
            assert path.resolve().is_relative_to(PHASE) and not path.is_symlink()
            assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
            files.append(path)
    rows = []
    for path in dict.fromkeys(files):
        assert path.stat().st_size < 2_000_000 and path.suffix in ('.py', '.json', '.md', '.diff', '.patch')
        raw = path.read_bytes()
        rows.append(dict(path=str(path.relative_to(PHASE)), bytes=len(raw), sha256=sha(path), data=base64.b64encode(raw).decode()))
    save('SOURCE_STAGE_INVENTORY.json', [{k:v for k,v in row.items() if k != 'data'} for row in rows])
    packed = zlib.compress(json.dumps(rows).encode(), 9)
    encoded = base64.b64encode(packed).decode()
    chunks = [encoded[n:n+40000] for n in range(0,len(encoded),40000)]
    for number, chunk in enumerate(chunks, 1):
        code = context() + 'staging=root/"source_chunks";staging.mkdir(parents=True,exist_ok=True)\nraw=' + repr(chunk) + '.encode()\nwith (staging/' + repr('part%03d.b64' % number) + ').open("xb") as s:s.write(raw)\nprint(json.dumps(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))\n'
        value = transport('chunk%03d' % number, code)
        assert value['sha256'] == hashlib.sha256(chunk.encode()).hexdigest()
    code = context() + 'import zlib\npacked=base64.b64decode(b"".join((root/"source_chunks"/("part%03d.b64"%n)).read_bytes() for n in range(1,' + str(len(chunks)+1) + ')),validate=True)\nassert hashlib.sha256(packed).hexdigest()==' + repr(hashlib.sha256(packed).hexdigest()) + '\nrows=json.loads(zlib.decompress(packed))\n'
    code += 'for row in rows:\n p=phase/row["path"];assert p.resolve().is_relative_to(phase);raw=base64.b64decode(row["data"],validate=True);assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"]\n if p.exists():assert not p.is_symlink() and p.read_bytes()==raw\n else:\n  p.parent.mkdir(parents=True,exist_ok=True)\n  with p.open("xb") as s:s.write(raw)\n'
    code += 'print(json.dumps(dict(status="REVIEWED_SOURCE_STAGED_CPU_DISABLED",files=len(rows),bytes=sum(r["bytes"] for r in rows),numerical_execution=False)))\n'
    save('SOURCE_STAGE_RECEIPT.json', transport('join', code))
    print('Reviewed CPU sources staged; no numerical execution.')


def admit():
    assert (HERE / 'SOURCE_STAGE_RECEIPT.json').exists()
    for name, pin, manifest in [('REVIEW.json', '63cc2abf61d322b3a406c53ecb55155627b361217a1ccdd71671c8d08fba994d', SUPERVISOR_PIN), ('COMBINED_CORE_REVIEW.json', 'ba9c33651534886c79a09383ce4b6ac2ee7b362accefff349fa061294f9d08c1', CORE_PIN)]:
        f = REVIEW / name
        d = json.loads(f.read_text())
        assert sha(f) == pin and d['status'] == 'PASS' and d['candidate_manifest_sha256'] == manifest
        assert d['execution_authorized'] is False and not d.get('blocking_findings')
    observed = json.loads((HERE / 'EXISTING_RUNTIME_PROFILE_OBSERVATION.json').read_text())
    assert observed['platform'] == 'linux' and observed['torch_imported'] is False and observed['torch_distribution_version'] == '2.7.1'
    inner = json.loads((CORE / 'CPU_ROOT_RELEASE_TEMPLATE.json').read_text())
    inner.update(status='APPROVED',authorized_stages=['fabricated_cpu'],root_authorization_reference=str(REMOTE/'ROOT_ADMISSION.json'),source_manifest_sha256=CORE_PIN,independent_source_review_path=str(REMOTE_PHASE/REVIEW.name/'COMBINED_CORE_REVIEW.json'),independent_source_review_sha256=sha(REVIEW/'COMBINED_CORE_REVIEW.json'),runtime_profile={k:observed[k] for k in ('python_executable','python_executable_sha256','torch_distribution_version')})
    save('ROOT_RELEASE_fabricated_cpu.json', inner)
    outer = json.loads((SUPERVISOR / 'ROOT_RELEASE_TEMPLATE.json').read_text())
    command = [observed['python_executable'],'-I','-B',str(REMOTE_PHASE/CORE.name/'qualify_cpu.py'),'--root-release',str(REMOTE/'ROOT_RELEASE_fabricated_cpu.json'),'--release-sha256',sha(HERE/'ROOT_RELEASE_fabricated_cpu.json')]
    command_pin = hashlib.sha256(json.dumps(command,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    outer.update(status='APPROVED',authorized_stages=['fabricated_cpu_owned_supervisor'],root_authorization_reference=str(REMOTE/'ROOT_ADMISSION.json'),supervisor_manifest_sha256=SUPERVISOR_PIN,independent_supervisor_review_path=str(REMOTE_PHASE/REVIEW.name/'REVIEW.json'),independent_supervisor_review_sha256=sha(REVIEW/'REVIEW.json'),inner_cpu_release_path=str(REMOTE/'ROOT_RELEASE_fabricated_cpu.json'),inner_cpu_release_sha256=sha(HERE/'ROOT_RELEASE_fabricated_cpu.json'),runtime_profile={k:observed[k] for k in ('platform','python_executable','python_executable_sha256','ps_executable','ps_executable_sha256')},child_argv=command,child_command_sha256=command_pin,child_cwd=str(REMOTE_PHASE/CORE.name))
    save('ROOT_RELEASE_owned_supervisor.json', outer)
    save('ROOT_ADMISSION.json', dict(UTC=datetime.now(timezone.utc).isoformat(),status='APPROVED_ONE_PHYSICALLY_SUPERVISED_FABRICATED_CPU_ORACLE',source_manifest_sha256=CORE_PIN,supervisor_manifest_sha256=SUPERVISOR_PIN,limits_accepted=['Sampled aggregate session RSS is not an instantaneous OS memory limit.','No process escape sandbox; the reviewed oracle does not create child processes.','F02 endpoint-range and L01 native float32/ragged assurance remain native integration blockers.'],GPU_data_access=False,scientific_fits=0,TEST_supported=False,automatic_retry=False,runtime_observation_sha256=sha(HERE/'EXISTING_RUNTIME_PROFILE_OBSERVATION.json')))
    payload = []
    for name in ['ROOT_RELEASE_fabricated_cpu.json','ROOT_RELEASE_owned_supervisor.json','ROOT_ADMISSION.json']:
        f = HERE / name
        payload.append(dict(path=str(REMOTE/name),sha256=sha(f),data=base64.b64encode(f.read_bytes()).decode()))
    code = context() + 'rows=' + repr(payload) + '\nfor row in rows:\n p=Path(row["path"]);assert p.resolve().is_relative_to(root);raw=base64.b64decode(row["data"],validate=True);assert hashlib.sha256(raw).hexdigest()==row["sha256"]\n with p.open("xb") as s:s.write(raw)\nprint(json.dumps(dict(status="EXACT_CPU_RELEASES_STAGED",numerical_execution=False)))\n'
    save('RELEASE_STAGE_RECEIPT.json', transport('release', code))
    remote_source = REMOTE_PHASE / SUPERVISOR.name
    gate_code = 'import sys;from pathlib import Path;sys.path.insert(0,' + repr(str(remote_source)) + ');from supervise_cpu import preflight;preflight(Path(' + repr(str(REMOTE/'ROOT_RELEASE_owned_supervisor.json')) + '),' + repr(sha(HERE/'ROOT_RELEASE_owned_supervisor.json')) + ');print("PASS_CPU_STDLIB_GATE")'
    gate_command = [observed['python_executable'],'-I','-B','-c',gate_code]
    code = context() + 'r=subprocess.run(' + repr(gate_command) + ',cwd=repo,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1"),capture_output=True,text=True,timeout=30)\nprint(json.dumps(dict(status="PASS" if r.returncode==0 else "FAILED",exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,numerical_execution=False)))\n'
    gate = transport('gate', code)
    save('PRENUMERICAL_GATE.json', gate)
    assert gate['status'] == 'PASS', gate
    command = [observed['python_executable'],'-I','-B',str(remote_source/'supervise_cpu.py'),'--root-release',str(REMOTE/'ROOT_RELEASE_owned_supervisor.json'),'--release-sha256',sha(HERE/'ROOT_RELEASE_owned_supervisor.json')]
    code = context() + 'command=' + repr(command) + '\nassert not (root/"owned_supervisor/run01").exists() and not (root/"fabricated_cpu/run01").exists()\nwith (root/"DETACHED_STDOUT.txt").open("xb") as out,(root/"DETACHED_STDERR.txt").open("xb") as err:\n child=subprocess.Popen(command,cwd=repo,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1"),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)\nraw=(Path("/proc")/str(child.pid)/"stat").read_text();fields=raw[raw.rfind(")")+2:].split();v=dict(UTC=datetime.now(timezone.utc).isoformat(),status="OWNED_FABRICATED_CPU_LAUNCHED",supervisor_PID=child.pid,supervisor_start_ticks=int(fields[19]),command=command,GPU_data_access=False,scientific_fits=0,TEST_supported=False)\nwith (root/"DETACHED_LAUNCH.json").open("x") as s:json.dump(v,s,indent=2);s.write("\\n")\nprint(json.dumps(v))\n'
    launch = transport('launch', code)
    save('DETACHED_LAUNCH.json', launch)
    print(json.dumps({k:launch[k] for k in ('UTC','status','supervisor_PID','supervisor_start_ticks')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['stage','admit'])
    args = parser.parse_args()
    stage() if args.mode == 'stage' else admit()
