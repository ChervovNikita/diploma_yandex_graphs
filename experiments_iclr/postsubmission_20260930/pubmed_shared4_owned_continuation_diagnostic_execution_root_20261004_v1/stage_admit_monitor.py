"""Explicit source staging, bounded root admission and metadata-only observation.

No invocation retries, scientific fits, TEST access or tensor transfers.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import base64
import hashlib
import importlib.util
import json
import zlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / 'pubmed_shared4_owned_continuation_diagnostic_source_20261004_v1'
REVIEW = PHASE / 'pubmed_shared4_owned_continuation_diagnostic_fresh_review_20261004_v1'
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
REMOTE = REMOTE_PHASE / HERE.name
EXPECTED_MANIFEST = 'e9795ede88f0630b1ca4e8f097ea612e50e355e913e30b6dbe9dbd9bdfa08fe4'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def transport(identity, code):
    path = PHASE / 'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py'
    spec = importlib.util.spec_from_file_location('safe_guarded_transport', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    return module.run('pubmed_contdiag_v1_' + identity + '_20261004', code)


def context():
    return ('from pathlib import Path\nfrom datetime import datetime,timezone\n'
            'import base64,hashlib,json,os,subprocess\n'
            'repo=Path(' + repr(str(REPO)) + ');phase=Path(' + repr(str(REMOTE_PHASE)) + ');root=Path(' + repr(str(REMOTE)) + ')\n'
            'assert Path.cwd()==repo and os.uname().nodename=="peptide"\n')


def validate_manifest(folder):
    manifest = json.loads((folder / 'MANIFEST.json').read_text())
    paths = []
    for row in manifest['files']:
        path = folder / row['path']
        assert path.resolve().is_relative_to(folder) and not path.is_symlink()
        assert sha(path) == row['sha256'] and path.stat().st_size == row.get('bytes', row.get('size'))
        paths.append(path)
    paths.append(folder / 'MANIFEST.json')
    if (folder / 'SEAL.json').exists():
        paths.append(folder / 'SEAL.json')
    return paths


def stage():
    assert sha(SOURCE / 'MANIFEST.json') == EXPECTED_MANIFEST
    review = REVIEW / 'REVIEW.json'
    value = json.loads(review.read_text())
    assert value['status'] == 'PASS' and value['candidate_manifest_sha256'] == EXPECTED_MANIFEST
    assert value['execution_authorized'] is False
    files = validate_manifest(SOURCE) + validate_manifest(REVIEW)
    closure = json.loads((SOURCE / 'METADATA_CLOSURE.json').read_text())
    for row in closure['external_source_files']:
        path = PHASE / row['path']
        assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
        files.append(path)
    for row in list(closure['scalar_prerequisite_metadata'].values()) + [closure['additional_required_metadata']]:
        path = PHASE / row['path']
        if path.is_file():
            assert sha(path) == row['sha256']
            files.append(path)
    rows = []
    for path in dict.fromkeys(files):
        assert path.resolve().is_relative_to(PHASE) and path.stat().st_size < 2000000
        raw = path.read_bytes()
        rows.append(dict(path=str(path.relative_to(PHASE)), bytes=len(raw), sha256=sha(path), data=base64.b64encode(raw).decode()))
    save(HERE / 'SOURCE_METADATA_STAGE_INVENTORY.json', [{k: v for k, v in row.items() if k != 'data'} for row in rows])
    packed = zlib.compress(json.dumps(rows).encode(), 9)
    encoded = base64.b64encode(packed).decode()
    chunks = [encoded[n:n+40000] for n in range(0, len(encoded), 40000)]
    for n, chunk in enumerate(chunks, 1):
        code = context() + 'staging=root/"source_chunks";staging.mkdir(parents=True,exist_ok=True)\nraw=' + repr(chunk) + '.encode()\n'
        code += 'with (staging/' + repr('part%03d.b64' % n) + ').open("xb") as stream:stream.write(raw)\n'
        code += 'print(json.dumps(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))\n'
        observed = transport('chunk%03d' % n, code)
        assert observed['sha256'] == hashlib.sha256(chunk.encode()).hexdigest()
    code = context() + 'import zlib\n'
    code += 'packed=base64.b64decode(b"".join((root/"source_chunks"/("part%03d.b64"%n)).read_bytes() for n in range(1,' + str(len(chunks)+1) + ')),validate=True)\n'
    code += 'assert hashlib.sha256(packed).hexdigest()==' + repr(hashlib.sha256(packed).hexdigest()) + '\nrows=json.loads(zlib.decompress(packed))\n'
    code += 'for row in rows:\n path=phase/row["path"];assert path.resolve().is_relative_to(phase);raw=base64.b64decode(row["data"],validate=True);assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"]\n if path.exists():assert not path.is_symlink() and path.read_bytes()==raw\n else:\n  path.parent.mkdir(parents=True,exist_ok=True)\n  with path.open("xb") as stream:stream.write(raw)\n'
    code += 'specifications='+repr(closure['scalar_prerequisite_metadata'])+'\nprerequisites={}\n'
    code += 'for name,row in specifications.items():\n path=phase/row["path"];assert path.resolve().is_relative_to(phase) and not path.is_symlink();raw=path.read_bytes();assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"];value=json.loads(raw);assert value["status"]==row["required_status"];prerequisites[name]=dict(path=row["path"],sha256=row["sha256"],bytes=len(raw),status=value["status"],verified_by_root=True)\n'
    code += 'print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status="SOURCE_AND_COMPLETE_METADATA_STAGED",files=len(rows),bytes=sum(row["bytes"] for row in rows),prerequisites=prerequisites,numerical_execution=False)))\n'
    save(HERE / 'SOURCE_STAGE_RECEIPT.json', transport('join', code))
    print('Reviewed source and complete metadata staged; numerical execution disabled.')


def admit():
    assert sha(SOURCE / 'MANIFEST.json') == EXPECTED_MANIFEST
    assert (HERE / 'SOURCE_STAGE_RECEIPT.json').is_file()
    plan = json.loads((SOURCE / 'PLAN.json').read_text())
    binding = json.loads((SOURCE / 'SOURCE_BINDING.json').read_text())
    review = REVIEW / 'REVIEW.json'
    reviewed = json.loads(review.read_text())
    assert reviewed['status'] == 'PASS' and reviewed['candidate_manifest_sha256'] == EXPECTED_MANIFEST
    assert reviewed['execution_authorized'] is False
    release = json.loads((SOURCE / 'ROOT_RELEASE_TEMPLATE.json').read_text())
    failed = PHASE / binding['failed_qualification_execution_root']
    pins = {}
    for name, row in binding['owned_failure_metadata'].items():
        local = failed / 'owned_monitor03' / Path(row['path']).relative_to(binding['failed_qualification_execution_root'])
        if name == 'owned_serialized_identity':
            local = failed / 'owned_failure_identity_observation01/private_SERIALIZED_IDENTITY.json'
        assert local.is_file() and local.stat().st_size < 2000000
        pins[name] = dict(path=row['path'], bytes=local.stat().st_size, sha256=sha(local))
        release[name + '_sha256'] = sha(local)
    prerequisites = {}
    observed_prerequisites=json.loads((HERE/'SOURCE_STAGE_RECEIPT.json').read_text())['prerequisites']
    for name, row in binding['prerequisite_receipts'].items():
        observed=observed_prerequisites[name]
        assert observed['path']==row['path'] and observed['sha256']==row['sha256']
        assert observed['status']==row['required_status'] and observed['verified_by_root'] is True
        prerequisites[name] = dict(path=row['path'],sha256=row['sha256'],verified_by_root=True)
    stage = 'continuation_diagnostic'
    release.update(status='APPROVED',authorized_stages=[stage],automatic_retry=False,
        root_source_review_approved=True,source_manifest_sha256=EXPECTED_MANIFEST,plan_sha256=sha(SOURCE/'PLAN.json'),
        caps=plan['stages'][stage]['caps'],invocation=plan['stages'][stage]['invocation'],input_authority=plan['input_authority'],
        existing_qualification_reused=True,prerequisites=prerequisites,
        independent_source_review_path=str(REMOTE_PHASE / review.relative_to(PHASE)),independent_source_review_sha256=sha(review),
        root_authorization_reference=str(REMOTE/'ROOT_ADMISSION.json'))
    admission = dict(UTC=datetime.now(timezone.utc).isoformat(),status='APPROVED_ONE_OWNED_ENGINEERING_DIAGNOSTIC',
        source_manifest_sha256=EXPECTED_MANIFEST,independent_review_sha256=sha(review),
        scope='Two original full epoch6 continuations;72 Adam updates;zero VALID serves;one owned epoch5 state load.',
        root_review='Unchanged restore helper is intentionally exercised; pristine observation clone and full pre-state/Adam alias evidence precede numeric interpretation.',
        failed_attempt_metadata_pins=pins,previous_failure_preserved=True,model_runtime_inputs_recipe_tolerance_changed=False,
        scientific_fits=0,TEST_access=False,state_donor_allowed=False,automatic_retry=False)
    save(HERE/'ROOT_ADMISSION.json',admission)
    save(HERE/'ROOT_RELEASE_continuation_diagnostic.json',release)
    payload=[]
    for local in (HERE/'ROOT_ADMISSION.json',HERE/'ROOT_RELEASE_continuation_diagnostic.json'):
        payload.append(dict(path=str(REMOTE/local.name),sha256=sha(local),data=base64.b64encode(local.read_bytes()).decode()))
    code=context()+'pins='+repr(pins)+'\n'
    code+='for row in pins.values():\n path=phase/row["path"];assert not path.is_symlink() and path.stat().st_size==row["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest()==row["sha256"]\n'
    code+='q=subprocess.run(["nvidia-smi","--query-gpu=uuid,memory.free,memory.total,name","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=15)\n'
    code+='gpu_rows=[[v.strip() for v in line.split(",")] for line in q.stdout.splitlines()];selected=next(row for row in gpu_rows if row[0]=='+repr(plan['execution_profile']['environment']['CUDA_VISIBLE_DEVICES'])+')\n'
    code+='assert int(selected[1])>=73728 and int(selected[2])==81920 and selected[3]=="NVIDIA A100 80GB PCIe"\n'
    code+='available=next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"));assert available>=40*1024**3\n'
    code+='assert not (root/"continuation_diagnostic/run01").exists() and not (root/"supervision/continuation_diagnostic/run01").exists()\n'
    code+='rows='+repr(payload)+'\nfor row in rows:\n path=Path(row["path"]);assert path.resolve().is_relative_to(phase);raw=base64.b64decode(row["data"],validate=True);assert hashlib.sha256(raw).hexdigest()==row["sha256"]\n with path.open("xb") as stream:stream.write(raw)\n'
    code+='print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status="EXACT_FAILURE_AND_RESOURCES_AUTHENTICATED",GPU_rows=gpu_rows,host_MemAvailable_bytes=available,numerical_execution=False)))\n'
    save(HERE/'FAILED_METADATA_AND_RESOURCE_ADMISSION.json',transport('pins_and_resources',code))
    environment=dict(plan['execution_profile']['environment'],GNNM_SSH_DESTINATION='shmelev@192.168.18.77',PYTHONDONTWRITEBYTECODE='1')
    gate_code='import sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(REMOTE_PHASE/SOURCE.name))+')\nfrom common import gate\ngate(Path('+repr(str(REMOTE/'ROOT_RELEASE_continuation_diagnostic.json'))+'),'+repr(sha(HERE/'ROOT_RELEASE_continuation_diagnostic.json'))+',"continuation_diagnostic")\nprint("EXACT_STDLIB_GATE_PASS")\n'
    code=context()+'environment='+repr(environment)+';command='+repr([plan['execution_profile']['interpreter_path'],'-B','-c',gate_code])+'\n'
    code+='r=subprocess.run(command,cwd=repo,env=dict(os.environ,**environment),capture_output=True,text=True,timeout=45)\nprint(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status="PASS" if r.returncode==0 else "FAILED",exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,numerical_execution=False)))\n'
    gate=transport('complete_gate',code)
    save(HERE/'PRENUMERICAL_GATE.json',gate)
    assert gate['status']=='PASS',gate
    command=[plan['execution_profile']['interpreter_path'],'-B',str(REMOTE_PHASE/SOURCE.name/'supervise.py'),'--stage',stage,'--root-release',str(REMOTE/'ROOT_RELEASE_continuation_diagnostic.json'),'--release-sha256',sha(HERE/'ROOT_RELEASE_continuation_diagnostic.json')]
    code=context()+'environment='+repr(environment)+';command='+repr(command)+'\n'
    code+='assert not (root/"continuation_diagnostic/run01").exists() and not (root/"supervision/continuation_diagnostic/run01").exists()\n'
    code+='with (root/"DETACHED_STDOUT.txt").open("xb") as out,(root/"DETACHED_STDERR.txt").open("xb") as err:\n child=subprocess.Popen(command,cwd=repo,env=dict(os.environ,**environment),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)\n'
    code+='raw=(Path("/proc")/str(child.pid)/"stat").read_text();fields=raw[raw.rfind(")")+2:].split();result=dict(UTC=datetime.now(timezone.utc).isoformat(),status="OWNED_CONTINUATION_DIAGNOSTIC_LAUNCHED",supervisor_PID=child.pid,supervisor_start_ticks=int(fields[19]),command=command,environment=environment,scientific_fits=0,engineering_updates_planned=72,TEST_access=False)\n'
    code+='with (root/"DETACHED_LAUNCH.json").open("x") as stream:json.dump(result,stream,indent=2);stream.write("\\n")\nprint(json.dumps(result))\n'
    result=transport('launch',code)
    save(HERE/'DETACHED_LAUNCH.json',result)
    print(json.dumps({k:result[k] for k in ('UTC','status','supervisor_PID','supervisor_start_ticks')}))


def monitor(sequence):
    out=HERE/('owned_monitor%02d'%sequence)
    out.mkdir()
    launch=json.loads((HERE/'DETACHED_LAUNCH.json').read_text())
    code=context()+'expected='+repr(launch)+'\nfiles=[]\n'
    code+='proc=Path("/proc")/str(expected["supervisor_PID"]);identity=None\nif proc.exists():\n raw=(proc/"stat").read_text();v=raw[raw.rfind(")")+2:].split();assert int(v[19])==expected["supervisor_start_ticks"];identity=dict(pid=expected["supervisor_PID"],start_ticks=int(v[19]),state=v[0])\n'
    code+='for relative in ["continuation_diagnostic/run01/DIAGNOSTIC.json","continuation_diagnostic/run01/FAILURE.json","continuation_diagnostic/run01/RESTORE_ALIAS_PROGRESS.json","continuation_diagnostic/run01/FINAL_CUSTODY.json","continuation_diagnostic/run01/STATUS.json","supervision/continuation_diagnostic/run01/SUPERVISOR_TERMINAL.json","supervision/continuation_diagnostic/run01/SUPERVISOR_STATUS.json","supervision/continuation_diagnostic/run01/SUPERVISOR_LAUNCH.json","DETACHED_STDERR.txt"]:\n path=root/relative\n if path.is_file():\n  assert not path.is_symlink() and path.stat().st_size<3000000;raw=path.read_bytes();raw.decode("utf8");files.append(dict(path=relative,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(raw).decode()))\n'
    code+='print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),supervisor_identity=identity,metadata_only=True,signals_sent=False,retry=False,files=files)))\n'
    value=transport('monitor%02d'%sequence,code)
    rows=value.pop('files');descriptors=[];decoded={}
    for row in rows:
        raw=base64.b64decode(row['data'],validate=True)
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        path=out/row['path'];assert path.resolve().is_relative_to(out);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(raw)
        descriptors.append({k:v for k,v in row.items() if k!='data'})
        if path.suffix=='.json':decoded[row['path']]=json.loads(raw)
    save(out/'MONITOR_RECEIPT.json',dict(value,files=descriptors))
    diagnostic=decoded.get('continuation_diagnostic/run01/DIAGNOSTIC.json')
    terminal=decoded.get('supervision/continuation_diagnostic/run01/SUPERVISOR_TERMINAL.json')
    summary=dict(UTC=value['UTC'],supervisor_identity=value['supervisor_identity'],diagnostic_status=diagnostic.get('status') if diagnostic else None,terminal_status=terminal.get('status') if terminal else None,
        loss=diagnostic['diagnostic_evidence']['loss'] if diagnostic else None,
        saved_tree_mutated_after_first_epoch=diagnostic['diagnostic_evidence']['saved_tree_mutated_after_first_epoch'] if diagnostic else None,
        progress=diagnostic['progress'] if diagnostic else None,failure=decoded.get('continuation_diagnostic/run01/FAILURE.json'))
    save(out/'SUMMARY.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['stage','admit','monitor'])
    parser.add_argument('--sequence',type=int)
    args=parser.parse_args()
    if args.mode=='stage':stage()
    elif args.mode=='admit':admit()
    else:
        assert args.sequence and 1<=args.sequence<=99
        monitor(args.sequence)
