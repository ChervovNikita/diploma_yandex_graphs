"""Reusable explicit stage/admit/launch client; no automatic operation."""
import argparse
import base64
import importlib.util
import json
from pathlib import Path
import re
import zlib

from execution_common import ROOT, PHASE, pin, save, sha, source_inventory, utc

REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE = REPO/'experiments_iclr/postsubmission_20260930'
REMOTE = REMOTE_PHASE/ROOT.name
TRANSPORT_RELATIVE = 'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py'


def plan():
    return json.loads((ROOT/'PLAN.json').read_text())


def run(identity, code):
    """Use the existing relay; every remote operation bootstraps pinned3.12."""
    current = plan()
    rows = json.loads((ROOT/'STAGE_CLOSURE.json').read_text())['external_files']
    for relative in (TRANSPORT_RELATIVE,'gpu77_connection_recovery_v1/run_gpu77_v3.py'):
        pin(next(row for row in rows if row['path']==relative))
    source = PHASE/TRANSPORT_RELATIVE
    spec = importlib.util.spec_from_file_location('reviewed_pilot_transport',source)
    transport = importlib.util.module_from_spec(spec); spec.loader.exec_module(transport)
    transport.HERE = ROOT
    # The relay's system Python only authenticates/starts the already pinned3.12.
    bootstrap = 'from pathlib import Path\nimport hashlib,subprocess\n'
    bootstrap += 'python=Path('+repr(current['interpreter'])+');assert hashlib.sha256(python.read_bytes()).hexdigest()=='+repr(current['interpreter_sha256'])+'\n'
    bootstrap += 'r=subprocess.run('+repr([current['interpreter'],'-B','-c',code])+',capture_output=True,text=True,timeout=90)\n'
    bootstrap += 'assert r.returncode==0,dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)\nprint(r.stdout.strip())\n'
    return transport.run('ddi_pilot_'+identity+'_20261004',bootstrap)


def context():
    return ('from pathlib import Path\nfrom datetime import datetime,timezone\n'
        'import base64,hashlib,json,os,subprocess,zlib\n'
        'repo=Path('+repr(str(REPO))+');phase=Path('+repr(str(REMOTE_PHASE))+');root=Path('+repr(str(REMOTE))+')\n'
        'assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
        'def sha(p):\n return hashlib.sha256(Path(p).read_bytes()).hexdigest()\n')


def stage(args):
    inventory = source_inventory()
    rows=[]
    for row in inventory:
        raw=pin(row).read_bytes()
        assert len(raw)<2_000_000
        rows.append(dict(row,data=base64.b64encode(raw).decode()))
    save(ROOT/('STAGE_INVENTORY_'+args.attempt+'.json'),inventory)
    packed=zlib.compress(json.dumps(rows).encode(),9)
    encoded=base64.b64encode(packed).decode()
    chunks=[encoded[start:start+40000] for start in range(0,len(encoded),40000)]
    for number,chunk in enumerate(chunks,1):
        code=context()+'folder=root/"source_chunks"/'+repr(args.attempt)+';folder.mkdir(parents=True,exist_ok=True)\n'
        code+='raw='+repr(chunk)+'.encode();path=folder/'+repr('part%03d.b64'%number)+'\nwith path.open("xb") as s:s.write(raw)\n'
        code+='print(json.dumps(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))\n'
        value=run('stage_'+args.attempt+'_part%03d'%number,code)
        assert value['sha256']==__import__('hashlib').sha256(chunk.encode()).hexdigest()
    code=context()+'folder=root/"source_chunks"/'+repr(args.attempt)+'\n'
    code+='packed=base64.b64decode(b"".join((folder/("part%03d.b64"%n)).read_bytes() for n in range(1,'+str(len(chunks)+1)+')),validate=True)\n'
    code+='assert hashlib.sha256(packed).hexdigest()=='+repr(__import__('hashlib').sha256(packed).hexdigest())+'\nrows=json.loads(zlib.decompress(packed))\n'
    code+='for row in rows:\n p=phase/row["path"];assert p.resolve().is_relative_to(phase)\n for ancestor in (p,*p.parents):\n  if ancestor==phase.parent:break\n  assert not ancestor.is_symlink()\n raw=base64.b64decode(row["data"],validate=True);assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"]\n if p.exists():assert p.is_file() and p.read_bytes()==raw\n else:\n  p.parent.mkdir(parents=True,exist_ok=True)\n  with p.open("xb") as s:s.write(raw)\n'
    code+='print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status="COMPLETE_PILOT_SOURCE_COST_REVIEW_CLOSURE_STAGED",files=len(rows),numerical_execution=False,predictive_values_read=False)))\n'
    receipt=run('stage_'+args.attempt+'_join',code)
    save(ROOT/('STAGE_RECEIPT_'+args.attempt+'.json'),dict(receipt,inventory_sha256=sha(ROOT/('STAGE_INVENTORY_'+args.attempt+'.json'))))
    print(json.dumps(receipt,indent=2))


def admit(args):
    current=plan(); inventory=source_inventory()
    stage_path=ROOT/('STAGE_RECEIPT_'+args.attempt+'.json')
    assert stage_path.is_file()
    assert json.loads((ROOT/('STAGE_INVENTORY_'+args.attempt+'.json')).read_text())==inventory
    authorization=args.authorization_file.resolve()
    assert authorization.is_relative_to(PHASE) and authorization.is_file()
    authorized=json.loads(authorization.read_text())
    assert authorized['approved'] is True and authorized['authorization_reference']
    capacity_path=ROOT/'CAPACITY_OBSERVATION_01.json'
    capacity=json.loads(capacity_path.read_text()); observation=capacity.get('observation',capacity)
    assert observation['GPU_UUID']==current['GPU_UUID']
    assert observation['memory_free_MiB']>=current['minimum_free_MiB_before_each_cell']
    review=json.loads((PHASE/current['review_packet']/'REVIEW.json').read_text())
    assert review['status']=='PASS' and review['blocking_findings']==[]
    assert review['candidate_manifest_sha256']==current['candidate_manifest_sha256']
    scientific=json.loads((PHASE/current['candidate_packet']/'config.json').read_text())
    scientific['release_enabled']=True
    scientific['root_admission']=dict(approved=True,supervisor=str(REMOTE/'supervise_queue.py'),
        cuda_allocator_fraction=0.30,host_cap_GiB=32,per_cell_wall_cap_seconds=8*3600,
        capacity_observation=str(REMOTE/'CAPACITY_OBSERVATION_01.json'))
    assert scientific['empirical_qualification']['full500_budget'] is False
    save(ROOT/'SCIENTIFIC_RELEASE.json',scientific)
    release=dict(schema='ddi-fixed12-pilot-root-release-v1',UTC=utc(),execution_authorized=True,
        plan_sha256=sha(ROOT/'PLAN.json'),root_source_manifest_sha256=sha(ROOT/'ROOT_SOURCE_MANIFEST.json'),
        root_source_seal_sha256=sha(ROOT/'ROOT_SOURCE_SEAL.json'),source_inventory=inventory,
        scientific_release_sha256=sha(ROOT/'SCIENTIFIC_RELEASE.json'),
        authorization_sha256=sha(authorization),authorization_reference=authorized['authorization_reference'],
        capacity_observation_sha256=sha(capacity_path),capacity_observation_name=capacity_path.name,
        stage_receipt_sha256=sha(stage_path),predictive_values_read=False,automatic_retry=False,full500_qualification=False)
    save(ROOT/'ROOT_RELEASE.json',release)
    save(ROOT/'ROOT_ADMISSION.json',dict(UTC=utc(),status='ROOT_ADMITTED_FIXED12_PILOT',
        root_release_sha256=sha(ROOT/'ROOT_RELEASE.json'),capacity_observation_sha256=sha(capacity_path),
        authorization_sha256=sha(authorization),GPU_UUID=current['GPU_UUID'],queue=current['queue'],
        cap_scope='RSS is maximum sampled sum over owned session, not true peak; wall/CUDA/headroom caps fixed.',
        predictive_values_read=False,full500_qualification=False,automatic_retry=False))
    print(json.dumps(release,indent=2))


def launch(args):
    current=plan(); inventory=source_inventory()
    release=json.loads((ROOT/'ROOT_RELEASE.json').read_text())
    assert release['execution_authorized'] is True and release['source_inventory']==inventory
    assert release['plan_sha256']==sha(ROOT/'PLAN.json')
    assert release['scientific_release_sha256']==sha(ROOT/'SCIENTIFIC_RELEASE.json')
    assert release['capacity_observation_sha256']==sha(ROOT/'CAPACITY_OBSERVATION_01.json')
    assert (ROOT/('STAGE_RECEIPT_'+args.attempt+'.json')).is_file()
    rows=[]
    for name in ('ROOT_RELEASE.json','SCIENTIFIC_RELEASE.json','ROOT_ADMISSION.json','CAPACITY_OBSERVATION_01.json'):
        path=ROOT/name;rows.append(dict(name=name,sha256=sha(path),data=base64.b64encode(path.read_bytes()).decode()))
    environment=dict(CUDA_VISIBLE_DEVICES=current['GPU_UUID'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(REPO/'.gnnm_runtime/buddy_extra_v1/site'))
    command=[current['interpreter'],'-B',str(REMOTE/'supervise_queue.py'),'--release-sha256',sha(ROOT/'ROOT_RELEASE.json')]
    code=context()+'inventory='+repr(inventory)+'\n'
    code+='for row in inventory:\n p=phase/row["path"];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row["bytes"] and sha(p)==row["sha256"]\n'
    code+='assert not any((root/x).exists() for x in ("QUEUE_ATTEMPT_SPENT.json","DETACHED_LAUNCH.json","QUEUE_STARTED.json","fits","supervision"))\n'
    code+='q=subprocess.run(["nvidia-smi","-i",'+repr(current['GPU_UUID'])+',"--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=15);g=q.stdout.strip().split(",");assert g[0].strip()=='+repr(current['GPU_UUID'])+' and int(g[1])>=34*1024\n'
    code+='rows='+repr(rows)+'\nfor row in rows:\n raw=base64.b64decode(row["data"],validate=True);assert hashlib.sha256(raw).hexdigest()==row["sha256"]\n with (root/row["name"]).open("xb") as s:s.write(raw)\n'
    code+='with (root/"QUEUE_ATTEMPT_SPENT.json").open("x") as s:json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),root_release_sha256='+repr(sha(ROOT/'ROOT_RELEASE.json'))+',automatic_retry=False),s)\n'
    code+='command='+repr(command)+';environment='+repr(environment)+'\n'
    code+='with (root/"QUEUE_PRIVATE_STDOUT.txt").open("xb") as out,(root/"QUEUE_PRIVATE_STDERR.txt").open("xb") as err:\n child=subprocess.Popen(command,cwd=repo,env=dict(os.environ,**environment),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)\n'
    code+='raw=(Path("/proc")/str(child.pid)/"stat").read_text();f=raw[raw.rfind(")")+2:].split();assert int(f[2])==int(f[3])==child.pid\n'
    code+='value=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,start_ticks=int(f[19]),group=int(f[2]),session=int(f[3]),command=command,root_release_sha256='+repr(sha(ROOT/'ROOT_RELEASE.json'))+',predictive_values_read=False,automatic_retry=False)\n'
    code+='with (root/"DETACHED_LAUNCH.json").open("x") as s:json.dump(value,s,indent=2);s.write("\\n")\nprint(json.dumps(value))\n'
    value=run('launch_'+args.attempt,code);save(ROOT/'DETACHED_LAUNCH.json',value)
    print(json.dumps(value,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('inventory','stage','admit','launch'))
    parser.add_argument('--attempt',default='run01')
    parser.add_argument('--authorization-file',type=Path)
    args=parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+',args.attempt)
    if args.operation=='inventory': print(json.dumps(source_inventory(),indent=2))
    elif args.operation=='admit':
        assert args.authorization_file is not None
        admit(args)
    else: {'stage':stage,'launch':launch}[args.operation](args)
