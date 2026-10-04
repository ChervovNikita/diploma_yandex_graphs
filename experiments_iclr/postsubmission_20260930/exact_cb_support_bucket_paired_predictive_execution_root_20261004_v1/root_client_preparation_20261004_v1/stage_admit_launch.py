"""Three independent root operations for the sealed, externally reviewed nine-fit queue."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import base64
import hashlib
import importlib.util
import json
import os
import zlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PHASE=ROOT.parent
REPO=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE=REPO/'experiments_iclr/postsubmission_20260930'
REMOTE=REMOTE_PHASE/ROOT.name
REMOTE_CLIENT=REMOTE/HERE.name
SOURCE=PHASE/'exact_cb_support_bucket_paired_predictive_preparation_20261004_v1'
SOURCE_SHA='0d0899d94f0a20d317f7e30491f3ed5b35c293d462f28a0baf3f4ea350e9d229'
SOURCE_SEAL_SHA='00cfa256ce00a9dc19ae32f27b8535f5ea9b069ff92b48377a5244efe6f33e7d'
RELAY_REL='ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py'
RELAY_SHA='cfb554db15ebdb7e5f82284bc73f0c112c74e1b9bd7dc238b41ba1dce1933276'
WRAPPER_REL='gpu77_connection_recovery_v1/run_gpu77_v3.py'
WRAPPER_SHA='035b740ceb50eefcfa3cec2aef6dbd1294c769d133e2bc4cf88fb52b088421ff'
PYTHON='/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())


def descriptor(path,remote=False):
    return {'path':str(REMOTE_PHASE/path.relative_to(PHASE)) if remote else str(path.relative_to(PHASE)),
            'bytes':path.stat().st_size,'sha256':sha(path)}


def packet(folder,manifest_sha,seal_sha):
    assert folder.resolve().is_relative_to(PHASE) and not folder.is_symlink()
    assert sha(folder/'MANIFEST.json')==manifest_sha and sha(folder/'SEAL.json')==seal_sha
    rows=json.loads((folder/'MANIFEST.json').read_text())['files'];files=[]
    for row in rows:
        path=folder/row['path']
        assert path.resolve().is_relative_to(folder) and not path.is_symlink()
        assert path.stat().st_size==row.get('bytes',row.get('size')) and sha(path)==row['sha256']
        files.append(path)
    seal=json.loads((folder/'SEAL.json').read_text())
    assert seal.get('manifest_sha256',seal.get('manifest',{}).get('sha256'))==manifest_sha
    return files+[folder/'MANIFEST.json',folder/'SEAL.json']


def reviewed_local():
    assert ROOT.name=='exact_cb_support_bucket_paired_predictive_execution_root_20261004_v1'
    packet(SOURCE,SOURCE_SHA,SOURCE_SEAL_SHA)
    packet(HERE,sha(HERE/'MANIFEST.json'),sha(HERE/'SEAL.json'))
    plan=json.loads((HERE/'CLIENT_PLAN.json').read_text())
    assert plan['source_manifest_sha256']==SOURCE_SHA and plan['source_seal_sha256']==SOURCE_SEAL_SHA
    pins=plan['external_review_pins']
    paths={}
    for name,pin in pins.items():
        path=PHASE/pin['path']
        assert path.resolve().is_relative_to(PHASE) and not path.is_symlink()
        assert path.stat().st_size==pin['bytes'] and sha(path)==pin['sha256'];paths[name]=path
    folder=paths['review_file'].parent
    assert folder==paths['review_manifest'].parent==paths['review_seal'].parent
    packet(folder,pins['review_manifest']['sha256'],pins['review_seal']['sha256'])
    review=json.loads(paths['review_file'].read_text())
    assert review['status']=='PASS' and not review.get('blocking_findings')
    assert review['candidate_manifest_sha256']==SOURCE_SHA and review['candidate_seal_sha256']==SOURCE_SEAL_SHA
    assert review['execution_authorized'] is False and review['reviewer_is_source_preparer'] is False
    return plan,paths


def context():
    return ('from pathlib import Path\nimport base64,hashlib,json,os,subprocess\n'
            'from datetime import datetime,timezone\n'
            'repo=Path('+repr(str(REPO))+');phase=Path('+repr(str(REMOTE_PHASE))+');root=Path('+repr(str(REMOTE))+')\n'
            'assert Path.cwd()==repo and os.uname().nodename=="peptide"\n')


def transport(identity,code):
    assert sha(PHASE/RELAY_REL)==RELAY_SHA and sha(PHASE/WRAPPER_REL)==WRAPPER_SHA
    spec=importlib.util.spec_from_file_location('predictive_native_relay',PHASE/RELAY_REL)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.HERE=ROOT
    return module.run('ncnc_exact_cb_paired_v1_'+identity+'_20261004',code)


def payload(files):
    rows=[]
    for path in dict.fromkeys(files):
        assert path.resolve().is_relative_to(PHASE) and not path.is_symlink()
        raw=path.read_bytes();raw.decode('utf8')
        assert len(raw)<8*1024**2 and (path.suffix in ('.py','.json','.md','.txt','.diff','.patch') or path.name=='.gitignore')
        rows.append({'path':str(path.relative_to(PHASE)),'bytes':len(raw),'sha256':sha(path),
                     'data':base64.b64encode(raw).decode()})
    return rows


def stage():
    plan,review_paths=reviewed_local()
    save(ROOT/'LOCAL_STAGE_ATTEMPT_SPENT.json',{'UTC':utc(),'source_manifest_sha256':SOURCE_SHA,
         'client_manifest_sha256':sha(HERE/'MANIFEST.json'),'no_blind_retry':True})
    files=packet(SOURCE,SOURCE_SHA,SOURCE_SEAL_SHA)+packet(HERE,sha(HERE/'MANIFEST.json'),sha(HERE/'SEAL.json'))
    files+=packet(review_paths['review_file'].parent,plan['external_review_pins']['review_manifest']['sha256'],
                  plan['external_review_pins']['review_seal']['sha256'])
    source_plan=json.loads((SOURCE/'PILOT_PLAN.json').read_text())
    for pin in source_plan['source_pins']:
        path=PHASE/pin['path']
        assert path.stat().st_size==pin['bytes'] and sha(path)==pin['sha256'];files.append(path)
    rows=payload(files)
    inventory=[{k:v for k,v in row.items() if k!='data'} for row in rows]
    save(ROOT/'SOURCE_STAGE_INVENTORY.json',inventory)
    compressed=zlib.compress(json.dumps(rows).encode(),9);encoded=base64.b64encode(compressed).decode()
    chunks=[encoded[i:i+40000] for i in range(0,len(encoded),40000)]
    for number,chunk in enumerate(chunks,1):
        code=context()+'root.mkdir(parents=True,exist_ok=True)\n'
        if number==1:
            code+='with (root/"STAGE_ATTEMPT_SPENT.json").open("x") as s:json.dump('+repr({'source_manifest_sha256':SOURCE_SHA,'client_manifest_sha256':sha(HERE/'MANIFEST.json'),'no_blind_retry':True})+',s)\n'
        code+='staging=root/"source_chunks";staging.mkdir(exist_ok=True);raw='+repr(chunk)+'.encode()\n'
        code+='with (staging/'+repr('part%03d.b64'%number)+').open("xb") as s:s.write(raw)\n'
        code+='print(json.dumps(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))\n'
        result=transport('source_chunk%03d'%number,code)
        assert result['sha256']==hashlib.sha256(chunk.encode()).hexdigest()
    owned_folders=[str(REMOTE_PHASE/SOURCE.relative_to(PHASE)),str(REMOTE_CLIENT),
                   str(REMOTE_PHASE/review_paths['review_file'].parent.relative_to(PHASE))]
    code=context()+'import zlib\nstaging=root/"source_chunks"\n'
    code+='compressed=base64.b64decode(b"".join((staging/("part%03d.b64"%n)).read_bytes() for n in range(1,'+str(len(chunks)+1)+')),validate=True)\n'
    code+='assert hashlib.sha256(compressed).hexdigest()=='+repr(hashlib.sha256(compressed).hexdigest())+'\nrows=json.loads(zlib.decompress(compressed))\n'
    code+='for row in rows:\n p=phase/row["path"];assert p.resolve().is_relative_to(phase);raw=base64.b64decode(row["data"],validate=True);raw.decode("utf8");assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"]\n if p.exists():assert p.is_file() and not p.is_symlink() and p.read_bytes()==raw\n else:\n  p.parent.mkdir(parents=True,exist_ok=True)\n  with p.open("xb") as s:s.write(raw)\n'
    code+='for folder in '+repr(owned_folders)+':\n p=Path(folder)\n for f in p.iterdir():\n  assert f.is_file() and not f.is_symlink();f.chmod(0o444)\n p.chmod(0o555)\n'
    code+='value=dict(UTC=datetime.now(timezone.utc).isoformat(),status="EXACT_PREDICTIVE_SOURCE_REVIEW_AND_CLIENT_STAGED",files=len(rows),bytes=sum(r["bytes"] for r in rows),source_manifest_sha256='+repr(SOURCE_SHA)+',client_manifest_sha256='+repr(sha(HERE/'MANIFEST.json'))+',numerical_execution=False,TEST_access=False)\n'
    code+='with (root/"SOURCE_STAGE_RECEIPT.json").open("x") as s:json.dump(value,s,indent=2);s.write("\\n")\nprint(json.dumps(value))\n'
    save(ROOT/'SOURCE_STAGE_RECEIPT.json',transport('source_join',code))


def controller(mode,queue_sha):
    command=[PYTHON,'-B',str(REMOTE_CLIENT/'remote_control.py'),mode,'--queue-release-sha256',queue_sha]
    code=context()+'control=Path('+repr(str(REMOTE_CLIENT/'remote_control.py'))+')\n'
    code+='assert hashlib.sha256(control.read_bytes()).hexdigest()=='+repr(sha(HERE/'remote_control.py'))+'\n'
    code+='r=subprocess.run('+repr(command)+',cwd=repo,capture_output=True,text=True,timeout=60)\n'
    code+='assert r.returncode==0,dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)\nprint(r.stdout.strip())\n'
    return code


def admit(authorization_reference):
    plan,review_paths=reviewed_local()
    assert authorization_reference and (ROOT/'SOURCE_STAGE_RECEIPT.json').is_file()
    save(ROOT/'LOCAL_ADMIT_ATTEMPT_SPENT.json',{'UTC':utc(),'source_manifest_sha256':SOURCE_SHA,
         'client_manifest_sha256':sha(HERE/'MANIFEST.json'),'authorization_reference':authorization_reference,'no_blind_retry':True})
    save(ROOT/'EXTERNAL_REVIEW_PINS.json',plan['external_review_pins'])
    admission={'UTC':utc(),'status':'APPROVED_FIXED_NINE_FITS','authorization_reference':authorization_reference,
               'source_manifest_sha256':SOURCE_SHA,'source_seal_sha256':SOURCE_SEAL_SHA,
               'client_manifest_sha256':sha(HERE/'MANIFEST.json'),'client_seal_sha256':sha(HERE/'SEAL.json'),
               'independent_review_sha256':plan['external_review_pins']['review_file']['sha256'],
               'queue_order_sha256':plan['queue_order_sha256'],'GPU_UUID':plan['GPU_UUID'],'fit_caps':plan['fit_caps'],
               'fits':9,'optimizer_updates':15300,'complete_VALID_traversals':918,
               'queue_order_frozen_before_predictive_outcome_reads':True,'automatic_retry_or_resume':False,'TEST_access':False}
    save(ROOT/'ROOT_ADMISSION.json',admission)
    release_pins=[]; files=[ROOT/'EXTERNAL_REVIEW_PINS.json',ROOT/'ROOT_ADMISSION.json']
    for cell in plan['queue_order']:
        release=json.loads((SOURCE/'ROOT_RELEASE_TEMPLATE.json').read_text())
        release.update(execution_enabled=True,root_authorization_reference=str(REMOTE/'ROOT_ADMISSION.json'),
                       driver_manifest_sha256=SOURCE_SHA,fit_supervision_manifest_sha256=SOURCE_SHA,
                       independent_source_review=descriptor(review_paths['review_file'],remote=True),
                       authorized_invocations=[cell['invocation']])
        release.pop('template_only_NOT_AUTHORIZATION',None)
        for key in ('prospective_cell_paths_NOT_AUTHORIZATION','example_invocations_NOT_AUTHORIZED'):
            release.pop(key,None)
        path=ROOT/'releases'/('ROOT_'+cell['cell']+'_RELEASE.json');save(path,release)
        release_pins.append(descriptor(path,remote=True));files.append(path)
    q={'schema':'ncnc-exact-CB-fixed-nine-fit-queue-release-v1','execution_enabled':True,
       'root_authorization_reference':str(REMOTE/'ROOT_ADMISSION.json'),
       'root_admission':descriptor(ROOT/'ROOT_ADMISSION.json',remote=True),
       'external_review_pins':descriptor(ROOT/'EXTERNAL_REVIEW_PINS.json',remote=True),
       'family_id':plan['family_id'],'source_manifest_sha256':SOURCE_SHA,'source_seal_sha256':SOURCE_SEAL_SHA,
       'source_plan_sha256':sha(SOURCE/'PILOT_PLAN.json'),'client_manifest_sha256':sha(HERE/'MANIFEST.json'),
       'client_seal_sha256':sha(HERE/'SEAL.json'),'queue_order':plan['queue_order'],
       'queue_order_sha256':plan['queue_order_sha256'],'cell_releases':release_pins,
       'GPU_UUID':plan['GPU_UUID'],'fit_caps':plan['fit_caps'],'automatic_retry_or_resume':False,'TEST_access':False}
    save(ROOT/'QUEUE_RELEASE.json',q);files.append(ROOT/'QUEUE_RELEASE.json')
    rows=payload(files)
    code=context()+'with (root/"ADMIT_ATTEMPT_SPENT.json").open("x") as s:json.dump('+repr({'queue_release_sha256':sha(ROOT/'QUEUE_RELEASE.json'),'no_blind_retry':True})+',s)\n'
    code+='rows='+repr(rows)+'\nfor row in rows:\n p=phase/row["path"];assert p.resolve().is_relative_to(root);raw=base64.b64decode(row["data"],validate=True);assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"];p.parent.mkdir(parents=True,exist_ok=True)\n with p.open("xb") as s:s.write(raw)\n'
    code+=controller('admit',sha(ROOT/'QUEUE_RELEASE.json'))
    save(ROOT/'PRENUMERICAL_ADMISSION.json',transport('admit',code))


def launch():
    reviewed_local()
    admission=json.loads((ROOT/'PRENUMERICAL_ADMISSION.json').read_text())
    queue_sha=sha(ROOT/'QUEUE_RELEASE.json')
    assert admission['status']=='PASS_FIXED_NINE_CELL_STDLIB_ADMISSION' and admission['queue_release_sha256']==queue_sha
    save(ROOT/'LOCAL_LAUNCH_ATTEMPT_SPENT.json',{'UTC':utc(),'queue_release_sha256':queue_sha,
         'client_manifest_sha256':sha(HERE/'MANIFEST.json'),'no_blind_retry':True})
    save(ROOT/'DETACHED_LAUNCH.json',transport('launch',controller('launch',queue_sha)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('stage','admit','launch'))
    parser.add_argument('--authorization-reference')
    args=parser.parse_args()
    if args.mode=='stage':stage()
    elif args.mode=='admit':
        parser.error('--authorization-reference is required for admission') if not args.authorization_reference else admit(args.authorization_reference)
    else:launch()
