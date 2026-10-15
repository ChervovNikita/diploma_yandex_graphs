"""Allocation-only source custody and one-attempt admission; stdlib only."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys

ROOT=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
RUNTIME_SHA='99711f3bbbb21175ca4574daa9dc7b284b749756007169e2292218dffb6c40fc'


def require(ok,message):
    if not ok:raise ValueError(message)


def guard():
    require(socket.gethostname()=='anogena-2-0','Wrong allocation; no project access')
    require(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[UUID],
            'Wrong sole allocation GPU UUID')


def inside(path):
    p=Path(path).resolve();require(p.is_relative_to(ROOT),'Path escapes approved root');return p


def receipt(path):
    p=Path(path);data=p.read_bytes();return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('w')as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    temporary.replace(path)


def package(folder,expected):
    folder=inside(folder);manifest=folder/'MANIFEST.json'
    require(receipt(manifest)['sha256']==expected,'Frozen manifest differs: '+str(folder))
    rows=[receipt(manifest)]
    for entry in json.loads(manifest.read_text())['files']:
        p=inside(folder/entry['path']);require(p.is_relative_to(folder),'Manifest member escaped package')
        require(p.suffix in('.py','.json','.md','.txt'),'Source package cannot open tensors/data')
        value=receipt(p);require(all(value[k]==entry[k]for k in('bytes','sha256')),'Frozen source changed: '+str(p));rows.append(value)
    return rows


def sources(here,driver_sha):
    here=inside(here);expansion=here.parents[1]
    rows=package(here,driver_sha)+package(expansion/'portable_runtime_authority_v4',RUNTIME_SHA)
    closure=json.loads((here/'SOURCE_CLOSURE.json').read_text())
    for pin in closure['source_metadata_receipts']:
        p=inside(pin['path']);require(p.suffix in('.py','.json','.md','.txt','.html','.csv'),'Only source metadata may be verified')
        require(receipt(p)==pin,'Explicit source/approval receipt differs: '+str(p));rows.append(pin)
    review=json.loads((expansion/'benchmark_reporting_protocol_v1/ROOT_REVIEW.json').read_text())
    require(review['status']=='accepted'and review['final_root_artifact_review_complete']is True,'Approved exploratory reporting review required')
    return rows


def validate_interpreter(here):
    value=json.loads((here/'SOURCE_CLOSURE.json').read_text())['interpreter_environment']
    require(receipt(inside(value['path']))==value,'Interpreter environment receipt differs')
    spec=json.loads(Path(value['path']).read_text())
    require(sys.executable==spec['lexical_executable']and sys.prefix==spec['prefix']and sys.base_prefix==spec['base_prefix']and sys.prefix!=sys.base_prefix,
            'Actual lexical venv/base prefix differs before tensor imports')
    binary=Path(sys.executable).resolve()
    require(str(binary)==spec['resolved_executable']and receipt(binary)['sha256']==spec['binary_sha256'],'Actual Python binary differs')
    require(receipt(inside(spec['pyvenv_config']['path']))==spec['pyvenv_config'],'Actual venv configuration differs')
    return {'lexical_executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'resolved_executable':str(binary),'Torch_imported':'torch'in sys.modules}


def load_admission(path,here):
    path=inside(path);value=json.loads(path.read_text());here=inside(here)
    require(value.get('schema')=='portablev3-link-queued-single-case-root-admission-v1','Exact root engineering admission required')
    require(value.get('template_is_not_admission')is False and value.get('published')is True and len(value.get('publication_commit',''))==40
        and all(c in'0123456789abcdef'for c in value['publication_commit']),'Publication and actual reviewed admission required')
    cases=json.loads((here/'CASES.json').read_text())['cases'];case=cases.get(value.get('case_id'))
    require(case is not None,'One named NCN or BUDDY case required')
    require(value.get('device')=='cuda:0'and value.get('maximum_attempts')==1 and value.get('automatic_retries')==0,
            'One explicit CUDA attempt per family; no fallback or retries')
    require(value.get('limits')==case['limits']and value.get('wall_watchdog_seconds')==case['wall_watchdog_seconds'],
            'Exact frozen work/watchdog bounds required')
    require(value.get('runtime_manifest_sha256')==RUNTIME_SHA and value.get('TEST_access')is False and value.get('full_campaign_GO')is False,
            'Frozen runtime authority v4 and engineering-only closed-TEST scope required')
    sources(here,value['driver_manifest_sha256'])
    job=here/('JOB_'+case['case_id']+'_CUDA.json')
    require(receipt(job)['sha256']==value['job_sha256'],'Frozen own engineering JOB differs')
    require(value.get('reporting_ROOT_REVIEW')==json.loads((here/'SOURCE_CLOSURE.json').read_text())['reporting_ROOT_REVIEW'],
            'Attach accepted exploratory reporting review')
    require(value.get('GPU_resource_evidence_applicable')is True and isinstance(value.get('GPU_resource_evidence'),dict)
        and type(value.get('own_GPU_reservation_bytes'))is int and value['own_GPU_reservation_bytes']>0,
        'Root must separately admit applicable GPU evidence/envelope; source templates are not profiles')
    prior=json.loads((here/'PRELAUNCH_RETARGET.json').read_text())['original_queue_outputs'][case['case_id']]
    require(not inside(prior['output']).exists() and not inside(prior['attempt_claim']).exists(),
            'Original queued attempt already has evidence/claim; root must review rather than duplicate it')
    # Validate both actual engineering schemas and pinned parent JSON before child launch/Torch.
    folder=here.parents[1]/'portable_runtime_authority_v4'
    sys.path.insert(0,str(folder))
    import host_binding, descriptor_binding, ncn_repair_authority
    require(all(Path(module.__file__).resolve()==folder/(module.__name__+'.py')
                for module in (host_binding,descriptor_binding,ncn_repair_authority)),
            'Exact reviewed runtime authority modules required')
    host_binding.validate_host_job(json.loads(job.read_text()),ROOT)
    target=json.loads((here/'PRELAUNCH_RETARGET.json').read_text())['attempts'][case['case_id']]
    require(value.get('output')==target['output'] and value.get('attempt_claim')==target['attempt_claim'],
            'Exact original BUDDY or separately named NCN repair identity required')
    if case['case_id']=='NCN':
        contract,verified=ncn_repair_authority.verify_prior_ncn_failure(ROOT)
        ncn_repair_authority.validate_repair_admission(value,contract)
        value['_verified_prior_failure']=verified
    else:
        require(value.get('repair_attempt_id') is None and value.get('prior_failed_attempt') is None,
                'BUDDY remains its original unexecuted queued case, with no repair identity')
    return value,job,case
