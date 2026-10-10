"""Extract stored scalar evidence only after the entire declared study closes."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import zlib

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = ROOT/'experiments_iclr/postsubmission_20260930'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def read(q):
    q=q.resolve(strict=True)
    assert q.is_relative_to(PHASE) and q.stat().st_size<100*1024**2
    return json.loads(q.read_text())

def descriptor(q):
    b=q.read_bytes()
    return dict(path=str(q.relative_to(PHASE)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def typed_closed(stem,expected):
    a=PHASE/stem
    if not (a/'TERMINAL.json').is_file():return None
    terminal=read(a/'TERMINAL.json')
    assert terminal['complete'] and terminal['exit_code']==0 and terminal['stop_reason'] is None
    assert terminal['child_reaped'] and terminal['original_pid_absent'] and terminal['owned_CUDA_absence_verified']
    launch=read(a/'LAUNCH.json')
    assert not Path('/proc',str(launch['child']['pid'])).exists()
    assert not Path('/proc',str(launch['parent']['pid'])).exists()
    release=read(a/'RELEASE.json')
    output=Path(release['output_directory'])
    report=read(output/'COHORT_REPORT.json')
    assert report['status']=='complete' and report['complete'] and len(report['runs'])==expected
    assert all(r['complete'] and r['status']=='complete' for r in report['runs'])
    if stem == 'typed_context_candidate_science_activation_source_20261010_v1':
        assert report['mode'] == 'scientific_development_comparison'
        assert all(report[k] is False for k in ('TEST_truth','TEST_membership_known','TEST_file_access'))
    else:
        assert stem == 'typed_context_reference_science_activation_source_20261010_v1'
        assert report['mode'] == 'scientific_reference_acquisition'
        assert report['TEST_access'] is False and report['automatic_retry'] is False
        assert report['qualification_passed'] is False and len(report['ensembles']) == 6
        assert all(e['complete'] and e['four_full_fits_charged'] for e in report['ensembles'])
    rows=[]
    for r in report['runs']:
        rows.append({k:v for k,v in r.items() if k!='history'})
    compact={k:v for k,v in report.items() if k not in ('runs','runtime')}
    compact['runs']=rows
    return dict(report=compact,authority=[descriptor(a/n) for n in ('RELEASE.json','LAUNCH.json','TERMINAL.json')]+[descriptor(output/'COHORT_REPORT.json')])

def cmcl_closed():
    a=PHASE/'private_cmcl_polyformer_root_preparation_20261010_v2'
    family=a/'science'
    if not (family/'FAMILY_COMPLETE.json').is_file():return None
    assert not (family/'FAMILY_FAILURE.json').exists()
    start=read(family/'FAMILY_START.json')
    assert not Path('/proc',str(start['owner']['PID'])).exists()
    plan=read(a/'SCIENCE_OWNER_PLAN.json')
    assert descriptor(a/'SCIENCE_OWNER_PLAN.json')['sha256']==start['plan_sha256']
    done=read(family/'FAMILY_COMPLETE.json')
    expected=[f'seed{s}__{c}' for s in (9101,9203,9307) for c in ('own_floor','private_cmcl','all_block_cmcl','vanilla_cmcl','private_uniform','private_constant_credit')]
    assert done['complete'] and done['all_records_directly_waited'] and done['records']==expected
    assert [r['record_id'] for r in plan['records']]==expected
    rows=[]
    authority=[descriptor(family/n) for n in ('FAMILY_START.json','FAMILY_COMPLETE.json')]+[descriptor(a/'SCIENCE_OWNER_PLAN.json')]
    for r in plan['records']:
        terminal_path=a/'owners'/r['owner_id']/'TERMINAL_CUSTODY.json'
        terminal=read(terminal_path)
        assert terminal['complete'] and terminal['directly_waited'] and terminal['child_exit_code']==0
        assert terminal['cap_or_owner_failure'] is None and terminal['owned_process_absence_verified'] and terminal['owned_CUDA_absence_verified']
        assert terminal['release_sha256']==r['release_sha256']
        release=read(PHASE/r['release'])
        complete_path=Path(release['output'])/'COMPLETE.json'
        assert descriptor(complete_path)['sha256']==terminal['complete_sha256']
        q=read(complete_path)
        assert q['complete'] and q['science_enabled'] and q['record_id']==r['record_id'] and q['TEST_access'] is False
        assert q['release_sha256']==r['release_sha256'] and q['fresh_reconstruction_verified']
        rows.append(q)
        authority.extend([descriptor(terminal_path),descriptor(complete_path)])
    reference=PHASE/'pubmed_factor1_complete6_scalar_comparison_execution_20261010_v1'/'COMPARISON.json'
    return dict(runs=rows,contextual_reference=read(reference),authority=authority+[descriptor(reference)])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',choices=('cmcl18','typed42'),required=True)
    parser.add_argument('--output-id',required=True)
    args=parser.parse_args()
    assert args.output_id.isidentifier()
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[GPU]
    started=time.monotonic()
    if args.study=='cmcl18':data=cmcl_closed()
    else:
        stems=('typed_context_candidate_science_activation_source_20261010_v1','typed_context_reference_science_activation_source_20261010_v1')
        ready=all((PHASE/stem/'TERMINAL.json').is_file() for stem in stems)
        if ready:
            assert all(read(PHASE/stem/'TERMINAL.json')['complete'] for stem in stems)
            candidate=typed_closed(stems[0],18)
            reference=typed_closed(stems[1],24)
            data=dict(candidate=candidate,reference=reference)
        else:data=None
        if data is not None:
            assert len(reference['report']['ensembles'])==6
            assert candidate['report']['matched_context_identity_by_role']==reference['report']['matched_context_identity_by_role']
    if data is None:
        print(json.dumps(dict(pending=True,study=args.study,quality_opened=False)))
        return
    output=PHASE/args.output_id
    output.mkdir(exist_ok=False)
    payload=dict(schema='whole-family-stored-scalar-evidence-v1',UTC=datetime.now(timezone.utc).isoformat(),study=args.study,
        data=data,reader_source=descriptor(Path(__file__)),scalar_reader_seconds=time.monotonic()-started,
        TEST_access=False,new_model_or_inference=False,original_scores_changed=False,
        arrays_or_checkpoints_read=False,whole_family_closed_before_quality=True)
    raw=(json.dumps(payload,indent=2)+'\n').encode()
    assert len(raw)<4*1024**2
    (output/'CLOSED_SCALAR_EVIDENCE.json').write_bytes(raw)
    print(json.dumps(dict(pending=False,study=args.study,output=str(output.relative_to(PHASE)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64_zlib=base64.b64encode(zlib.compress(raw,9)).decode())))

if __name__=='__main__':main()
