"""Wait once for both existing studies, then run the frozen raw-only join."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
def now():return datetime.now(timezone.utc).isoformat()
def item(p):
    p=Path(p).resolve();assert p.is_relative_to(R);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):assert not p.exists();p.write_text(json.dumps(v,indent=2)+'\n')

def main():
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID]
    H=Path(__file__).resolve().parent;assert H.is_relative_to(P)
    E=P/'native_SAGE_endpoint_value_execution_root_20261011_v1'
    S=P/'native_SAGE_PE_E_joined_raw_reader_source_20261011_v1'
    bound=[item(S/'analysis.py'),item(S/'PROTOCOL.json'),item(H/'PROSPECTIVE_COMPARISON.json')]
    write(H/'JOINED_READER_WATCH_START_V1.json',dict(UTC=now(),PID=os.getpid(),source_bindings=bound,wait_cap_seconds=87000,reader_cap_seconds=300,new_calibration=0,native_forwards=0,partial_predictive_metrics_read=False))
    ended=[H/'fit_owner_v1/OWNER_END.json',E/'READER_WATCH_END_V1.json']
    began=time.monotonic()
    while not all(p.exists() for p in ended):
        fail=H/'FIT_WATCH_END_V1.json'
        if fail.exists():
            closed=json.loads(fail.read_text())
            if closed.get('status')=='E_FAILURE_RETAINED_NO_PE_FIT' or closed.get('timed_out') or closed.get('exit_code',0)!=0:
                write(H/'JOINED_READER_WATCH_END_V1.json',dict(UTC=now(),status='FIT_FAILURE_RETAINED_NO_JOIN',fit_watcher=item(fail)));return 1
        if time.monotonic()-began>87000:raise TimeoutError('Finite complete-study wait expired; no raw join attempted')
        time.sleep(30)
    assert [item(Path(v['path'])) for v in bound]==bound
    pe_owner=json.loads(ended[0].read_text());e_owner=json.loads(ended[1].read_text())
    if not pe_owner.get('success') or not e_owner.get('complete_analysis'):
        write(H/'JOINED_READER_WATCH_END_V1.json',dict(UTC=now(),status='TERMINAL_FAILURE_RETAINED_NO_JOIN',PE_owner=item(ended[0]),E_reader_owner=item(ended[1])));return 1
    F=H/'actual_PE_family_v1';complete=F/'COMPLETE_PE_BANKS.json'
    snapshot=dict(schema_version=1,made_after_complete_owner_closure=True,complete_banks=item(complete),owner_end=item(ended[0]),banks=[])
    for seed in (7301,7403,7507):
        d=F/f'PE4_gamma1_F_seed{seed}'
        snapshot['banks'].append(dict(arm='PE4_gamma1_F',seed=seed,bank_result=item(d/'RESULT.json'),unit_result=item(d/'member0/RESULT.json'),checkpoint=item(d/'member0/selected.pt'),archive=item(d/'selected_VALID.npz')))
    snappath=H/'PE_COMPLETE_SNAPSHOT_V1.json';write(snappath,snapshot)
    spec=dict(schema_version=1,study_kind='native_SAGE_PE_E_joined_raw',study_id='native_SAGE_PE4_BASE_budget_comparison_20261011_v1',
        reader_source=bound[0],protocol=bound[1],E_input_spec=item(E/'COMPLETE_READER_INPUT_SPEC_V1.json'),
        E_complete_summary=item(E/'complete_analysis_v1/COMPLETE_ANALYSIS_SUMMARY.json'),E_reader_owner_end=item(ended[1]),
        PE=dict(family_root=str(F),release=item(H/'FIT_RELEASE_V1.json'),config=item(H/'SCIENTIFIC_CONFIG_V1.json'),
            complete_banks=item(complete),source_hashes=item(F/'SOURCE_HASHES.json'),consumer_complete=item(F/'CONSUMER_COMPLETE.json'),owner_end=item(ended[0]),snapshot=item(snappath)))
    specpath=H/'JOINED_READER_INPUT_SPEC_V1.json';write(specpath,spec)
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    command=[str(R/'.venv/bin/python'),'-B',str(S/'analysis.py'),'--input-spec',str(specpath),'--output',str(H/'complete_joined_raw_analysis_v1')]
    with (H/'joined_reader_stdout.log').open('x') as out,(H/'joined_reader_stderr.log').open('x') as err:
        child=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        write(H/'JOINED_READER_EXECUTION_START_V1.json',dict(UTC=now(),child_PID=child.pid,command=command,whole_studies_closed_before_join=True))
        timed_out=False
        try:code=child.wait(timeout=300)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=15)
    summary=H/'complete_joined_raw_analysis_v1/COMPLETE_JOINED_RAW_SUMMARY.json'
    write(H/'JOINED_READER_WATCH_END_V1.json',dict(UTC=now(),exit_code=code,timed_out=timed_out,complete_join=code==0 and summary.exists(),summary=item(summary) if summary.exists() else None,native_fits_added=0,native_forwards=0,new_calibration=0,TEST_access=False))
    return code

if __name__=='__main__':raise SystemExit(main())
