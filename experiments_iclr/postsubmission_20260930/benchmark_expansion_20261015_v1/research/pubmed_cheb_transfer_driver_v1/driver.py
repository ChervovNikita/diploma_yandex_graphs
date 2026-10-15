"""Finite original-runner fits, serial selected collection, CPU saved analysis.

Default preflight launches nothing. Root source/resource/timing admission is
mandatory for every execution mode. No automatic retry/resume or TEST.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback


def preflight(s,h,root,here):
    spec = json.loads((here/'SPEC.json').read_text())
    h.require(Path(spec['project_root']).resolve() == root,'Frozen repository root differs')
    for row in json.loads((here/'MANIFEST.json').read_text())['files']:
        path = (here/row['path']).resolve(); h.require(path.is_relative_to(here),'Source escapes package')
        h.verified(root,dict(row,path=str(path)))
    for row in spec['source_pins']+spec['data_pins']:h.verified(root,row)
    ext = s.load(s.COMPOSITION/'extension.py','_pubmed_driver_composition')
    ext.source_bindings(root,spec['composition_manifest']['sha256'])
    rows,cohorts = s.jobs(spec)
    for completed in spec['completed_CPU_qualifications']:
        value = json.loads(h.verified(root,completed['authority']).read_text())['native']
        expected = next(r for r in rows if r['cohort_seed'] == 101 and r['kind'] == completed['variant'])
        h.require(value['status'] == 'passed' and value['planned_job_hash'] == expected['job_hash'] and
            value['optimizer_calls'] == {'attempts':[2],'completed':[2]} and value['VALID_completed'] == 3 and
            value['fresh_restore_VALID_bit_exact'] is True and value['TEST_accessed'] is False,
            'Already-completed actual CPU authority differs; do not repeat qualification')
    sys.path.insert(0,str(s.E/'runtime'))
    from run_fit import validate_job
    for row in rows:validate_job(row['job'])
    h.require(spec['new_fits'] == 24 and spec['TEST_access'] is False and spec['full_campaign_GO'] is False,'Exact24-fit scope required')
    return spec,rows,cohorts


def admission(s,h,root,here,args):
    h.require(args.admission and args.admission_sha256,'Actual root admission and hash required')
    path = h.safe(root,args.admission); h.require(h.pin(path)['sha256'] == args.admission_sha256,'Admission changed')
    a = json.loads(path.read_text())
    wanted = 'fits' if args.mode == 'fits' else 'collect'
    h.require(a.get('schema') == 'root-reviewed-pubmed-cheb-transfer-v1' and a.get('mode') == wanted and
        a.get('approved') is True and a.get('publication_approved') is True and a.get('resource_admission_approved') is True and
        a.get('manifest_sha256') == h.pin(here/'MANIFEST.json')['sha256'] and
        a.get('SPEC_sha256') == h.pin(here/'SPEC.json')['sha256'] and a.get('maximum_new_workers') == 1 and
        a.get('allocation_GPU_UUID') == h.GPU and a.get('full_campaign_GO') is False and a.get('TEST_access') is False,
        'Exact root publication/resource/scope admission required')
    h.require(Path(sys.prefix).resolve() == (root/'.venv').resolve(),'Original repository virtualenv required')
    if args.mode != 'analyze':
        h.require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip() == a['publication_commit'],
            'Published checkout differs from admitted commit')
        h.require(all(type(a.get(k)) is int and a[k] > 0 for k in ('maximum_wall_seconds_per_child','reserved_GPU_bytes','reserved_host_bytes')),
            'Finite measured root time/memory admission required')
    if args.mode == 'fits':
        h.require(a.get('new_fits') == 24 and a.get('all_numerical_lifecycles_approved') is True and
            a.get('isolated_production_timing_approved') is True and a.get('evidence_review_note'),
            'CPU/CUDA lifecycle and isolated production timing review remain prerequisites')
        for key in ('lifecycle_evidence','production_timing_evidence'):
            h.require(isinstance(a.get(key),list) and a[key],'Actual root-reviewed evidence receipts required')
            for receipt in a[key]:h.verified(root,receipt)
    else:
        h.require(a.get('new_collections') == 12 and a.get('new_training_fits') == 0,'Exactly12 zero-training collections required')
    return a,path


def terminal(s,h,root,study,rows,a):
    end_path,outcomes_path = study/'END.json',study/'OUTCOMES.json'
    for path,key in ((end_path,'terminal_END_sha256'),(outcomes_path,'terminal_OUTCOMES_sha256')):
        h.require(h.pin(path)['sha256'] == a[key],'Actual all24 terminal receipt differs')
    end,outcomes = json.loads(end_path.read_text()),json.loads(outcomes_path.read_text())
    s.terminal_gate(end,outcomes,rows)
    fits = [s.fit(root,study,row,outcome)[0] for row,outcome in zip(rows,outcomes)]
    return fits,h.pin(end_path),h.pin(outcomes_path)


def serial(s,h,root,output,rows,commands,timeout,allowed,check_result,start):
    """Existing process/receipt helpers; one child, no retry or resume."""
    h.require(not output.exists(),'Fresh output required')
    output.mkdir(parents=True)
    outcomes = [{'id':r['id'],'index':r['index'],'status':'not_started'} for r in rows]
    h.write(output/'START.json',start)
    child,failure,row = None,None,None; started = time.monotonic()
    signal.signal(signal.SIGTERM,h.interrupted)
    try:
        for binding,command in zip(rows,commands):
            h.check_peers(allowed); row = outcomes[binding['index']]
            row.update(status='started',command=command); h.write(output/'OUTCOMES.json',outcomes,True)
            before = time.monotonic()
            try:
                with (output/('child_'+str(binding['index']).zfill(3)+'.log')).open('x') as log:
                    child = subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
                    row.update(worker_pid=child.pid,worker_start_ticks=h.ticks(child.pid))
                    h.write(output/'OUTCOMES.json',outcomes,True)
                    try:row['returncode'] = child.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        row['timed_out'] = True; h.terminate(child); row['returncode'] = child.returncode
            finally:
                h.terminate(child); row['wall_seconds'] = time.monotonic()-before
            row.update(check_result(binding,command))
            h.write(output/'OUTCOMES.json',outcomes,True)
            h.require(row['returncode'] == 0 and row['status'] in ('complete','complete_with_sanity_warning') and
                not row.get('timed_out') and not row.get('close_error'),'Child failed; stop later admission, preserve all work')
    except BaseException as error:
        failure = {'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=8)}
        if row is not None and row['status'] == 'started':row.update(status='owner_or_child_failed',unreceipted_work='unknown, not zero')
        raise
    finally:
        h.terminate(child); h.write(output/'OUTCOMES.json',outcomes,True)
        complete = failure is None and all(r['status'] in ('complete','complete_with_sanity_warning') for r in outcomes)
        h.write(output/'END.json',{'status':'complete' if complete else 'stopped_no_retry','outcomes':outcomes,
            'failure':failure,'wall_seconds':time.monotonic()-started,'automatic_retries':0,'TEST_accessed':False,**start})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',required=True); parser.add_argument('--mode',choices=('preflight','fits','collect','analyze'),default='preflight')
    parser.add_argument('--admission'); parser.add_argument('--admission-sha256'); args = parser.parse_args()
    if socket.gethostname() != 'anogena-2-0' or subprocess.check_output(
        ['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines() != ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Dedicated allocation required before metadata reads')
    import study as s
    h = s.helpers(); root,here = Path(args.project_root).resolve(),Path(__file__).resolve().parent
    spec,rows,cohorts = preflight(s,h,root,here)
    if args.mode == 'preflight':
        print(json.dumps({'status':'preflight_passed_no_launch','fits':24,'banks':18,'jobs':[r['job_hash'] for r in rows]}));return 0
    a,admission_path = admission(s,h,root,here,args)
    study_output = h.safe(root,spec['study_output'])
    start = {'SPEC':h.pin(here/'SPEC.json'),'manifest':h.pin(here/'MANIFEST.json'),'admission':h.pin(admission_path),'mode':args.mode}
    if args.mode == 'analyze':
        h.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '','Analysis is CPU saved-prediction only')
        fits,end_pin,outcomes_pin = terminal(s,h,root,study_output,rows,a)
        analyzer = s.load(here/'analyze.py','_pubmed_selected_analysis')
        return analyzer.run(root,spec,rows,cohorts,fits,a,start,end_pin,outcomes_pin)
    h.require(os.environ.get('CUDA_VISIBLE_DEVICES') in (None,'0',h.GPU),'Only authorized GPU0 permitted')
    lock_path = h.safe(root,a['slot_lock_path']); lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); h.check_peers(a['allowed_other_GPU_processes'])
        if args.mode == 'fits':
            h.require(not study_output.exists(),'Fresh scientific output required; no retry/resume')
            # Job files are materialized only after the actual root fit admission.
            job_root = study_output.with_name(study_output.name+'_jobs')
            h.require(not job_root.exists(),'Fresh job descriptor directory required'); job_root.mkdir(parents=True)
            commands = []
            for row in rows:
                path = job_root/(row['token']+'.json');h.write(path,row['job'])
                commands.append([sys.executable,'-B','-u',str(s.E/'runtime/run_fit.py'),'--project-root',str(root),
                    '--job',str(path),'--output',str(study_output/'fits'/row['token'])])
            def fitted(binding,command):
                p = Path(command[-1])/'RESULT.json'
                if not p.is_file():return {'status':'missing_RESULT','unreceipted_work':'unknown, not zero'}
                value = json.loads(p.read_text())
                h.require(value['job_hash'] == binding['job_hash'] and value['job_id'] == binding['id'] and value['TEST_accessed'] is False,'Child fit identity differs')
                return {'RESULT':h.pin(p),'status':value['status'],**{k:value.get(k) for k in
                    ('selected_epoch','completed_epochs','stop_reason','convergence_established','optimizer_calls','optimizer_calls_this_invocation','resources','invocation_wall_seconds','failure','close_error')}}
            serial(s,h,root,study_output,rows,commands,a['maximum_wall_seconds_per_child'],a['allowed_other_GPU_processes'],fitted,start)
        else:
            fits,end_pin,outcomes_pin = terminal(s,h,root,study_output,rows,a)
            collect = h.safe(root,a['collection_output']); commands = []; selected = []
            h.require(not collect.is_relative_to(study_output) and not study_output.is_relative_to(collect),
                'Collection output must be separate from scientific fit tree')
            for row in rows:
                if row['kind'] == 'native':continue
                binding = dict(row,index=len(selected),fit_index=row['index']);selected.append(binding)
                commands.append([sys.executable,'-B','-u',str(here/'collect.py'),'--project-root',str(root),
                    '--index',str(row['index']),'--admission',str(admission_path),'--admission-sha256',args.admission_sha256,
                    '--output',str(collect/row['token'])])
            def collected(binding,command):
                p = Path(command[-1])/'RESULT.json'
                if not p.is_file():return {'status':'missing_RESULT','unreceipted_work':'unknown, not zero'}
                v = json.loads(p.read_text());h.require(v['job_id'] == binding['id'] and v['TEST_accessed'] is False,'Collection identity differs')
                return {'RESULT':h.pin(p),'status':v['status'],**{k:v.get(k) for k in ('actual_work','wall_seconds','resources','failure','close_error')}}
            serial(s,h,root,collect,selected,commands,a['maximum_wall_seconds_per_child'],a['allowed_other_GPU_processes'],collected,
                dict(start,terminal_END=end_pin,terminal_OUTCOMES=outcomes_pin))
    return 0


if __name__ == '__main__':raise SystemExit(main())
