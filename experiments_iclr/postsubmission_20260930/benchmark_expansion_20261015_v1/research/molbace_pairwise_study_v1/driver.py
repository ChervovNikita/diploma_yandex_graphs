"""Prospective21-fit native runner +9 old collectors +CPU saved analysis.

Preflight is the default; every execution mode requires separate root review.
The unchanged serial owner and molecular collector are reused, not forked.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def preflight(s,h,root,here):
    spec=json.loads((here/'SPEC.json').read_text())
    h.require(Path(spec['project_root']).resolve()==root,'Frozen allocation repository differs')
    for row in json.loads((here/'MANIFEST.json').read_text())['files']:
        path=(here/row['path']).resolve();h.require(path.is_relative_to(here),'Package source escapes')
        h.verified(root,dict(row,path=str(path)))
    for row in spec['source_pins']:h.verified(root,row)
    producer=s.load(s.PRODUCER/'extension.py','_pair_scientific_producer')
    producer.verify_sources(root)
    for row in json.loads((s.PRODUCER/'MANIFEST.json').read_text())['files']:h.verified(root,dict(row,path=str(s.PRODUCER/row['path'])))
    rows=s.jobs(spec)
    for completed in spec['completed_CPU_lifecycles']:
        v=json.loads(h.verified(root,completed['receipt']).read_text())['result']
        wanted=next(r for r in rows if r['cohort_seed']==101 and r['kind']==completed['arm'])
        h.require(v['status']=='passed' and v['planned_job']==wanted['job'] and v['manifest']['sha256']==spec['producer_manifest']['sha256'] and
            v['optimizer_calls']=={'attempts':[2],'completed':[2]} and v['VALID_completed']==3 and
            v['fresh_restore_VALID_bit_exact']is True and v['TEST_accessed']is False,'Completed actual CPU lifecycle authority differs')
    sys.path.insert(0,str(s.E/'runtime'))
    from run_fit import validate_job
    for row in rows:validate_job(row['job'])
    h.require(spec['new_fits']==21 and spec['new_collections']==9 and spec['TEST_access']is False and spec['full_campaign_GO']is False,'Frozen21-fit scope differs')
    return spec,rows


def admitted(s,h,root,here,args):
    h.require(args.admission and args.admission_sha256,'Actual root admission and hash required')
    path=h.safe(root,args.admission);h.require(h.pin(path)['sha256']==args.admission_sha256,'Admission changed')
    a=json.loads(path.read_text());wanted='fits' if args.mode=='fits' else 'collect'
    h.require(a.get('schema')=='root-reviewed-molbace-pair21-v1' and a.get('mode')==wanted and a.get('approved')is True and
        a.get('publication_approved')is True and a.get('resource_admission_approved')is True and a.get('maximum_new_workers')==1 and
        a.get('manifest_sha256')==h.pin(here/'MANIFEST.json')['sha256'] and a.get('SPEC_sha256')==h.pin(here/'SPEC.json')['sha256'] and
        a.get('allocation_GPU_UUID')==h.GPU and a.get('TEST_access')is False and a.get('full_campaign_GO')is False,
        'Exact root publication/resource/source/scope review required')
    h.require(Path(sys.prefix).resolve()==(root/'.venv').resolve(),'Original repository virtualenv required')
    if args.mode!='analyze':
        h.require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==a['publication_commit'],'Admitted published checkout differs')
        h.require(all(type(a.get(k))is int and a[k]>0 for k in ('maximum_wall_seconds_per_child','reserved_GPU_bytes','reserved_host_bytes')),
            'Root chooses finite wall/memory limits from actual evidence')
    if args.mode=='fits':
        h.require(a.get('new_fits')==21 and a.get('all_numerical_lifecycles_approved')is True and
            a.get('production_timing_reviewed')is True and a.get('evidence_review_note'),'Root numerical/production review remains prerequisite')
        for key in ('lifecycle_evidence','production_evidence'):
            h.require(isinstance(a.get(key),list) and a[key],'Actual evidence receipts required')
            for pin in a[key]:h.verified(root,pin)
    else:h.require(a.get('new_collections')==9 and a.get('new_training_fits')==0,'Exactly9 new shared collectors, zero training')
    return a,path


def terminal(s,h,root,study,rows,a):
    end_path,out_path=study/'END.json',study/'OUTCOMES.json'
    for path,key in ((end_path,'terminal_END_sha256'),(out_path,'terminal_OUTCOMES_sha256')):
        h.require(h.pin(path)['sha256']==a[key],'Actual all21 terminal receipt differs')
    end,outcomes=json.loads(end_path.read_text()),json.loads(out_path.read_text());s.terminal_gate(end,outcomes,rows)
    fits=[s.prior().fit(root,study,row,outcome)[0] for row,outcome in zip(rows,outcomes)]
    return fits,h.pin(end_path),h.pin(out_path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',required=True);parser.add_argument('--mode',choices=('preflight','fits','collect','analyze'),default='preflight')
    parser.add_argument('--admission');parser.add_argument('--admission-sha256');args=parser.parse_args()
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(
        ['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Exact dedicated allocation required before project metadata')
    import study as s
    h=s.helpers();root,here=Path(args.project_root).resolve(),Path(__file__).resolve().parent
    spec,rows=preflight(s,h,root,here)
    if args.mode=='preflight':print(json.dumps({'status':'preflight_no_launch','new_fits':21,'banks':27,'job_hashes':[r['job_hash'] for r in rows]}));return 0
    a,path=admitted(s,h,root,here,args);study=h.safe(root,spec['study_output'])
    start={'SPEC':h.pin(here/'SPEC.json'),'manifest':h.pin(here/'MANIFEST.json'),'admission':h.pin(path),'mode':args.mode}
    if args.mode=='analyze':
        h.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Analysis is CPU saved-prediction only')
        fits,end_pin,out_pin=terminal(s,h,root,study,rows,a)
        return s.load(here/'analyze.py','_pair_complete_saved_analysis').run(root,spec,rows,fits,a,start,end_pin,out_pin)
    h.require(os.environ.get('CUDA_VISIBLE_DEVICES')in (None,'0',h.GPU),'Only authorized GPU0 permitted')
    lock_path=h.safe(root,a['slot_lock_path']);lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);h.check_peers(a['allowed_other_GPU_processes'])
        if args.mode=='fits':
            h.require(not study.exists(),'Fresh study output; no automatic retry/resume')
            job_root=study.with_name(study.name+'_jobs');h.require(not job_root.exists(),'Fresh job descriptor directory required');job_root.mkdir(parents=True)
            commands=[]
            for row in rows:
                job_path=job_root/(row['token']+'.json');h.write(job_path,row['job'])
                commands.append([sys.executable,'-B','-u',str(s.E/'runtime/run_fit.py'),'--project-root',str(root),'--job',str(job_path),'--output',str(study/'fits'/row['token'])])
            def result(binding,command):
                p=Path(command[-1])/'RESULT.json'
                if not p.is_file():return {'status':'missing_RESULT','unreceipted_work':'unknown, not zero'}
                v=json.loads(p.read_text());h.require(v['job_id']==binding['id'] and v['job_hash']==binding['job_hash'] and v['TEST_accessed']is False,'Own fit identity differs')
                return {'RESULT':h.pin(p),'status':v['status'],**{k:v.get(k) for k in ('selected_epoch','completed_epochs','stop_reason','convergence_established','optimizer_calls','optimizer_calls_this_invocation','resources','invocation_wall_seconds','failure','close_error')}}
            s.serial()(s,h,root,study,rows,commands,a['maximum_wall_seconds_per_child'],a['allowed_other_GPU_processes'],result,start)
        else:
            fits,end_pin,out_pin=terminal(s,h,root,study,rows,a);output=h.safe(root,a['collection_output'])
            h.require(not output.is_relative_to(study) and not study.is_relative_to(output),'Collection must be separate from fit tree')
            selected=[];commands=[]
            for row in rows:
                if row['kind']=='native_common':continue
                selected.append(dict(row,index=len(selected),fit_index=row['index']))
                commands.append([sys.executable,'-B','-u',str(s.E/'analysis/selected_molecular.py'),'--project-root',str(root),'--fit',str(study/'fits'/row['token']),'--output',str(output/row['token'])])
            def result(binding,command):
                p=Path(command[-1])/'RESULT.json'
                if not p.is_file():return {'status':'missing_RESULT','unreceipted_work':'unknown, not zero'}
                v=json.loads(p.read_text());s.collection(v,fits[binding['fit_index']],h.pin(study/'fits'/binding['token']/'RESULT.json'))
                h.require(v['analysis_source']['sha256']==h.pin(s.E/'analysis/selected_molecular.py')['sha256'],'Unchanged collector required')
                return {'RESULT':h.pin(p),'status':v['status'],**{k:v.get(k) for k in ('actual_work','wall_seconds','own_peak_rss_bytes','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','failure','close_error','sanity_against_saved_export')}}
            s.serial()(s,h,root,output,selected,commands,a['maximum_wall_seconds_per_child'],a['allowed_other_GPU_processes'],result,
                dict(start,terminal_END=end_pin,terminal_OUTCOMES=out_pin))
    return 0


if __name__=='__main__':raise SystemExit(main())
