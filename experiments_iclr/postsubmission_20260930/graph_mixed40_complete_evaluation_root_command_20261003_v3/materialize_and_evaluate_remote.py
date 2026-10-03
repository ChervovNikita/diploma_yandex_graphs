"""Root command: exact closure/exit first, then unchanged reviewed CPU evaluator."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys,time
sys.dont_write_bytecode=True
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
OUTPUT=PHASE/'graph_mixed40_complete_evaluation_execution_root_20261003_v3'
RUN=PHASE/'graph_mixed_block_training_preparation_20261003_v2/runs/root_full40_run01'
TRANSPORT=PHASE/'graph_mixed_block_execution_root_20261003_v1/transport/root_full40_run01'
PACKET=PHASE/'graph_mixed_block_closed_family_evaluation_preparation_20261003_v3'
MANIFEST_SHA='3d2c3549afba16ee2b1689aa82e168b3251d8d3bb544f2f38dd8883372dcfb59'
REVIEW=PHASE/'mixed_closed_family_evaluation_source_review_20261003_v3/REVIEW.json'
REVIEW_SHA='cc91457f036f7bbaf42cd78a2de248424da71813a9bf2153a197a4def2679f37'
NATIVE=PHASE/'graph_heterogeneous_dblp_native15_audit_execution_root_20261003_v1/NATIVE15_AUDIT_run01.json'
NATIVE_SHA='c697a571abcc9bdef6f5d543ba8cda58cb394fa4784405653138d75e690f01ce'
SEEDS=[131,137,139,149,151];DATASETS=['HGB-DBLP','HGB-ACM'];POLICIES=['own/own','pool/pool','pool/own','own/pool']

def need(ok,message):
    if not ok:raise ValueError(message)

def descriptor(p):
    p=Path(p);need(p.is_absolute() and p.is_relative_to(PHASE) and '..' not in p.parts,'Authorized phase path required')
    need(not any(q.is_symlink() for q in [p,*p.parents] if q.is_relative_to(PHASE)),'Symlink input/output forbidden')
    b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}

def read(p):
    d=descriptor(p);return json.loads(Path(p).read_text()),d

def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')

def process(pid,start):
    p=Path('/proc')/str(pid)
    if not p.exists():return {'PID':pid,'expected_handle_present':False}
    try:
        stat=p.joinpath('stat').read_text();tail=stat[stat.rfind(')')+2:].split()
        return {'PID':pid,'state':tail[0],'PPID':int(tail[1]),'starttime_ticks':tail[19],'expected_handle_present':tail[19]==start}
    except (FileNotFoundError,ProcessLookupError):return {'PID':pid,'expected_handle_present':False}

os.chdir(REPO)
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
need(Path.cwd()==REPO and gpu.returncode==0 and gpu.stdout.strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac','Authorized route/repository/single UUID differs')
need(descriptor(PACKET/'MANIFEST.json')['sha256']==MANIFEST_SHA,'Reviewed evaluator manifest differs')
review,review_rec=read(REVIEW);need(review_rec['sha256']==REVIEW_SHA and review['source']['manifest']['sha256']==MANIFEST_SHA and review['status']=='passed' and not review['blocking_findings'],'Independent V3 one-line source review differs')
spec=importlib.util.spec_from_file_location('unchanged_reviewed_mixed_evaluator',PACKET/'evaluate.py')
evaluator=importlib.util.module_from_spec(spec);sys.modules[spec.name]=evaluator;spec.loader.exec_module(evaluator)
provenance,actual=evaluator.source_guard();need(actual==MANIFEST_SHA,'Evaluator source guard differs')
child,child_rec=read(TRANSPORT/'CHILD.json');need(child['PID']==379193 and child['command'][0]==str(REPO/'.venv/bin/python'),'Original training handle/interpreter differs')
processes=[process(379192,'5981324833'),process(379193,'5981324843')]
slots=[]
for dataset in DATASETS:
    for seed in SEEDS:
        for policy in POLICIES:
            p=RUN/dataset/('seed'+str(seed))/policy.replace('/','__')/'TERMINAL.json'
            row={'dataset':dataset,'seed':seed,'policy':policy,'terminal_present':p.is_file()}
            if p.is_file():
                d,_=read(p);row.update({k:d[k] for k in ('status','selected_state_replay','checkpoint_bindings_verified','reused','final_labels_closed') if k in d})
            slots.append(row)
selected=sum(r.get('status')=='selected' for r in slots)
terminal_path=TRANSPORT/'SUPERVISOR_TERMINAL.json'
closed=(RUN/'STUDY.json').is_file() and terminal_path.is_file() and selected==40 and not any(r['expected_handle_present'] for r in processes)
if not closed:
    print(json.dumps({'status':'pending_full40_and_actual_process_exit','UTC':datetime.now(timezone.utc).isoformat(),'selected':selected,'remaining_slots':[r for r in slots if r.get('status')!='selected'],'processes':processes,'study_present':(RUN/'STUDY.json').is_file(),'supervisor_terminal_present':terminal_path.is_file(),'evaluation_release_created':False,'evaluator_started':False,'remote_writes':False}))
    raise SystemExit(3)
terminal,terminal_rec=read(terminal_path)
need(terminal['exit_code']==0 and terminal['timed_out'] is False and terminal['automatic_restart'] is False and terminal['unrelated_processes_touched'] is False,'Training child did not close normally')
study,study_rec=read(RUN/'STUDY.json')
need(study['summary']['status']=='complete_development_summary' and study['summary']['all40_terminals'] is True and study['summary']['successful_subset_scored'] is False and study['originals_preserved'] is True and study['final_labels_closed'] is True,'Closed full40 source summary differs')
need([(r['dataset'],r['seed'],r['policy']) for r in study['rows']]==[(d,s,p) for d in DATASETS for s in SEEDS for p in POLICIES],'Full40 ordered family differs')
need(all(r['status']=='selected' and r['selected_state_replay'] is True and r['checkpoint_bindings_verified'] is True and r['reused'] is False and r['final_labels_closed'] is True for r in study['rows']),'Full40 replay/binding custody is incomplete')
native,native_rec=read(NATIVE);need(native_rec['sha256']==NATIVE_SHA and native_rec['bytes']==72174,'Required native15 audit differs')
need(all(native[k] is True for k in ('root_observed','all15_closed','training_child_reaped','all15_checkpoint_replays_audited','originals_preserved','heldout_labels_closed')),'Required all15 native audit is unclosed')
need([(r['seed'],r['arm']) for r in native['cases']]==[(s,a) for s in SEEDS for a in ('native_GAT','native_Simple_HGN','native_SeHGNN')],'Required native15 complete denominator differs')
freeze,_=read(PACKET/'EVALUATION_FREEZE_TEMPLATE.json')
freeze.update(status='root_released_closed40_required_native15',study=study_rec,expected_output_path=str(OUTPUT/'EVALUATION_run01.json'),native_secondary=native_rec)
release,_=read(PACKET/'EVALUATION_RELEASE_TEMPLATE.json')
release.update(status='root_released_after_observed_full40_and_actual_exit',execution_authorized=True,evaluation_manifest_sha256=MANIFEST_SHA,study_sha256=study_rec['sha256'],root_observed_closed_full40=True,training_child_reaped=True,independent_source_review=review_rec,independent_source_review_observed=True)
need(len(sys.argv)==2 and sys.argv[1] in ('--materialize','--execute'),'Use --materialize or --execute')
need(not OUTPUT.exists(),'Fresh external evaluation directory required; no automatic retry or overwrite')
OUTPUT.mkdir()
write(OUTPUT/'ROOT_PROCESS_EXIT_AND_CLOSURE.json',{'UTC':datetime.now(timezone.utc).isoformat(),'training_terminal':terminal_rec,'training_terminal_metadata':terminal,'original_child':child_rec,'exact_old_handles':processes,'study':study_rec,'all40_selected':True,'all15_native_audit':native_rec,'heldout_labels_closed':True})
write(OUTPUT/'EVALUATION_FREEZE.json',freeze)
release['evaluation_freeze_sha256']=descriptor(OUTPUT/'EVALUATION_FREEZE.json')['sha256']
release['root_process_exit_and_closure_receipt']=descriptor(OUTPUT/'ROOT_PROCESS_EXIT_AND_CLOSURE.json')
write(OUTPUT/'EVALUATION_RELEASE.json',release)
# Admission and full-family source custody precede any numerical execution.
evaluator.admission(OUTPUT/'EVALUATION_FREEZE.json',OUTPUT/'EVALUATION_RELEASE.json')
command=[str(REPO/'.venv/bin/python'),'-B',str(PACKET/'evaluate.py'),'--freeze',str(OUTPUT/'EVALUATION_FREEZE.json'),'--admission',str(OUTPUT/'EVALUATION_RELEASE.json'),'--output',str(OUTPUT/'EVALUATION_run01.json')]
env_options={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
write(OUTPUT/'COMMAND.json',{'command':command,'cwd':str(REPO),'environment_overrides':env_options,'reviewed_V3_evaluator_unchanged_after_one_line_source_repair':True,'new_predictive_fits':False,'required_native_secondary_cases':15,'automatic_retry':False})
if sys.argv[1]=='--materialize':
    print(json.dumps({'status':'exact_release_and_command_materialized','output_directory':str(OUTPUT),'evaluator_started':False}));raise SystemExit(0)
env=os.environ.copy();env.update(env_options);start=time.perf_counter()
with (OUTPUT/'EVALUATOR.stdout.log').open('x') as stdout,(OUTPUT/'EVALUATOR.stderr.log').open('x') as stderr:
    result=subprocess.run(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
write(OUTPUT/'EVALUATION_PROCESS_EXIT.json',{'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':result.returncode,'elapsed_seconds':time.perf_counter()-start,'automatic_retry':False,'process_killed_or_restarted':False})
if result.returncode:raise SystemExit(result.returncode)
evaluation,ev_rec=read(OUTPUT/'EVALUATION_run01.json')
secondary=evaluation.get('native_secondary') or {}
complete=evaluation['status']=='complete' and len(evaluation['rows'])==40 and len(secondary.get('native_rows',[]))==15 and evaluation['heldout_labels_opened'] is False and evaluation['originals_preserved'] is True
write(OUTPUT/'AUDIT_CLOSURE.json',{'UTC':datetime.now(timezone.utc).isoformat(),'status':'complete40_and_required_native15' if complete else 'required_native15_secondary_not_admitted','evaluation':ev_rec,'primary_rows':len(evaluation['rows']),'native_secondary_rows':len(secondary.get('native_rows',[])),'native_secondary_failure':secondary.get('reason'),'heldout_labels_closed':True,'source_evaluator_modified':False,'scientific_gate_decision_or_partial_ranking_added':False})
print(json.dumps({'status':'complete40_and_required_native15' if complete else 'required_native15_secondary_not_admitted','evaluation':ev_rec,'primary_rows':len(evaluation['rows']),'native_secondary_rows':len(secondary.get('native_rows',[]))}))
raise SystemExit(0 if complete else 4)
