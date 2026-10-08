"""Disabled thin retrospective analyzer: one already-selected common C&S per seed."""
import argparse, hashlib, importlib.util, json, os, resource, signal, sys, time
from collections import deque
from pathlib import Path
HERE=Path(__file__).resolve().parent; P=HERE.parent; SEEDS=(6101,6203,6307)
def require(ok,message):
    if not ok: raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def bound(row):
    f=(P/row['path']).resolve(strict=True)
    require(f.is_relative_to(P) and sha(f)==row['sha256'] and ('bytes' not in row or f.stat().st_size==row['bytes']),'Exact project file binding')
    return f
def module(path):
    s=importlib.util.spec_from_file_location('_selected_CS_author_helper',path); m=importlib.util.module_from_spec(s)
    sys.modules[s.name]=m; s.loader.exec_module(m); return m
def probabilities(torch,scores):
    require(bool(torch.isfinite(scores).all()) and bool((scores>=0).all()),'Existing nonnegative score-map domain')
    v=scores.to(torch.float64); total=v.sum(-1,keepdim=True); zero=total[:,0]==0; p=torch.empty_like(v)
    p[~zero]=v[~zero]/total[~zero]; p[zero]=0.1
    return p,int(zero.sum())
def confidence(torch,p,truth,mask):
    if not bool(mask.any()): return dict(support=0,top1=None,true_class=None,top1_minus_second=None)
    values=dict(top1=p.max(-1).values,true_class=p.gather(-1,truth[:,None])[:,0],top1_minus_second=p.topk(2,-1).values.diff(dim=-1).neg()[:,0])
    return dict(support=int(mask.sum()),**{k:dict(mean=float(v[mask].mean()),q10_median_q90=[float(x) for x in torch.quantile(v[mask],torch.tensor([.1,.5,.9],dtype=v.dtype))]) for k,v in values.items()})
def flows(before,after,truth,mask):
    a,b=before[mask]==truth[mask],after[mask]==truth[mask]
    repairs=int((~a&b).sum()); harms=int((a&~b).sum())
    require(repairs-harms==int(b.sum())-int(a.sum()),'Exact repair-harm identity')
    return dict(support=int(mask.sum()),native_correct=int(a.sum()),CS_correct=int(b.sum()),repairs=repairs,harms=harms,persistent_errors=int((~a&~b).sum()),wrong_label_churn=int((~a&~b&(before[mask]!=after[mask])).sum()))
def run(*,release_path,release_sha256,later_execution_authorized=False):
    require(later_execution_authorized is True,'Disabled retrospective analyzer; exact root release required')
    require(sha(release_path)==release_sha256,'Exact root release hash'); cfg=read(release_path); fixed=read(HERE/'ROOT_RELEASE_TEMPLATE_DISABLED.json')
    variable={'enabled','source_review_approved','root_complete_families_opened','analyzer_manifest_sha256','CS_receipt','collector_complete'}
    require(set(cfg)==set(fixed) and all(cfg[k]==fixed[k] for k in fixed if k not in variable) and all(cfg[k] is True for k in ('enabled','source_review_approved','root_complete_families_opened')),'Exact enabled reviewed complete-family release')
    require(sha(HERE/'MANIFEST.json')==cfg['analyzer_manifest_sha256'],'Reviewed analyzer manifest')
    for row in read(HERE/'MANIFEST.json')['files']: require(sha(HERE/row['path'])==row['sha256'] and (HERE/row['path']).stat().st_size==row['bytes'],'Analyzer payload unchanged')
    out=P/cfg['output_relative']; require(not out.exists(),'Fresh once-only output'); out.mkdir(mode=0o700)
    start=time.monotonic(); cpu=time.process_time(); rows=[]; error=None; work=dict(CS_calls_attempted=0,CS_calls_completed=0,propagation_passes=0,native_forwards=0,refits=0,configurations_selected=0,new_raw_files=0)
    def stop(n,f): raise TimeoutError('Retrospective analysis stop/deadline '+str(n))
    old={n:signal.signal(n,stop) for n in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}; signal.setitimer(signal.ITIMER_REAL,180)
    try:
        pins=read(HERE/'SOURCE_BINDINGS.json')
        for row in pins['files']: bound(row)
        owner=read(bound(pins['owner_bindings'])); require(P==Path(owner['runtime']['phase']) and sys.executable==owner['runtime']['python'],'Server-only pinned runtime')
        qualification=read(bound(pins['qualification'])); require(qualification['complete'] is True and qualification['metrics_computed'] is False,'Published sparse runtime qualification')
        receipt_path=bound(cfg['CS_receipt']); receipt=read(receipt_path); complete_path=bound(cfg['collector_complete']); complete=read(complete_path)
        require(receipt['schema']=='fixed-native-own-best-TRAIN-only-CS-evaluation-v1' and receipt['complete'] is True
            and receipt['protocol_sha256']==pins['cs_protocol']['sha256']
            and receipt['evaluator_program_sha256']==pins['cs_evaluator']['sha256']
            and receipt['evaluator_manifest_sha256']==pins['cs_evaluator_manifest']['sha256']
            and receipt['root_release_sha256']==pins['CS_original_evaluator_release_sha256'],'Exact completed C&S family source/role release')
        require(len(receipt['records'])==72 and all(r['status']=='complete' for r in receipt['records'])
            and [(r['seed'],r['configuration_id']) for r in receipt['records']]==[(s,'CS%02d'%i) for s in SEEDS for i in range(1,25)]
            and [r['seed'] for r in receipt['native_records']]==list(SEEDS)
            and all(r['status']=='complete' and r['probability_storage_complete'] for r in receipt['native_records']),
            'All72 records and3stored native probabilities complete')
        require(complete['complete'] is True and complete['work']['reconstructions_completed']==complete['work']['native_forwards_completed']==complete['work']['raw_files_completed']==12 and [(r['seed'],r['arm']) for r in complete['selected_states']]==[(s,a) for s in SEEDS for a in ('C4','S_joint4head','U4_sharedB','S_one_path')],'Complete12-arm collector')
        proto=read(bound(pins['cs_protocol'])); chosen=receipt['selection']['configuration_id']; selected=[next(r for r in receipt['records'] if r['seed']==s and r['configuration_id']==chosen) for s in SEEDS]
        recipe=next(r['configuration'] for r in proto['configurations'] if r['id']==chosen); require(all(r['configuration']==recipe for r in selected),'Existing COMMON selected recipe only')
        native={r['seed']:r for r in receipt['native_records']}; raw={r['seed']:r for r in complete['raw_predictions'] if r['arm']=='C4'}
        require(set(raw)==set(native)==set(SEEDS),'All3 C4/native raw identities')
        native_paths={s:bound(dict(path=str((receipt_path.parent/native[s]['probability_file']).relative_to(P)),sha256=native[s]['probability_file_sha256'],bytes=native[s]['probability_file_bytes'])) for s in SEEDS}
        c4_paths={s:bound(dict(path=str((complete_path.parent/raw[s]['path']).relative_to(P)),sha256=raw[s]['sha256'])) for s in SEEDS}
        train_path=bound(owner['roles']['train']); valid_path=bound(owner['roles']['valid'])
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='1'
        import numpy as np
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1); cs=module(bound(pins['cs_helper']))
        require(str(torch.__version__)==owner['runtime']['torch'] and np.__version__==owner['runtime']['numpy'] and torch.get_num_threads()==torch.get_num_interop_threads()==1,'Qualified providers and fixed1threads')
        with np.load(train_path,allow_pickle=False) as f: edges,anchors,labels=[torch.from_numpy(f[k].copy()) for k in ('edge_index','ids','y')]
        with np.load(valid_path,allow_pickle=False) as f: ids,truth=[torch.from_numpy(f[k].copy()) for k in ('ids','y')]
        require(edges.shape==(2,442907) and anchors.shape==labels.shape==(580,) and ids.shape==truth.shape==(5274,) and not bool(torch.isin(ids,anchors).any()),'Same complete safe roles')
        self_mask=edges[0]==edges[1]; require(int(self_mask.sum())==11701 and torch.equal(edges[0,self_mask].sort().values,torch.arange(11701)),'Known prepared self records only'); edges=edges[:,~self_mask]
        require(edges.shape==(2,431206),'Fixed directed nonself records'); adjacency=[[] for _ in range(11701)]
        for u,v in edges.T.tolist(): adjacency[u].append(v)
        distance=[-1]*11701; queue=deque(anchors.tolist())
        for u in queue: distance[u]=0
        while queue:
            u=queue.popleft()
            for v in adjacency[u]:
                if distance[v]<0: distance[v]=distance[u]+1; queue.append(v)
        d=torch.tensor(distance)[ids]; cohort_masks=dict(whole=torch.ones(5274,dtype=torch.bool),covered=d==1,uncovered=d!=1)
        distance_masks={str(k):d==k for k in sorted(set(d.tolist()))}
        class_support=dict(TRAIN=torch.bincount(labels,minlength=10).tolist(),development=torch.bincount(truth,minlength=10).tolist())
        for seed,record in zip(SEEDS,selected):
            p0=torch.load(native_paths[seed],map_location='cpu',weights_only=False); c4=torch.load(c4_paths[seed],map_location='cpu',weights_only=False)
            require(p0.shape==(11701,10) and torch.equal(c4['valid_ids'],ids) and c4['seed']==seed and c4['arm']=='C4' and c4['served_probabilities'].shape==(5274,10) and c4['selected_sha256']==next(r['sha256'] for r in complete['selected_states'] if r['seed']==seed and r['arm']=='C4'),'Existing ordered raw arrays')
            work['CS_calls_attempted']+=1
            result=cs.correct_and_smooth_train_only(probabilities=p0,edge_index=edges,train_ids=anchors,train_labels=labels,classes=10,configuration=recipe,device='cpu',later_execution_authorized=True)
            work['CS_calls_completed']+=1; work['propagation_passes']+=result['propagation_passes']
            score=result['smoothed_scores'][ids]; native_score=p0[ids]; cp=c4['served_probabilities']; n=native_score.argmax(-1); q=score.argmax(-1); c=cp.argmax(-1)
            require(int((q==truth).sum())==record['readout']['correctcount'] and int((n==truth).sum())==native[seed]['readout']['correctcount'],'Saved selected/native correctcounts reproduced')
            pn,nzero=probabilities(torch,native_score); pq,qzero=probabilities(torch,score); pt=pn.gather(-1,truth[:,None])[:,0]; qt=pq.gather(-1,truth[:,None])[:,0]; ct=cp.gather(-1,truth[:,None])[:,0]
            require(int((qt==0).sum())==record['readout']['NLL']['zero_target_count'] and int((pt==0).sum())==native[seed]['readout']['NLL']['zero_target_count'] and qzero==record['readout']['uniform_zero_rows'] and nzero==native[seed]['readout']['uniform_zero_rows'],'Exact saved zero-target/row counts; no clipping')
            def panel(mask):
                a,b=c[mask]==truth[mask],q[mask]==truth[mask]
                return dict(**flows(n,q,truth,mask),C4_correct=int(a.sum()),both_correct=int((a&b).sum()),C4_only_correct=int((a&~b).sum()),CS_only_correct=int((~a&b).sum()),neither_correct=int((~a&~b).sum()),native_zero_true_probability=int(((pt==0)&mask).sum()),CS_zero_true_probability=int(((qt==0)&mask).sum()),C4_zero_true_probability=int(((ct==0)&mask).sum()))
            groups=dict(repaired=(n!=truth)&(q==truth),persistent=(n!=truth)&(q!=truth),harmed=(n==truth)&(q!=truth))
            rows.append(dict(seed=seed,configuration_id=chosen,configuration=recipe,
                CS_native_selected_epoch=native[seed]['native_epoch'],C4_native_selected_epoch=c4['selected_epoch'],
                native_selected_epochs_equal=native[seed]['native_epoch']==c4['selected_epoch'],
                native_probability_sha256=native[seed]['probability_file_sha256'],C4_raw_sha256=raw[seed]['sha256'],
                C4_selected_state_sha256=c4['selected_sha256'],CS_native_state_sha256=native[seed]['native_state_sha256'],
                overlap_caveat='Own-native-best C&S versus C4-selected native endpoints; actual epochs/equality disclosed. Overlap is descriptive, not a causal correction effect.',
                selected_CS_readout=record['readout'],
                cohorts={k:panel(m) for k,m in cohort_masks.items()},
                directed_shortest_TRAIN_anchor_distance={k:panel(m) for k,m in distance_masks.items()},
                native_confidence={k:confidence(torch,pn,truth,m) for k,m in groups.items()},
                class_support_and_zero_targets=[dict(class_id=k,support=int((truth==k).sum()),
                    native_zero=int(((truth==k)&(pt==0)).sum()),CS_zero=int(((truth==k)&(qt==0)).sum()),
                    C4_zero=int(((truth==k)&(ct==0)).sum())) for k in range(10)],
                uniform_zero_rows=dict(native=nzero,CS=qzero)))
            require(time.monotonic()-start<180,'Finite180s complete diagnostic budget')
        write(out/'RESULT.json',dict(complete=True,seeds=rows,class_support=class_support,work=work,
            root_release_sha256=release_sha256,CS_receipt=cfg['CS_receipt'],collector_complete=cfg['collector_complete'],
            retrospective_consumed_development=True,no_gate_change_or_subset_rescue=True,
            distance_policy='Directed source->receiver nonself BFS; -1 unreachable; TRAIN anchors distance0; one-hop covered distance1.',
            raw_arrays_server_only=True))
    except BaseException as caught:
        error=dict(type=type(caught).__name__,message=str(caught)); write(out/'FAILURE.json',dict(error=error,completed_seed_summaries=rows,work=work,automatic_retry=False)); raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        write(out/'COST_TERMINAL.json',dict(complete=error is None and len(rows)==3,error=error,work=work,
            partial_propagation_passes_unobserved=work['CS_calls_attempted']>work['CS_calls_completed'],
            wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
            max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            threads=1,active_seconds=180,release_sha256=release_sha256,automatic_retry=False))
        for n,h in old.items(): signal.signal(n,h)
    return dict(complete=True,output=str(out),result_sha256=sha(out/'RESULT.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--release',type=Path,required=True); p.add_argument('--release-sha256',required=True); p.add_argument('--authorized',action='store_true'); a=p.parse_args()
    print(json.dumps(run(release_path=a.release,release_sha256=a.release_sha256,later_execution_authorized=a.authorized),sort_keys=True))
