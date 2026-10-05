"""Audit the complete frozen 36-fit cohort and compare all seven VALID families.

No training, selector change, coefficient fitting, TEST load, or GPU work.
The early completion gate precedes every prediction/history/checkpoint read.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import itertools
import json
import math
import socket
import statistics
import subprocess
import time
from selection_audit import audit_selection

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
COHORT=PHASE/'citeseer_endpoint_frame_paired_development_20261005_v1'
ROOT=PHASE/'citeseer_frame_complete_analysis_20261005_v2'
PLAN_SHA='fd9fec451a81d0512cd8431ba2a990c580592ff6a6e8e12d784386a99c5e9daa'
SOURCE_MANIFEST_SHA='efa95806d86e3cc261a8506042204d8d32e39506386d6faf90a625d52be12ff9'
FAMILIES=('native_single','ordinary_independent4','unframed_f4','shared_frame_f4',
          'private_frame_f4','same_four_frames_single','independent_frame4')
PRIMARY=('native_single','ordinary_independent4','same_four_frames_single','independent_frame4')
SECONDARY=('shared_frame_f4','unframed_f4')
# Two-sided t(2) .975 quantile; these intervals describe seed variation on one split.
T2_975=4.302652729696142


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        while part:=stream.read(1024*1024):value.update(part)
    return value.hexdigest()


def save(name,value):
    with (ROOT/name).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def seed_summary(values):
    assert len(values)==3 and all(math.isfinite(x) for x in values)
    mean=statistics.mean(values);sd=statistics.stdev(values)
    return dict(values=values,mean=mean,sd=sd,
                descriptive_t95_seed_interval=[mean-T2_975*sd/math.sqrt(3),mean+T2_975*sd/math.sqrt(3)],
                inference_scope='Three training-seed blocks conditional on this fixed graph/split; normal-seed approximation, not graph/query independence.')


def contrast(candidate,reference):
    delta=[a-b for a,b in zip(candidate,reference)]
    observed=abs(statistics.mean(delta))
    possibilities=[abs(statistics.mean([s*d for s,d in zip(signs,delta)])) for signs in itertools.product((-1,1),repeat=3)]
    result=seed_summary(delta)
    result.update(candidate_higher_blocks=sum(d>0 for d in delta),ties=sum(d==0 for d in delta),
                  exact_two_sided_seed_signflip_p=sum(x>=observed-1e-15 for x in possibilities)/8,
                  p_assumption='Independent seed-block differences with sign symmetry under null; not certified by n=3. Minimum attainable two-sided p is0.25.')
    return result


def main():
    started=time.monotonic()
    stage='preflight_and_complete_cohort_gate'
    authorized=False
    try:
        assert Path.cwd()==REPO and socket.gethostname()=='anogena-2-0'
        assert Path(__file__).resolve().parent==ROOT
        assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
        authorized=True
        plan_path=COHORT/'PLAN.json';assert sha(plan_path)==PLAN_SHA
        plan=json.loads(plan_path.read_text());rows=plan['physical_fits'];assert len(rows)==36
        terminal=COHORT/'COHORT_FREEZE.json'
        if not terminal.exists():
            # Read owned progress only; do not import numerical libraries or open outcomes.
            progress=COHORT/'QUEUE_PROGRESS.json'
            print(json.dumps(dict(status='WAIT_FOR_COMPLETE_COHORT',progress=json.loads(progress.read_text()) if progress.exists() else None,
                                  prediction_history_checkpoint_reads=0,TEST_access=False)))
            return
        completed=json.loads(terminal.read_text())
        assert completed['complete'] is True and completed['TEST_access'] is False
        assert completed['plan_sha256']==PLAN_SHA and completed['source_manifest_sha256']==SOURCE_MANIFEST_SHA
        assert len(completed['completed_physical_fits'])==36, 'Duplicate or extra completed physical fits'
        registered={r['id']:r for r in completed['completed_physical_fits']}
        assert set(registered)=={r['id'] for r in rows} and len(registered)==36
        assert not (ROOT/'RESULTS.json').exists() and not (ROOT/'FAILURE.json').exists()
        protocol_path=ROOT/'PROTOCOL.json';protocol=json.loads(protocol_path.read_text())
        assert protocol['plan_sha256']==PLAN_SHA and protocol['analysis_source_sha256']==sha(Path(__file__))
        manifest_path=ROOT/'ANALYSIS_MANIFEST.json'
        assert sha(manifest_path)==protocol['analysis_manifest_sha256']
        for source in json.loads(manifest_path.read_text())['files']:
            assert sha(ROOT/source['path'])==source['sha256']
        # Authenticate every physical terminal before any comparative-outcome access.
        for row in rows:
            assert sha(COHORT/'fits'/row['id']/'FREEZE.json')==registered[row['id']]['freeze_sha256']
        save('START.json',dict(UTC=datetime.now(timezone.utc).isoformat(),cohort_freeze_sha256=sha(terminal),
                               protocol_sha256=sha(protocol_path),analysis_source_sha256=sha(Path(__file__)),
                               fit_count=0,optimizer_updates=0,TEST_access=False))
        stage='complete_frozen_cohort_analysis'
        import torch
        torch.set_num_threads(2)
        predictions={};audits=[];identities=None
        for row in rows:
            folder=COHORT/'fits'/row['id'];freeze=json.loads((folder/'FREEZE.json').read_text())
            assert freeze['TEST_access'] is False and freeze['cohort_plan_sha256']==PLAN_SHA
            assert freeze['arm']==row['arm'] and freeze['seed']==row['seed'] and freeze['member_count']==row['member_count']
            assert freeze['paired_seed_block']==row['paired_seed_block']
            assert freeze['source_manifest_sha256']==SOURCE_MANIFEST_SHA
            for name,key in [('CONFIG.json','config_sha256'),('VALID_HISTORY.jsonl','history_sha256'),
                             ('selected_checkpoint.pt','checkpoint_sha256'),('selected_VALID_logits.pt','VALID_logits_sha256')]:
                assert sha(folder/name)==freeze[key],row['id']+': '+name
            config=json.loads((folder/'CONFIG.json').read_text());job=config['job']
            assert job['cohort_plan_sha256']==PLAN_SHA and job['source_manifest_sha256']==SOURCE_MANIFEST_SHA
            for key,value in row.items():assert job[key]==value
            if identities is None:identities=freeze['input_identities']
            assert freeze['input_identities']==identities==config['input_identities']
            history=[json.loads(line) for line in (folder/'VALID_HISTORY.jsonl').read_text().splitlines()]
            selection_receipt=audit_selection(history,freeze,config)
            saved=torch.load(folder/'selected_VALID_logits.pt',map_location='cpu',weights_only=True)
            assert saved['checkpoint_sha256']==freeze['checkpoint_sha256'] and saved['selected_epoch']==freeze['selected_epoch']
            assert saved['input_identities']==identities
            pos,neg=saved['pos'],saved['neg']
            assert pos.shape==(227,) and neg.shape==(227,500)
            assert pos.dtype==neg.dtype==torch.float32 and torch.isfinite(pos).all() and torch.isfinite(neg).all()
            rank=1+.5*((neg>=pos[:,None]).sum(1)+(neg>pos[:,None]).sum(1))
            reciprocal=1/rank.float();mrr=reciprocal.mean().item()
            assert round(mrr,4)==freeze['selected_VALID_MRR']
            predictions[row['id']]=dict(pos=pos,neg=neg)
            audits.append(dict(id=row['id'],selected_epoch=freeze['selected_epoch'],last_epoch=freeze['last_epoch'],
                               MRR_unrounded=mrr,MRR_selector_rounded4=round(mrr,4),updates=freeze['updates'],
                               inclusive_seconds=freeze['inclusive_seconds'],peak_CUDA_allocated_bytes=freeze['peak_CUDA_allocated_bytes'],
                               peak_CUDA_reserved_bytes=freeze['peak_CUDA_reserved_bytes'],checkpoint_sha256=freeze['checkpoint_sha256'],
                               VALID_logits_sha256=freeze['VALID_logits_sha256'],selection_audit=selection_receipt))
        served={};family_scores={f:[] for f in FAMILIES};query_rows=[];components={}
        for base in range(3):
            block='citeseer_native_b'+str(base)
            ordinary=[f'b{base}_native_single_seed{base}']+[f'b{base}_independent_member_m{m}_seed{base+5*m}' for m in range(1,4)]
            framed=[f'b{base}_independent_frame_member_m{m}_seed{base+5*m}' for m in range(4)]
            for family in FAMILIES:
                ids=ordinary if family=='ordinary_independent4' else framed if family=='independent_frame4' else [f'b{base}_{family}_seed{base}']
                # All component selectors are fixed. Never select a better ensemble epoch.
                pos=torch.stack([predictions[i]['pos'] for i in ids]).mean(0)
                neg=torch.stack([predictions[i]['neg'] for i in ids]).mean(0)
                rank=1+.5*((neg>=pos[:,None]).sum(1)+(neg>pos[:,None]).sum(1))
                rr=1/rank.float();mrr=rr.mean().item()
                family_scores[family].append(mrr)
                key=block+'/'+family;served[key]=dict(pos=pos,neg=neg,rank=rank,reciprocal_rank=rr)
                components[key]=ids
                query_rows.extend(dict(seed_block=block,family=family,query_index=i,rank=float(rank[i]),reciprocal_rank=float(rr[i])) for i in range(227))
        primary={reference:contrast(family_scores['private_frame_f4'],family_scores[reference]) for reference in PRIMARY}
        ordered=sorted(PRIMARY,key=lambda name:primary[name]['exact_two_sided_seed_signflip_p'])
        running=0.
        for index,name in enumerate(ordered):
            running=max(running,(4-index)*primary[name]['exact_two_sided_seed_signflip_p'])
            primary[name]['Holm_adjusted_primary_family_p']=min(1.,running)
        secondary={reference:contrast(family_scores['private_frame_f4'],family_scores[reference]) for reference in SECONDARY}
        torch.save(dict(served=served,components=components,input_identities=identities,cohort_freeze_sha256=sha(terminal)),ROOT/'served_VALID_predictions.pt')
        save('PER_QUERY.json',query_rows)
        result=dict(UTC=datetime.now(timezone.utc).isoformat(),status='complete_audited_development_comparison',
                    source_plan_sha256=PLAN_SHA,cohort_freeze_sha256=sha(terminal),source_manifest_sha256=SOURCE_MANIFEST_SHA,
                    fit_audits=audits,family_summaries={f:seed_summary(v) for f,v in family_scores.items()},
                    primary_paired_contrasts=primary,secondary_descriptive_contrasts=secondary,
                    primary_p_family_size=4,serving='mean raw logits; independent native constituents selected individually',
                    metric='HeaRT MRR over each released positive and all500fixed negatives, native midpoint ties; unrounded reporting',
                    data_scope='TRAIN/VALID development only, one fixed Citeseer graph/split; three paired seed blocks',
                    limitations=['VALID chose checkpoints and this is a development comparison.',
                                 'Training seeds do not sample independent graphs, edge splits or query populations.',
                                 'Three paired blocks cannot establish statistical significance in the exact two-sided signflip test.',
                                 'Wall times include concurrently running owned Amazon work; not dedicated throughput comparisons.',
                                 'A positive development contrast does not establish novelty or publication readiness.'],
                    independent_aggregate_retuning=False,TEST_access=False,fit_count=0,optimizer_updates=0,
                    predictive_outcome_selection_performed=False,
                    served_predictions_sha256=sha(ROOT/'served_VALID_predictions.pt'),per_query_sha256=sha(ROOT/'PER_QUERY.json'),
                    inclusive_seconds=time.monotonic()-started)
        save('RESULTS.json',result)
        print(json.dumps(dict(status=result['status'],family_means={f:s['mean'] for f,s in result['family_summaries'].items()},
                              TEST_access=False,source_plan_sha256=PLAN_SHA)))
    except (Exception,KeyboardInterrupt) as error:
        failure=dict(UTC=datetime.now(timezone.utc).isoformat(),error=type(error).__name__+': '+str(error),
                     stage=stage,fits=0,optimizer_updates=0,TEST_access=False,partial_outputs_preserved=True,
                     inclusive_seconds=time.monotonic()-started)
        if authorized and ROOT.is_dir() and not (ROOT/'FAILURE.json').exists():
            save('FAILURE.json',failure)
        else:
            print(json.dumps(failure))
        raise


if __name__=='__main__':
    main()
