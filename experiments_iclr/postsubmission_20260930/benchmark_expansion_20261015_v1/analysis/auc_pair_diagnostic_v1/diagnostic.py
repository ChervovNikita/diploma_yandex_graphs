"""Diagnose native binary AUC orderings using saved VALID tensors only."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time


def run(root, output):
    started=time.perf_counter()
    root=Path(root).resolve();output=Path(output).resolve()
    if not output.is_relative_to(root):raise ValueError('Project output only')
    helper=Path(__file__).resolve().parents[1]/'fixed_probability_aggregation_v1/diagnostic.py'
    assert hashlib.sha256(helper.read_bytes()).hexdigest()==HELPER_SHA
    spec=importlib.util.spec_from_file_location('saved_probability_helper_for_auc_pairs',helper)
    authority=importlib.util.module_from_spec(spec);spec.loader.exec_module(authority)
    authority.guard()
    inputs_path=helper.with_name('INPUTS.json')
    assert hashlib.sha256(inputs_path.read_bytes()).hexdigest()==INPUTS_SHA
    inputs=json.loads(inputs_path.read_text())
    os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
    import torch
    from ogb.graphproppred import Evaluator
    torch.set_num_threads(2)
    evaluator=Evaluator(name='ogbg-molbace')
    results=[];margins={};ref_ids=ref_labels=None
    for case in inputs['cases']:
        route_path=authority.verified(case['route_tensor'],root)
        data=torch.load(route_path,map_location='cpu',weights_only=False)
        assert data['TEST_accessed'] is False and data['role']=='VALID'
        ids,labels=data['ids'],data['labels']
        if ref_ids is None:ref_ids,ref_labels=ids,labels
        else:assert torch.equal(ids,ref_ids) and torch.equal(labels,ref_labels)
        z=data['member_raw_logits'][:,:,0].double()
        pooled=data['pooled_raw_logits'][:,0].double()
        assert z.shape[0]==case['expected_members'] and len(labels)==151
        y=labels[:,0]
        pos,neg=(y==1),(y==0)
        assert int(pos.sum())==130 and int(neg.sum())==21
        member=z[:,pos,None]-z[:,None,neg]
        pool=pooled[pos,None]-pooled[None,neg]
        correct,wrong,tied=member>0,member<0,member==0
        pc,pw,pt=pool>0,pool<0,pool==0
        pair_count=pool.numel()
        native=float(evaluator.eval({'y_true':labels.numpy(),'y_pred':data['pooled_raw_logits'].numpy()})['rocauc'])
        manual=float((pc.double()+.5*pt.double()).mean())
        assert abs(manual-native)<1e-12
        r=dict(name=case['name'],pair_count=pair_count,selected_epoch=data['selected_epoch'],
               actual_pool_AUC=native,pool_correct=int(pc.sum()),pool_tied=int(pt.sum()),pool_reversed=int(pw.sum()),
               member_AUC=[float(v) for v in (correct.double()+.5*tied.double()).mean((1,2))],
               all_members_strictly_reversed=int(wrong.all(0).sum()),
               no_member_strictly_correct=int((~correct.any(0)).sum()),
               member_sign_disagreement_pairs=int((correct.any(0)&~correct.all(0)).sum()),
               pool_reversed_despite_some_member_correct=int((pw&correct.any(0)).sum()),
               pool_reversed_despite_strict_majority_correct=int((pw&(correct.sum(0)>len(z)/2)).sum()),
               pool_correct_with_some_member_nonpositive=int((pc&~correct.all(0)).sum()),
               pool_strict_repairs_of_all_member_reversals=int((pc&wrong.all(0)).sum()),
               per_member_mean_absolute_pair_margin=member.abs().mean((1,2)).tolist(),
               input_receipt=case['route_tensor'])
        results.append(r);margins[case['name']]=pool
    assert len(results)==4 and 'base' in margins
    base=margins['base'];comparisons=[]
    for name,margin in margins.items():
        if name=='base':continue
        base_credit=(base>0).double()+.5*(base==0).double()
        credit=(margin>0).double()+.5*(margin==0).double()
        comparisons.append(dict(name=name,versus='base',
            strictly_repaired_pairs=int(((base<0)&(margin>0)).sum()),
            strictly_harmed_pairs=int(((base>0)&(margin<0)).sum()),
            changed_credit_pairs=int((credit!=base_credit).sum()),
            AUC_difference=float((credit-base_credit).mean())))
    result=dict(schema='saved-selected-VALID-AUC-pair-diagnostic-v1',
        scope='posthoc selected-VALID seed101 only; not independent confirmation',
        graphs=151,positives=130,negatives=21,pairs_are_independent_observations=False,
        cases=results,comparisons=comparisons,new_forwards=0,new_models=0,
        optimizer_updates=0,selector_changes=0,TEST_accessed=False,
        wall_seconds=time.perf_counter()-started,peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        INPUTS_sha256=INPUTS_SHA)
    output.mkdir(parents=True,exist_ok=False)
    (output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


HELPER_SHA='d9154f83ace6064bdd592777f87464b6c65a79ce353391b7e8b1364273343e4e'
INPUTS_SHA='81feb334627062eb8dc01d9421393780cb29dbd706fe19b98d5b942295eb378c'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();r=run(args.project_root,args.output)
    print(json.dumps({'comparisons':r['comparisons'],'cases':[{k:v[k] for k in ['name','pool_reversed','all_members_strictly_reversed','pool_reversed_despite_some_member_correct','pool_reversed_despite_strict_majority_correct','member_AUC','per_member_mean_absolute_pair_margin']} for v in r['cases']]}))
