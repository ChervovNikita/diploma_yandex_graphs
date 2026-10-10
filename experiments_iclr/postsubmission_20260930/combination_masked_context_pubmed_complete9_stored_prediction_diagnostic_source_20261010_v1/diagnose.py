"""Disabled complete-nine stored-logit error counts. No models or forwards."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
from common import HERE,PHASE,SEEDS,CONDITIONS,LIMITS,SERVER_PHASE,GPU,bind,inside,sha,sources,fingerprint,write


def admit(path,digest):
    path=inside(path)
    if sha(path)!=digest:raise ValueError('Exact separate root diagnostic release')
    spec=json.loads(path.read_text())
    if spec.get('schema')!='masked-context-complete9-stored-prediction-release-v1':raise ValueError('Complete-nine diagnostic release required')
    for key in ('enabled','root_diagnostic_authorized','whole_nine_owned_completion_verified','source_review_approved','finite_owner_bound','fresh_resource_readiness_confirmed'):
        if spec.get(key) is not True:raise ValueError('Stored diagnostic remains disabled: '+key)
    for key in ('TEST_access','automatic_retry','new_forwards','new_fits','thresholds_changed'):
        if spec.get(key) is not False:raise ValueError('No TEST, retries, forwards, fits or new thresholds')
    if any(key in spec for key in ('train_bundle','test_bundle','test_ids','test_y','checkpoint_bundle')):raise ValueError('VALID and stored logits only')
    if spec.get('limits')!=LIMITS:raise ValueError('Exact finite300+10-second diagnostic envelope')
    if socket.gethostname()!='anogena-2-0' or str(PHASE)!=SERVER_PHASE:raise ValueError('Exact authorized allocation route')
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).splitlines()!=[GPU] or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('Exact sole allocation GPU')
    binding=sources(spec['source_manifest_sha256'])
    if spec.get('stage1_source_manifest_sha256')!=binding['stage1_manifest']['sha256'] or spec.get('frozen_providers')!=binding['frozen_providers']:raise ValueError('Exact original source/provider binding')
    expected={f'seed{s}__{c}':(s,c) for s in SEEDS for c in CONDITIONS}
    if set(spec.get('records',{}))!=set(expected):raise ValueError('All nine required before any outcome/VALID access')
    terminals={}
    for identity,rows in spec['records'].items():
        terminal=json.loads(bind(rows['terminal_custody']).read_text())
        if terminal.get('schema')!='masked-context-stage1-owned-terminal-custody-v1' or terminal.get('record_id')!=identity:raise ValueError('Exact owned terminal identity')
        if terminal.get('directly_waited') is not True or terminal.get('child_exit_code')!=0 or terminal.get('cap_or_owner_failure') is not None:raise ValueError('Actual successful direct wait required for every record')
        if terminal.get('owned_process_absence_verified') is not True or terminal.get('owned_CUDA_absence_verified') is not True:raise ValueError('Owned process/CUDA absence required')
        raw=json.loads(bind(terminal['raw_owner_terminal']).read_text())
        absent=json.loads(bind(terminal['owned_absence_evidence']).read_text())
        if raw.get('directly_waited') is not True or raw.get('child_exit_code')!=0 or raw.get('cap_or_owner_failure') is not None or terminal['release_sha256'] not in raw.get('argv',[]):raise ValueError('Raw actual wait/release evidence')
        if absent.get('owned_process_absence_verified') is not True or absent.get('owned_CUDA_absence_verified') is not True:raise ValueError('Raw actual owned absence evidence')
        if absent.get('child_identity')!=raw.get('child_identity'):raise ValueError('Absence must refer to the same owned child birth')
        if terminal.get('source_manifest_sha256')!=binding['stage1_manifest']['sha256'] or terminal.get('complete_sha256')!=rows['complete']['sha256']:raise ValueError('Original source/completion custody')
        bind(rows['complete'])  # Hash every completion before parsing any selected outcome.
        terminals[identity]=terminal
    records={}
    for identity,rows in spec['records'].items():
        f=bind(rows['complete']);r=json.loads(f.read_text());seed,name=expected[identity]
        if r.get('schema')!='masked-context-PubMed-stage1-complete-v1' or r.get('complete') is not True or (r.get('seed'),r.get('condition'),r.get('record_id'))!=(seed,name,identity):raise ValueError('Exact whole-nine completed identity')
        if r.get('source_manifest_sha256')!=binding['stage1_manifest']['sha256'] or r.get('release_sha256')!=terminals[identity]['release_sha256']:raise ValueError('Original source/release lineage')
        if r.get('split_seed')!=190111 or r.get('split_identity')!='PubMed-class-stratified-floor60-20-20':raise ValueError('Exact original split')
        if r.get('selector')!='first strict maximum factual mean-member-probability pooled VALID accuracy':raise ValueError('Exact original selected-predictor rule')
        if r.get('max_epochs')!=2000 or r.get('patience')!=250 or not 1<=r['selected_epoch']<=r['epochs_executed']<=2000:raise ValueError('Original complete trajectory/selection')
        if r.get('stopped_by')=='patience250':
            if r['epochs_executed']-r['selected_epoch']!=250:raise ValueError('Exact patience closure')
        elif r.get('stopped_by')!='max2000' or r['epochs_executed']!=2000:raise ValueError('No shortened record')
        if r.get('TEST_scored') is not False or r.get('TEST_labels_or_id_inputs') is not False:raise ValueError('Original TEST closure')
        signatures=r.get('selected_prediction_signatures',{})
        if set(signatures)!={'pooled_predictions','member_predictions','pooled_correct_mask','member_correct_masks'} or any(not isinstance(v,str) or len(v)!=64 for v in signatures.values()):raise ValueError('Every original exact signature required')
        saved=r['complete_saved_member_logits'];target=(f.parent/saved['path']).resolve(strict=True)
        expected_keys=['factual_member_logits']+(['owned_masked_member_logits'] if name=='shared4_core' else [])
        if not target.is_relative_to(f.parent) or saved.get('server_only') is not True or saved.get('factual_shape')!=[4,19717,3] or saved.get('keys')!=expected_keys or saved.get('contains_labels_or_role_ids') is not False:raise ValueError('Complete original server-only factual logits')
        bind(dict(path=str(target),sha256=saved['sha256'],bytes=saved['bytes']))
        r['_logits_path']=str(target);records[identity]=r
    if len({(r['train_bundle_sha256'],r['valid_bundle_sha256']) for r in records.values()})!=1:raise ValueError('All nine share exact original roles')
    # Only after all nine terminals/headers/logit payloads are bound, bind VALID.
    valid=bind(spec['valid_bundle']);custody=json.loads(bind(spec['validation_custody']).read_text())
    if custody.get('schema')!='masked-context-PubMed-VALID-role-custody-v1' or custody.get('VALID_count')!=3942 or custody.get('VALID_class_counts')!=[820,1547,1575]:raise ValueError('Exact original projected VALID custody')
    if custody.get('split_seed')!=190111 or custody.get('split_identity')!='PubMed-class-stratified-floor60-20-20' or custody.get('split_protocol')!='class_stratified_floor60_20_20':raise ValueError('VALID split identity')
    if custody.get('valid_bundle_sha256')!=spec['valid_bundle']['sha256'] or any(r['valid_bundle_sha256']!=spec['valid_bundle']['sha256'] for r in records.values()):raise ValueError('Exact original VALID bytes for all nine')
    if custody.get('TEST_labels_loaded') is not False or custody.get('model_constructed') is not False or set(custody.get('array_fingerprints',{}))!={'valid_ids','valid_y'}:raise ValueError('Only projected VALID arrays')
    runtime=spec['runtime'];python=Path(os.path.abspath(runtime['python']['path']))
    if not python.is_relative_to(PHASE) or sha(python)!=runtime['python']['sha256'] or python.resolve()!=Path(sys.executable).resolve() or os.environ.get('PYTHONPATH','')!=runtime['PYTHONPATH']:raise ValueError('Exact existing ordinary runtime')
    owner=json.loads(bind(spec['external_owner_release']).read_text())
    if owner.get('enabled') is not True or owner.get('limits')!=LIMITS or owner.get('automatic_retry') is not False:raise ValueError('Exact reviewed finite diagnostic owner')
    for key in ('separate_process_group','direct_wait_required','resource_caps_enforced','output_and_log_caps_enforced'):
        if owner.get(key) is not True:raise ValueError('Finite owner contract: '+key)
    bind(owner['owner_source']);bind(spec['resource_readiness_evidence'])
    output=inside(spec['output'],existing=False)
    if output.exists() or output.is_relative_to(HERE):raise ValueError('Fresh separate server diagnostic output')
    spec['_validation_custody']=custody;spec['_release_sha256']=digest
    return spec,records,output


def bank_state(bank,ids,labels):
    # Exact original serving operation on the full contiguous FP32 bank.
    probabilities=bank.softmax(-1).mean(0)
    member=bank[:,ids].argmax(-1);pooled=probabilities[ids].argmax(-1)
    member_correct=member==labels[None,:];pool_correct=pooled==labels
    role_logits=bank[:,ids]
    target=role_logits.gather(-1,labels[None,:,None].expand(4,-1,1))
    strict_common_rival=(role_logits>target).all(0)  # True class is never strictly above itself.
    common_wrong_rival=(member==member[0:1]).all(0)&~member_correct.any(0)
    signatures=dict(pooled_predictions=fingerprint(pooled.detach().cpu().numpy()),
        member_predictions=fingerprint(member.detach().cpu().numpy()),
        pooled_correct_mask=fingerprint(pool_correct.detach().cpu().numpy()),
        member_correct_masks=fingerprint(member_correct.detach().cpu().numpy()))
    return dict(member_correct=member_correct.cpu().numpy(),pool_correct=pool_correct.cpu().numpy(),
        strict_rivals=strict_common_rival.cpu().numpy(),unanimous_wrong=common_wrong_rival.cpu().numpy(),
        unanimous_rival_class=member[0].cpu().numpy()),signatures


def count_bank(state,labels,subset):
    import numpy as np
    correct=state['member_correct'][:,subset];pooled=state['pool_correct'][subset]
    k=correct.sum(0);strict=state['strict_rivals'][subset];blocked=strict.any(1)
    unanimous=state['unanimous_wrong'][subset];rival=state['unanimous_rival_class'][subset]
    truth=labels[subset]
    return dict(nodes=int(subset.sum()),correct_member_count_histogram=[int((k==i).sum()) for i in range(5)],
        member_error_counts=[int((~row).sum()) for row in correct],pool_error_count=int((~pooled).sum()),
        any_correct_member_count=int((k>0).sum()),all_correct_members_count=int((k==4).sum()),
        no_correct_member_count=int((k==0).sum()),pool_correct_without_correct_member_count=int((pooled&(k==0)).sum()),
        pool_wrong_with_correct_alternative_count=int((~pooled&(k>0)).sum()),
        unanimous_same_wrong_rival_count=int(unanimous.sum()),
        unanimous_wrong_rival_by_class=[int((unanimous&(rival==c)).sum()) for c in range(3)],
        strict_common_false_over_true_in_every_member_count=int(blocked.sum()),
        strict_common_rival_by_true_and_false_class=[[int((strict[:,f]&(truth==t)).sum()) for f in range(3)] for t in range(3)],
        strict_common_rival_class_count_histogram=[int((strict.sum(1)==i).sum()) for i in range(3)],
        all_wrong_without_strict_common_rival_count=int(((k==0)&~blocked).sum()),
        strict_common_rival_and_FP32_pool_correct_count=int((blocked&pooled).sum()))


def count_pair(candidate,baseline,subset,matched_members=False):
    pool_c,pool_b=candidate['pool_correct'][subset],baseline['pool_correct'][subset]
    any_c=candidate['member_correct'][:,subset].any(0);any_b=baseline['member_correct'][:,subset].any(0)
    strict_c=candidate['strict_rivals'][subset];strict_b=baseline['strict_rivals'][subset]
    row=dict(nodes=int(subset.sum()),pooled_repairs=int((~pool_b&pool_c).sum()),
        pooled_introduced_errors=int((pool_b&~pool_c).sum()),both_pool_wrong=int((~pool_b&~pool_c).sum()),
        correct_alternative_gains=int((~any_b&any_c).sum()),correct_alternative_losses=int((any_b&~any_c).sum()),
        both_no_correct_member=int((~any_b&~any_c).sum()),
        no_correct_member_cleared=int((~any_b&any_c).sum()),no_correct_member_introduced=int((any_b&~any_c).sum()),
        strict_common_rival_cleared=int((strict_b.any(1)&~strict_c.any(1)).sum()),
        strict_common_rival_introduced=int((~strict_b.any(1)&strict_c.any(1)).sum()),
        same_false_rival_persists_by_class=[int((strict_b[:,c]&strict_c[:,c]).sum()) for c in range(3)],
        same_false_rival_cleared_by_class=[int((strict_b[:,c]&~strict_c[:,c]).sum()) for c in range(3)],
        same_false_rival_appeared_by_class=[int((~strict_b[:,c]&strict_c[:,c]).sum()) for c in range(3)])
    if matched_members:
        c,b=candidate['member_correct'][:,subset],baseline['member_correct'][:,subset]
        row['fixed_shared_route_member_repairs']=[int((~b[m]&c[m]).sum()) for m in range(4)]
        row['fixed_shared_route_member_introduced_errors']=[int((b[m]&~c[m]).sum()) for m in range(4)]
    return row


def check_integer_counts(state,labels,record):
    selected=record['selected_VALID']
    if int(state['pool_correct'].sum())!=selected['pooled']['correct'] or selected['pooled']['count']!=3942:raise ValueError('Saved pooled correct count changed')
    if [int(row.sum()) for row in state['member_correct']]!=[m['correct'] for m in selected['members']]:raise ValueError('Saved member correct counts changed')
    for c in range(3):
        chosen=labels==c;row=selected['classes'][c]
        if row['class_id']!=c or row['count']!=int(chosen.sum()) or row['correct']!=int(state['pool_correct'][chosen].sum()) or [int(m[chosen].sum()) for m in state['member_correct']]!=[m['correct'] for m in row['members']]:raise ValueError('Saved class correct counts changed')


def run(spec,records,output,started):
    output.mkdir(parents=True,exist_ok=False);current=None;expected=derived=None
    try:
        import numpy as np
        import torch
        providers=dict(torch=str(torch.__version__),numpy=str(np.__version__))
        for name in ('torch-geometric','scipy','torch-scatter','torch-sparse'):
            try:providers[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:providers[name]=None
        if providers!=spec['frozen_providers']:raise ValueError('Exact original providers; no alternate backend')
        torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats(0)
        with np.load(bind(spec['valid_bundle']),allow_pickle=False) as archive:
            if set(archive.files)!={'valid_ids','valid_y'}:raise ValueError('Only projected VALID arrays')
            valid={key:archive[key].copy() for key in archive.files}
        if {k:fingerprint(v) for k,v in valid.items()}!=spec['_validation_custody']['array_fingerprints']:raise ValueError('Exact original VALID array order/bytes')
        ids,y=valid['valid_ids'],valid['valid_y']
        if ids.dtype!=np.int64 or y.dtype!=np.int64 or ids.shape!=(3942,) or y.shape!=(3942,) or len(np.unique(ids))!=3942 or ids.min()<0 or ids.max()>=19717 or y.min()<0 or y.max()>2 or np.bincount(y,minlength=3).tolist()!=[820,1547,1575]:raise ValueError('Complete original VALID role')
        gpu_ids=torch.from_numpy(ids).to('cuda:0');gpu_y=torch.from_numpy(y).to('cuda:0')
        states={};signatures={}
        with torch.no_grad():
            for current,record in records.items():
                expected=record['selected_prediction_signatures'];derived=None
                if time.monotonic()-started>=300:raise TimeoutError('Frozen diagnostic active cap')
                with np.load(record['_logits_path'],allow_pickle=False) as archive:
                    if set(archive.files)!=set(record['complete_saved_member_logits']['keys']):raise ValueError('Original archive header')
                    logits=archive['factual_member_logits'].copy()  # No masked array/labels/checkpoint loaded.
                if logits.dtype!=np.float32 or logits.shape!=(4,19717,3) or not logits.flags.c_contiguous:raise ValueError('Original contiguous FP32 full factual bank')
                bank=torch.from_numpy(logits).to('cuda:0')
                if not torch.isfinite(bank).all():raise ValueError('Original logits must remain finite')
                state,derived=bank_state(bank,gpu_ids,gpu_y)
                if derived!=expected:raise ValueError('Stored pooled/member prediction or correct-mask signature differs; stop without mismatch chasing')
                check_integer_counts(state,y,record)
                states[current]=state;signatures[current]=derived
                del bank,logits
        all_nodes=np.ones(3942,dtype=bool);subsets={'all_VALID':all_nodes,**{f'class{c}':y==c for c in range(3)}}
        banks={identity:{name:count_bank(state,y,mask) for name,mask in subsets.items()} for identity,state in states.items()}
        paired=[]
        for seed in SEEDS:
            core,own,ind=[states[f'seed{seed}__{c}'] for c in ('shared4_core','shared4_own','independent4_native')]
            row=dict(seed=seed,core_minus_own={name:count_pair(core,own,mask,True) for name,mask in subsets.items()},
                core_minus_independent={name:count_pair(core,ind,mask) for name,mask in subsets.items()},
                independent_minus_own={name:count_pair(ind,own,mask) for name,mask in subsets.items()},
                repair_intersections={})
            for name,mask in subsets.items():
                c,o,i=core['pool_correct'][mask],own['pool_correct'][mask],ind['pool_correct'][mask]
                row['repair_intersections'][name]=dict(core_repairs_both_baselines=int((c&~o&~i).sum()),
                    core_repairs_own_only=int((c&~o&i).sum()),core_repairs_independent_only=int((c&o&~i).sum()),
                    core_introduces_error_against_both=int((~c&o&i).sum()),
                    all_three_pool_wrong=int((~c&~o&~i).sum()))
            paired.append(row)
        torch.cuda.synchronize()
        if time.monotonic()-started>=300:raise TimeoutError('Frozen diagnostic cap before report')
        report=dict(schema='masked-context-complete9-stored-error-counts-v1',complete=True,whole_nine_bound=True,
            banks=banks,paired_seeds=paired,verified_signatures=signatures,
            selected_epochs={identity:r['selected_epoch'] for identity,r in records.items()},
            source_manifest_sha256=spec['source_manifest_sha256'],stage1_source_manifest_sha256=spec['stage1_source_manifest_sha256'],
            release_sha256=spec['_release_sha256'],record_custody=spec['records'],valid_bundle=spec['valid_bundle'],
            providers=providers,new_model_forwards=0,new_fits=0,new_thresholds=0,accuracy_or_NLL_scores_recomputed=False,
            further18_activated=False,TEST_access=False,automatic_retry=False,
            interpretation='All-member argmax wrong does not itself block multiclass convex pooling. A common false class strictly above the true class in every member is a sufficient real-arithmetic obstruction after softmax; its absence does not prove that feasible pooling weights exist. The actual original FP32 pool crossing count is reported without forcing ideal-arithmetic assumptions or changing masks.',
            inclusive_seconds=time.monotonic()-started,CPU_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
            CPU_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_CUDA_bytes=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()),
            owner_success_not_inferred=True,derived_arrays_saved=False)
        write(output/'COUNTS.json',report)
        return report
    except BaseException as error:
        write(output/'FAILURE.json',dict(complete=False,error_type=type(error).__name__,error=str(error),
            current_record=current,expected_signatures=expected,derived_signatures=derived,input_custody=spec['records'],
            inclusive_seconds=time.monotonic()-started,automatic_retry=False,new_forwards=0,TEST_access=False,
            partial_counts_published=False,mismatch_chasing=False))
        raise


def main():
    started=time.monotonic();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('describe','diagnose'),default='describe')
    parser.add_argument('--release',type=Path);parser.add_argument('--release-sha256')
    args=parser.parse_args()
    if args.mode=='describe':print(json.dumps(dict(enabled=False,required_records=9,limits=LIMITS,new_forwards=0,new_thresholds=0)));return
    if args.release is None or args.release_sha256 is None:parser.error('Separate exact root complete-nine diagnostic release')
    spec,records,output=admit(args.release,args.release_sha256);report=run(spec,records,output,started)
    print(json.dumps(dict(complete=report['complete'],output=str(output),new_forwards=0,further18_activated=False)))


if __name__=='__main__':main()
