"""Disabled, whole-family selected WikiCS VALID readout; no fit or owner loop."""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
N = 5274


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def phase_path(relative):
    value = Path(relative)
    require(not value.is_absolute() and '..' not in value.parts, 'Canonical phase-relative path')
    path = (PHASE/value).resolve()
    require(path != PHASE and path.is_relative_to(PHASE), 'Authorized phase only')
    return path


def bound(row):
    path = phase_path(row['path'])
    require(path.is_file() and sha(path) == row['sha256']
        and ('bytes' not in row or path.stat().st_size == row['bytes']), 'Exact bound artifact')
    return path


def sealed(row):
    manifest = bound(row)
    for item in read(manifest)['files']:
        path = (manifest.parent/item['path']).resolve()
        require(path.is_relative_to(manifest.parent) and path.stat().st_size == item['bytes']
            and sha(path) == item['sha256'], 'Immutable source payload')
    return manifest.parent


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.replace(temporary, path)


def admission(release_path, release_sha256):
    require(sha(release_path) == release_sha256, 'Exact root readout release')
    release = read(release_path); pins = read(HERE/'SOURCE_BINDINGS.json')
    require(all(release.get(key) is True for key in ('enabled','root_execution_authorized',
        'source_review_approved','whole9_closure_approved','existing_owner_binding_approved')), 'Disabled until whole-family root admission')
    require(release['readout_manifest_sha256'] == sha(HERE/'MANIFEST.json')
        and release['TEST_access'] is False and release['training_authorized'] is False
        and release['automatic_retry'] is False, 'Exact source, selected VALID only, no fit or retry')
    sealed(dict(path=str((HERE/'MANIFEST.json').relative_to(PHASE)),sha256=release['readout_manifest_sha256']))
    sealed(pins['study_manifest']); sealed(pins['public_manifest']); sealed(pins['live_manifest'])
    for row in pins['source_files']:
        bound(row)
    criteria = read(bound(pins['quality_criteria']))
    require(criteria['whole9_before_comparison'] is True and criteria['TEST_access'] is False
        and [(r['kind'],r['seed']) for r in release['cells']] == [(r['kind'],r['seed']) for r in criteria['roster']], 'Frozen full nine roster and unchanged criteria')
    runtime = pins['runtime']
    require(socket.gethostname() == runtime['hostname'] and Path.cwd().resolve() == Path(runtime['repository'])
        and str(Path(sys.executable).absolute()) == runtime['python'] and os.environ.get('PYTHONPATH','') == ''
        and not os.environ.get('PYTHONHOME') and release['physical_gpu_uuid'] in runtime['physical_gpu_inventory']
        and os.environ.get('CUDA_VISIBLE_DEVICES') == release['physical_gpu_uuid'], 'Original77 CUDA runtime and physical singleton')
    require(all(importlib.metadata.version(name) == runtime[name] for name in
        ('torch','numpy','torch-geometric','torch-scatter','torch-sparse','ogb')), 'Original numerical providers')
    owner_plan = read(bound(release['fit_owner_plan'])); family = read(bound(release['family_complete']))
    require(owner_plan['enabled'] is True and owner_plan['automatic_retry'] is False
        and owner_plan['adapter_manifest_sha256'] == pins['study_manifest']['sha256']
        and owner_plan['existing_run_fit_helper'] == pins['run_fit_helper']
        and owner_plan['existing_ownership_helper'] == pins['ownership_helper']
        and owner_plan['physical_gpu_uuid'] == release['physical_gpu_uuid']
        and family['complete'] is True and family['all9_directly_waited'] is True
        and family['all9_owned_absence_verified'] is True and family['TEST_access'] is False
        and family['adapter_manifest_sha256'] == pins['study_manifest']['sha256']
        and family['scientific_source_manifest_sha256'] == pins['live_manifest']['sha256']
        and family['records'] == [r['cell_id'] for r in release['cells']], 'Actual whole owner closure before checkpoint access')
    require(len(owner_plan['records']) == 9, 'Original full owner roster')
    costs = []
    # Complete ALL metadata/owner checks before hashing or opening any checkpoint.
    for cell, owned in zip(release['cells'], owner_plan['records']):
        require((cell['kind'],cell['seed'],cell['cell_id']) == (owned['kind'],owned['seed'],owned['cell_id'])
            and cell['fit_release'] == owned['release'], 'Exact original owner cell/release')
        fit = read(bound(cell['fit_release'])); complete = read(bound(cell['completion']))
        terminal = read(bound(cell['owner_exit'])); physical = read(bound(cell['physical_terminal']))
        preflight = read(bound(cell['preflight']))
        require(fit['enabled'] is True and fit['root_execution_authorized'] is True
            and fit['adapter_manifest_sha256'] == pins['study_manifest']['sha256']
            and fit['scientific_source_manifest'] == pins['live_manifest']
            and fit['kind'] == cell['kind'] and fit['seed'] == cell['seed'] and fit['epochs'] == 1100
            and fit['train'] == pins['author_train'] and fit['valid'] == pins['author_valid']
            and fit['TEST_access'] is False and fit['automatic_retry'] is False, 'Original source/role/full-fit release')
        require(complete['complete'] is True and complete['epochs'] == complete['steps'] == 1100
            and complete['root_cell_release_sha256'] == cell['fit_release']['sha256']
            and complete['selected_sha256'] == cell['selected']['sha256'] and complete['TEST_scoring'] is False,
            'Actual complete fit and exact selected checksum')
        require(terminal['exit_code'] == 0 and terminal['terminal_wait_observed'] is True
            and terminal['reason'] is None and terminal['signals_sent'] == []
            and terminal['attempts'] == 1 and terminal['retry'] is False
            and terminal['job_sha256'] == cell['fit_release']['sha256']
            and terminal['raw_identity_observation']['argv'] == owned['argv']
            and physical['terminal_wait_observed'] is True and physical['owned_PID_absent'] is True
            and physical['owned_PID_no_CUDA_rows'] is True and physical['physical_gpu_uuid'] == release['physical_gpu_uuid']
            and physical['child_PID'] == terminal['raw_identity_observation']['PID']
            and physical['child_start_ticks'] == terminal['raw_identity_observation']['start_ticks'], 'Exact actual child argv, wait and physical absence')
        root = phase_path(fit['output'])
        require(phase_path(cell['selected']['path']) == root/'selected.pt'
            and phase_path(cell['completion']['path']) == root/'COMPLETE.json'
            and phase_path(cell['trace']['path']) == root/'VALID_TRACE.json'
            and not (root/'FAILURE.json').exists(), 'Original saved selected file; no alternate checkpoint or failure omission')
        costs.append(dict(cell_id=cell['cell_id'], fit_release=cell['fit_release'], completion=cell['completion'],
            owner_exit=cell['owner_exit'], physical_terminal=cell['physical_terminal'], preflight=cell['preflight'],
            driver_seconds=complete['seconds'], adapter_driver_inclusive_seconds=complete['adapter_inclusive_seconds'],
            owner_child_inclusive_seconds=terminal['elapsed_seconds'], resource_wait_seconds=preflight['elapsed_seconds'],
            owner_resource_and_terminal_record=terminal, complete_actual_live_work=complete['actual_live_work'],
            parameter_counts=complete['parameter_counts'],selected_artifact=cell['selected']))
    supervision = read(bound(release['external_supervision'])); limits = supervision['resource_limits']
    require(supervision['enabled'] is True and supervision['root_owns_finite_launch_and_actual_costs'] is True
        and supervision['existing_run_fit_helper'] == pins['run_fit_helper']
        and supervision['existing_ownership_helper'] == pins['ownership_helper']
        and supervision['physical_gpu_uuid'] == release['physical_gpu_uuid']
        and supervision['entry_program'] == dict(path=str(Path(__file__).resolve().relative_to(PHASE)),bytes=Path(__file__).stat().st_size,sha256=sha(__file__))
        and supervision['argv_prefix'] == [runtime['python'],'-B',str(Path(__file__).resolve())]
        and supervision['release_argument_path'] == str(Path(release_path).resolve()) and supervision['output'] == release['output']
        and 0 < supervision['active_seconds'] < supervision['hard_seconds']
        and supervision['hard_seconds'] == supervision['active_seconds']+supervision['cleanup_seconds']
        and supervision['cleanup_seconds'] > 0 and release['owned_GPU_allocator_cap_bytes'] == 64*1024**3
        and limits['owned_tree_GPU_memory_cap_bytes'] == 68*1024**3
        and limits['minimum_fresh_GPU_free_bytes'] >= 72*1024**3
        and limits['owned_tree_RSS_cap_bytes'] > 0 and limits['own_fit_output_cap_bytes'] > 0
        and 0 < limits['combined_child_log_cap_bytes'] <= 8*1024**2, 'Existing finite readout owner; no release/SHA cycle')
    output = phase_path(release['output'])
    forbidden = [HERE,sealed(pins['study_manifest']),sealed(pins['public_manifest']),sealed(pins['live_manifest']),
        phase_path(pins['author_train']['path']).parent,*[phase_path(c['selected']['path']).parent for c in release['cells']]]
    require(not output.exists() and output.parent.is_dir() and all(not output.is_relative_to(p) for p in forbidden), 'Fresh separate readout output')
    return release,pins,criteria,family,costs,output


def ids(mask, objects):
    return objects[mask].tolist()


def score(session, batch, labels, state):
    torch = session.torch
    require(session.device.type == 'cuda' and not session.model.training and tuple(labels.shape) == (N,), 'Original CUDA selected eval and full VALID labels')
    with torch.no_grad():
        logits, representation = session.forward(batch); del representation
        pooled = session.serving(logits)
        session.core['selection'].finite_predictions(logits, pooled)
        require(tuple(logits.shape) == (4,N,10) and tuple(pooled.shape) == (N,10), 'Complete selected VALID logits only')
        y = labels.to(session.device); member_pred = logits.argmax(-1); pool_pred = pooled.argmax(-1)
        mc = member_pred == y; pc = pool_pred == y; coverage = mc.any(0)
        lp = logits.double().log_softmax(-1)
        member_loss = -lp.gather(-1,y.view(1,-1,1).expand(4,-1,1)).squeeze(-1)
        pool_loss = -(torch.logsumexp(lp,dim=0)-math.log(4)).gather(-1,y[:,None]).squeeze(-1)
        require(torch.isfinite(member_loss).all() and torch.isfinite(pool_loss).all(), 'Finite unclipped float64 proper losses')
        member_counts = mc.sum(1).tolist(); pool_count = int(pc.sum())
        member_acc = [100*x/N for x in member_counts]
        metrics = dict(pool_correct=pool_count,pool_accuracy_percent=100*pool_count/N,member_correct=member_counts,
            member_accuracy_percent=member_acc,mean_member_accuracy_percent=sum(member_acc)/4,worst_member_accuracy_percent=min(member_acc),
            pool_NLL=float(pool_loss.mean()),member_NLL=member_loss.mean(1).tolist(),mean_member_NLL=float(member_loss.mean()),
            coverage=int(coverage.sum()),pooling_losses=int((coverage & ~pc).sum()),pooled_only_rescues=int((~coverage & pc).sum()))
        classes = []
        for label in range(10):
            mask = y == label; size = int(mask.sum())
            classes.append(dict(label=label,population=size,pool_correct=int(pc[mask].sum()),
                pool_accuracy_percent=100*int(pc[mask].sum())/size if size else None,
                member_correct=mc[:,mask].sum(1).tolist(),pool_NLL=float(pool_loss[mask].mean()) if size else None))
        original = dict(epoch=state['epoch'],global_mode=state['global'],pool_accuracy_percent=100*state['selected_VALID'],
            member_accuracy_percent=[100*x for x in state['member_VALID']],
            pool_correct_recovered_from_TRACE_float=round(state['selected_VALID']*N),
            member_correct_recovered_from_TRACE_float=[round(x*N) for x in state['member_VALID']])
        differences = [pool_count-original['pool_correct_recovered_from_TRACE_float']]+[
            a-b for a,b in zip(member_counts,original['member_correct_recovered_from_TRACE_float'])]
        pool_margin = pooled.topk(2,-1).values; pool_margin = pool_margin[:,0]-pool_margin[:,1]
        member_margin = logits.softmax(-1).topk(2,-1).values
        member_margin = member_margin[:,:,0]-member_margin[:,:,1]
        arrays = dict(ids=batch['ids'].cpu(),y=y.cpu(),member_correct=mc.cpu(),pool_correct=pc.cpu(),coverage=coverage.cpu(),
            common_rival=((member_pred == member_pred[0:1]).all(0) & ~coverage & ~pc).cpu())
        payload = dict(schema='selected-full-VALID-node-readout-v1',ids=arrays['ids'],labels=arrays['y'],
            member_logits=logits.cpu(),served_probability=pooled.cpu(),member_prediction=member_pred.cpu(),pool_prediction=pool_pred.cpu(),
            member_probability_top2_margin=member_margin.cpu(),pool_probability_top2_margin=pool_margin.cpu())
        low = torch.argsort(pool_margin)[:10]
        replay = dict(pool_then_four_member_correct_count_deltas=differences,
            meaningful_discrepancy=max(abs(x) for x in differences)>1,
            one_node_difference_is_uncertainty=any(differences) and max(abs(x) for x in differences)<=1,
            smallest_pool_margins=[dict(id=int(batch['ids'][i]),margin=float(pool_margin[i])) for i in low])
        return metrics,classes,original,replay,arrays,payload


def pair(before, after):
    require(before['ids'].equal(after['ids']) and before['y'].equal(after['y']), 'Exact paired VALID IDs and truth')
    objects=before['ids']; old,new=before['pool_correct'],after['pool_correct']
    acquisition=~before['coverage'] & after['coverage']; lost=before['coverage'] & ~after['coverage']
    masks=dict(acquired_coverage=acquisition,lost_old_coverage=lost,acquired_alternatives_served=acquisition & new,
        acquired_alternatives_lost_by_pool=acquisition & ~new,repairs=~old & new,harms=old & ~new,
        common_success=old & new,pooled_only_rescues=~after['coverage'] & new,
        reference_common_false_rival_repairs=before['common_rival'] & new,
        pooled_only_actual_repairs=~old & new & ~after['coverage'])
    result={name:dict(count=int(mask.sum()),ids=ids(mask,objects)) for name,mask in masks.items()}
    require(int(new.sum())-int(old.sum()) == result['repairs']['count']-result['harms']['count'], 'Exact correct-count reconciliation')
    old_lost=int((before['coverage'] & ~old).sum());new_lost=int((after['coverage'] & ~new).sum())
    old_only=int((~before['coverage'] & old).sum());new_only=int((~after['coverage'] & new).sum())
    require(int(new.sum())-int(old.sum()) == int(acquisition.sum())-int(lost.sum())-(new_lost-old_lost)+(new_only-old_only), 'Exact coverage/pooling-loss/pooled-only flow reconciliation')
    result['pooling_flow']=dict(before_pooling_losses=old_lost,after_pooling_losses=new_lost,
        before_pooled_only_correct=old_only,after_pooled_only_correct=new_only)
    result['classes']=[dict(label=k,population=int((before['y']==k).sum()),
        pool_correct_delta=int(new[before['y']==k].sum())-int(old[before['y']==k].sum()),
        pool_accuracy_delta_pp=100*(int(new[before['y']==k].sum())-int(old[before['y']==k].sum()))/int((before['y']==k).sum()) if int((before['y']==k).sum()) else None,
        member_correct_delta=(after['member_correct'][:,before['y']==k].sum(1)-before['member_correct'][:,before['y']==k].sum(1)).tolist(),
        repair_ids=ids(masks['repairs'] & (before['y']==k),objects),harm_ids=ids(masks['harms'] & (before['y']==k),objects)) for k in range(10)]
    result['members']=[dict(member=m,correct_delta=int(after['member_correct'][m].sum())-int(before['member_correct'][m].sum()),
        repair_ids=ids(~before['member_correct'][m] & after['member_correct'][m],objects),
        harm_ids=ids(before['member_correct'][m] & ~after['member_correct'][m],objects)) for m in range(4)]
    return result


def run(release_path, release_sha256):
    started=time.monotonic(); release,pins,criteria,family,costs,output=admission(release_path,release_sha256)
    output.mkdir(); work=dict(selected_load_attempts=0,selected_load_completions=0,constructor_attempts=0,
        constructor_completions=0,VALID_joint_forward_attempts=0,VALID_joint_forward_completions=0,training_updates=0)
    write(output/'FIT_COSTS.json',dict(cells=costs,family_owner_inclusive_seconds=family['owner_seconds'],
        aggregation='Worker/adapter/child-owner/family-owner times are nested; do not add them. Resource waits are separately identified.',
        owner_closure=release['family_complete'],fit_owner_plan=release['fit_owner_plan']))
    (output/'FROZEN_QUALITY_CRITERIA.json').write_bytes(bound(pins['quality_criteria']).read_bytes())
    rows=[]; arrays={}; old_path=list(sys.path); session=None
    def progress(stage,cell=None):
        write(output/'READOUT_PROGRESS.json',dict(stage=stage,cell_id=cell,actual_work=work,
            elapsed_seconds=time.monotonic()-started,last_observed_native_work=copy.deepcopy(session.live_work) if session else None))
    try:
        spec=importlib.util.spec_from_file_location('_complete9_study_adapter',bound(pins['study_adapter']))
        adapter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=adapter;spec.loader.exec_module(adapter)
        adapter_pins=adapter.source_gate(); public,_driver,_live=adapter.dependencies(adapter_pins)
        data=adapter.module('data_interface',bound(pins['data_interface']))
        train,valid,origin=data.load_train_valid('wikics',bound(pins['author_train']),bound(pins['author_valid']))
        import torch
        torch.cuda.set_device(0);total=torch.cuda.get_device_properties(0).total_memory
        require(release['owned_GPU_allocator_cap_bytes']<=total,'Declared CUDA allocator cap')
        torch.cuda.set_per_process_memory_fraction(release['owned_GPU_allocator_cap_bytes']/total,0)
        # Freeze each baseline's declared Q before any candidate prediction is opened.
        ordered=sorted(release['cells'],key=lambda c:(c['kind']!='baseline',c['seed'],c['kind']))
        for cell in ordered:
            label=cell['cell_id']; trace=read(bound(cell['trace']))
            require(len(trace)==1100 and [x['epoch'] for x in trace]==list(range(1,1101)), 'Complete immutable original TRACE')
            first_max=max(trace,key=lambda x:x['VALID_selector'])
            work['selected_load_attempts']+=1;progress('selected_load_attempt',label)
            selected=bound(cell['selected']);state=torch.load(selected,map_location='cpu',weights_only=False)
            work['selected_load_completions']+=1
            require(state['epoch']==first_max['epoch'] and state['selected_VALID']==first_max['VALID_selector']
                and state['member_VALID']==first_max['members'] and state['run']['seed']==cell['seed']
                and state['global']==(state['epoch']>100) and state['live_route_study']['kind']==cell['kind']
                and state['run']['arm']=='live_route_native_init_v3__'+cell['kind']
                and state['run']['data']['train_npz_sha256']==pins['author_train']['sha256']
                and state['run']['data']['valid_npz_sha256']==pins['author_valid']['sha256'], 'Original first strict maximum and exact source roles; no reselection')
            work['constructor_attempts']+=1;progress('selected_reconstruction_attempt',label)
            session=adapter.reconstruct_selected(state,'cuda:0');work['constructor_completions']+=1
            batches=list(data.validation_batches('wikics',train,valid,'cuda:0'));require(len(batches)==1,'One complete VALID batch')
            batch,truth=batches[0];work['VALID_joint_forward_attempts']+=1;progress('VALID_forward_attempt',label)
            metrics,classes,original,replay,a,payload=score(session,batch,truth,state)
            work['VALID_joint_forward_completions']+=1
            require(sha(selected)==cell['selected']['sha256'],'Selected checkpoint unchanged during readout')
            payload.update(selected=cell['selected'],source_manifest=pins['study_manifest'],scientific_source=pins['live_manifest'])
            artifact=output/(label+'.VALID.pt');torch.save(payload,artifact)
            rows.append(dict(cell_id=label,kind=cell['kind'],seed=cell['seed'],selected=cell['selected'],original_TRACE=original,
                measured_replay=metrics,classes=classes,replay_difference=replay,actual_native_work=copy.deepcopy(session.live_work),
                VALID_artifact=dict(path=str(artifact.relative_to(PHASE)),bytes=artifact.stat().st_size,sha256=sha(artifact))))
            arrays[(cell['seed'],cell['kind'])]=a
            if len(rows)==3:
                write(output/'BASELINE_Q_FROZEN.json',dict(definition='All four baseline argmax predictions agree on a false class; served pool also false.',
                    frozen_before_candidate_readout=True,seeds={str(s):ids(arrays[(s,'baseline')]['common_rival'],arrays[(s,'baseline')]['ids']) for s in (6101,6203,6307)},
                    genuine_I4_coverage_available=False,outside_I4_acquisition_unmeasured=True))
            del session,state,payload,batch,truth,batches;session=None;gc.collect();torch.cuda.empty_cache();progress('cell_complete',label)
        require(work['selected_load_completions']==work['constructor_completions']==work['VALID_joint_forward_completions']==9,'Full nine selected readouts')
        blocked=any(r['replay_difference']['meaningful_discrepancy'] for r in rows)
        paired=[];overlaps=[]
        if not blocked:
            for seed in (6101,6203,6307):
                for before,after in (('baseline','exchange'),('baseline','separable'),('separable','exchange')):
                    result=pair(arrays[(seed,before)],arrays[(seed,after)])
                    old=next(r for r in rows if r['seed']==seed and r['kind']==before)['measured_replay']
                    new=next(r for r in rows if r['seed']==seed and r['kind']==after)['measured_replay']
                    old_recorded=next(r for r in rows if r['seed']==seed and r['kind']==before)['original_TRACE']
                    new_recorded=next(r for r in rows if r['seed']==seed and r['kind']==after)['original_TRACE']
                    result.update(seed=seed,contrast=after+'-'+before,pool_accuracy_delta_pp=new['pool_accuracy_percent']-old['pool_accuracy_percent'],
                        pool_NLL_delta=new['pool_NLL']-old['pool_NLL'],mean_member_delta_pp=new['mean_member_accuracy_percent']-old['mean_member_accuracy_percent'],
                        worst_member_delta_pp=new['worst_member_accuracy_percent']-old['worst_member_accuracy_percent'],
                        worst_paired_member_delta_pp=min(a-b for a,b in zip(new['member_accuracy_percent'],old['member_accuracy_percent'])),
                        original_TRACE_pool_accuracy_delta_pp=new_recorded['pool_accuracy_percent']-old_recorded['pool_accuracy_percent'],
                        original_TRACE_mean_member_delta_pp=sum(new_recorded['member_accuracy_percent'])/4-sum(old_recorded['member_accuracy_percent'])/4,
                        original_TRACE_worst_member_delta_pp=min(new_recorded['member_accuracy_percent'])-min(old_recorded['member_accuracy_percent']),
                        reference_Q_condition=before)
                    paired.append(result)
                b,x,s=(arrays[(seed,k)] for k in ('baseline','exchange','separable'))
                xr=~b['pool_correct'] & x['pool_correct'];sr=~b['pool_correct'] & s['pool_correct']
                overlaps.append(dict(seed=seed,common_repair_ids=ids(xr & sr,b['ids']),exchange_exclusive_repair_ids=ids(xr & ~sr,b['ids']),
                    separable_exclusive_repair_ids=ids(sr & ~xr,b['ids']),interpretation='Repair overlap between two conditions; no A+B interaction was measured.'))
        persistent={}
        for contrast in ('exchange-baseline','separable-baseline','exchange-separable'):
            sets=[set(r['repairs']['ids']) for r in paired if r['contrast']==contrast]
            persistent[contrast]=sorted(set.intersection(*sets)) if len(sets)==3 else None
        write(output/'SELECTED_VALID_METRICS.json',dict(cells=rows,original_TRACE_unchanged=True,
            NLL_definition='Float64 log-softmax of preserved float32 logits; unclipped log-mean member probability. Argmax uses native float32 served probability.',
            selected_development_reused=True,TEST_access=False))
        mean_deltas={contrast:{field:sum(r[field] for r in paired if r['contrast']==contrast)/3 for field in
            ('pool_accuracy_delta_pp','pool_NLL_delta','mean_member_delta_pp','worst_member_delta_pp','original_TRACE_pool_accuracy_delta_pp',
             'original_TRACE_mean_member_delta_pp','original_TRACE_worst_member_delta_pp')} for contrast in persistent} if not blocked else {}
        write(output/'PAIRED_ERRORS.json',dict(paired=paired,paired_mean_deltas=mean_deltas,repair_overlap=overlaps,persistent_repair_ids_all_three_seeds=persistent,
            interpretation_blocked=blocked,reason='More than one correct-count difference from original TRACE in a pool/member' if blocked else None,
            one_node_uncertainties_retained=True,genuine_I4_coverage_available=False,outside_I4_claims_authorized=False))
        write(output/'COMPLETE.json',dict(complete=True,selected_readouts=9,actual_work=work,interpretation_blocked=blocked,
            root_release_sha256=release_sha256,source_manifest_sha256=sha(HERE/'MANIFEST.json'),
            study_manifest=pins['study_manifest'],scientific_source=pins['live_manifest'],quality_criteria=pins['quality_criteria'],
            elapsed_seconds=time.monotonic()-started,max_reserved_GPU_bytes=torch.cuda.max_memory_reserved(0),
            screen_decision_issued=False,TEST_access=False,training_updates=0,novelty_or_superiority_claimed=False))
    except BaseException as error:
        write(output/'FAILURE.json',dict(complete=False,error_type=type(error).__name__,error=str(error),actual_work=work,
            observed_cells=rows,elapsed_seconds=time.monotonic()-started,automatic_retry=False,TEST_access=False))
        raise
    finally:
        sys.path[:]=old_path


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args();run(args.release,args.release_sha256)
