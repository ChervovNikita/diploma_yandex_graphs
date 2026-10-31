"""Executable fixed Citeseer growth qualification/cost/pilot; disabled by default."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parent
PROVIDERS = {
    '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs':
        ('shared_backbone_private_transfer_training_source_20261005_v2','05649f101ca8ddc5bd549c3ece48a93752a934d68cc1ea3da58b8353495235c4'),
    '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git':
        ('citeseer_known_ranking_control_gpu77_b0_preparation_20261006_v1','b400c9fdd5ed885a3538a1af4e0214f24056149e135b3e5fcb1f8c37ff81d46a')}
INITIALIZER = 'citeseer_gnnm_initialization_pilot_preparation_20261006_v1/initialize.py'
INITIALIZER_SHA = '64c3a064336d523e1dabd968e83a4938c0e5cfaf263cb4e3b6536ceb36adc7bb'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name,path):
    path = Path(path).resolve(strict=True)
    if name in sys.modules and Path(sys.modules[name].__file__).resolve() != path:
        raise ValueError('Pinned growth dependency shadowed: '+name)
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def authorize(args):
    supplied = json.loads(args.job.read_text())
    relative,expected = PROVIDERS[supplied['repository']]
    phase = Path(supplied['repository'])/'experiments_iclr/postsubmission_20260930'
    provider = phase/relative
    if sha(provider/'custody.py') != expected: raise ValueError('Pinned provider custody changed')
    custody = load('growth_provider_custody',provider/'custody.py')
    manifest = ROOT/'SOURCE_MANIFEST.json'
    if sha(manifest) != supplied.get('growth_source_manifest_sha256'):
        raise ValueError('Growth source manifest binding changed')
    for row in json.loads(manifest.read_text())['files']:
        path = (ROOT/row['path']).resolve(strict=True)
        if not path.is_relative_to(ROOT) or sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Growth source bytes changed: '+row['path'])
    for row in json.loads((ROOT/'INPUT_BINDINGS.json').read_text())['files']:
        path = custody.phase_file(row['path'])
        if sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Growth input/source bytes changed: '+row['path'])
    phase_name = supplied.get('phase')
    purposes = {'qualify':'GNNM_growth_TRAIN_only_qualification',
                'cost':'GNNM_growth_complete_TRAIN_cycle_cost',
                'fit':'GNNM_growth_TRAIN_VALID_fit'}
    if phase_name not in purposes: raise ValueError('Explicit growth phase required')
    # Provider verifies exact host/GPU/runtime authorities and its own unchanged
    # source manifest; the additional growth manifest was checked above.
    job,output = custody.authorize(__file__,args,purposes[phase_name])
    plan_path = custody.phase_file(job['plan_relative'])
    if sha(plan_path) != job['plan_sha256']: raise ValueError('Root-bound growth plan changed')
    plan = json.loads(plan_path.read_text())
    if plan.get('root_adopted') is not True or plan.get('TEST_closed') is not True:
        raise ValueError('Root must adopt the prospective growth plan')
    if (plan['warm_cycles'],plan['post_cycles'],plan['eval_every'],job['outer_size'],job['inner_size']) != (20,60,5,64,256):
        raise ValueError('Fixed native warm/post/exposure horizon changed')
    if [row['seed'] for row in plan['blocks']] != [0,1,2]: raise ValueError('Fixed three seeds required')
    match = [row for row in plan['blocks'] if (row['seed'],row['factor_seed']) == (job['seed'],job['factor_seed'])]
    if len(match) != 1: raise ValueError('Fixed paired seed/factor identity changed')
    if job.get('VALID_values_access') is not (phase_name == 'fit') or job.get('fits_authorized') is not (phase_name == 'fit'):
        raise ValueError('Only adopted fit may load VALID or run full training')
    if phase_name != 'qualify' and job.get('optimizer_updates_authorized') is not True:
        raise ValueError('Complete cost/fit optimizer work requires explicit root adoption')
    growth = load('native_preaggregation_growth',ROOT/'growth.py')
    if phase_name != 'qualify' and job.get('condition') not in growth.CONDITIONS:
        raise ValueError('Exact fixed growth condition required')
    if phase_name != 'qualify' and job['condition'] == 'no_growth':
        raise ValueError('No-growth warmidentity uses retained source/results; no duplicate fit/cost here')
    if phase_name == 'fit':
        for key,purpose in [('qualification_receipt','qualify'),('complete_cycle_cost_receipt','cost')]:
            record = job.get(key,{})
            receipt = custody.phase_file(record['path'])
            if sha(receipt) != record.get('sha256'): raise ValueError('Growth admission receipt changed: '+key)
            value = json.loads(receipt.read_text())
            if value.get('phase') != purpose or value.get('success') is not True or value.get('growth_source_manifest_sha256') != job['growth_source_manifest_sha256']:
                raise ValueError('Exact growth qualification/cost source receipt required')
            if purpose == 'qualify' and value.get('seed') != job['seed']:
                raise ValueError('Same-seed actual calibration/growth gate required')
            if purpose == 'cost' and (value.get('condition') != job['condition'] or value.get('complete_cycles') != 1):
                raise ValueError('Same-arm complete-cycle cost receipt required')
    return custody,job,output,plan,growth


def own_loss(torch,p,n):
    return -torch.nn.functional.logsigmoid(p).mean()-torch.nn.functional.logsigmoid(-n).mean()


def ordinary_step(torch,steps,model,optimizer,x,support,inner,outer,streams):
    calls = [(0,(torch.cat([q[0] for q in inner],1),torch.cat([q[1] for q in inner],1)))] if model.member_count == 1 else list(enumerate(inner))
    values = []
    for pass_index in range(3):
        optimizer.zero_grad(set_to_none=True)
        parameters = dict(model.named_parameters())
        if pass_index == 1:
            p,n = steps.functional_forward(torch,model,parameters,x,support,outer,training=False)
            value = (own_loss(torch,p,n) if model.member_count == 1 else
                     .5*own_loss(torch,p.mean(1),n.mean(1))+.5*own_loss(torch,p,n))
        else:
            losses = []
            for member,queries in calls:
                p,n = steps.functional_forward(torch,model,parameters,x,support,queries,
                    training=True,stream=(streams,member),route=member,advance=pass_index == 2)
                losses.append(own_loss(torch,p,n))
            value = torch.stack(losses).mean()
        if not bool(torch.isfinite(value)): raise FloatingPointError('Nonfinite fixed native task objective')
        value.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError('Every base/private/growth parameter must remain connected and finite')
        optimizer.step(); values.append(float(value.detach()))
    return values


def metric(torch,p,n):
    rank = 1+.5*((n >= p[:,None]).sum(1)+(n > p[:,None]).sum(1))
    return {'MRR':round((1/rank.float()).mean().item(),4),'Hits10':round((rank <= 10).float().mean().item(),4)}


def checkpoint(torch,model,optimizers,streams,endpoint,random,job):
    return {'model':{n:v.detach().cpu() for n,v in model.state_dict().items()},
        'optimizers':[o.state_dict() for o in optimizers],
        'dropout_streams':[s.state_dict() for s in streams],
        'torch_RNG':{'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()},
        'endpoint_sampling':[[{k:v.getstate() for k,v in row.items()} for row in rows] for rows in endpoint],
        'random_sampling':[[{k:v.getstate() for k,v in row.items()} for row in rows] for rows in random],
        'source_configuration':job,'postwarm_optimizer_dropout_reset_explicit':True}


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    custody,job,output,plan,growth = authorize(args)
    output.mkdir(); custody.write_json(output/'START.json',{'job_sha256':sha(args.job),'phase':job['phase'],'TEST_access':False})
    try:
        torch,device,versions = custody.runtime(job)
        torch.cuda.reset_peak_memory_stats()
        native,heads,geometry,cycle,adjoint,models,steps = custody.load_sources()
        initializer_path = custody.phase_file(INITIALIZER)
        if sha(initializer_path) != INITIALIZER_SHA: raise ValueError('Warm lift source changed')
        initializer = custody.load_module('growth_original_warm_initializer',initializer_path)
        gates = load('native_growth_gates',ROOT/'gates.py')
        x,train,valid,pool,identities = custody.load_inputs(torch,job,include_valid=job['phase'] == 'fit')
        x = x.to(device)
        warm,_,_,_ = models.build(native,heads,arm='capable_single',seed=job['seed'],factor_seed=job['factor_seed'],device=device)
        frozen_path = custody.phase_file(job['warm_freeze_relative'])
        if sha(frozen_path) != job['warm_freeze_sha256']: raise ValueError('Bound common warm freeze changed')
        frozen = json.loads(frozen_path.read_text())
        if frozen['phase'] != 'warm' or frozen['completed_cycles'] != 20 or frozen['seed'] != job['seed'] or frozen['VALID_TEST_access'] is not False:
            raise ValueError('Require exact terminal20 TRAIN-only warm state')
        warm_path = frozen_path.parent/'warm_checkpoint.pt'
        if sha(warm_path) != frozen['checkpoint_sha256']: raise ValueError('Common warm checkpoint changed')
        state = torch.load(warm_path,map_location='cpu',weights_only=True)
        if state['seed'] != job['seed'] or any(state['inputs'][k] != identities[k] for k in ('train_pos.txt','gnn_feature')):
            raise ValueError('Common warm input custody differs')
        warm.load_state_dict(state['model'],strict=True); del state

        # Fixed first complete-cycle episode. No model-dependent draw or retry.
        negative,order,episodes = custody.make_pair(torch,geometry,cycle,train,len(x),job['seed'],0,64,256,
            geometry.route_streams(job['seed'],4),geometry.route_streams(job['seed'],4,control=True))
        if len(order) != 3870 or len(episodes) != 61: raise ValueError('Full TRAIN episode cycle required')
        endpoint,random_episode,kept,description = episodes[0]
        calibration_support = custody.support_tensor(torch,train,kept,len(x),device)
        _,queries = custody.queries(torch,train,negative,endpoint,device)
        calibration_draw = {'sampling_seed':job['seed'],'cycle_index':0,'episode_index':0,
            'negative_bank':negative.tolist(),'outer_order':order,'endpoint':endpoint,
            'matched_random':random_episode,'kept_positive_ids':kept,'description':description,
            'target_mask':'endpoint and matched-random positive union, exactly provider geometry',
            'labels':'TRAIN only; no VALID/TEST or predicted pseudo-labels'}
        custody.write_json(output/'CALIBRATION_DRAW.json',calibration_draw)
        torch.cuda.synchronize(); calibration_started = time.monotonic()
        h,g,calibration = growth.calibrate(torch,warm,x,calibration_support,queries)
        required = (None if job['phase'] == 'qualify' else 'single' if job['condition'] == 'capable_single_rank8'
                    else 'unfiltered' if job['condition'] == 'unfiltered_growth' else 'graph')
        basis,basis_record = growth.bases(torch,warm.encoder,h,g,calibration_support,required=required)
        torch.cuda.synchronize(); calibration_seconds = time.monotonic()-calibration_started
        custody.write_json(output/'CALIBRATION.json',{**calibration,'bases':basis_record,
            'draw_sha256':sha(output/'CALIBRATION_DRAW.json'),'seconds':calibration_seconds,
            'warm_checkpoint_sha256':sha(warm_path),'inputs':identities})
        torch.save({k:v.detach().cpu() for k,v in basis.items()},output/'INITIAL_BASES.pt')
        del h,g,negative,episodes

        if job['phase'] == 'qualify':
            results = {}
            for condition in growth.CONDITIONS:
                model,partition = growth.make(torch,native,heads,models,initializer,warm,condition,job['seed'],job['factor_seed'],device,basis)
                old_condition = ('single' if condition == 'capable_single_rank8' else
                    'independent_warm4' if condition == 'independent_graph_growth4' else 'warm_identity')
                reference,_ = initializer.make(torch,native,heads,models,warm,old_condition,job['seed'],job['factor_seed'],device,{})
                results[condition] = gates.qualify(torch,steps,model,reference,warm,x,calibration_support,queries,job['seed'],device)
                results[condition]['partition'] = partition
                del model,reference
                if time.monotonic()-started > job['soft_seconds']:
                    raise TimeoutError('Fixed native qualification cap exceeded; no retry')
            custody.write_json(output/'NATIVE_GATES.json',results)
            custody.write_json(output/'FREEZE.json',{'phase':'qualify','success':True,'seed':job['seed'],
                'growth_source_manifest_sha256':job['growth_source_manifest_sha256'],
                'gates_sha256':sha(output/'NATIVE_GATES.json'),'calibration_sha256':sha(output/'CALIBRATION.json'),
                'optimizer_updates':0,'VALID_TEST_access':False,'inclusive_seconds':time.monotonic()-started,'runtime':versions,
                'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_CUDA_reserved_bytes':torch.cuda.max_memory_reserved(),
                'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
            return

        model,partition = growth.make(torch,native,heads,models,initializer,warm,job['condition'],job['seed'],job['factor_seed'],device,basis)
        del calibration_support,queries,basis,warm
        custody.write_json(output/'PARAMETER_PARTITION.json',partition)
        independent = job['condition'] == 'independent_graph_growth4'
        routes = list(model.routes) if independent else [model]
        seeds = [job['seed']+5*m for m in range(4)] if independent else [job['seed']]
        optimizers = [torch.optim.Adam(list(route.parameters()),lr=.001,betas=(.9,.999),eps=1e-8,
                       weight_decay=0.,foreach=False,fused=False) for route in routes]
        optimizer_ids = [{id(p) for group in opt.param_groups for p in group['params']} for opt in optimizers]
        if any(optimizer_ids[a]&optimizer_ids[b] for a in range(len(optimizers)) for b in range(a+1,len(optimizers))):
            raise ValueError('Independent optimizer ownership overlaps')
        streams = [steps.DropoutStreams(torch,device,seed+200000) for seed in seeds]
        endpoint_streams = [geometry.route_streams(seed,4) for seed in seeds]
        random_streams = [geometry.route_streams(seed,4,control=True) for seed in seeds]
        aggregation = {'calls':0,'column_histogram':{}}
        def count(module,args,result):
            width = str(args[0].shape[1]); aggregation['calls'] += 1
            aggregation['column_histogram'][width] = aggregation['column_histogram'].get(width,0)+1
        handles = [route.encoder.convs[0].register_forward_hook(count) for route in routes]
        full_support = custody.support_tensor(torch,train,list(range(len(train))),len(x),device)
        counters = {'episodes':0,'Adam_updates':0,'complete_train_cycles_per_member':0}
        torch.save(checkpoint(torch,model,optimizers,streams,endpoint_streams,random_streams,job),output/'INITIAL_STATE.pt')
        cycles = 60 if job['phase'] == 'fit' else 1
        selected,selected_cycle = None,None
        cycle_seconds = []
        with (output/'HISTORY.jsonl').open('x') as history,gzip.open(output/'CYCLE_DRAWS.jsonl.gz','wt',encoding='utf-8') as draws:
            for cycle_index in range(cycles):
                torch.cuda.synchronize(); cycle_started = time.monotonic()
                for member,(route,seed) in enumerate(zip(routes,seeds)):
                    negative,order,episodes = custody.make_pair(torch,geometry,cycle,train,len(x),seed,cycle_index,64,256,endpoint_streams[member],random_streams[member])
                    if len(order) != 3870 or len(episodes) != 61: raise ValueError('No truncated TRAIN cycle permitted')
                    draws.write(json.dumps({'cycle':cycle_index+1,'member':member,'seed':seed,'negative_bank':negative.tolist(),
                        'outer_order':order,'episodes':[{'endpoint':e,'matched_random':r,'kept_positive_ids':k} for e,r,k,_ in episodes]})+'\n'); draws.flush()
                    for endpoint,_,kept,_ in episodes:
                        support = custody.support_tensor(torch,train,kept,len(x),device)
                        inner,outer = custody.queries(torch,train,negative,endpoint,device)
                        ordinary_step(torch,steps,route,optimizers[member],x,support,inner,outer,streams[member])
                        counters['episodes'] += 1; counters['Adam_updates'] += 3
                        if time.monotonic()-started > job['soft_seconds']:
                            raise TimeoutError('Fixed complete-cycle cap exceeded; no shortening/retry')
                        del support,inner,outer
                counters['complete_train_cycles_per_member'] += 1
                torch.cuda.synchronize(); cycle_seconds.append(time.monotonic()-cycle_started)
                row = {'cycle':cycle_index+1,'cycle_seconds':cycle_seconds[-1],'counters':dict(counters),'native_encoder_P':json.loads(json.dumps(aggregation))}
                if job['phase'] == 'fit' and (cycle_index+1)%5 == 0:
                    with torch.no_grad():
                        p,n = steps.functional_forward(torch,model,dict(model.named_parameters()),x,full_support,
                            (valid.t().to(device),pool.permute(2,0,1).reshape(2,-1).to(device)),training=False)
                        n = n.reshape(227,500,model.member_count)
                        result = metric(torch,p.mean(1),n.mean(2)); row['complete_VALID'] = result
                        row['members'] = [metric(torch,p[:,m],n[:,:,m]) for m in range(model.member_count)]
                    if selected is None or result['MRR'] > selected:
                        selected,selected_cycle = result['MRR'],cycle_index+1
                        torch.save({**checkpoint(torch,model,optimizers,streams,endpoint_streams,random_streams,job),
                            'selected_cycle':selected_cycle,'selected_VALID_MRR':selected},output/'selected_checkpoint.pt')
                        torch.save({'member_pos':p.cpu(),'member_neg':n.cpu(),'mean_pos':p.mean(1).cpu(),'mean_neg':n.mean(2).cpu(),
                            'inputs':identities,'selected_cycle':selected_cycle,'checkpoint_sha256':sha(output/'selected_checkpoint.pt')},output/'selected_VALID_logits.pt')
                history.write(json.dumps(row)+'\n'); history.flush()
                custody.write_json(output/'PROGRESS.json',row)
        if job['phase'] == 'fit' and selected is None: raise ValueError('No complete cadence VALID selector')
        torch.save(checkpoint(torch,model,optimizers,streams,endpoint_streams,random_streams,job),output/'FINAL_STATE.pt')
        for handle in handles: handle.remove()
        custody.write_json(output/'FREEZE.json',{'phase':job['phase'],'success':True,'seed':job['seed'],'condition':job['condition'],
            'growth_source_manifest_sha256':job['growth_source_manifest_sha256'],'complete_cycles':cycles,
            'counters':counters,'native_encoder_P':aggregation,'cycle_seconds':cycle_seconds,
            'calibration_seconds':calibration_seconds,'calibration_and_basis_generation_paid':True,
            'warm20_cost_must_be_charged_from_bound_warm_receipt':True,'warm_freeze_sha256':sha(frozen_path),
            'inclusive_seconds':time.monotonic()-started,'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_CUDA_reserved_bytes':torch.cuda.max_memory_reserved(),
            'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'output_bytes_before_freeze':sum(f.stat().st_size for f in output.rglob('*') if f.is_file()),
            'parameter_count':sum(p.numel() for p in model.parameters()),
            'runtime':versions,'inputs':identities,
            'VALID_TEST_access':job['phase'] == 'fit','TEST_access':False,'selected_cycle':selected_cycle,
            'selected_VALID_MRR':selected,'independent_optimizer_count':len(optimizers),
            'independent_common_warm_and_pooled_selector_limitation':independent,
            'serving':'mean raw native logits; all member trajectories execute; no inference adaptation',
            'provenance_sha256':{n:sha(output/n) for n in ('INITIAL_STATE.pt','FINAL_STATE.pt','CYCLE_DRAWS.jsonl.gz','HISTORY.jsonl','INITIAL_BASES.pt','CALIBRATION_DRAW.json')},
            'retry':False})
    except BaseException as error:
        custody.write_json(output/'FAILURE.json',{'error':type(error).__name__+': '+str(error),
            'partial_outputs_preserved':True,'retry':False,'TEST_access':False})
        raise


if __name__ == '__main__': main()
