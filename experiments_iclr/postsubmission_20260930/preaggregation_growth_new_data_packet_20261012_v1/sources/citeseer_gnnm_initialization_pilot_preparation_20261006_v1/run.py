"""Fixed real HeaRT Citeseer GNNM initialization pilot; root adoption required."""
import gc
import gzip
import argparse
import json
from pathlib import Path
import time
import sys

def own_loss(torch, p, n):
    return -torch.nn.functional.logsigmoid(p).mean()-torch.nn.functional.logsigmoid(-n).mean()


def ordinary_step(torch, steps, model, optimizer, x, support, inner, outer, streams):
    calls = [(0, (torch.cat([q[0] for q in inner], 1), torch.cat([q[1] for q in inner], 1)))] if model.member_count == 1 else list(enumerate(inner))
    losses = []
    for index in range(3):
        optimizer.zero_grad(set_to_none=True)
        parameters = dict(model.named_parameters())
        if index == 1:
            p, n = steps.functional_forward(torch, model, parameters, x, support, outer, training=False)
            value = .5*own_loss(torch, p.mean(1), n.mean(1))+.5*own_loss(torch, p, n)
        else:
            values = []
            for member, queries in calls:
                p, n = steps.functional_forward(torch, model, parameters, x, support, queries,
                    training=True, stream=(streams, member), route=member, advance=index == 2)
                values.append(own_loss(torch, p, n))
            value = torch.stack(values).mean()
        if not bool(torch.isfinite(value)): raise FloatingPointError('Nonfinite fixed BCE task loss')
        value.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError('All dense/private parameters must remain connected and finite')
        optimizer.step(); losses.append(float(value.detach()))
    return losses


def adam(torch, model):
    return torch.optim.Adam(list(model.parameters()), lr=.001, betas=(.9, .999), eps=1e-8,
                            weight_decay=0., foreach=False, fused=False)


def metric(torch, p, n):
    rank = 1+.5*((n >= p[:, None]).sum(1)+(n > p[:, None]).sum(1))
    return {'MRR': round((1/rank.float()).mean().item(), 4), 'Hits10': round((rank <= 10).float().mean().item(), 4)}


def saved_state(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def checkpoint_state(torch, model, optimizers, streams, endpoint, random, job):
    return {'model':saved_state(model),'optimizers':[v.state_dict() for v in optimizers],
        'dropout_streams':[v.state_dict() for v in streams],
        'torch_RNG':{'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()},
        'endpoint_sampling':[[{k:v.getstate() for k,v in row.items()} for row in rows] for rows in endpoint],
        'random_sampling':[[{k:v.getstate() for k,v in row.items()} for row in rows] for rows in random],
        'source_configuration':job,'postwarm_optimizer_and_dropout_reset_explicit':True}


def train_audit(torch, steps, model, warm, x, support, train, negative):
    positives, negatives, donor_p, donor_n = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(train), 512):
            queries = (train[start:start+512].t().to(x.device), negative[start:start+512].t().to(x.device))
            p, n = steps.functional_forward(torch, model, dict(model.named_parameters()), x, support, queries, training=False)
            wp, wn = steps.functional_forward(torch, warm, dict(warm.named_parameters()), x, support, queries, training=False)
            positives.append(p.cpu()); negatives.append(n.cpu()); donor_p.append(wp.cpu()); donor_n.append(wn.cpu())
    p, n, wp, wn = map(torch.cat, (positives, negatives, donor_p, donor_n))
    return {'scope': 'fixed_complete_TRAIN_bank_only', 'positive_queries': len(p), 'negative_queries': len(n),
        'member_BCE': [float(own_loss(torch, p[:, i], n[:, i])) for i in range(model.member_count)],
        'pooled_BCE': float(own_loss(torch, p.mean(1), n.mean(1))),
        'mean_member_logit_variance_positive': float(p.var(1, unbiased=False).mean()),
        'mean_member_logit_variance_negative': float(n.var(1, unbiased=False).mean()),
        'maximum_positive_donor_difference': float((p-wp).abs().max()),
        'maximum_negative_donor_difference': float((n-wn).abs().max()),
        'all_dense_and_private_parameters_trainable': all(v.requires_grad for v in model.parameters()),
        'no_strength_change_or_initialization_rescue': True}


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); proposed = json.loads(args.job.read_text())
    providers = {
        '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs': ('shared_backbone_private_transfer_training_source_20261005_v2','05649f101ca8ddc5bd549c3ece48a93752a934d68cc1ea3da58b8353495235c4'),
        '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git': ('citeseer_known_ranking_control_gpu77_b0_preparation_20261006_v1','b400c9fdd5ed885a3538a1af4e0214f24056149e135b3e5fcb1f8c37ff81d46a')}
    relative, expected = providers[proposed['repository']]
    source = Path(proposed['repository'])/'experiments_iclr/postsubmission_20260930'/relative
    import hashlib
    if hashlib.sha256((source/'custody.py').read_bytes()).hexdigest() != expected:
        raise ValueError('Existing qualified provider custody changed')
    sys.path.insert(0, str(source)); import custody
    if Path(custody.__file__).resolve() != (source/'custody.py').resolve(): raise ValueError('Provider custody shadowed')
    phase = proposed.get('phase')
    purpose = 'GNNM_init_TRAIN_only_common_warm' if phase == 'warm' else 'GNNM_init_TRAIN_VALID_fit' if phase == 'fit' else None
    if purpose is None: raise ValueError('Explicit warm/fit phase required')
    job, output = custody.authorize(__file__, args, purpose)
    plan_path = custody.phase_file(job['plan_relative'])
    if custody.sha(plan_path) != job['plan_sha256']: raise ValueError('Fixed initialization plan changed')
    plan = json.loads(plan_path.read_text())
    if plan.get('root_adopted') is not True or plan.get('TEST_closed') is not True or (plan['warm_cycles'], plan['post_cycles'], plan['eval_every']) != (20,60,5):
        raise ValueError('Root must adopt the exact fixed20/60 plan before science')
    matches = [row for row in plan['blocks'] if row['seed'] == job['seed'] and row['factor_seed'] == job['factor_seed']]
    if len(matches) != 1: raise ValueError('Fixed paired initialization seed required')
    if (job['outer_size'],job['inner_size']) != (64,256): raise ValueError('Fixed full episode geometry required')
    is_fit = phase == 'fit'
    if job['fits_authorized'] is not is_fit or job['VALID_values_access'] is not is_fit: raise ValueError('Warm never opens VALID')
    helper = Path(__file__).resolve().parent/'initialize.py'
    if custody.sha(helper) != job['initialization_sha256']: raise ValueError('Reviewed initialization bytes changed')
    initialization = custody.load_module('citeseer_gnnm_initializer', helper)
    if is_fit and job['condition'] not in initialization.CONDITIONS: raise ValueError('Fixed condition required')
    output.mkdir(); custody.write_json(output/'START.json', {'phase':phase,'job_sha256':custody.sha(args.job),'TEST_access':False})
    try:
        torch, device, versions = custody.runtime(job)
        native, heads, geometry, cycle, adjoint, models, steps = custody.load_sources()
        x, train, valid, pool, identities = custody.load_inputs(torch, job, include_valid=is_fit); x = x.to(device)
        warm, _, _, _ = models.build(native, heads, arm='capable_single', seed=job['seed'], factor_seed=job['factor_seed'], device=device)
        full_support = custody.support_tensor(torch, train, list(range(len(train))), len(x), device)
        if is_fit:
            warm_freeze = custody.phase_file(job['warm_freeze_relative'])
            frozen = json.loads(warm_freeze.read_text())
            if frozen['phase'] != 'warm' or frozen['completed_cycles'] != 20 or frozen['seed'] != job['seed'] or frozen['VALID_TEST_access'] is not False:
                raise ValueError('Exact fixed TRAIN-only warm state required')
            if custody.sha(warm_freeze) != job['warm_freeze_sha256']:
                raise ValueError('Root-bound common warm freeze changed')
            checkpoint = warm_freeze.parent/'warm_checkpoint.pt'
            if custody.sha(checkpoint) != frozen['checkpoint_sha256']: raise ValueError('Frozen warm checkpoint changed')
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if state['seed'] != job['seed'] or any(state['inputs'][k] != identities[k] for k in ('train_pos.txt','gnn_feature')):
                raise ValueError('Warm TRAIN/features custody differs')
            warm.load_state_dict(state['model'], strict=True)
            variations, covariances = initialization.variation(torch, warm, x, full_support, train, job['factor_seed'])
            model, partition = initialization.make(torch, native, heads, models, warm, job['condition'], job['seed'], job['factor_seed'], device, variations)
            custody.write_json(output/'PARAMETER_PARTITION.json', partition)
            negative, _, _ = custody.make_pair(torch,geometry,cycle,train,len(x),job['seed'],0,64,256,
                geometry.route_streams(job['seed'],4),geometry.route_streams(job['seed'],4,control=True))
            audit = train_audit(torch, steps, model, warm, x, full_support, train, negative)
            custody.write_json(output/'INITIAL_TRAIN_AUDIT.json', {'condition':job['condition'],'audit':audit,'covariances':covariances,
                'warm_checkpoint_sha256':frozen['checkpoint_sha256'],'perturbation_RMS':initialization.RMS,
                'realized_first_R_RMS_around_one': float((model.predictor.xlin.ops[0].r.detach().double()-1).square().mean().sqrt()) if job['condition'] not in ('single','independent_warm4') else None,
                'realized_first_R_maximum_member_mean_perturbation': float((model.predictor.xlin.ops[0].r.detach().double()-1).mean(0).abs().max()) if job['condition'] not in ('single','independent_warm4') else None})
            if job['condition'] in ('warm_identity','single','independent_warm4') and (audit['maximum_positive_donor_difference'] != 0 or audit['maximum_negative_donor_difference'] != 0):
                raise ValueError('Neutral warm lift must preserve full TRAIN deterministic native logits exactly')
        else: model = warm
        independent = is_fit and job['condition'] == 'independent_warm4'
        routes = list(model.routes) if independent else [model]
        route_seeds = [job['seed']+5*i for i in range(4)] if independent else [job['seed']]
        optimizers = [adam(torch, route) for route in routes]
        streams = [steps.DropoutStreams(torch, device, seed+200000 if is_fit else seed) for seed in route_seeds]
        endpoint_streams = [geometry.route_streams(seed,4) for seed in route_seeds]
        random_streams = [geometry.route_streams(seed,4,control=True) for seed in route_seeds]
        cycles = 60 if is_fit else 20; selected = selected_cycle = None; counters = {'native_Adam_updates':0,'episodes':0}; costs = []
        if is_fit: valid, pool = valid.to(device), pool.to(device)
        torch.save(checkpoint_state(torch,model,optimizers,streams,endpoint_streams,random_streams,job),output/'INITIAL_STATE.pt')
        torch.cuda.reset_peak_memory_stats()
        with custody.native_adjoint(torch,native,adjoint.make_constant_adjacency_spmm), (output/'HISTORY.jsonl').open('x') as history, gzip.open(output/'CYCLE_DRAWS.jsonl.gz','wt',encoding='utf-8',compresslevel=6) as draws:
            for cycle_index in range(cycles):
                cycle_started = time.monotonic()
                for member,(route,seed) in enumerate(zip(routes,route_seeds)):
                    negative,order,episodes = custody.make_pair(torch,geometry,cycle,train,len(x),seed,cycle_index,64,256,endpoint_streams[member],random_streams[member])
                    if len(order) != 3870 or len(episodes) != 61: raise ValueError('Complete full TRAIN tail coverage required')
                    draws.write(json.dumps({'cycle':cycle_index+1,'member':member,'sampling_seed':seed,'negative_bank':negative.tolist(),'outer_order':order,
                        'paired_episodes':[{'endpoint':e,'matched_random':r,'kept_positive_ids':k} for e,r,k,_ in episodes]})+'\n'); draws.flush()
                    for endpoint,_,kept,_ in episodes:
                        support = custody.support_tensor(torch,train,kept,len(x),device)
                        inner,outer = custody.queries(torch,train,negative,endpoint,device)
                        ordinary_step(torch,steps,route,optimizers[member],x,support,inner,outer,streams[member])
                        counters['native_Adam_updates'] += 3; counters['episodes'] += 1
                        del support,inner,outer
                        if time.monotonic()-started > job['soft_seconds']: raise TimeoutError('Fixed phase cap exceeded; no shortening/retry')
                torch.cuda.synchronize(); costs.append(time.monotonic()-cycle_started)
                row = {'cycle':cycle_index+1,'cycle_seconds':costs[-1],'counters':dict(counters)}
                if is_fit and (cycle_index+1)%5 == 0:
                    with torch.no_grad():
                        pos,neg = steps.functional_forward(torch,model,dict(model.named_parameters()),x,full_support,
                            (valid.t(),pool.permute(2,0,1).reshape(2,-1)),training=False)
                        neg = neg.reshape(227,500,model.member_count)
                        result = metric(torch,pos.mean(1),neg.mean(2)); row['complete_VALID'] = result
                        row['members'] = [metric(torch,pos[:,i],neg[:,:,i]) for i in range(model.member_count)]
                    if selected is None or result['MRR'] > selected:
                        selected,selected_cycle = result['MRR'],cycle_index+1
                        torch.save({**checkpoint_state(torch,model,optimizers,streams,endpoint_streams,random_streams,job),'seed':job['seed'],'condition':job['condition'],'job':job,
                            'selected_cycle':selected_cycle,'selected_VALID_MRR':selected},output/'selected_checkpoint.pt')
                        torch.save({'member_pos':pos.cpu(),'member_neg':neg.cpu(),'mean_pos':pos.mean(1).cpu(),'mean_neg':neg.mean(2).cpu(),
                            'inputs':identities,'selected_cycle':selected_cycle,'checkpoint_sha256':custody.sha(output/'selected_checkpoint.pt')},output/'selected_VALID_logits.pt')
                history.write(json.dumps(row)+'\n'); history.flush()
                custody.write_json(output/'PROGRESS.json',{'phase':phase,'cycle':cycle_index+1,'condition':job.get('condition'),'counters':counters})
        torch.save(checkpoint_state(torch,model,optimizers,streams,endpoint_streams,random_streams,job),output/'FINAL_STATE.pt')
        if not is_fit:
            torch.save({**checkpoint_state(torch,model,optimizers,streams,endpoint_streams,random_streams,job),'seed':job['seed'],'inputs':identities},output/'warm_checkpoint.pt')
        freeze = {'phase':phase,'seed':job['seed'],'condition':job.get('condition'),'completed_cycles':cycles,'cycle_seconds':costs,
            'job_sha256':custody.sha(args.job),'plan_sha256':job['plan_sha256'],'inputs':identities,'runtime':versions,'counters':counters,
            'VALID_TEST_access':is_fit,'TEST_access':False,'selected_cycle':selected_cycle,'selected_VALID_MRR':selected,
            'checkpoint_sha256':custody.sha(output/('selected_checkpoint.pt' if is_fit else 'warm_checkpoint.pt')),
            'inclusive_seconds':time.monotonic()-started,'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_CUDA_reserved_bytes':torch.cuda.max_memory_reserved(),'native_alias_restored':True,'retry':False}
        freeze['provenance_sha256'] = {name:custody.sha(output/name) for name in ('CYCLE_DRAWS.jsonl.gz','INITIAL_STATE.pt','FINAL_STATE.pt','HISTORY.jsonl')}
        if is_fit and selected is None: raise ValueError('No complete native-cadence selector')
        custody.write_json(output/'FREEZE.json',freeze)
    except BaseException as error:
        custody.write_json(output/'FAILURE.json',{'error':type(error).__name__+': '+str(error),'partial_outputs_preserved':True,'retry':False,'TEST_access':False})
        raise


if __name__ == '__main__': main()
