"""One admitted TRAIN-only selected-checkpoint shared-gradient diagnostic.

No optimizer/model update, validation array/score use, TEST, fit or parity gate.
Numerical imports occur only inside the explicitly admitted finite entrypoint.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SEEDS = (7301, 7403, 7507)
ARMS = ('BASE', 'E4', 'N4', 'equipped_M1', 'equipped_I4')
OPERATIONS = dict(zip(ARMS, ('none', 'endpoint', 'node', 'endpoint', 'endpoint')))
SHARED = ARMS[:3]
WALL, CUDA, RSS = 240, 12 * 1024 ** 3, 8 * 1024 ** 3
N, EDGES, TRAIN = 11701, 442907, 580
ATOMIC = ('stem', 'graph_layer0', 'graph_layer1', 'FFN_layer0', 'FFN_layer1',
          'normalization', 'head', 'builder_layer0_source', 'builder_layer0_target',
          'builder_layer1_source', 'builder_layer1_target')
AGGREGATE = dict(native_graph=('graph_layer0', 'graph_layer1'),
                 native_FFN=('FFN_layer0', 'FFN_layer1'),
                 conditional_builders=ATOMIC[7:], native_all=ATOMIC[:7], all_shared=ATOMIC)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def receipt(item):
    require(isinstance(item, dict) and set(item) == {'path', 'bytes', 'sha256'}, 'Exact receipt')
    path = Path(item['path']).resolve()
    require(path.is_relative_to(R) and path.is_file() and path.stat().st_size == item['bytes']
            and sha(path) == item['sha256'], 'Pinned original input changed: ' + str(path))
    return path


def write(path, value):
    require(not path.exists(), 'One fresh terminal or bank record; retain failures')
    temporary = path.with_suffix(path.suffix + '.tmp')
    require(not temporary.exists(), 'Retain partial writes')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def interrupt(signum, frame):
    raise TimeoutError('Fixed TRAIN-gradient diagnostic cap; no retry or survivor result')


def roster(plan, cfg):
    """Use closure/receipt fields only; serialized metadata also carries scores."""
    closed = json.loads(receipt(plan['complete_family']).read_text())
    require(closed.get('complete') is True and closed.get('full_same_runtime_stage_complete') is True
            and closed.get('groups') == 15 and closed.get('optimizer_acquisition_bundles') == 24
            and closed.get('native_body_fit_records') == 24 and closed.get('fit_units') == 24
            and closed.get('fixed_seed_roster') == list(SEEDS) and closed.get('fixed_arm_roster') == list(ARMS)
            and closed.get('TEST_access') is False and closed.get('config_sha256') == plan['config']['sha256'],
            'Authoritative COMPLETE15-bank/24-state encountered family')
    banks = closed['results']
    require(len(banks) == 15 and {(r['seed'], r['arm']) for r in banks} == {(s, a) for s in SEEDS for a in ARMS},
            'All BASE/E4/N4/M1/I4 banks, not the previous18-state diagnostic projection')
    require(cfg['source_receipts'] == plan['source_receipts'] and cfg['safe_roles']['train'] == plan['train']
            and cfg['seeds'] == list(SEEDS) and cfg['device'] == 'cuda:0', 'Exact original source/config/TRAIN custody')
    result = {}
    for r in banks:
        arm = r['arm']; bundles = 4 if arm == 'equipped_I4' else 1
        members = 1 if arm == 'equipped_M1' else 4
        require(r['value_operation'] == OPERATIONS[arm] and len(r['fits']) == bundles
                and r['optimizer_acquisition_bundles'] == r['native_bodies_fitted'] == bundles
                and r['native_trajectories_per_forward'] == members, 'Exact actual body/bundle/route count')
        require({f['member'] for f in r['fits']} == set(range(bundles)), 'All distinct optimizer members')
        for f in r['fits']:
            require(f['factorized'] is True and f['learning_objective'] == 'own_only'
                    and f['within_route_supcon'] is False, 'Original own-CE route objective')
            receipt(f['selected_state_receipt'])
        result[r['seed'], arm] = [{k: f[k] for k in ('member', 'selected_step', 'selected_state_receipt')}
                                  for f in sorted(r['fits'], key=lambda f: f['member'])]
    files = [f['selected_state_receipt'] for bank in result.values() for f in bank]
    require(len(files) == 24 and len({r['path'] for r in files}) == 24, 'All24 actual distinct original checkpoints')
    return result


def partition(family, model, body, members):
    """Partition actual registered parameter objects, including normalization."""
    names = {id(p): name for name, p in model.named_parameters()}
    params = {id(p): p for p in model.parameters()}
    require(len(names) == len(params) and all(p.requires_grad for p in params.values())
            and not list(model.named_buffers()), 'Original trainable native ownership; no mutable buffers')
    groups = {key: set() for key in ATOMIC}
    private = {id(p) for m in body.modules() if isinstance(m, family.factors.FactorLinear) for p in (m.r, m.s)}
    def add(key, module): groups[key].update(id(p) for p in module.parameters())
    add('stem', body.input_linear); add('head', body.output_linear)
    add('normalization', body.output_normalization)
    require(len(body.residual_modules) == 2, 'Actual native depth two')
    for i, residual in enumerate(body.residual_modules):
        add('normalization', residual.normalization)
        add('FFN_layer' + str(i), residual.module.feed_forward_module)
        conv = residual.module.conv
        add('graph_layer' + str(i), conv.lin_l); add('graph_layer' + str(i), conv.lin_r)
        if family._operation != 'none':
            for end in ('source', 'target'):
                add('builder_layer' + str(i) + '_' + end, getattr(conv, 'value_' + end + '_builder'))
    flattened = [key for ids in groups.values() for key in ids]
    require(len(flattened) == len(set(flattened)) and set(flattened) == set(params),
            'Every actual parameter in exactly one named native/builder group')
    require(all(p.shape[0] == members for key, p in params.items() if key in private), 'Actual registered r/s row count')
    if members == 4:
        roles = model.parameter_roles()
        require(set(roles['private_fast']) == {names[key] for key in private}, 'Original wrapper private-factor authority')
        require(set(roles['native_slow']) == {names[key] for group in ATOMIC[:7] for key in groups[group] - private}
                and set(roles['shared_block']) == {names[key] for group in ATOMIC[7:] for key in groups[group] - private},
                'Native slow/builder shared ownership from actual wrapper')
    ordered = {key: sorted(ids, key=lambda q: names[q]) for key, ids in groups.items()}
    inventory = [{ 'name': names[key], 'shape': list(params[key].shape), 'numel': params[key].numel(),
                   'group': group, 'ownership': 'private_factor_rows' if key in private else 'shared_within_bank' if members == 4 else 'own_body'}
                 for group, ids in ordered.items() for key in ids]
    return params, names, private, ordered, inventory


def gradient_summary(torch, params, names, private, groups, route):
    vectors, private_rows, unused = {}, {}, []
    for group, ids in groups.items():
        shared, own = [], []
        for key in ids:
            p = params[key]; gradient = p.grad
            if gradient is None:
                unused.append(names[key]); gradient = torch.zeros_like(p)
            require(bool(torch.isfinite(gradient).all()), 'Finite route own-CE gradient; failed bank retained otherwise')
            if key in private:
                require(not bool(torch.count_nonzero(gradient[:route])) and not bool(torch.count_nonzero(gradient[route + 1:])),
                        'An own-route loss touches only its registered private row')
                own.append(gradient[route].detach().to(device='cpu', dtype=torch.float64).reshape(-1))
            else:
                shared.append(gradient.detach().to(device='cpu', dtype=torch.float64).reshape(-1))
        vectors[group] = torch.cat(shared) if shared else torch.empty(0, dtype=torch.float64)
        own = torch.cat(own) if own else torch.empty(0, dtype=torch.float64)
        sq = float(own.square().sum())
        private_rows[group] = dict(active_row_dimensions=own.numel(), squared_norm=sq, norm=math.sqrt(sq), zero_norm=sq == 0,
                                   present=own.numel() > 0, coordinate_cosine_or_cancellation_comparison=False)
    for group, components in AGGREGATE.items():
        vectors[group] = torch.cat([vectors[key] for key in components])
    total = sum(value['squared_norm'] for value in private_rows.values())
    private_rows['all_private'] = dict(active_row_dimensions=sum(v['active_row_dimensions'] for v in private_rows.values()),
                                      squared_norm=total, norm=math.sqrt(total), zero_norm=total == 0,
                                      coordinate_cosine_or_cancellation_comparison=False)
    return vectors, private_rows, unused


def norm_record(vector):
    sq = float(vector.square().sum())
    return dict(dimensions=vector.numel(), present=vector.numel() > 0,
                squared_norm=sq, norm=math.sqrt(sq), zero_norm=sq == 0)


def shared_comparison(torch, vectors):
    """Only the four routes of this one shared bank have common coordinates."""
    require(len(vectors) == 4, 'Exactly four within-bank shared gradient vectors')
    records = {}
    for group in (*ATOMIC, *AGGREGATE):
        rows = [v[group] for v in vectors]; norms = [norm_record(v) for v in rows]
        pairs = []
        for i in range(4):
            for j in range(i + 1, 4):
                dot = float(torch.dot(rows[i], rows[j])); denominator = norms[i]['norm'] * norms[j]['norm']
                pairs.append(dict(route_i=i, route_j=j, dot=dot, norm_i=norms[i]['norm'], norm_j=norms[j]['norm'],
                                  cosine=dot / denominator if denominator else None,
                                  cosine_status='defined' if denominator else 'undefined_zero_norm',
                                  no_epsilon_no_clipping=True))
        mean = torch.stack(rows).mean(0); mean_sq = float(mean.square().sum())
        denominator = sum(r['squared_norm'] for r in norms) / 4
        records[group] = dict(route_norms=norms, pairwise=pairs, mean_gradient_squared_norm=mean_sq,
                              mean_route_squared_norm=denominator,
                              cancellation_ratio=mean_sq / denominator if denominator else None,
                              ratio_status='defined' if denominator else 'undefined_all_zero_norm',
                              all_zero_norm=denominator == 0, one_common_parameterization_within_this_bank=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(socket.gethostname() == 'anogena-2-0' and Path.cwd() == R, 'Authorized allocation/repository before project reads')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines() == [UUID],
            'Sole authorized GPU before project metadata')
    path = Path(__file__).resolve().with_name('PLAN.json')
    source_sha, plan_sha = sha(__file__), sha(path)
    plan, admission = json.loads(path.read_text()), json.loads(args.admission.read_text())
    require(admission.get('admitted') is True and admission.get('action') == 'closed_WikiCS_TRAIN_shared_gradient_diagnostic'
            and admission.get('source_sha256') == source_sha and admission.get('plan_sha256') == plan_sha
            and admission.get('complete_original_family_verified') is True and admission.get('original_runtime_verified') is True
            and admission.get('external_owner_bound') is True and admission.get('validation_access') is False
            and admission.get('TEST_access') is False and admission.get('current_Physics_Penn_access') is False
            and admission.get('model_updates') == 0 and admission.get('model_fits') is False,
            'Explicit root-owned exact-source TRAIN-only diagnostic admission')
    require(type(admission.get('wall_seconds')) is int and 0 < admission['wall_seconds'] <= WALL
            and type(admission.get('external_owner_wall_seconds')) is int and 0 < admission['external_owner_wall_seconds'] <= WALL,
            'Finite internal/external240s-or-shorter owners with actual exit/reap')
    require(plan['seeds'] == list(SEEDS) and plan['arms'] == list(ARMS)
            and plan['finite_budget'] == dict(wall_seconds_max=WALL, cuda_bytes=CUDA, rss_bytes=RSS), 'Exact frozen roster/budgets')
    output = args.output.resolve()
    require(output.is_relative_to(P) and not output.exists(), 'Fresh root-owned output in scientific phase')
    require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'Dedicated externally bounded owner process')
    previous = signal.getsignal(signal.SIGALRM); signal.signal(signal.SIGALRM, interrupt)
    started = time.monotonic(); signal.alarm(admission['wall_seconds']); output.mkdir()
    phase, restores, backwards = 'custody', 0, 0
    bank_rows = [dict(seed=s, arm=a, status='unstarted_retained', route_records=[], state_restores=[])
                 for s in SEEDS for a in ARMS]
    try:
        cfg = json.loads(receipt(plan['config']).read_text())
        index = roster(plan, cfg)
        for item in plan['source_receipts'].values(): receipt(item)
        receipt(plan['train'])
        import numpy as np
        import torch
        from torch_geometric.nn.conv import SAGEConv, MessagePassing
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        require(torch.get_default_dtype() == torch.float32, 'Actual original float32 convention')
        sys.path.insert(0, str(R))
        from models import Model
        from run_common import set_seed
        for key, obj in (('models', Model), ('run_common', set_seed), ('sage_conv', SAGEConv), ('message_passing', MessagePassing)):
            require(Path(inspect.getfile(obj)).resolve() == Path(plan['source_receipts'][key]['path']).resolve(), 'Pinned native import')
        owner = load('TRAIN_gradient_original_owner', plan['source_receipts']['training_family']['path'])
        family = object.__new__(owner.Family)  # No __init__: no validation arrays or fitting admission.
        family.cfg, family.torch, family.np, family.F = cfg, torch, np, torch.nn.functional
        family.Model, family.set_seed = Model, set_seed
        for attribute, key in (('routes', 'common_routes'), ('factors', 'factors'),
                               ('value_installer', 'value_installer'), ('value_helper', 'value_helper')):
            setattr(family, attribute, load('TRAIN_gradient_' + attribute, plan['source_receipts'][key]['path']))
        family._fit_model = family._single_adapter = None
        family.device = device = torch.device('cuda:0'); family.cuda = 0
        torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        def guard():
            require(time.monotonic() - started < admission['wall_seconds']
                    and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= RSS
                    and torch.cuda.max_memory_allocated(device) <= CUDA, 'Fixed wall/CUDA/RSS caps')
        phase = 'full_TRAIN_load'; guard()
        with np.load(receipt(plan['train']), allow_pickle=False) as archive:
            require(set(archive.files) == {'x', 'edge_index', 'ids', 'y'}, 'Actual safe TRAIN container only')
            x, edges, ids, y = (archive[k].copy() for k in ('x', 'edge_index', 'ids', 'y'))
        require(x.dtype == np.float32 and x.shape == (N, 300) and np.isfinite(x).all()
                and edges.dtype == np.int64 and edges.shape == (2, EDGES) and edges.min() >= 0 and edges.max() < N
                and ids.dtype == y.dtype == np.int64 and ids.shape == y.shape == (TRAIN,)
                and len(np.unique(ids)) == TRAIN and ids.min() >= 0 and ids.max() < N and np.isin(y, np.arange(10)).all(),
                'Original full11701-node graph and all580TRAIN targets')
        family.edge_metadata = dict(nodes=N, edges=EDGES, self_edges=int((edges[0] == edges[1]).sum()))
        require(family.edge_metadata['self_edges'] == N, 'Original factual self support')
        tensor = lambda a: torch.from_numpy(a).to(device)
        family.train_ids = tensor(ids)
        labels = torch.zeros(N, dtype=torch.long, device=device); labels[family.train_ids] = tensor(y)
        mask = torch.zeros(N, dtype=torch.bool, device=device); mask[family.train_ids] = True
        family.data = SimpleNamespace(x=tensor(x), edge_index=tensor(edges), y=labels, train_mask=mask)
        family.batch = dict(x=family.data.x, edge_index=family.data.edge_index,
                            graph=SimpleNamespace(edge_index=family.data.edge_index))
        del x, edges, ids, y
        phase = 'selected_TRAIN_route_gradients'
        for bank in bank_rows:
            seed, arm = bank['seed'], bank['arm']; shared = arm in SHARED; vectors = []
            bank['status'] = 'running'; family._operation = OPERATIONS[arm]
            for fit in index[seed, arm]:
                guard(); before = time.monotonic(); member = fit['member']; members = 4 if shared else 1
                model, optimizer, initial_streams = family.make(seed, member, 'baseline', True, members, 'shared' if shared else 'single')
                state = torch.load(receipt(fit['selected_state_receipt']), map_location='cpu', weights_only=True)
                require(state['step'] == fit['selected_step'], 'Actual selected checkpoint step; no selection score read')
                model.load_state_dict(state['model'], strict=True); optimizer.load_state_dict(state['optimizer'])
                streams = state['streams']  # Original next-training dropout streams, not fresh make streams.
                require(len(streams) == members and all(set(s) == {'cpu', 'cuda'} for s in streams), 'Actual selected persistent route RNG')
                body = model.body if shared else model
                params, names, private, groups, inventory = partition(family, model, body, members)
                versions = {key: p._version for key, p in params.items()}; model.train(); restores += 1
                bank['state_restores'].append(dict(member=member, selected_step=fit['selected_step'],
                     selected_state_receipt=fit['selected_state_receipt'], constructor_optimizer_restore_seconds=time.monotonic() - before,
                     actual_Family_make=True, optimizer_state_restored=True, optimizer_steps=0, parameter_inventory=inventory))
                for route in range(members):
                    guard(); before = time.monotonic(); optimizer.zero_grad(set_to_none=True)
                    rng_before = {key: hashlib.sha256(value.numpy().tobytes()).hexdigest() for key, value in streams[route].items()}
                    if shared:
                        # The exact baseline wrapper route: original body call,
                        # original RNG scope and factor row, then TRAIN IDs.
                        # Sequential forwards avoid retaining four full graphs.
                        with family.scope(streams, route), family.factors.member_context(body, route):
                            logits = body(family.batch['graph'], family.batch['x'])[family.train_ids]
                        target = family.data.y[family.train_ids]
                    else:
                        logits, hidden = family.training_outputs(model, streams, False)
                        logits = logits[0]; target = family.data.y[family.data.train_mask]; del hidden
                    require(logits.shape == (TRAIN, 10) and bool(torch.isfinite(logits).all()), 'All580 actual own-loss rows')
                    loss = family.F.cross_entropy(logits, target, reduction='mean')
                    require(bool(torch.isfinite(loss)), 'Finite own CE')
                    loss.backward(); backwards += 1
                    shared_vectors, private_norms, unused = gradient_summary(torch, params, names, private, groups, route)
                    bank['route_records'].append(dict(route=route if shared else member, checkpoint_member=member,
                        own_CE=float(loss.detach()), objective_scale=1, TRAIN_rows=TRAIN,
                        shared_or_own_parameter_norms={g: norm_record(v) for g, v in shared_vectors.items()},
                        private_active_row_norms=private_norms, unused_parameters_as_zero_with_identity_retained=unused,
                        RNG_before=rng_before, RNG_after={key: hashlib.sha256(value.numpy().tobytes()).hexdigest() for key, value in streams[route].items()},
                        forward_backward_reduce_seconds=time.monotonic() - before, model_updates=0))
                    if shared: vectors.append(shared_vectors)
                    require(versions == {key: p._version for key, p in params.items()}, 'No model parameter update')
                    del logits, target, loss, shared_vectors; guard()
                family._fit_model = family._single_adapter = None
                del model, body, optimizer, state, streams, initial_streams, params
                torch.cuda.empty_cache(); guard()
            expected = 1 if arm == 'equipped_M1' else 4
            require(len(bank['route_records']) == expected, 'Every prescribed route retained')
            bank['shared_gradients'] = shared_comparison(torch, vectors) if shared else None
            bank['reference_scope'] = 'within_shared_bank_common_coordinates' if shared else 'own_loss_and_gradient_size_only_no_coordinate_alignment'
            bank['status'] = 'complete'; del vectors
            write(output / (arm + '_seed' + str(seed) + '.json'), bank)
        require(len(bank_rows) == 15 and all(b['status'] == 'complete' for b in bank_rows)
                and restores == 24 and backwards == 51, 'COMPLETE15banks/24states/51backwards before interpretation')
        for item in plan['source_receipts'].values(): receipt(item)
        receipt(plan['config']); receipt(plan['complete_family']); receipt(plan['train'])
        for fits in index.values():
            for fit in fits: receipt(fit['selected_state_receipt'])
        require(sha(__file__) == source_sha and sha(path) == plan_sha, 'Source/plan unchanged')
        guard()
        value = dict(complete=True, banks=15, restored_states=restores, route_backward_passes=backwards, records=bank_rows,
                     source_sha256=source_sha, plan_sha256=plan_sha, original_complete_family=plan['complete_family'], TRAIN=plan['train'],
                     model_updates=0, new_fits=False, only_closure_identity_and_receipt_fields_used=True,
                     VALID_label_or_archive_deserialized=False, VALID_scores_used_or_recomputed=False, TEST_access=False,
                     current_Physics_Penn_access=False, private_cross_route_cosines=False, independent_body_coordinate_comparisons=False,
                     one_original_next_training_dropout_sample_per_route=True, causal_or_retraining_success_claim=False,
                     entire_training_history_or_AdamW_update_measured=False, validation_error_causation_established=False,
                     numerical_parity_gate=False, automatic_followup=False, seconds=time.monotonic() - started,
                     peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(device))
        write(output / 'COMPLETE_TRAIN_GRADIENT_DIAGNOSTIC.json', value)
    except BaseException as error:
        write(output / 'FAILURE.json', dict(complete=False, status='FAILED_RETAINED', phase=phase, records=bank_rows,
               restored_states=restores, route_backward_passes=backwards, type=type(error).__name__, message=str(error),
               traceback=traceback.format_exc(), interpretation_eligible=False, retry=False, survivor_mean=False, seconds=time.monotonic() - started))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    main()
