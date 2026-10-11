"""One frozen forward-only source-context/message-association diagnostic.

Archived factual flags retain authority; replay is a separate event.
No fits/calibration/TEST/current Physics-Penn reads or numerical utility gate.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SEEDS=(7301,7403,7507)
ARMS=('E4','equipped_M1','equipped_I4')
CONDITIONS={'factual':(),'layer0':(0,),'layer1':(1,),'both':(0,1)}
MAP_SEED=2026101101
N,E,ROWS=11701,442907,5274

def require(value, message):
    if not value:
        raise ValueError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            digest.update(part)
    return digest.hexdigest()

def receipt(item):
    path = Path(item['path']).resolve()
    require(path.is_relative_to(R) and path.is_file() and path.stat().st_size == item['bytes']
            and sha(path) == item['sha256'], 'Pinned original input changed: ' + str(path))
    return path

def file_receipt(path):
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))

def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def interrupted(signum, frame):
    raise TimeoutError('Finite diagnostic wall cap; retain all partial records without interpretation')

def strict_rival(np, logits, y):
    truth = logits[:, np.arange(len(y)), y]
    rival = (logits > truth[:, :, None]).all(0)
    rival[np.arange(len(y)), y] = False
    return rival.any(-1)

def contrast(np, candidate, reference, mask):
    cp, rp = ~candidate['pooled_errors'], ~reference['pooled_errors']
    cv, rv = (~candidate['member_errors']).any(0), (~reference['member_errors']).any(0)
    repair, harm = cp & ~rp & mask, ~cp & rp & mask
    cs, rs = candidate['strict_rival'], reference['strict_rival']
    result = dict(n=int(mask.sum()), repairs=int(repair.sum()), harms=int(harm.sum()),
                  net_correct=int(((cp.astype(int) - rp.astype(int)) * mask).sum()),
                  candidate_correct=int((cp & mask).sum()), reference_correct=int((rp & mask).sum()),
                  coverage_added=int((cv & ~rv & mask).sum()), coverage_removed=int((~cv & rv & mask).sum()),
                  candidate_pool_losses=int((cv & ~cp & mask).sum()), reference_pool_losses=int((rv & ~rp & mask).sum()),
                  repairs_new_member_alternative=int((repair & cv & ~rv).sum()),
                  repairs_both_have_member_alternative=int((repair & cv & rv).sum()),
                  repairs_pool_only=int((repair & ~cv).sum()),
                  harms_available_alternative_lost=int((harm & cv).sum()),
                  harms_removed_member_alternative=int((harm & ~cv & rv).sum()),
                  harms_lost_pool_only=int((harm & ~cv & ~rv).sum()),
                  strict_rival_removed=int((rs & ~cs & mask).sum()), strict_rival_created=int((~rs & cs & mask).sum()),
                  reference_strict_rival_pooled_errors_repaired=int((rs & repair).sum()))
    require(result['net_correct'] == result['repairs'] - result['harms'], 'Exact gross/net accounting')
    result['coverage_delta'] = result['coverage_added'] - result['coverage_removed']
    result['pool_loss_delta'] = result['candidate_pool_losses'] - result['reference_pool_losses']
    result['pool_only_delta'] = int(((~cv & cp & mask).sum()) - ((~rv & rp & mask).sum()))
    require(result['net_correct'] == result['coverage_delta'] - result['pool_loss_delta'] + result['pool_only_delta'],
            'Exact gross/net availability and pooling-loss identity')
    require(result['repairs'] == sum(result[k] for k in ('repairs_new_member_alternative', 'repairs_both_have_member_alternative', 'repairs_pool_only'))
            and result['harms'] == sum(result[k] for k in ('harms_available_alternative_lost', 'harms_removed_member_alternative', 'harms_lost_pool_only')),
            'Complete gross repair/harm partitions')
    return result


def receiver_mapping(np, edges):
    """One label-free cyclic derangement in original COO record coordinates."""
    source, receiver = edges
    nonself = source != receiver
    candidates = np.flatnonzero(nonself)
    order = np.lexsort((candidates, source[candidates], receiver[candidates]))
    ordered = candidates[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(receiver[ordered])) + 1, len(ordered)]
    mapping = np.arange(E, dtype=np.int64)
    eligible = np.zeros(E, dtype=bool)
    rng = np.random.Generator(np.random.PCG64(MAP_SEED))
    groups = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group = ordered[start:end]
        if len(group) >= 2:
            offset = int(rng.integers(1, len(group)))
            mapping[group] = np.roll(group, -offset)
            eligible[group] = True
            groups += 1
    require(np.array_equal(np.sort(mapping), np.arange(E))
            and np.array_equal(receiver[mapping], receiver)
            and np.array_equal(mapping[~nonself], np.flatnonzero(~nonself))
            and (mapping[eligible] != np.flatnonzero(eligible)).all()
            and (source[mapping[eligible]] != source[eligible]).all(),
            'Original support: receiver-preserving permutation, self fixed, another eligible source/record')
    return mapping, dict(seed=MAP_SEED, generator='PCG64', canonical_order='receiver,source,original_COO_index',
                         eligible_receiver_groups=groups, eligible_nonself_records=int(eligible.sum()),
                         self_records_fixed=int((~nonself).sum()), one_mapping_only=True,
                         restricted_nonzero_cyclic_offsets_not_independent_averaging=True,
                         context_multisets_local_to_supplied_variant_states=True)


def hooked_forward(torch, family, model, body, members, batch, ids, layers, mapping):
    """Change only the first half of each builder's input, not helper context/x_j."""
    handles, hits = [], {}
    def before(key):
        def hook(module, arguments):
            require(len(arguments) == 1, 'One original endpoint-builder context')
            context = arguments[0]
            require(context.shape == (E, 256) and context.dtype == torch.float32
                    and context.device == mapping.device, 'Original live ordered endpoint context in COO order')
            hits[key] += 1
            # Self/ineligible rows map to themselves. Actual helper x_j and the
            # original context remain untouched, including receiver coordinates.
            changed = torch.cat((context[mapping, :128], context[:, 128:]), dim=-1)
            return (changed,)
        return hook
    try:
        for layer in layers:
            conv = body.residual_modules[layer].module.conv
            require(conv.value_conditioning == 'endpoint', 'Perturb endpoint builders only')
            for name in ('value_source_builder', 'value_target_builder'):
                key = str(layer) + '/' + name
                hits[key] = 0
                handles.append(getattr(conv, name).register_forward_pre_hook(before(key)))
        with torch.no_grad():
            if members == 4:
                logits = model(dict(batch, ids=ids))[0]
            else:
                with family.factors.member_context(body, 0):
                    logits = model(batch['graph'], batch['x'])[ids].unsqueeze(0)
            require(logits.shape == (members, ROWS, 10) and logits.dtype == torch.float32
                    and bool(torch.isfinite(logits).all()), 'Complete selected eval-mode forward')
            result = logits.detach().cpu().clone()
        require(all(value == members for value in hits.values()), 'Both requested builders hooked once per native trajectory')
        return result, hits
    finally:
        for handle in handles:
            handle.remove()


def score(torch, np, logits, y):
    require(logits.dtype == torch.float32 and logits.device.type == 'cpu' and bool(torch.isfinite(logits).all()),
            'Finite CPU float32 selected scores')
    probabilities = logits.softmax(-1)
    pool = probabilities.mean(0)
    logp = logits.to(torch.float64).log_softmax(-1)
    member_loss = -logp.gather(-1, y[None, :, None].expand(len(logits), -1, 1)).squeeze(-1)
    log_pool = torch.logsumexp(logp, 0) - np.log(len(logits))
    pool_loss = -log_pool.gather(1, y[:, None]).squeeze(1)
    raw = logits.numpy().copy()
    return dict(raw_logits=raw, probability_mean=pool.numpy().copy(),
                member_errors=probabilities.argmax(-1).ne(y[None]).numpy().copy(),
                pooled_errors=pool.argmax(-1).ne(y).numpy().copy(),
                strict_rival=strict_rival(np, raw, y.numpy()),
                nll_by_node=pool_loss.numpy().copy(), member_nll_by_node=member_loss.numpy().copy())


def statistics(np, value, mask):
    n = int(mask.sum())
    pooled = ~value['pooled_errors'][mask]
    members = ~value['member_errors'][:, mask]
    k = members.sum(0)
    coverage = k > 0
    member_rate = members.mean(1) if n else None
    correct, available, loss, only = (int(pooled.sum()), int(coverage.sum()),
                                    int((coverage & ~pooled).sum()), int((~coverage & pooled).sum()))
    require(correct == available - loss + only, 'Exact served/coverage/loss/only accounting')
    return dict(n=n, pooled_correct=correct, coverage=available, pool_losses=loss, pool_only_correct=only,
                member_correct=members.sum(1).astype(int).tolist(),
                member_correct_histogram=[int((k == i).sum()) for i in range(len(members) + 1)],
                served_accuracy_pct=100 * float(pooled.mean()) if n else None,
                per_member_accuracy_pct=(100 * member_rate).tolist() if n else [None] * len(members),
                mean_member_accuracy_pct=100 * float(member_rate.mean()) if n else None,
                minimum_member_accuracy_pct=100 * float(member_rate.min()) if n else None,
                pooled_nll=float(value['nll_by_node'][mask].mean()) if n else None,
                per_member_nll=value['member_nll_by_node'][:, mask].mean(1).tolist() if n else [None] * len(members),
                mean_member_nll=float(value['member_nll_by_node'][:, mask].mean()) if n else None,
                strict_rival=int(value['strict_rival'][mask].sum()),
                strict_rival_pooled_errors=int((value['strict_rival'][mask] & ~pooled).sum()),
                strict_rival_served_correct=int((value['strict_rival'][mask] & pooled).sum()),
                small_readout_cell=n < 20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(socket.gethostname() == 'anogena-2-0' and Path.cwd() == R, 'Authorized allocation/repository before project reads')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines() == [UUID],
            'Sole authorized GPU allocation')
    plan_path = Path(__file__).resolve().with_name('PLAN.json')
    plan, admission = json.loads(plan_path.read_text()), json.loads(args.admission.read_text())
    source_sha, plan_sha = sha(__file__), sha(plan_path)
    require(admission.get('admitted') is True and admission.get('action') == 'closed_WikiCS_source_context_association_diagnostic'
            and admission.get('source_sha256') == source_sha and admission.get('plan_sha256') == plan_sha
            and admission.get('TEST_access') is False and admission.get('current_Physics_Penn_access') is False
            and admission.get('model_fits') is False and admission.get('external_owner_bound') is True,
            'Exact root-owned forward-only diagnostic admission')
    require(type(admission.get('wall_seconds')) is int and 0 < admission['wall_seconds'] <= 7200
            and type(admission.get('external_owner_wall_seconds')) is int
            and 0 < admission['external_owner_wall_seconds'] <= 7200, 'Finite two-hour-or-shorter internal/external owner')
    require(plan['permutation']['frozen_seed'] == MAP_SEED, 'One mapping seed frozen before arrays')
    require(plan['conditions'] == {key: list(value) for key, value in CONDITIONS.items()}
            and plan['seeds'] == list(SEEDS) and plan['arms'] == list(ARMS), 'Exact fixed forward roster')
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(admission['wall_seconds'])
    output, started, phase = args.output.resolve(), time.monotonic(), 'custody'
    require(output.is_relative_to(P) and not output.exists(), 'Fresh output in authorized phase')
    output.mkdir()
    try:
        cfg = json.loads(receipt(plan['config']).read_text())
        closed = json.loads(receipt(plan['complete_family']).read_text())
        analysis = json.loads(receipt(plan['complete_analysis']).read_text())
        require(closed['complete'] is True and closed['groups'] == 15 and closed['optimizer_acquisition_bundles'] == 24
                and closed['native_body_fit_records'] == 24 and closed['fixed_seed_roster'] == list(SEEDS)
                and closed['TEST_access'] is False and analysis['complete'] is True and analysis['banks'] == 15,
                'Original whole E family closed before arrays')
        require(cfg['source_receipts'] == plan['source_receipts'] and cfg['safe_roles'] == plan['safe_roles']
                and cfg['seeds'] == list(SEEDS), 'Exact original source/role/seed custody')
        pinned = [plan['config'], plan['complete_family'], plan['complete_analysis']] + list(plan['source_receipts'].values()) \
                 + list(plan['safe_roles'].values()) + plan['all_archive_receipts'] + plan['all_checkpoint_receipts']
        for value in pinned:
            receipt(value)
        import numpy as np
        import torch
        from torch_geometric.nn.conv import SAGEConv, MessagePassing
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(torch.get_default_dtype() == torch.float32, 'Original float32 provider')
        sys.path.insert(0, str(R))
        from models import Model
        from run_common import set_seed
        for key, obj in (('models', Model), ('run_common', set_seed), ('sage_conv', SAGEConv), ('message_passing', MessagePassing)):
            require(Path(inspect.getfile(obj)).resolve() == Path(plan['source_receipts'][key]['path']).resolve(), 'Pinned native import: ' + key)
        owner = load_source('association_selected_E_owner', plan['source_receipts']['training_family']['path'])
        family = object.__new__(owner.Family)
        family.cfg, family.torch, family.Model, family.set_seed = cfg, torch, Model, set_seed
        for attribute, key in (('routes', 'common_routes'), ('factors', 'factors'),
                               ('value_installer', 'value_installer'), ('value_helper', 'value_helper')):
            setattr(family, attribute, load_source('association_' + attribute, plan['source_receipts'][key]['path']))
        family._operation = 'endpoint'
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

        def guard():
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= 8 * 1024 ** 3
                    and torch.cuda.max_memory_allocated(device) <= 8 * 1024 ** 3, 'Finite 8GiB RSS/CUDA caps')

        with np.load(receipt(plan['safe_roles']['train']), allow_pickle=False) as a:
            require(set(a.files) == {'x', 'edge_index', 'ids', 'y'}, 'Exact safe graph container; TRAIN labels not decoded')
            x, edges = a['x'].copy(), a['edge_index'].copy()
        with np.load(receipt(plan['safe_roles']['valid']), allow_pickle=False) as a:
            require(set(a.files) == {'ids', 'y'}, 'Exact selected role container')
            valid_ids, valid_y = a['ids'].copy(), a['y'].copy()
        require(x.dtype == np.float32 and x.shape == (N, 300) and np.isfinite(x).all()
                and edges.dtype == np.int64 and edges.shape == (2, E) and edges.min() >= 0 and edges.max() < N,
                'Original full factual graph/features')
        require(valid_ids.dtype == valid_y.dtype == np.int64 and valid_ids.shape == valid_y.shape == (ROWS,)
                and len(np.unique(valid_ids)) == ROWS and valid_ids.min() >= 0 and valid_ids.max() < N
                and np.isin(valid_y, np.arange(10)).all(), 'Complete original selected IDs/classes')
        mapping, mapping_receipt = receiver_mapping(np, edges)
        mapping_path = output / 'PERMUTATION.npy'
        np.save(mapping_path, mapping, allow_pickle=False)
        mapping_receipt.update(file=file_receipt(mapping_path))
        write(output / 'PERMUTATION_RECEIPT.json', mapping_receipt)
        family.edge_metadata = dict(nodes=N, edges=E, self_edges=int((edges[0] == edges[1]).sum()))
        require(family.edge_metadata['self_edges'] == N, 'Original fixed self support')
        tx, te, permutation, ids = (torch.from_numpy(a).to(device) for a in (x, edges, mapping, valid_ids))
        y = torch.from_numpy(valid_y)
        batch = dict(x=tx, edge_index=te, graph=SimpleNamespace(edge_index=te))
        archive_index = {(r['seed'], r['arm']): r for r in plan['all_archive_receipts']}
        state_index = {(r['seed'], r['arm'], r['member']): r for r in plan['all_checkpoint_receipts']}
        original, values, forward_receipts, replays = {}, {}, {}, []
        trajectories, restored = 0, 0
        (output / 'PREDICTIONS').mkdir()
        phase = 'selected_forward_roster'
        for seed in SEEDS:
            for arm in ARMS:
                guard()
                members = 1 if arm == 'equipped_M1' else 4
                with np.load(receipt(archive_index[seed, arm]), allow_pickle=False) as a:
                    require(set(a.files) == {'ids', 'y', 'raw_logits', 'probability_mean', 'member_errors', 'pooled_errors'}, 'Original six-field archive')
                    stored = {key: a[key].copy() for key in a.files}
                require(np.array_equal(stored['ids'], valid_ids) and np.array_equal(stored['y'], valid_y)
                        and stored['raw_logits'].dtype == np.float32 and stored['raw_logits'].shape == (members, ROWS, 10)
                        and stored['probability_mean'].dtype == np.float32 and stored['probability_mean'].shape == (ROWS, 10)
                        and stored['member_errors'].dtype == stored['pooled_errors'].dtype == np.bool_
                        and stored['member_errors'].shape == (members, ROWS) and stored['pooled_errors'].shape == (ROWS,),
                        'Original aligned scores/probabilities and archived decision authority')
                base = score(torch, np, torch.from_numpy(stored['raw_logits']), y)
                base.update(probability_mean=stored['probability_mean'], member_errors=stored['member_errors'], pooled_errors=stored['pooled_errors'])
                original[seed, arm] = base
                by_condition = {condition: [] for condition in CONDITIONS}
                hook_counts = {condition: [] for condition in CONDITIONS}
                for member in range(4 if arm == 'equipped_I4' else 1):
                    count = 4 if arm == 'E4' else 1
                    body, wrapper = family.native_body(seed, member, member, True, count)
                    model = wrapper if count == 4 else body
                    state = torch.load(receipt(state_index[seed, arm, member]), map_location='cpu', weights_only=True)
                    model.load_state_dict(state['model'], strict=True)
                    for parameter in model.parameters():
                        parameter.requires_grad_(False)
                    model.to(device).eval()
                    parameter_versions = [p._version for p in model.parameters()]
                    restored += 1
                    for condition, layers in CONDITIONS.items():
                        guard()
                        logits, hits = hooked_forward(torch, family, model, body, count, batch, ids, layers, permutation)
                        by_condition[condition].append(logits)
                        hook_counts[condition].append(dict(optimizer_member=member, actual_builder_hook_calls=hits))
                        trajectories += count
                        require(parameter_versions == [p._version for p in model.parameters()], 'Frozen selected weights unchanged by hooks')
                    family._single_adapter = None
                    del model, body, wrapper, state
                    torch.cuda.empty_cache()
                    guard()
                for condition in CONDITIONS:
                    value = score(torch, np, torch.cat(by_condition[condition]), y)
                    values[seed, arm, condition] = value
                    filename = ('factual_replay' if condition == 'factual' else condition) + '_selected_VALID.npz'
                    folder = output / 'PREDICTIONS' / (arm + '_seed' + str(seed))
                    folder.mkdir(exist_ok=True)
                    path = folder / filename
                    np.savez(path, ids=valid_ids, y=valid_y, **{key: value[key] for key in
                             ('raw_logits', 'probability_mean', 'member_errors', 'pooled_errors')})
                    forward_receipts[seed, arm, condition] = dict(file=file_receipt(path), hooks=hook_counts[condition],
                                                                 role='factual replay only; original archived flags authoritative' if condition == 'factual' else 'perturbed selected-model forward')
                factual = values[seed, arm, 'factual']
                replays.append(dict(seed=seed, arm=arm, max_abs_score_difference=float(np.abs(factual['raw_logits'] - stored['raw_logits']).max()),
                                    member_flag_differences=int((factual['member_errors'] != stored['member_errors']).sum()),
                                    pooled_flag_differences=int((factual['pooled_errors'] != stored['pooled_errors']).sum()),
                                    utility_gate=False))
                write(output / 'PROGRESS.json', dict(complete=False, closed_forward_banks=len(values), restored_states=restored,
                                                    native_trajectories=trajectories, interpretation_allowed=False))
        require(len(values) == 36 and trajectories == 108 and restored == 18, 'Entire frozen forward cohort before interpretation')
        phase = 'complete_descriptive_readout'
        allmask = np.ones(ROWS, dtype=bool)
        rows, comparisons, originals = [], [], []

        def primary(seed, arm, condition):
            return original[seed, arm] if condition == 'factual' else values[seed, arm, condition]

        for seed in SEEDS:
            for arm in ARMS:
                anchor = original[seed, arm]
                originals.append(dict(seed=seed, arm=arm, metrics=statistics(np, anchor, allmask),
                                      original_archive=archive_index[seed, arm], archived_decisions_are_authority=True))
                for condition in CONDITIONS:
                    value = primary(seed, arm, condition)
                    E_anchor = original[seed, 'E4']
                    old_E_strict_errors = E_anchor['strict_rival'] & E_anchor['pooled_errors']
                    rows.append(dict(seed=seed, arm=arm, condition=condition, metrics=statistics(np, value, allmask),
                                     factual_authority='original archive' if condition == 'factual' else 'perturbed selected-model forward',
                                     forward=forward_receipts[seed, arm, condition],
                                     versus_original_same_method=contrast(np, value, anchor, allmask),
                                     versus_factual_replay_same_method=contrast(np, values[seed, arm, condition], values[seed, arm, 'factual'], allmask),
                                     versus_old_E_strict_rival_errors=contrast(np, value, E_anchor, old_E_strict_errors),
                                     factual_classes=[dict(class_id=cl, metrics=statistics(np, value, valid_y == cl),
                                                          versus_original_same_method=contrast(np, value, anchor, valid_y == cl)) for cl in range(10)]))
            for condition in CONDITIONS:
                for ref in ('equipped_M1', 'equipped_I4'):
                    a, b = primary(seed, 'E4', condition), primary(seed, ref, condition)
                    comparisons.append(dict(seed=seed, condition=condition, candidate='E4', reference=ref,
                                            whole_population=contrast(np, a, b, allmask),
                                            factual_classes=[dict(class_id=cl, **contrast(np, a, b, valid_y == cl)) for cl in range(10)]))
        equal_seed = []
        for arm in ARMS:
            for condition in CONDITIONS:
                selected = [next(r for r in rows if r['seed'] == s and r['arm'] == arm and r['condition'] == condition) for s in SEEDS]
                metric = {key: sum(r['metrics'][key] for r in selected) / 3 for key in
                          ('served_accuracy_pct', 'pooled_nll', 'mean_member_accuracy_pct', 'minimum_member_accuracy_pct', 'mean_member_nll')}
                counts = {key: sum(r['metrics'][key] for r in selected) for key in
                          ('n', 'pooled_correct', 'coverage', 'pool_losses', 'pool_only_correct', 'strict_rival', 'strict_rival_pooled_errors')}
                transitions = {key: sum(r['versus_original_same_method'][key] for r in selected) for key in selected[0]['versus_original_same_method']}
                equal_seed.append(dict(arm=arm, condition=condition, equal_seed_mean_metrics=metric,
                                       summed_dependent_readout_counts=counts, summed_own_original_transitions=transitions,
                                       per_seed_correct_delta=[r['versus_original_same_method']['net_correct'] for r in selected], distinct_nodes=ROWS))
        for value in pinned:
            receipt(value)
        require(sha(__file__) == source_sha and sha(plan_path) == plan_sha, 'Exact admitted source/plan unchanged')
        guard()
        summary = dict(complete=True, served_banks=36, native_trajectories=108, restored_selected_states=18,
                       source_sha256=source_sha, plan_sha256=plan_sha, mapping=mapping_receipt,
                       originals=originals, conditions=rows, same_condition_strong_reference_comparisons=comparisons,
                       equal_seed_summary=equal_seed, factual_replay=replays, SELECT_rows=ROWS,
                       original_archive_decisions_are_authority=True, numerical_replay_utility_gate=False,
                       model_fits=0, head_fits=0, calibration_fits=0, TEST_access=False, current_Physics_Penn_access=False,
                       scope=plan['scope'], intervention=plan['intervention'], permutation_limits=plan['permutation'],
                       N_receiver_only_analytic_control='concat(x_i,silu(x_i)) contains no sender context; no redundant N trajectories',
                       dependency='Three seed readouts of same5274 selected nodes:15822 dependent readouts; one shared fixed cyclic mapping',
                       seconds=time.monotonic() - started, process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       backend=dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                                    cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
                       performance_scope='Forward-only GPU diagnostic overlaps unchanged live fits; no isolated speed claim',
                       automatic_new_study_or_reviewer=False)
        write(output / 'COMPLETE_ASSOCIATION_DIAGNOSTIC.json', summary)
        signal.alarm(0)
        print(json.dumps(dict(complete=True, served_banks=36, native_trajectories=108, model_fits=0, TEST_access=False)), flush=True)
    except BaseException as error:
        write(output / 'FAILURE.json', dict(complete=False, status='FAILED_RETAINED', phase=phase,
                                           type=type(error).__name__, message=str(error), source_sha256=source_sha, plan_sha256=plan_sha,
                                           model_fits=0, TEST_access=False, current_Physics_Penn_access=False, retry_or_survivor_mean=False))
        raise


if __name__ == '__main__':
    main()
