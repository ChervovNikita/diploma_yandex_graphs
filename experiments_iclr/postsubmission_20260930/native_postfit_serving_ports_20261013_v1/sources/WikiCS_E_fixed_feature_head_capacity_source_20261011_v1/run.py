"""One finite anchored fixed-feature diagnostic, not a proposed training method.

Stdlib-only until exact root admission and original closed custody checks.
Restore selected states once; collect canonical features; fit only CPU heads.
Saved SELECT logits/errors retain original authority. No TEST/current fits read.
"""
import argparse
import copy
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

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SEEDS = (7301, 7403, 7507)
ARMS = ('E4', 'equipped_M1', 'equipped_I4')
CONDITIONS = {
    'E4': ('BE_head_continuation', 'common_affine_residual', 'private_affine_residual'),
    'equipped_M1': ('BE_head_continuation', 'affine_residual'),
    'equipped_I4': ('BE_head_continuation', 'private_affine_residual'),
}


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


def score(torch, np, logits, y):
    require(logits.dtype == torch.float32 and logits.device.type == 'cpu' and logits.shape[1:] == (len(y), 10)
            and bool(torch.isfinite(logits).all()), 'Finite original-rule CPU float32 logits')
    probabilities = logits.softmax(-1)
    pool = probabilities.mean(0)
    member_errors = probabilities.argmax(-1).ne(y[None])
    pooled_errors = pool.argmax(-1).ne(y)
    log_pool = torch.logsumexp(logits.to(torch.float64).log_softmax(-1), dim=0) - np.log(len(logits))
    raw = logits.detach().numpy().copy()
    return dict(raw_logits=raw, probability_mean=pool.detach().numpy().copy(),
                member_errors=member_errors.numpy().copy(), pooled_errors=pooled_errors.numpy().copy(),
                pooled_nll=float(-log_pool.gather(1, y[:, None]).mean()),
                strict_rival=strict_rival(np, raw, y.numpy()))


def statistics(np, value, mask):
    n = int(mask.sum())
    correct = ~value['pooled_errors'][mask]
    members = ~value['member_errors'][:, mask]
    coverage = members.any(0)
    rates = members.mean(1) if n else None
    return dict(n=n, pooled_correct=int(correct.sum()), member_correct=members.sum(1).astype(int).tolist(),
                coverage=int(coverage.sum()), pool_losses=int((coverage & ~correct).sum()),
                pool_only_correct=int((~coverage & correct).sum()),
                strict_rival=int(value['strict_rival'][mask].sum()),
                strict_rival_pooled_errors=int((value['strict_rival'][mask] & ~correct).sum()),
                strict_rival_served_correct=int((value['strict_rival'][mask] & correct).sum()),
                served_accuracy_pct=100 * float(correct.mean()) if n else None,
                mean_member_accuracy_pct=100 * float(rates.mean()) if n else None,
                minimum_member_accuracy_pct=100 * float(rates.min()) if n else None)


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
    return result


def delta_head(torch, original_heads, members, condition):
    nn = torch.nn

    class Delta(nn.Module):
        def __init__(self):
            super().__init__()
            self.be = condition == 'BE_head_continuation'
            if self.be:
                require(len(original_heads) == 1, 'One original head per joint bank or independent own-body fit')
                self.changed = copy.deepcopy(original_heads[0]).cpu()
                self.initial = copy.deepcopy(original_heads[0]).cpu()
                for parameter in self.changed.parameters():
                    parameter.requires_grad_(True)
                for parameter in self.initial.parameters():
                    parameter.requires_grad_(False)
            else:
                number = members if condition == 'private_affine_residual' else 1
                with torch.random.fork_rng(devices=[]):
                    self.maps = nn.ModuleList(nn.Linear(128, 10, bias=True, device='cpu', dtype=torch.float32)
                                              for _ in range(number))
                with torch.no_grad():
                    for affine in self.maps:
                        affine.weight.zero_()
                        affine.bias.zero_()

        def head_outputs(self, head, H):
            old, outputs = head.member, []
            try:
                for member in range(members):
                    head.member = member
                    outputs.append(head(H[member]))
            finally:
                head.member = old
            return torch.stack(outputs)

        def forward(self, H):
            if self.be:
                # Both calls use native FactorLinear's sequential FP32 r/W/s/b.
                return self.head_outputs(self.changed, H) - self.head_outputs(self.initial, H)
            if len(self.maps) == 1:
                return self.maps[0](H)
            return torch.stack([affine(H[member]) for member, affine in enumerate(self.maps)])

    return Delta()


def fit_head(torch, np, folder, data, heads, condition, guard):
    folder.mkdir()
    started = time.monotonic()
    Ht, Hv, Zt, Zv, yt, yv = (data[name] for name in ('H_train', 'H_SELECT', 'Z_train', 'Z_SELECT', 'y_train', 'y_SELECT'))
    members = len(Ht)
    head = delta_head(torch, heads, members, condition)
    parameters = [p for p in head.parameters() if p.requires_grad]
    with torch.no_grad():
        require(torch.count_nonzero(head(Ht)).item() == 0 and torch.count_nonzero(head(Hv)).item() == 0,
                'Constructed delta is exactly zero; this is not native replay tolerance')
        require(torch.equal(Zv + head(Hv), Zv), 'Saved SELECT logits are exact step0 anchors')
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=0)
    best, stale, selected_step, selected_state = -1., 0, None, None
    with (folder / 'trace.jsonl').open('x') as trace:
        for step in range(1, 1001):
            guard()
            optimizer.zero_grad(set_to_none=True)
            logits = Zt + head(Ht)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), yt.repeat(members))
            require(bool(torch.isfinite(loss)), 'Finite ordinary own CE')
            loss.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters), 'Finite head gradients')
            optimizer.step()
            require(all(bool(torch.isfinite(p).all()) for p in parameters), 'Finite head parameters')
            with torch.no_grad():
                valid_logits = Zv + head(Hv)
                require(bool(torch.isfinite(valid_logits).all()), 'Finite full SELECT scores')
                prediction = valid_logits.softmax(-1).mean(0).argmax(-1)
                accuracy = prediction.eq(yv).float().mean().item()
            improved = accuracy > best
            if improved:
                best, stale, selected_step = accuracy, 0, step
                selected_state = {name: value.detach().clone() for name, value in head.state_dict().items()}
            else:
                stale += 1
            trace.write(json.dumps(dict(step=step, TRAIN_own_CE=float(loss), SELECT_accuracy=accuracy,
                                        strict_improvement=improved, stale_updates=stale), allow_nan=False) + '\n')
            trace.flush()
            if stale >= 300:
                break
    require(selected_state is not None, 'Head selected endpoint exists')
    head.load_state_dict(selected_state, strict=True)
    with torch.no_grad():
        selected = Zv + head(Hv)
    selected_path = folder / 'selected_head.pt'
    torch.save(dict(head=selected_state, selected_step=selected_step, completed_updates=step,
                    original_backbone_frozen=True, original_head_frozen=condition != 'BE_head_continuation'), selected_path)
    record = dict(condition=condition, members=members, selected_step=selected_step, completed_updates=step,
                  selected_accuracy=best, optimizer='AdamW', learning_rate=.001, weight_decay=0,
                  fresh_optimizer_moments=True, own_CE_updates=step, backbone_updates=0,
                  trainable_parameters=sum(p.numel() for p in parameters), seconds=time.monotonic() - started,
                  selected_head=file_receipt(selected_path), TEST_access=False)
    write(folder / 'RESULT.json', record)
    return selected, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(socket.gethostname() == 'anogena-2-0' and Path.cwd() == R, 'Authorized host/repository before project reads')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines() == [UUID],
            'Sole authorized allocation')
    plan_path = Path(__file__).resolve().with_name('PLAN.json')
    plan, admission = json.loads(plan_path.read_text()), json.loads(args.admission.read_text())
    source_sha, plan_sha = sha(__file__), sha(plan_path)
    require(admission.get('admitted') is True and admission.get('action') == 'fixed_feature_E_head_capacity_diagnostic'
            and admission.get('source_sha256') == source_sha and admission.get('plan_sha256') == plan_sha
            and admission.get('TEST_access') is False and admission.get('current_Physics_Penn_access') is False
            and admission.get('backbone_fitting') is False, 'Root exact source/plan diagnostic admission')
    require(type(admission.get('wall_seconds')) is int and 0 < admission['wall_seconds'] <= 7200, 'Finite root wall cap')
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(admission['wall_seconds'])
    started, output, phase = time.monotonic(), args.output.resolve(), 'custody'
    require(output.is_relative_to(P) and not output.exists(), 'Fresh diagnostic output in authorized phase')
    output.mkdir()
    try:
        cfg = json.loads(receipt(plan['config']).read_text())
        complete = json.loads(receipt(plan['complete_family']).read_text())
        analysis = json.loads(receipt(plan['complete_analysis']).read_text())
        require(complete['complete'] and complete['groups'] == 15 and complete['optimizer_acquisition_bundles'] == 24
                and complete['native_body_fit_records'] == 24 and complete['fixed_seed_roster'] == list(SEEDS)
                and complete['TEST_access'] is False and analysis['complete'] and analysis['banks'] == 15, 'Original full E family closed')
        require(cfg['source_receipts'] == plan['source_receipts'] and cfg['safe_roles'] == plan['safe_roles']
                and cfg['seeds'] == list(SEEDS) and (cfg['learning_rate'], cfg['max_updates'], cfg['patience_updates']) == (.001, 1000, 300),
                'Exact original source, roles, roster and optimizer/selector budget')
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
        owner = load_source('closed_E_head_owner', plan['source_receipts']['training_family']['path'])
        family = object.__new__(owner.Family)  # Constructor methods only; no original training/data admission or optimizer.
        family.cfg, family.torch, family.Model, family.set_seed = cfg, torch, Model, set_seed
        family.routes = load_source('closed_E_head_routes', plan['source_receipts']['common_routes']['path'])
        family.factors = load_source('closed_E_head_factors', plan['source_receipts']['factors']['path'])
        family.value_installer = load_source('closed_E_head_value_installer', plan['source_receipts']['value_installer']['path'])
        family.value_helper = load_source('closed_E_head_value_helper', plan['source_receipts']['value_helper']['path'])
        family._operation = 'endpoint'
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

        def guard():
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            require(rss <= 8 * 1024 ** 3 and torch.cuda.max_memory_allocated(device) <= 8 * 1024 ** 3,
                    'Finite diagnostic RSS/CUDA cap')

        with np.load(receipt(plan['safe_roles']['train']), allow_pickle=False) as a:
            require(set(a.files) == {'x', 'edge_index', 'ids', 'y'}, 'Exact safe TRAIN graph/role container')
            x, edges, train_ids, train_y = (a[k].copy() for k in ('x', 'edge_index', 'ids', 'y'))
        with np.load(receipt(plan['safe_roles']['valid']), allow_pickle=False) as a:
            require(set(a.files) == {'ids', 'y'}, 'Exact safe SELECT container')
            valid_ids, valid_y = (a[k].copy() for k in ('ids', 'y'))
        require(x.dtype == np.float32 and x.shape == (11701, 300) and np.isfinite(x).all()
                and edges.dtype == np.int64 and edges.shape == (2, 442907) and edges.min() >= 0 and edges.max() < 11701,
                'Unchanged full factual graph/features')
        for ids, y, rows in ((train_ids, train_y, 580), (valid_ids, valid_y, 5274)):
            require(ids.dtype == y.dtype == np.int64 and ids.shape == y.shape == (rows,) and len(np.unique(ids)) == rows
                    and ids.min() >= 0 and ids.max() < 11701 and np.isin(y, np.arange(10)).all(), 'Exact full role rows/classes')
        require(not np.intersect1d(train_ids, valid_ids).size, 'Disjoint TRAIN and SELECT')
        family.edge_metadata = dict(nodes=11701, edges=edges.shape[1], self_edges=int((edges[0] == edges[1]).sum()))
        tx, te = torch.from_numpy(x).to(device), torch.from_numpy(edges).to(device)
        batch = dict(x=tx, edge_index=te, graph=SimpleNamespace(edge_index=te))
        ids = torch.from_numpy(np.concatenate((train_ids, valid_ids))).to(device)
        yt, yv = torch.from_numpy(train_y), torch.from_numpy(valid_y)
        archive_index = {(v['seed'], v['arm']): v for v in plan['all_archive_receipts']}
        state_index = {(v['seed'], v['arm'], v['member']): v for v in plan['all_checkpoint_receipts']}
        features, originals, all_fits, predictions, collection = {}, {}, [], {}, []
        (output / 'FEATURES').mkdir()
        (output / 'FITS').mkdir()
        phase, native_states, trajectories = 'feature_collection', 0, 0
        for seed in SEEDS:
            for arm in ARMS:
                guard()
                item = archive_index[seed, arm]
                members = 1 if arm == 'equipped_M1' else 4
                with np.load(receipt(item), allow_pickle=False) as a:
                    require(set(a.files) == {'ids', 'y', 'raw_logits', 'probability_mean', 'member_errors', 'pooled_errors'}, 'Original exact6-field SELECT archive')
                    original = {name: a[name].copy() for name in a.files}
                require(np.array_equal(original['ids'], valid_ids) and np.array_equal(original['y'], valid_y)
                        and original['raw_logits'].dtype == np.float32 and original['raw_logits'].shape == (members, 5274, 10)
                        and original['probability_mean'].dtype == np.float32 and original['probability_mean'].shape == (5274, 10)
                        and original['member_errors'].dtype == original['pooled_errors'].dtype == np.bool_
                        and original['member_errors'].shape == (members, 5274) and original['pooled_errors'].shape == (5274,),
                        'Exact original aligned scores and native archived flag authority')
                require(np.isfinite(original['raw_logits']).all() and np.isfinite(original['probability_mean']).all(), 'Finite original anchors')
                original['strict_rival'] = strict_rival(np, original['raw_logits'], valid_y)
                original['pooled_nll'] = score(torch, np, torch.from_numpy(original['raw_logits']), yv)['pooled_nll']
                originals[seed, arm] = original
                representations, scores, heads = [], [], []
                bundles = 4 if arm == 'equipped_I4' else 1
                for member in range(bundles):
                    count = 4 if arm == 'E4' else 1
                    body, wrapper = family.native_body(seed, member, member, True, count)
                    model = wrapper if count > 1 else body
                    state = torch.load(receipt(state_index[seed, arm, member]), map_location='cpu', weights_only=True)
                    model.load_state_dict(state['model'], strict=True)
                    head = body.output_linear
                    require(isinstance(head, family.factors.FactorLinear) and head.weight.shape == (10, 128)
                            and head.bias is not None and head.bias.shape == (10,) and head.r.shape == (count, 128)
                            and head.s.shape == (count, 10), 'Actual original final affine/factor head ownership')
                    for parameter in model.parameters():
                        parameter.requires_grad_(False)
                    model.to(device).eval()
                    with torch.no_grad():
                        if count > 1:
                            z, H = model(dict(batch, ids=ids))
                        else:
                            with family.factors.member_context(body, 0):
                                zfull, Hfull = family._single_adapter.native(batch)
                            z, H = zfull[ids].unsqueeze(0), Hfull[ids].unsqueeze(0)
                        require(z.shape == (count, 5854, 10) and H.shape == (count, 5854, 128)
                                and z.dtype == H.dtype == torch.float32 and bool(torch.isfinite(z).all()) and bool(torch.isfinite(H).all()),
                                'Complete canonical pre-final-affine H and same-call original scores')
                        representations.append(H.detach().cpu().clone())
                        scores.append(z.detach().cpu().clone())
                    heads.append(copy.deepcopy(body.output_linear).cpu())
                    native_states += 1
                    trajectories += count
                    del model, body, wrapper, state, z, H, head
                    if count == 1:
                        del zfull, Hfull
                    family._single_adapter = None
                    torch.cuda.empty_cache()
                    guard()
                H, Z = torch.cat(representations), torch.cat(scores)
                replay = score(torch, np, Z[:, 580:], yv)
                event = dict(seed=seed, arm=arm, native_states=bundles, native_member_trajectories=members,
                             SELECT_replay_max_abs=float((Z[:, 580:] - torch.from_numpy(original['raw_logits'])).abs().max()),
                             SELECT_replay_member_flag_differences=int((replay['member_errors'] != original['member_errors']).sum()),
                             SELECT_replay_pooled_flag_differences=int((replay['pooled_errors'] != original['pooled_errors']).sum()),
                             replay_is_utility_gate=False)
                folder = output / 'FEATURES' / (arm + '_seed' + str(seed))
                folder.mkdir()
                path = folder / 'fixed_inputs.npz'
                np.savez(path, H_train=H[:, :580].numpy(), H_SELECT=H[:, 580:].numpy(), frozen_train_logits=Z[:, :580].numpy(),
                         saved_SELECT_logits=original['raw_logits'], train_ids=train_ids, train_y=train_y, SELECT_ids=valid_ids, SELECT_y=valid_y)
                head_path = folder / 'initial_heads.pt'
                torch.save([h.state_dict() for h in heads], head_path)
                event.update(features=file_receipt(path), initial_heads=file_receipt(head_path))
                collection.append(event)
                features[seed, arm] = (dict(H_train=H[:, :580], H_SELECT=H[:, 580:], Z_train=Z[:, :580],
                                           Z_SELECT=torch.from_numpy(original['raw_logits']), y_train=yt, y_SELECT=yv), heads)
                write(output / 'PROGRESS.json', dict(phase=phase, closed_feature_banks=len(collection), native_states=native_states,
                                                     native_member_trajectories=trajectories, complete=False, comparative_interpretation=False))
        require(native_states == 18 and trajectories == 27 and len(features) == 9, 'Entire original feature cohort collected')
        del tx, te, batch, ids, family
        torch.cuda.empty_cache()
        write(output / 'COMPLETE_FEATURES.json', dict(complete=True, banks=9, native_states=18, trajectories=27, collection=collection,
                                                    source_sha256=source_sha, plan_sha256=plan_sha, backbone_updates=0, TEST_access=False))
        phase = 'CPU_head_fits'
        for seed in SEEDS:
            for arm in ARMS:
                data, heads = features[seed, arm]
                for condition in CONDITIONS[arm]:
                    selected, records = [], []
                    for member in range(4 if arm == 'equipped_I4' else 1):
                        if arm == 'equipped_I4':
                            own = {name: value[member:member + 1] if name in ('H_train', 'H_SELECT', 'Z_train', 'Z_SELECT') else value
                                   for name, value in data.items()}
                            own_heads = [heads[member]]
                        else:
                            own, own_heads = data, heads
                        folder = output / 'FITS' / (arm + '_' + condition + '_seed' + str(seed) + '_member' + str(member))
                        logits, record = fit_head(torch, np, folder, own, own_heads, condition, guard)
                        record.update(seed=seed, arm=arm, optimizer_member=member)
                        selected.append(logits)
                        records.append(record)
                        all_fits.append(record)
                        write(output / 'PROGRESS.json', dict(phase=phase, closed_head_bundles=len(all_fits),
                                                             complete=False, comparative_interpretation=False))
                    value = score(torch, np, torch.cat(selected), yv)
                    prediction_path = output / (arm + '_' + condition + '_seed' + str(seed) + '_selected_VALID.npz')
                    np.savez(prediction_path, ids=valid_ids, y=valid_y, **{name: value[name] for name in
                             ('raw_logits', 'probability_mean', 'member_errors', 'pooled_errors')})
                    value['receipt'] = file_receipt(prediction_path)
                    predictions[seed, arm, condition] = value
        require(len(all_fits) == 39 and len(predictions) == 21, 'Entire fixed head roster closed before interpretation')
        phase = 'complete_descriptive_readout'
        allmask = np.ones(5274, dtype=bool)
        rows, comparisons, anchors = [], [], []
        for seed in SEEDS:
            for arm in ARMS:
                original = originals[seed, arm]
                anchors.append(dict(seed=seed, arm=arm, metrics=statistics(np, original, allmask), pooled_nll=original['pooled_nll'],
                                    custody=archive_index[seed, arm], archived_flags_are_authority=True))
                for condition in CONDITIONS[arm]:
                    fitted = predictions[seed, arm, condition]
                    E_original = originals[seed, 'E4']
                    E_common_error = E_original['strict_rival'] & E_original['pooled_errors']
                    rows.append(dict(seed=seed, arm=arm, condition=condition, metrics=statistics(np, fitted, allmask),
                                     pooled_nll=fitted['pooled_nll'], prediction=fitted['receipt'],
                                     versus_own_original=contrast(np, fitted, original, allmask),
                                     versus_E_original_strict_rival_pooled_errors=contrast(np, fitted, E_original, E_common_error),
                                     factual_classes=[dict(class_id=cl, metrics=statistics(np, fitted, valid_y == cl),
                                                          versus_own_original=contrast(np, fitted, original, valid_y == cl)) for cl in range(10)]))
            pairs = [('private_affine_residual', 'E4', 'common_affine_residual'),
                     ('private_affine_residual', 'E4', 'BE_head_continuation'),
                     ('BE_head_continuation', 'equipped_M1', 'BE_head_continuation'),
                     ('BE_head_continuation', 'equipped_I4', 'BE_head_continuation'),
                     ('common_affine_residual', 'equipped_M1', 'affine_residual'),
                     ('common_affine_residual', 'equipped_I4', 'private_affine_residual'),
                     ('private_affine_residual', 'equipped_M1', 'affine_residual'),
                     ('private_affine_residual', 'equipped_I4', 'private_affine_residual')]
            for candidate_condition, reference_arm, reference_condition in pairs:
                a, b = predictions[seed, 'E4', candidate_condition], predictions[seed, reference_arm, reference_condition]
                comparisons.append(dict(seed=seed, candidate='E4/' + candidate_condition,
                                        reference=reference_arm + '/' + reference_condition,
                                        whole_population=contrast(np, a, b, allmask),
                                        factual_classes=[dict(class_id=cl, **contrast(np, a, b, valid_y == cl)) for cl in range(10)]))
            for reference_arm in ('equipped_M1', 'equipped_I4'):
                for condition in ('original',) + CONDITIONS['E4']:
                    a = originals[seed, 'E4'] if condition == 'original' else predictions[seed, 'E4', condition]
                    b = originals[seed, reference_arm]
                    comparisons.append(dict(seed=seed, candidate='E4/' + condition, reference=reference_arm + '/original',
                                            whole_population=contrast(np, a, b, allmask),
                                            factual_classes=[dict(class_id=cl, **contrast(np, a, b, valid_y == cl)) for cl in range(10)]))
        for value in pinned:
            receipt(value)
        require(sha(__file__) == source_sha and sha(plan_path) == plan_sha, 'Admitted source and plan remained fixed')
        guard()
        summary = dict(complete=True, banks=21, head_optimizer_bundles=39, restored_native_states=18, native_member_feature_trajectories=27,
                       source_sha256=source_sha, plan_sha256=plan_sha, original_anchors=anchors, conditions=rows, comparisons=comparisons,
                       collection=collection, fits=all_fits, TRAIN_rows=580, SELECT_rows=5274,
                       seconds=time.monotonic() - started, process_lifetime_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       CPU_threads=1, head_device='cpu', backbone_updates=0, calibration_fits=0,
                       backend=dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                                    cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                    float32_matmul_precision=torch.get_float32_matmul_precision()),
                       TEST_access=False, current_Physics_Penn_access=False, numerical_replay_utility_gate=False,
                       scope='Retrospective fixed representations and original selected development; no head-bottleneck cause, global capacity certificate, new principle, two-stage method or unused confirmation',
                       dependency='Three readouts reuse same5274 nodes; no subgroup p values/CIs',
                       performance_scope='GPU feature extraction and CPU head fitting overlap unchanged live fits; no isolated speed claim',
                       automatic_new_GNN_fit=False)
        write(output / 'COMPLETE_HEAD_DIAGNOSTIC.json', summary)
        signal.alarm(0)
        print(json.dumps(dict(complete=True, banks=21, head_optimizer_bundles=39, backbone_updates=0, TEST_access=False)), flush=True)
    except BaseException as error:
        write(output / 'FAILURE.json', dict(complete=False, status='FAILED_RETAINED', phase=phase, type=type(error).__name__, message=str(error),
                                           source_sha256=source_sha, plan_sha256=plan_sha, TEST_access=False, current_Physics_Penn_access=False,
                                           backbone_updates=0, retry_or_rescue=False))
        raise


if __name__ == '__main__':
    main()
