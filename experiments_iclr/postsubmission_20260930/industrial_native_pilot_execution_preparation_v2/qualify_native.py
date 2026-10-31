"""Full admitted graph disposable checks. No useful fit/checkpoint is retained."""
from __future__ import annotations
import copy
import json
import time
import numpy as np
import scipy.special
import torch
import delu
from sklearn.metrics import average_precision_score
from native_model import Model
from native_optimizer import make_optimizer, make_parameter_groups
from public_preprocessor import load_public, transform_public
from public_reference import reference_public_load, postprocess_reference, reference_transform
from torch_coo import replace_native_primitives
from boundary_adapter import build_boundary_family, transport_adamw, replace_graph_primitives_for_ad
from ad_qualifier import bind_common_model, qualify_gradient_interface
from closed_inputs import write_json

CHECK_LOG = None
PROGRESS_PATH = None
PROGRESS = None


def mark(stage, counter=None):
    if PROGRESS is None:
        raise RuntimeError('Disposable operation lacks qualification cost registry')
    PROGRESS['stage'] = stage
    if counter is not None:
        PROGRESS[counter] = PROGRESS.get(counter, 0) + 1
    write_json(PROGRESS_PATH, PROGRESS)


def optimizer_step(optimizer, scope):
    mark(scope + ' optimizer step attempted', 'attempted_optimizer_steps')
    optimizer.step()
    mark(scope + ' optimizer step completed', 'completed_optimizer_steps')


def difference(a, b, atol=1e-6, rtol=1e-5):
    if a.shape != b.shape:
        raise ValueError('Parity shapes differ')
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError('Nonfinite parity tensor')
    passed = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
    error = float((a.detach().double() - b.detach().double()).abs().max()) if a.numel() else 0.
    return {'passed': passed, 'max_absolute_error': error, 'atol': atol, 'rtol': rtol}


def require(check, name):
    if CHECK_LOG is not None:
        with CHECK_LOG.open('a') as file:
            file.write(json.dumps({'name': name, **check}, allow_nan=False) + '\n')
    if not check['passed']:
        raise ValueError('Qualification failed: ' + name + ' ' + str(check))
    return {'name': name, **check}


def make_model(public, config, device):
    model_config = copy.deepcopy(config['model'])
    if model_config['n_classes'] != 2:
        raise ValueError('Known binary class count differs')
    return Model(n_features=public.width, n_num_features=public.numerical_width, **model_config).to(device)


def preprocessing_checks(inputs, roles, compact, config):
    public = load_public(inputs / 'tolokers-2', roles, compact)
    reference = postprocess_reference(reference_public_load(inputs / 'tolokers-2', internal_split_name='RL'))
    records = []
    for key, candidate in public.features.items():
        original = reference[key]
        passed = (candidate is None and original is None) or (
            candidate is not None and original is not None and candidate.dtype == original.dtype
            and candidate.shape == original.shape and np.array_equal(candidate, original, equal_nan=True))
        records.append(require({'passed': passed}, 'raw public feature parity ' + key))
    for role in ('train', 'val', 'test'):
        records.append(require({'passed': np.array_equal(public.official_masks[role], reference['masks'][role])},
                               'official public mask parity ' + role))
    for role in ('train', 'val'):
        effective = np.zeros(public.node_count, dtype=bool)
        effective[compact[role]['rows']] = True
        effective &= reference['masks'][role]
        records.append(require({'passed': np.array_equal(public.source_masks[role], effective)},
                               'compact nonmissing source mask parity ' + role))
    src, dst = public.graph.edges(order='eid')
    ref_src, ref_dst = reference['graph'].edges(order='eid')
    records.append(require({'passed': torch.equal(src, ref_src) and torch.equal(dst, ref_dst)
                            and public.graph.idtype == reference['graph'].idtype
                            and public.graph.num_nodes() == reference['graph'].num_nodes()}, 'native graph eid/dtype/node parity'))
    records.append(require({'passed': not bool((src == dst).any())}, 'source-faithful no loops'))
    pairs = set(zip(src.tolist(), dst.tolist()))
    records.append(require({'passed': len(pairs) == len(src) and all((v, u) in pairs for u, v in pairs)},
                           'simple bidirected full topology'))
    policies = config['transform']['features']
    public = transform_public(public, policies)
    reference = reference_transform(reference, policies)
    for key, candidate in public.features.items():
        original = reference[key]
        passed = (candidate is None and original is None) or (
            candidate is not None and original is not None and candidate.dtype == original.dtype
            and candidate.shape == original.shape and np.array_equal(candidate, original))
        records.append(require({'passed': passed}, 'native transformed feature parity ' + key))
    return public, records


def train_once(model, optimizer, graph, numerical, other, train_rows, train_labels, family=False):
    model.eval()  # Frozen qualification dropout-off; this is not a useful fit.
    optimizer.zero_grad(set_to_none=True)
    mark('disposable family' if family else 'disposable native', 'attempted_training_forwards')
    logits = model(graph, numerical, other)
    mark('disposable training forward completed', 'completed_training_forwards')
    if family:
        loss = torch.stack([torch.nn.functional.cross_entropy(z[train_rows], train_labels) for z in logits]).mean()
    else:
        loss = torch.nn.functional.cross_entropy(logits[train_rows], train_labels)
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite disposable source loss')
    mark('disposable backward attempted', 'attempted_backwards')
    loss.backward()
    mark('disposable backward completed', 'completed_backwards')
    return logits, loss


def native_torch_parity(donor, donor_optimizer, config, graph, numerical, other, rows, labels):
    native = copy.deepcopy(donor).eval()
    original_opt = make_optimizer(native, config['optimizer'])
    original_opt.load_state_dict(copy.deepcopy(donor_optimizer.state_dict()))
    coo = replace_native_primitives(copy.deepcopy(donor), graph).eval()
    options = {k: tuple(v) if k == 'betas' else v for k, v in config['optimizer'].items() if k != 'type'}
    coo_opt = torch.optim.AdamW(make_parameter_groups(coo), **options)
    coo_opt.load_state_dict(copy.deepcopy(donor_optimizer.state_dict()))
    checks = []
    zn, _ = train_once(native, original_opt, graph, numerical, other, rows, labels)
    zt, _ = train_once(coo, coo_opt, graph, numerical, other, rows, labels)
    checks.append(require(difference(zn, zt), 'native/Torch full output'))
    pn, pt = dict(native.named_parameters()), dict(coo.named_parameters())
    if list(pn) != list(pt):
        raise ValueError('Native/Torch named parameter order differs')
    for name in pn:
        checks.append(require(difference(pn[name].grad, pt[name].grad, rtol=1e-4), 'native/Torch gradient ' + name))
    optimizer_step(original_opt, 'DGL parity')
    optimizer_step(coo_opt, 'Torch parity')
    for name in pn:
        checks.append(require(difference(pn[name], pt[name]), 'native/Torch AdamW parameter ' + name))
        sn, st = original_opt.state[pn[name]], coo_opt.state[pt[name]]
        if sn.keys() != st.keys():
            raise ValueError('Native/Torch AdamW keys differ')
        for key in sn:
            checks.append(require(difference(sn[key], st[key]), 'native/Torch AdamW ' + name + '/' + key))
    with torch.no_grad():
        checks.append(require(difference(native(graph, numerical, other), coo(graph, numerical, other)),
                              'native/Torch post-step logits'))
    return {'passed': True, 'checks': checks, 'populated_native_history': True,
            'dropout_off': True, 'full_graph': True, 'useful_checkpoint_retained': False}


def boundary_parity(donor, donor_optimizer, config, graph, numerical, other, rows, labels, members):
    native = copy.deepcopy(donor).eval()
    native_opt = make_optimizer(native, config['optimizer'])
    native_opt.load_state_dict(copy.deepcopy(donor_optimizer.state_dict()))
    family = build_boundary_family(native, members).eval()
    family_opt = transport_adamw(native, native_opt, family)
    from boundary_adapter import map_native_parameter_name
    named_native, named_family = dict(native.named_parameters()), dict(family.named_parameters())
    native_groups = {id(parameter): group for group in native_opt.param_groups for parameter in group['params']}
    family_groups = {id(parameter): group for group in family_opt.param_groups for parameter in group['params']}
    option_checks = []
    for name, parameter in named_native.items():
        target_name, private = map_native_parameter_name(name)
        original_options = {key: value for key, value in native_groups[id(parameter)].items() if key not in ('params', 'param_names')}
        expected_options = copy.deepcopy(original_options)
        if private:
            expected_options['eps'] /= members
        actual_options = {key: value for key, value in family_groups[id(named_family[target_name])].items() if key not in ('params', 'param_names')}
        option_checks.append(require({'passed': expected_options == actual_options}, 'native/family optimizer group options ' + name))
    for boundary, native_weight in (('stem', 'input.0.weight'), ('head', 'output.3.weight')):
        expected_options = {key: value for key, value in native_groups[id(named_native[native_weight])].items() if key not in ('params', 'param_names')}
        for suffix in ('R', 'S'):
            parameter = named_family[boundary + '.' + suffix]
            actual_options = {key: value for key, value in family_groups[id(parameter)].items() if key not in ('params', 'param_names')}
            option_checks.append(require({'passed': expected_options == actual_options and not family_opt.state.get(parameter)},
                                         'native/family new factor options and empty history ' + boundary + '.' + suffix))
    for name, parameter in family.named_parameters():
        if name.endswith('.R') or name.endswith('.S'):
            parameter.requires_grad_(False)
    zn, _ = train_once(native, native_opt, graph, numerical, other, rows, labels)
    zf, _ = train_once(family, family_opt, graph, numerical, other, rows, labels, family=True)
    checks = option_checks + [require(difference(zn, member), 'native/K' + str(members) + ' identity') for member in zf]
    native_params, family_params = dict(native.named_parameters()), dict(family.named_parameters())
    for name, p in native_params.items():
        target_name, private = map_native_parameter_name(name)
        q = family_params[target_name]
        expected = p.grad.expand_as(q) if private else p.grad
        actual = q.grad * members if private else q.grad
        checks.append(require(difference(expected, actual, rtol=1e-4), 'native/family gradient ' + name))
    optimizer_step(native_opt, 'native boundary parity')
    optimizer_step(family_opt, 'K' + str(members) + ' boundary parity')
    for name, p in native_params.items():
        target_name, private = map_native_parameter_name(name)
        q = family_params[target_name]
        checks.append(require(difference(p.expand_as(q) if private else p, q), 'native/family parameter ' + name))
        left, right = native_opt.state[p], family_opt.state[q]
        if left.keys() != right.keys():
            raise ValueError('Native/family optimizer keys differ')
        for key in left:
            scale = members if private and key == 'exp_avg' else members ** 2 if private and key in ('exp_avg_sq', 'max_exp_avg_sq') else 1
            expected = left[key].expand_as(right[key]) if private and key != 'step' else left[key]
            checks.append(require(difference(expected, right[key] * scale), 'native/family state ' + name + '/' + key))
    with torch.no_grad():
        zn, zf = native(graph, numerical, other), family(graph, numerical, other)
        checks += [require(difference(zn, member), 'native/family post-step logits') for member in zf]
    return {'passed': True, 'members': members, 'checks': checks, 'dropout_off': True,
            'factor_updates_frozen': True, 'populated_history': True, 'full_graph': True,
            'useful_checkpoint_retained': False}


def qualify(inputs, roles, compact, matrix, device, output):
    global CHECK_LOG, PROGRESS_PATH, PROGRESS
    records = []
    for row in matrix['configurations']:
        CHECK_LOG = output / (row['id'] + '_checks.jsonl')
        PROGRESS_PATH = output / (row['id'] + '_progress.json')
        PROGRESS = {'config_id': row['id'], 'seed': 17, 'status': 'STARTED',
                    'scheduled_disposable_optimizer_steps': 7, 'completed_optimizer_steps': 0,
                    'attempted_optimizer_steps': 0, 'useful_fit_updates': 0,
                    'scheduled_AD': {'directions': 3, 'central_FD_epsilons': [1e-3, 3e-4]},
                    'test_labels': 'CLOSED'}
        mark('public preprocessing and independent loader parity')
        config = row['native_config']
        started = time.perf_counter()
        delu.random.seed(17)
        public, preprocessing = preprocessing_checks(inputs, roles, compact, config)
        graph, numerical, other = public.tensors(device)
        train_rows = torch.tensor(compact['train']['rows'], device=device, dtype=torch.long)
        train_labels = torch.tensor(compact['train']['labels'], device=device, dtype=torch.long)
        donor = make_model(public, config, device).eval()
        require({'passed': sum(p.numel() for p in donor.parameters()) == 512 * public.width + 2907150},
                'source-derived native parameter formula')
        optimizer = make_optimizer(donor, config['optimizer'])
        train_once(donor, optimizer, graph, numerical, other, train_rows, train_labels)
        optimizer_step(optimizer, 'native populated-history donor')
        if not any(bool(state['exp_avg'].abs().sum() > 0) for state in optimizer.state.values()):
            raise ValueError('Disposable native donor lacks populated nonzero history')
        torch_check = native_torch_parity(donor, optimizer, config, graph, numerical, other, train_rows, train_labels)
        boundary_checks = [boundary_parity(donor, optimizer, config, graph, numerical, other,
                                          train_rows, train_labels, members) for members in (1, 4)]
        common = build_boundary_family(donor, 1).eval()
        src, dst = graph.edges(order='eid')
        replace_graph_primitives_for_ad(common, src.to(torch.int64), dst.to(torch.int64), graph.num_nodes())
        theta, logits_fn, binding = bind_common_model(common, 'NativeGAT-Tolokers2',
                                                     (graph, numerical, other), lambda z: z[0])
        mark('full-graph K1 cold JVP/VJP/FD attempted')
        ad_check = qualify_gradient_interface(logits_fn, theta, train_rows, train_labels, 17 + 90000)
        PROGRESS['completed_AD_receipt'] = ad_check
        mark('full-graph K1 cold JVP/VJP/FD completed')
        require(ad_check, 'full-graph K1 cold JVP/VJP/FD')
        with torch.no_grad():
            logits = donor(graph, numerical, other).float().cpu().numpy()
        val_rows = compact['val']['rows']
        val_labels = np.array(compact['val']['labels'], dtype=np.int64)
        probs = scipy.special.softmax(logits, axis=-1)[val_rows, 1]
        native_ap = float(average_precision_score(val_labels, probs))
        alternative_ap = float(average_precision_score(val_labels, torch.from_numpy(logits[val_rows]).softmax(-1)[:, 1].numpy()))
        require({'passed': abs(native_ap - alternative_ap) <= 1e-8}, 'native/closed AP rank parity')
        result = {'config_id': row['id'], 'seed': 17, 'full_graph': True, 'passed': True,
                  'node_count': public.node_count, 'edge_count': graph.num_edges(),
                  'encoded_width': public.width, 'numerical_width': public.numerical_width,
                  'native_parameter_count': sum(p.numel() for p in donor.parameters()),
                  'column_lineage': public.column_lineage, 'preprocessing_checks': preprocessing,
                  'native_torch_parity': torch_check, 'boundary_identity_transport_checks': boundary_checks,
                  'cold_AD': ad_check, 'factor_binding': binding, 'native_validation_ap': native_ap,
                  'seconds': time.perf_counter() - started,
                  'disposable_optimizer_steps': {'native_donor': 1, 'native_torch_pair': 2,
                                                 'native_K1_pair': 2, 'native_K4_pair': 2},
                  'useful_checkpoint_retained': False, 'test_access': False}
        write_json(output / (row['id'] + '_qualification.json'), result)
        PROGRESS['status'] = 'SUCCEEDED'
        mark('configuration qualification complete')
        records.append(result)
        # Avoid keeping callable pullbacks/models across configurations.
        del donor, optimizer, common, theta, logits_fn, graph, numerical, other, public
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    CHECK_LOG = None
    PROGRESS_PATH = None
    PROGRESS = None
    return {'passed': True, 'records': records, 'scope': 'All three configs, cold seed17, complete graph; '
            'no selected-fit or actual-warm extension qualification claim.', 'test_labels': 'CLOSED'}
