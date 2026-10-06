#!/usr/bin/env python3
"""Bounded discarded full-WikiCS TRAIN checks and six complete epoch costs."""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
REPO = ROOT.parents[2]
SOURCE = PHASE/'wikics_polynormer_r_balanced_graph_supervision_pilot_preparation_20261007_v1'
SOURCE_SHA = '7257d81cf4cb9623f89cbfd75e78bfa023b5c3d9013e7ea32d4acd975dbd8c71'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
TOLERANCES = {'logits': (1e-5, 1e-6), 'gradients_moments': (1e-4, 2e-6), 'parameters': (1e-5, 1e-6), 'weight_balance': (1e-5, 2e-7)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''): digest.update(part)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def bound(record):
    path = (PHASE/record['path']).resolve(strict=True)
    if not path.is_relative_to(PHASE.resolve()) or not path.is_file() or sha(path) != record['sha256']: raise ValueError('Exact phase binding changed')
    return path


def helpers(shared, device, torch, np, Polynormer, PolynormerBoundaryFamily):
    # These three helper ASTs must match the sealed actual trainer exactly.
    def core(model): return model.core if shared else model

    def construct(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = Polynormer(300, 512, 10, local_layers=7, global_layers=2, in_dropout=.5, dropout=.5,
            global_dropout=.5, heads=1, beta=-1, pre_ln=False).to(device)
        model.reset_parameters(); model._global = False
        if shared: model = PolynormerBoundaryFamily(model, members=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0., betas=(.9, .999), eps=1e-8)
        names = [name for name, _ in model.named_parameters()]
        if len({id(p) for p in model.parameters()}) != len(names) or {id(p) for group in optimizer.param_groups for p in group['params']} != {id(p) for p in model.parameters()}:
            raise ValueError('Optimizer must own exactly this complete model')
        streams = [{'cpu': torch.get_rng_state().clone(), 'cuda': torch.cuda.get_rng_state().clone()}]
        for member in range(1, 4 if shared else 1):
            streams.append({'cpu': torch.Generator(device='cpu').manual_seed(seed+1009*member).get_state(),
                'cuda': torch.Generator(device=device).manual_seed(seed+1009*member).get_state()})
        return model, optimizer, streams, names

    @contextmanager
    def route_rng(stream):
        old_cpu, old_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        torch.set_rng_state(stream['cpu']); torch.cuda.set_rng_state(stream['cuda'])
        try: yield
        finally:
            stream.update(cpu=torch.get_rng_state().clone(), cuda=torch.cuda.get_rng_state().clone())
            torch.set_rng_state(old_cpu); torch.cuda.set_rng_state(old_cuda)

    return core, construct, route_rng


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); job = json.loads(args.job.read_text()); output = args.output.resolve()
    if (job.get('root_execution_authorized') is not True or job.get('source_review_approved') is not True
        or any(job.get(key) is not False for key in ('fits_authorized', 'VALID_values_access', 'TEST_access', 'retry'))):
        raise ValueError('Root must release exact discarded TRAIN-only qualification')
    if Path.cwd().resolve() != REPO.resolve() or str(REPO) != '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs' or socket.gethostname() != 'anogena-2-0':
        raise ValueError('Exact existing allocation repository required')
    if subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines() != [GPU] or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise ValueError('Exact singleton visible allocation GPU required')
    if output.exists() or not output.is_relative_to(PHASE.resolve()) or str(output) != job['output_directory'] or not output.parent.is_dir():
        raise ValueError('Fresh exact output required')
    if sha(__file__) != job['program_sha256'] or sha(ROOT/'MANIFEST.json') != job['qualification_manifest_sha256'] or sha(SOURCE/'MANIFEST.json') != SOURCE_SHA:
        raise ValueError('Reviewed immutable source changed')
    for folder in (ROOT, SOURCE):
        for row in json.loads((folder/'MANIFEST.json').read_text())['files']:
            if sha(folder/row['path']) != row['sha256']: raise ValueError('Sealed source closure changed')
    if not job['source_review_evidence']: raise ValueError('Source review evidence required')
    for record in job['source_review_evidence']: bound(record)
    if job['soft_seconds'] != 900 or job['hard_seconds'] != 1200 or job['external_hard_bound_confirmed'] is not True:
        raise ValueError('Root must bind proposed900/1200 ceilings without implicit expansion')
    interpreter = PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'
    if str(Path(sys.executable).absolute()) != str(interpreter): raise ValueError('Existing native interpreter required')
    trees = [ast.parse(path.read_text()) for path in (SOURCE/'run.py', Path(__file__))]
    for name in ('core', 'construct', 'route_rng'):
        matches = [[node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name] for tree in trees]
        if any(len(nodes) != 1 for nodes in matches) or ast.dump(matches[0][0], include_attributes=False) != ast.dump(matches[1][0], include_attributes=False):
            raise ValueError('Actual trainer helper AST differs: '+name)
    import numpy as np
    import torch
    import torch_geometric
    import torch_scatter
    import torch_sparse
    sys.path.insert(0, str(SOURCE))
    from vendor.native_polynormer import Polynormer
    from vendor.backbone_boundary_adapter import PolynormerBoundaryFamily, BoundaryProjector, set_boundary_identity_
    from weights import make_weights
    versions = {'torch': str(torch.__version__), 'numpy': np.__version__, 'PyG': torch_geometric.__version__,
        'torch_scatter': torch_scatter.__version__, 'torch_sparse': torch_sparse.__version__, 'CUDA': torch.version.cuda}
    if versions != job['runtime_versions'] or torch.cuda.device_count() != 1: raise ValueError('Exact legacy operator/runtime differs')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    projection_path = bound(job['TRAIN_manifest']); authority = json.loads(projection_path.read_text())
    data_path = bound(authority['available']); data = torch.load(data_path, map_location='cpu', weights_only=True)
    if set(data) != {'x', 'edge_index', 'train_ids', 'train_y'} or tuple(data['x'].shape) != (11701, 300) or data['edge_index'].shape != (2, 442907) or len(data['train_ids']) != 580:
        raise ValueError('Full official TRAIN-only roles/dimensions required')
    if authority['qualified_reader_deserializes_VALID_TEST_values'] is not False: raise ValueError('No VALID/TEST values may enter qualifier')
    output.mkdir(); device = torch.device('cuda:0')
    saved_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
    results = {'source_manifest_sha256': SOURCE_SHA, 'TRAIN_manifest_sha256': job['TRAIN_manifest']['sha256'], 'runtime': versions,
        'actual_trainer_helper_AST_equality': True, 'copy_checks': {}, 'mean_gradient_identity': {}, 'serial_reference': {}, 'epoch_costs': []}

    def cpu_tree(value):
        if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
        if isinstance(value, dict): return {key: cpu_tree(v) for key, v in value.items()}
        if isinstance(value, (tuple, list)): return type(value)(cpu_tree(v) for v in value)
        return value

    def compare(left, right, kind):
        if left.shape != right.shape or left.dtype != right.dtype or not bool(torch.isfinite(left).all() and torch.isfinite(right).all()): raise ValueError('Comparison geometry/finiteness differs')
        rtol, atol = TOLERANCES[kind]
        maximum = float((left-right).abs().max()) if left.numel() else 0.
        if not torch.allclose(left, right, rtol=rtol, atol=atol): raise ValueError(kind+' mismatch; max_abs='+str(maximum))
        return maximum

    def deadline():
        if time.monotonic()-started > 900: raise TimeoutError('Fixed qualifier soft limit; no partial-cycle pass')
        if torch.cuda.max_memory_reserved() > 24*1024**3: raise MemoryError('Fixed24GiB qualifier CUDA reserve ceiling')

    def inactive(name, family, global_stage):
        if global_stage: return name.startswith('local_head.' if family else 'pred_local.')
        return name.startswith(('core.global_attn.', 'core.ln.', 'global_head.')) if family else name.startswith(('global_attn.', 'ln.', 'pred_global.'))

    def loss_for(logits, weights):
        log_prob = torch.nn.functional.log_softmax(logits, 1)[train_ids]
        if weights is None: return torch.nn.functional.nll_loss(log_prob, train_y)
        per_node = torch.nn.functional.nll_loss(log_prob, train_y, reduction='none')
        return (weights[train_ids]*per_node).sum()/580

    def active_gradients(model, family, global_stage):
        result = {}
        for name, parameter in model.named_parameters():
            if inactive(name, family, global_stage):
                if parameter.grad is not None: raise ValueError('Stage-inactive gradient is not None: '+name)
            elif parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()): raise ValueError('Disconnected/nonfinite active gradient: '+name)
            else: result[name] = parameter.grad.detach().cpu().clone()
        return result

    def epoch(model, optimizer, streams, family, global_stage, weights, rng_context, reference=False):
        model.train(); (model.core if family else model)._global = global_stage; optimizer.zero_grad(set_to_none=True)
        names = [(name, p) for name, p in model.named_parameters() if not inactive(name, family, global_stage)]
        accumulated = {name: torch.zeros_like(p) for name, p in names} if reference else None
        losses = []
        for route in range(4 if family else 1):
            with rng_context(streams[route]):
                logits = model.forward_member(x, edge, route) if family else model(x, edge)
                value = loss_for(logits, None if weights is None else weights[route])
                if not bool(torch.isfinite(value)): raise ValueError('Nonfinite complete TRAIN loss')
                scaled = value/(4 if family else 1)
                if reference:
                    gradients = torch.autograd.grad(scaled, [p for _, p in names], allow_unused=False)
                    for (name, _), gradient in zip(names, gradients): accumulated[name].add_(gradient)
                    del gradients
                else: scaled.backward()
                losses.append(float(value.detach()))
            del logits, value, scaled
        if reference:
            for name, parameter in names: parameter.grad = accumulated[name]
        gradients = active_gradients(model, family, global_stage); optimizer.step(); torch.cuda.synchronize()
        for parameter in model.parameters():
            if not bool(torch.isfinite(parameter).all()): raise ValueError('Nonfinite committed parameter')
        for state in optimizer.state.values():
            if float(state['step']) != 1 or not all(bool(torch.isfinite(state[key]).all()) for key in ('exp_avg', 'exp_avg_sq')): raise ValueError('Invalid first native Adam state')
        deadline()
        return gradients, losses

    try:
        field_started = time.monotonic(); before_cpu, before_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        gw, iw, field_audit = make_weights(torch, data['edge_index'], data['train_ids'], 11701, 17)
        if not torch.equal(before_cpu, torch.get_rng_state()) or not torch.equal(before_cuda, torch.cuda.get_rng_state()): raise ValueError('Fields consumed global RNG')
        expected_rms = .5/field_audit['common_full_node_bound']; field_checks = {}
        for name, weights in (('graph', gw), ('IID', iw)):
            if not bool(torch.isfinite(weights).all() and (weights >= .5).all() and (weights <= 1.5).all()): raise ValueError('Field range/finiteness failed')
            means = weights[:, data['train_ids']].double().mean(1); pointwise = weights.double().mean(0)
            rms = (weights[:, data['train_ids']].double()-1).square().mean(1).sqrt()
            compare(means, torch.ones_like(means), 'weight_balance'); compare(pointwise, torch.ones_like(pointwise), 'weight_balance')
            compare(rms, torch.full_like(rms, expected_rms), 'weight_balance')
            field_checks[name] = {'TRAIN_mass_mean': means.tolist(), 'TRAIN_RMS_around_one': rms.tolist(), 'maximum_pointwise_mean_drift': float((pointwise-1).abs().max())}
        results['fields'] = {'audit': field_audit, 'checks': field_checks, 'seconds': time.monotonic()-field_started, 'global_RNG_unchanged': True}
        x = data['x'].to(device); edge = data['edge_index'].to(device); train_ids = data['train_ids'].to(device); train_y = data['train_y'].to(device)
        gw, iw = gw.to(device), iw.to(device)
        native_core, native_construct, native_rng = helpers(False, device, torch, np, Polynormer, PolynormerBoundaryFamily)
        family_core, family_construct, family_rng = helpers(True, device, torch, np, Polynormer, PolynormerBoundaryFamily)
        for global_stage in (False, True):
            stage = 'global' if global_stage else 'local'; deadline(); setup = time.monotonic()
            native, no, ns, _ = native_construct(17); family, fo, fs, _ = family_construct(17); set_boundary_identity_(family)
            native._global = global_stage; family.core._global = global_stage
            native.eval(); family.eval()
            native_named, family_named = dict(native.named_parameters()), dict(family.named_parameters())
            mapping = {'lin_in': 'stem', 'pred_local': 'local_head', 'pred_global': 'global_head'}
            for name, parameter in native_named.items():
                prefix, _, suffix = name.partition('.')
                equivalent = family_named[mapping[prefix]+'.weight'] if prefix in mapping and suffix == 'weight' else family_named[mapping[prefix]+'.B'][0] if prefix in mapping else family_named['core.'+name]
                if not torch.equal(parameter, equivalent): raise ValueError('Native-copy parameter bytes differ: '+name)
                if prefix in mapping and suffix == 'bias' and not all(torch.equal(parameter, row) for row in family_named[mapping[prefix]+'.B']): raise ValueError('Native bias rows differ')
            if {p.data_ptr() for p in native.parameters()} & {p.data_ptr() for p in family.parameters()}: raise ValueError('Native and neutral-family storage aliases')
            for module in family.modules():
                if isinstance(module, BoundaryProjector) and (module.B.stride(0) < module.B.shape[1] or not bool((module.R == 1).all() and (module.S == 1).all())): raise ValueError('Neutral private rows overlap or are not identity')
            with torch.no_grad():
                ordinary = native(x, edge); deltas = []
                for route in range(4):
                    value = family.forward_member(x, edge, route); deltas.append(compare(value, ordinary, 'logits')); del value
                del ordinary
            results['copy_checks'][stage] = {'exact_parameter_copy': True, 'independent_storage': True, 'private_bias_rows_distinct': True,
                'maximum_logit_delta': max(deltas), 'neutral_numeric_check_only': True, 'setup_seconds': time.monotonic()-setup}
            private_ids = {id(p) for module in family.modules() if isinstance(module, BoundaryProjector) for p in (module.R, module.S, module.B)}
            common_names = [name for name, p in family.named_parameters() if id(p) not in private_ids and not inactive(name, True, global_stage)]
            def deterministic_gradients(weights):
                family.eval(); fo.zero_grad(set_to_none=True)
                for route in range(4):
                    logits = family.forward_member(x, edge, route); value = loss_for(logits, None if weights is None else weights[route])/4
                    value.backward(); del logits, value
                return active_gradients(family, True, global_stage)
            unit = deterministic_gradients(None); gradient_checks = {}
            for label, weights in (('graph', gw), ('IID', iw)):
                actual = deterministic_gradients(weights)
                gradient_checks[label] = {'maximum_shared_gradient_delta': max(compare(unit[name], actual[name], 'gradients_moments') for name in common_names), 'shared_parameter_tensors': len(common_names)}
                del actual; deadline()
            results['mean_gradient_identity'][stage] = gradient_checks
            del native, no, ns, family, fo, fs, native_named, family_named, unit; torch.cuda.empty_cache()
            # Actual non-neutral source GNNM with separate paired route dropout.
            family, optimizer, streams, _ = family_construct(17); family.core._global = global_stage
            original = {'model': cpu_tree(family.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()), 'streams': cpu_tree(streams)}
            actual_gradients, actual_losses = epoch(family, optimizer, streams, True, global_stage, gw, family_rng)
            actual_state = {'model': cpu_tree(family.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()), 'streams': cpu_tree(streams)}
            family.load_state_dict(original['model']); optimizer.load_state_dict(original['optimizer']); streams = cpu_tree(original['streams'])
            reference_gradients, reference_losses = epoch(family, optimizer, streams, True, global_stage, gw, family_rng, reference=True)
            grad_delta = max(compare(actual_gradients[name], reference_gradients[name], 'gradients_moments') for name in actual_gradients)
            parameter_delta = max(compare(actual_state['model'][name], value.detach().cpu(), 'parameters') for name, value in family.state_dict().items())
            actual_moments = actual_state['optimizer']['state']; ref_moments = cpu_tree(optimizer.state_dict())['state']
            moment_delta = max(compare(actual_moments[key][name], ref_moments[key][name], 'gradients_moments') for key in actual_moments for name in ('exp_avg', 'exp_avg_sq'))
            if not all(torch.equal(a['cpu'], b['cpu']) and torch.equal(a['cuda'], b['cuda']) for a, b in zip(actual_state['streams'], streams)): raise ValueError('Replayed route dropout state differs')
            results['serial_reference'][stage] = {'gradient_max_abs': grad_delta, 'parameter_max_abs': parameter_delta, 'moment_max_abs': moment_delta,
                'actual_TRAIN_losses': actual_losses, 'reference_TRAIN_losses': reference_losses, 'dropout_streams_exact': True,
                'reference': 'Per-route autograd.grad accumulation + native Adam; never retains four GAT graphs.'}
            del family, optimizer, streams, original, actual_gradients, reference_gradients, actual_state, actual_moments, ref_moments; torch.cuda.empty_cache()
            for architecture in ('native1', 'shared4', 'independent4'):
                deadline(); setup = time.monotonic(); family_kind = architecture == 'shared4'; size = 4 if architecture == 'independent4' else 1
                factory = family_construct if family_kind else native_construct; rng = family_rng if family_kind else native_rng
                bank = [factory(17+1009*member if size == 4 else 17) for member in range(size)]
                all_parameters = [p for model, _, _, _ in bank for p in model.parameters()]
                if len({id(p) for p in all_parameters}) != len(all_parameters) or len({p.data_ptr() for p in all_parameters}) != len(all_parameters) or len({id(opt) for _, opt, _, _ in bank}) != size:
                    raise ValueError('Independent model/optimizer ownership is not disjoint')
                torch.cuda.synchronize(); setup_seconds = time.monotonic()-setup
                torch.cuda.reset_peak_memory_stats(); epoch_started = time.monotonic()
                losses = []
                for member, (model, opt, stream, _) in enumerate(bank):
                    weights = gw if family_kind else gw[member:member+1] if size == 4 else None
                    gradients, values = epoch(model, opt, stream, family_kind, global_stage, weights, rng); losses.extend(values); del gradients
                torch.cuda.synchronize()
                results['epoch_costs'].append({'architecture': architecture, 'stage': stage, 'complete_TRAIN_nodes_per_member': 580,
                    'public_nodes': 11701, 'directed_edges': 442907, 'TRAIN_forward_backwards': 4 if architecture != 'native1' else 1,
                    'ordinary_Adam_updates': size, 'weight_case': 'ordinary' if architecture == 'native1' else 'graph',
                    'setup_seconds': setup_seconds, 'complete_epoch_seconds': time.monotonic()-epoch_started,
                    'peak_CUDA_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_CUDA_reserved_bytes': torch.cuda.max_memory_reserved(),
                    'TRAIN_losses': losses, 'fresh_native_member_seeds': [17+1009*m for m in range(size)], 'independent_ownership_verified': True,
                    'global_state_note': 'Fresh untrained global-mode cost; no local100 warm run or predictive claim.' if global_stage else 'Fresh local-mode cost.'})
                write(output/'PROGRESS.json', {'completed_epoch_costs': len(results['epoch_costs']), 'total_epoch_costs': 6, 'stage': stage,
                    'architecture': architecture, 'VALID_TEST_values_access': False, 'states_discarded': True})
                del bank, all_parameters, model, opt, stream; torch.cuda.empty_cache(); deadline()
        if len(results['epoch_costs']) != 6: raise ValueError('All six complete actual TRAIN costs required')
        results.update(passed=True, scope='discarded_full_WikiCS_TRAIN_numerical_and_complete_epoch_cost', thresholds=TOLERANCES,
            job_sha256=sha(args.job), qualification_manifest_sha256=job['qualification_manifest_sha256'], states_discarded=True,
            VALID_values_access=False, TEST_access=False, full1100_fits=0, predictive_verdict=None, inclusive_seconds=time.monotonic()-started)
    except BaseException as error:
        write(output/'FAILURE.json', {'error': type(error).__name__+': '+str(error), 'partial_checks_preserved': True,
            'states_discarded': True, 'VALID_values_access': False, 'TEST_access': False, 'retry': False, 'results': results})
        raise
    finally:
        random.setstate(saved_rng[0]); np.random.set_state(saved_rng[1]); torch.set_rng_state(saved_rng[2]); torch.cuda.set_rng_state(saved_rng[3])
    numpy_state = np.random.get_state()
    restored = random.getstate() == saved_rng[0] and numpy_state[0] == saved_rng[1][0] and numpy_state[2:] == saved_rng[1][2:] and np.array_equal(numpy_state[1], saved_rng[1][1]) and torch.equal(torch.get_rng_state(), saved_rng[2]) and torch.equal(torch.cuda.get_rng_state(), saved_rng[3])
    if not restored: raise ValueError('Qualification caller RNG restoration failed')
    results['caller_RNG_restored'] = True; write(output/'QUALIFICATION.json', results)
    print(json.dumps({'passed': True, 'cost_epochs': 6, 'TRAIN_only': True, 'states_discarded': True}), flush=True)


if __name__ == '__main__': main()
