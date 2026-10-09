"""Disabled full-input Q/K engineering qualifier; imports only stdlib at import.

One local and one global TRAIN F update per fresh operator/kind, never a fit.
No VALID prediction, metric, selector, checkpoint, resume or cleanup of others.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import resource
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ADAPTER_DIR = 'pre_sigmoid_qk_operator_adapter_source_20261009_v1'
ADAPTER_MANIFEST = '0e82d0c94d0f7bb5c2ca849aad0c5b37a7dac26c8d3eafa203ea4a8d6230187c'
ADAPTER_PROGRAM = 'be9974399e090b424c57fe4f77ba10126f575d8e89454945ce51f0f7c61b9631'
OPERATORS = ('native_tied', 'active_reversible_exp', 'pre_sigmoid_split', 'full_qk')
KINDS = ('single', 'independent4', 'be_init')
ENGINEERING_SEED = 20261091
TOLERANCE = 2e-5
OUTPUT_NAME = 'pre_sigmoid_qk_full_input_qualification_output_20261009_v2'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def own_resources(torch):
    torch.cuda.synchronize(0)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return dict(process_RSS_high_water_bytes=int(usage.ru_maxrss) * (1 if sys.platform == 'darwin' else 1024),
                process_user_seconds=usage.ru_utime, process_system_seconds=usage.ru_stime,
                CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))


def streams(session):
    return [{k: v.clone() for k, v in row.items()} for row in session.streams]


def same_streams(left, right, torch):
    return len(left) == len(right) and all(a.keys() == b.keys() and
        all(torch.equal(a[k], b[k]) for k in a) for a, b in zip(left, right))


def rng(session):
    return (random.getstate(), session.np.random.get_state(), session.torch.get_rng_state().clone(),
            session.torch.cuda.get_rng_state(0).clone(), streams(session))


def same_rng(before, session):
    after = rng(session)
    np, torch = session.np, session.torch
    return (before[0] == after[0] and before[1][0] == after[1][0]
            and np.array_equal(before[1][1], after[1][1]) and before[1][2:] == after[1][2:]
            and torch.equal(before[2], after[2]) and torch.equal(before[3], after[3])
            and same_streams(before[4], after[4], torch))


def ownership(session):
    named = list(session.model.named_parameters())
    ids = [id(p) for opt in session.optimizers for group in opt.param_groups for p in group['params']]
    require(len(ids) == len(set(ids)) == len(named) and set(ids) == {id(p) for _, p in named},
            'Every actual coordinate needs exactly one optimizer owner')
    require(len(session.optimizers) == (4 if session.arm == 'independent4' else 1), 'Declared native Adam ownership')
    if session.arm == 'independent4':
        sets = [{id(p) for p in body.parameters()} for body in session.model.models]
        require(all(not a & b for i, a in enumerate(sets) for b in sets[i+1:]), 'Independent bodies share a coordinate')
    for opt in session.optimizers:
        for group in opt.param_groups:
            require(group['lr'] == .001 and group['eps'] == 1e-8 and group['weight_decay'] == 0
                    and group['betas'] == (.9, .999) and not group['amsgrad'], 'Unchanged native Adam recipe')
    return dict(parameter_tensors=len(named), parameter_scalars=sum(p.numel() for _, p in named),
                optimizer_count=len(session.optimizers), unique_owned_parameter_tensors=len(ids))


def targets(session):
    return {name: p for name, p in session.model.named_parameters()
            if '.global_attn.k_lins.' in name or '.global_attn.q_lins.' in name}


def initial_coordinates(session, operator, original_keys, original_count):
    torch = session.torch
    mode_members = 1 if session.model.independent else session.model.members
    scale_count = 0
    for body_index, model in enumerate(session.model.models):
        attn = model.body.global_attn
        require(attn._operator_mode == operator and attn.qk_shared == (operator != 'full_qk'), 'Actual native Q/K mode')
        require(attn.num_layers == 2 and len(attn.k_lins) == 2 and attn.hidden_channels == 512 and attn.heads == 1,
                'Native two-layer full-width Q/K shape')
        for index, lin in enumerate(attn.k_lins):
            old_weight, old_bias = original_keys[body_index][index]
            require(lin.weight is old_weight and lin.bias is old_bias, 'Key W and outside bias must retain original objects')
            require(lin.r.shape == lin.s.shape == (mode_members, 512)
                    and torch.equal(lin.r, torch.ones_like(lin.r))
                    and torch.equal(lin.s, torch.ones_like(lin.s)), 'Global factors must start at one')
            extra = 'delta' if operator == 'pre_sigmoid_split' else 'gamma' if operator == 'active_reversible_exp' else None
            if extra:
                value = getattr(lin, extra)
                require(value.shape == (mode_members, 512) and int(torch.count_nonzero(value)) == 0,
                        'Additional relational scales must start at zero')
                scale_count += value.numel()
            if operator == 'full_qk':
                q = attn.q_lins[index]
                require(set(dict(q.named_parameters())) == set(dict(lin.named_parameters())), 'Copy-matched Q/K fields')
                for leaf, key in lin.named_parameters():
                    query = dict(q.named_parameters())[leaf]
                    require(query is not key and torch.equal(query, key), 'Query maps must copy values into independent parameters')
        require(hasattr(attn, 'q_lins') == (operator == 'full_qk'), 'Only full_qk has independent query projections')
    expected_scales = 2 * 512 * session.model.members if operator in ('pre_sigmoid_split', 'active_reversible_exp') else 0
    require(scale_count == expected_scales, 'Exact added direction/postscale count')
    ordinary = 2 * 2 * 512 * session.model.members if session.model.independent else 0
    added_query = len(session.model.models) * 2 * (512 * 512 + 512 + 2 * mode_members * 512) if operator == 'full_qk' else 0
    require(sum(p.numel() for p in session.model.parameters()) - original_count == ordinary + scale_count + added_query,
            'Exact added coordinate count')
    return dict(additional_baseline_global_factor_scalars=ordinary, added_delta_or_gamma_scalars=scale_count,
                added_full_query_scalars=added_query, original_key_weight_and_outside_bias_objects_preserved=True,
                outside_bias_convention='Pinned FactorLinear and immutable adapter: W(x*r)*(s±delta)+b; exp control after sigmoid')


def instrument(session, operator, counters):
    torch = session.torch
    original = session.model.member_forward
    def forward(self, batch, member):
        require(batch['x'].shape == (11701, 300) and batch['edge_index'].shape == (2, 442907)
                and batch['ids'].shape == (580,), 'Every member uses complete native graph and TRAIN query rows')
        counters['member_forward_attempts'] += 1
        result = original(batch, member)
        require(result[0].shape == (580, 10) and result[1].shape == (580, 512), 'Complete native prediction/representation fields')
        counters['member_forwards_completed'] += 1
        return result
    session.model.member_forward = types.MethodType(forward, session.model)
    for model in session.model.models:
        attn = model.body.global_attn
        if operator in ('pre_sigmoid_split', 'active_reversible_exp'):
            original_pair = attn._operator_qk_pair
            def pair(self, x, index, implementation=original_pair):
                q, k = implementation(x, index)
                require(x.shape == q.shape == k.shape == (11701, 512), 'Full native Q/K activation fields')
                if operator == 'active_reversible_exp':
                    require(q is k, 'Active reversible Q/K must use the same activation')
                counters['global_QK_pair_calls'] += 1
                counters['large_QK_projection_calls'] += 1
                return q, k
            attn._operator_qk_pair = types.MethodType(pair, attn)
        # The original class forward calls these key modules even in patched
        # modes. Candidate patched pairs bypass them and count their one direct
        # projection above, so reference and candidate events do not double-count.
        def projection_hook(_, inputs, output):
            require(inputs[0].shape == output.shape == (11701, 512), 'Full native Q/K projection fields')
            counters['large_QK_projection_calls'] += 1
        for lin in attn.k_lins:
            lin.register_forward_hook(projection_hook)
        if operator == 'full_qk':
            for lin in attn.q_lins:
                lin.register_forward_hook(projection_hook)
    for optimizer in session.optimizers:
        original_step = optimizer.step
        def step(*args, implementation=original_step, **kwargs):
            require(session.operator_work['member_view_backwards'] == 2 * session.model.members * (session.steps + 1),
                    'Every old-state member/view backward must precede every Adam')
            counters['observed_Adam_calls'] += 1
            return implementation(*args, **kwargs)
        optimizer.step = step


def same_state_identity(session, batch, receipt):
    """Original native and candidate forwards on this one current parameter bank."""
    torch = session.torch
    parameters = tuple(session.model.parameters())
    parameter_ids = tuple(id(p) for p in parameters)
    versions = tuple(p._version for p in parameters)
    batch_versions = {k:(id(v),v._version) for k,v in batch.items()}
    before_rng = rng(session)
    saved = []
    for model in session.model.models:
        attn = model.body.global_attn
        for lin in attn.k_lins:
            for name in ('delta','gamma'):
                if hasattr(lin,name):
                    require(int(torch.count_nonzero(getattr(lin,name))) == 0, 'Extra scales changed before first global use')
        if session.operator_binding['operator'] == 'full_qk':
            for query,key in zip(attn.q_lins,attn.k_lins):
                require(all(torch.equal(dict(query.named_parameters())[name],value)
                            for name,value in key.named_parameters()), 'Full query copy changed while globally unused')
    try:
        for model in session.model.models:
            attn = model.body.global_attn
            had_forward = 'forward' in attn.__dict__
            saved.append((attn,had_forward,attn.__dict__.get('forward'),attn.qk_shared))
            attn.forward = types.MethodType(type(attn).forward,attn)
            attn.qk_shared = True
        with torch.no_grad():
            native_logits,native_representation = session.forward(batch)
        require(bool(torch.isfinite(native_logits).all()) and bool(torch.isfinite(native_representation).all()),
                'Finite same-state original native TRAIN-query reference')
        reference = native_logits.detach().cpu()
        del native_logits,native_representation
    finally:
        for attn,had_forward,old_forward,old_flag in reversed(saved):
            if had_forward:
                attn.forward = old_forward
            else:
                delattr(attn,'forward')
            attn.qk_shared = old_flag
        require(all(attn.qk_shared == flag and (attn.__dict__.get('forward') is forward if had else 'forward' not in attn.__dict__)
                    for attn,had,forward,flag in saved), 'Native-reference method/flag restoration failed')
    require(same_rng(before_rng,session) and tuple(p._version for p in parameters) == versions
            and tuple(id(p) for p in session.model.parameters()) == parameter_ids
            and {k:(id(v),v._version) for k,v in batch.items()} == batch_versions,
            'Native reference changed parameters, inputs, RNG or streams')
    with torch.no_grad():
        logits,representation = session.forward(batch)
    require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(representation).all()), 'Finite same-state candidate TRAIN-query output')
    difference = float((logits.detach().cpu().double()-reference.double()).abs().max())
    del logits,representation,reference
    require(same_rng(before_rng,session) and tuple(p._version for p in parameters) == versions
            and tuple(id(p) for p in session.model.parameters()) == parameter_ids
            and {k:(id(v),v._version) for k,v in batch.items()} == batch_versions,
            'Candidate identity evaluation changed parameters, inputs, RNG or streams')
    receipt.update(identity_max_abs_logit_difference=difference,identity_tolerance=TOLERANCE,
                   identity_reference='Original native GlobalAttn class forward, qk_shared=True, exact same current bank and batch',
                   native_reference_methods_flags_restored=True,identity_parameter_versions_unchanged=True,
                   identity_batch_versions_unchanged=True,identity_RNG_and_streams_unchanged=True,
                   native_reference_member_forwards=session.model.members,candidate_identity_member_forwards=session.model.members)
    require(difference <= TOLERANCE,'Same-state native pre-update identity tolerance failed')


def stage(session, batch, labels, name, counters, receipt):
    torch = session.torch
    global_stage = name == 'global'
    session.model.set_global(global_stage)
    require(all(model.body._global == global_stage for model in session.model.models), 'Actual stage flag')
    torch.cuda.reset_peak_memory_stats(0)
    started = time.monotonic()
    before_counts, before_work = dict(counters), dict(session.operator_work)
    receipt.update(stage=name, started=True, passed=False, VALID_forwards=0, VALID_metrics=0)
    try:
        session.model.eval()
        same_state_identity(session,batch,receipt)
        coordinates = targets(session)
        before_values = {k: p.detach().cpu().clone() for k, p in coordinates.items()}
        result = session.train_step(batch, labels)
        receipt['TRAIN_own_CE_diagnostic'] = float(result['own_mean'])
        named = list(session.model.named_parameters())
        active = [(n, p) for n, p in named if p.grad is not None]
        require(active and all(bool(torch.isfinite(p.grad).all()) for _, p in active), 'Actual accumulated gradients finite')
        receipt['gradient_field_counts'] = dict(parameter_tensors=len(named), parameter_scalars=sum(p.numel() for _, p in named),
            grad_None_tensors=len(named)-len(active), grad_finite_tensors=len(active), grad_active_scalars=sum(p.numel() for _, p in active))
        owners = {id(p): opt for opt in session.optimizers for group in opt.param_groups for p in group['params']}
        fields = []
        for key, p in coordinates.items():
            changed = int(torch.count_nonzero(p.detach().cpu() != before_values[key]))
            field = dict(name=key, shape=list(p.shape), scalars=p.numel(), grad_is_None=p.grad is None,
                         changed_coordinates=changed)
            if global_stage:
                require(p.grad is not None and bool(torch.isfinite(p.grad).all()), 'Global Q/K/scale gradient absent or nonfinite: ' + key)
                field['gradient_nonzero_coordinates'] = int(torch.count_nonzero(p.grad))
                field['gradient_max_abs'] = float(p.grad.abs().max())
                require(field['gradient_nonzero_coordinates'] > 0 and changed > 0, 'Global Q/K/scale did not have a live actual update: ' + key)
                state = owners[id(p)].state[p]
                require(state and float(state['step']) == 1. and all(bool(torch.isfinite(v).all())
                        for v in state.values() if isinstance(v, torch.Tensor)), 'New global Adam fields/clock not finite and live')
                field['Adam_step'] = 1
            else:
                require(p.grad is None and changed == 0 and p not in owners[id(p)].state,
                        'Inactive local-stage Q/K/scale gradient, update or Adam state: ' + key)
            fields.append(field)
        receipt['QK_coordinate_fields'] = fields
        session.core['selection'].finite_state(session.model, session.optimizers)
        work = {k: session.operator_work[k]-before_work[k] for k in before_work}
        expected = dict(update_attempts=1, completed_updates=1, member_view_forwards=2*session.model.members,
                        member_view_backwards=2*session.model.members, Adam_steps=len(session.optimizers))
        require(work == expected, 'Exact streamed F/Adam counters')
        observed = {k: counters[k]-before_counts[k] for k in counters}
        require(observed['member_forwards_completed'] == observed['member_forward_attempts'] == 4*session.model.members
                and observed['observed_Adam_calls'] == len(session.optimizers), 'Same-state native/candidate identities plus two TRAIN views counted')
        projections = (2*session.model.members + 6*session.model.members*(2 if session.operator_binding['operator'] == 'full_qk' else 1)) if global_stage else 0
        require(observed['large_QK_projection_calls'] == projections, 'Full-input native Q/K projection event count')
        pair_calls = 6*session.model.members if global_stage and session.operator_binding['operator'] in ('pre_sigmoid_split','active_reversible_exp') else 0
        require(observed['global_QK_pair_calls'] == pair_calls, 'Actual patched global Q/K pair event count')
        receipt.update(streamed_F_work=work, observed_work=observed, finite_parameters_and_Adam=True, passed=True)
    finally:
        receipt['streamed_F_work'] = {k: session.operator_work[k]-before_work[k] for k in before_work}
        receipt['observed_work'] = {k: counters[k]-before_counts[k] for k in counters}
        receipt['resources'] = own_resources(torch)
        receipt['elapsed_seconds'] = time.monotonic() - started


def run_qualification(*, release=None, root_execution_authorized=False):
    """Callable entry; default refuses before numerical imports or input loading."""
    require(root_execution_authorized is True and isinstance(release, dict), 'Separate root engineering execution authorization required')
    for flag in ('enabled', 'root_execution_authorized', 'source_review_approved'):
        require(release.get(flag) is True, 'Disabled engineering release: ' + flag)
    seal = read(HERE / 'SEAL.json')
    require(seal['execution_enabled'] is False and seal['source_only'] is True
            and sha(HERE / 'MANIFEST.json') == seal['manifest_sha256'] == release['qualifier_manifest_sha256']
            and sha(__file__) == release['qualifier_program_sha256'], 'Exact immutable qualifier source')
    for row in read(HERE / 'MANIFEST.json')['files']:
        require((HERE/row['path']).stat().st_size == row['bytes'] and sha(HERE/row['path']) == row['sha256'], 'Changed qualifier payload')
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    for row in pins['source_files']:
        require((PHASE/row['path']).stat().st_size == row['bytes'] and sha(PHASE/row['path']) == row['sha256'], 'Changed bound source')
    require(release['engineering_seed'] == ENGINEERING_SEED and release['operators'] == list(OPERATORS)
            and release['kinds'] == list(KINDS) and release['updates_per_cell'] == ['local','global']
            and release['identity_absolute_tolerance'] == TOLERANCE
            and release['identity_reference'] == 'same_current_bank_native_class_forward'
            and release['adapter_manifest_sha256'] == ADAPTER_MANIFEST and release['adapter_program_sha256'] == ADAPTER_PROGRAM
            and release['output_relative'] == OUTPUT_NAME and release['VALID_scoring'] is False
            and release['TEST_access'] is False and release['scientific_launch_authorized'] is False,
            'Exact twelve-cell, twenty-four-update engineering scope')
    require(sha(PHASE/ADAPTER_DIR/'MANIFEST.json') == ADAPTER_MANIFEST
            and sha(PHASE/ADAPTER_DIR/'operator_adapter.py') == ADAPTER_PROGRAM, 'Exact parent-pinned adapter')
    adapter = module('_qualified_qk_operator_adapter', PHASE/ADAPTER_DIR/'operator_adapter.py')
    paths, public_root, _ = adapter._sources()
    runtime = adapter._route(paths)  # Normal existing host/env route, before numerical import.
    recipe = read(public_root/'recipes/wikics.json')
    require(ENGINEERING_SEED not in recipe['pilot_seeds'] + recipe['confirmation_seeds'], 'Engineering seed reserved scientifically')
    roles = pins['role_files']
    require(release['role_files'] == roles and release['runtime_sha256'] == sha(paths['runtime']), 'Exact full role/runtime bindings')
    for row in roles.values():
        require((PHASE/row['path']).stat().st_size == row['bytes'] and sha(PHASE/row['path']) == row['sha256'], 'Changed full role input')
    output = PHASE / OUTPUT_NAME
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    report = dict(schema='full-input-qk-engineering-qualification-v2', complete=False, scientific_fit=False,
                  engineering_seed=ENGINEERING_SEED, root_release=release, runtime=runtime, cells=[],
                  predecessor_engineering_metadata=pins['provenance_files'],
                  VALID_metrics=0, VALID_forwards=0, TEST_access=False, old_scientific_artifacts_read=False)
    try:
        public = module('_qk_qualification_public', public_root/'portable.py')
        data = public._module('_qk_qualification_full_data', 'data.py')
        torch = data.torch
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        train = data.load_npz(PHASE/roles['train']['path'], data.ROLE_KEYS['wikics']['train'])
        valid = data.load_npz(PHASE/roles['valid']['path'], data.ROLE_KEYS['wikics']['valid'])
        data.check_projection('wikics', train, valid, {'split_index':0})
        report['input_fields'] = {role:{k:dict(shape=list(v.shape), dtype=str(v.dtype), scalars=v.numel())
                                       for k,v in payload.items()} for role,payload in (('TRAIN',train),('VALID',valid))}
        report['VALID_role_use'] = 'Schema/count/disjointness validation only; no prediction or metric'
        batch = {k:train[k].to('cuda:0') for k in ('x','edge_index','ids')}
        labels = train['y'].to('cuda:0')
        del train, valid
        for row in roles.values():
            require(sha(PHASE/row['path']) == row['sha256'], 'Role changed during loading')
        report['input_load_resources'] = own_resources(torch)
        for operator in OPERATORS:
            for kind in KINDS:
                cell = dict(operator=operator, kind=kind, seed=ENGINEERING_SEED, passed=False, stages=[])
                report['cells'].append(cell)
                cell_started = time.monotonic()
                session = None
                counters = dict(member_forward_attempts=0, member_forwards_completed=0, global_QK_pair_calls=0,
                                large_QK_projection_calls=0, observed_Adam_calls=0)
                torch.cuda.reset_peak_memory_stats(0)
                try:
                    session = public.Session('wikics', kind, ENGINEERING_SEED, 'cuda:0', paths['native'])
                    originals = [[(lin.weight,lin.bias) for lin in model.body.global_attn.k_lins] for model in session.model.models]
                    original_count = sum(p.numel() for p in session.model.parameters())
                    before_rng = rng(session)
                    adapter.install(session, operator=operator, later_execution_authorized=True)
                    require(same_rng(before_rng,session), 'Installation consumed extra RNG or changed persistent streams')
                    cell['installation_zero_extra_RNG_draws'] = True
                    cell['ownership'] = ownership(session)
                    cell['initial_coordinates'] = initial_coordinates(session,operator,originals,original_count)
                    cell['operator_binding'] = session.operator_binding
                    cell['construction_install_resources'] = own_resources(torch)
                    instrument(session,operator,counters)
                    for name in ('local','global'):
                        receipt = {}
                        cell['stages'].append(receipt)
                        stage(session,batch,labels,name,counters,receipt)
                    require(session.steps == 2 and not session._operator_failed, 'Two complete disposable engineering updates')
                    cell.update(passed=True, observed_total_work=dict(counters), streamed_total_work=dict(session.operator_work))
                except BaseException as exc:
                    cell['error'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
                finally:
                    cell['observed_total_work'] = dict(counters)
                    if session is not None and hasattr(session, 'operator_work'):
                        cell['streamed_total_work'] = dict(session.operator_work)
                    cell['own_resources_end'] = own_resources(torch)
                    cell['elapsed_seconds'] = time.monotonic()-cell_started
                    write(output/(operator+'__'+kind+'.json'), cell)
                    session = None
                    originals = before_rng = None
                    gc.collect()
                    torch.cuda.empty_cache()  # This process's allocator only.
                write(output/'REPORT.json',report)
        report['complete'] = len(report['cells']) == 12 and all(cell['passed'] for cell in report['cells'])
        report['completed_updates'] = sum(c.get('streamed_total_work',{}).get('completed_updates',0) for c in report['cells'])
        report['completed_member_view_forwards'] = sum(c.get('streamed_total_work',{}).get('member_view_forwards',0) for c in report['cells'])
        report['completed_member_view_backwards'] = sum(c.get('streamed_total_work',{}).get('member_view_backwards',0) for c in report['cells'])
        report['completed_Adam_steps'] = sum(c.get('streamed_total_work',{}).get('Adam_steps',0) for c in report['cells'])
        report['observed_totals'] = {key:sum(c['observed_total_work'][key] for c in report['cells']) for key in counters}
        report['identity_member_forwards'] = report['observed_totals']['member_forwards_completed'] - report['completed_member_view_forwards']
        if report['complete']:
            require((report['completed_updates'],report['completed_member_view_forwards'],report['completed_member_view_backwards'],report['completed_Adam_steps'])
                    == (24,144,144,48), 'Complete qualification field/work totals')
            require(report['identity_member_forwards'] == 144 and report['observed_totals'] == dict(
                member_forward_attempts=288, member_forwards_completed=288, global_QK_pair_calls=108,
                large_QK_projection_calls=342, observed_Adam_calls=48), 'Complete observed full-input field/work totals')
        report['own_resources_final'] = own_resources(torch)
    except BaseException as exc:
        report['complete'] = False
        report['error'] = dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        write(output/'REPORT.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--release-sha256',required=True)
    args = parser.parse_args()
    require(sha(args.release) == args.release_sha256,'Exact separately reviewed root release')
    result = run_qualification(release=read(args.release),root_execution_authorized=True)
    print(json.dumps(dict(complete=result['complete'],output=str(PHASE/OUTPUT_NAME),engineering_only=True)))
    raise SystemExit(0 if result['complete'] else 1)


if __name__ == '__main__':
    main()
