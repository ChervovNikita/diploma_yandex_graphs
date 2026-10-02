"""One separately authorized Photo17 CE arithmetic diagnostic; old gate stays failed."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
PHASE = BASE.parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
REMOTE = REPO/'experiments_iclr/postsubmission_20260930'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
INITIAL_MANIFEST_SHA = '6e0549ab362731d9d7460f17d1130c6aaf9399f218fa34a251ec0136999f6717'
EPSILONS = (1e-3, 3e-4)
MODES = ('A_FP32_reduced_CE', 'B_FP32_per_example_CE_FP64_mean', 'C_FP64_CE_on_same_FP32_logits')


def helpers():
    """Verify the immutable stdlib helper source before importing it."""
    root = PHASE/'graph_init_execution_root_v1'
    manifest_path = root/'MANIFEST.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != INITIAL_MANIFEST_SHA:
        raise ValueError('Immutable initial stdlib helper manifest differs')
    manifest = json.loads(manifest_path.read_text())
    row = [r for r in manifest['payload'] if r['path'] == 'admission_support.py'][0]
    source = root/'admission_support.py'
    if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
        raise ValueError('Immutable stdlib helper source differs')
    spec = importlib.util.spec_from_file_location('photo_fd_stdlib_custody', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def own_guard(h, manifest_record, seal_record):
    h.require(manifest_record['path'] == h.remote(BASE/'MANIFEST.json') and
              seal_record['path'] == h.remote(BASE/'SEAL.json'), 'Exact diagnostic packet required')
    manifest = h.read(h.bound(manifest_record))
    seal = h.read(h.bound(seal_record))
    h.require(seal['manifest_sha256'] == manifest_record['sha256'], 'Diagnostic seal differs')
    names = set()
    for row in manifest['payload']:
        rel = Path(row['path'])
        h.require(not rel.is_absolute() and '..' not in rel.parts and str(rel) not in names, 'Unsafe diagnostic payload')
        names.add(str(rel))
        path = h.text_path(BASE/rel)
        h.require(h.sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Diagnostic source changed')


def rng_hash(rt):
    """State digest only; no RNG/model checkpoint is exported."""
    state = rt.integration.rng_snapshot()
    h = hashlib.sha256()
    def absorb(value):
        if rt.torch.is_tensor(value):
            h.update(str((str(value.dtype), tuple(value.shape))).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                h.update(key.encode()); absorb(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                absorb(item)
        else:
            h.update(repr(value).encode())
    absorb(state)
    return h.hexdigest()


def reproduce(core, rt, original_admission, out, ledger, expected_gate):
    """Literal sealed qualify_body prefix, followed by unchanged bind_and_qualify."""
    context = original_admission['context']
    graph, edges, train, _ = core.source_inputs(rt, context, ledger, validation=False)
    spec = rt.adapter.specification(context['backbone'], 'single_author', 0, context['seed'])
    native = ledger.measured(rt, 'disposable_native_construction', lambda: rt.adapter.TeacherFamily(spec).to(rt.device))
    optimizer = rt.adapter.optimizer_for(native)
    native.eval()
    prefix_rng = rng_hash(rt)
    for global_stage in (False, True):
        native.set_global_stage(global_stage)
        native.eval()
        def populate():
            optimizer.zero_grad(set_to_none=True)
            logits = native(graph)[0]
            loss = rt.torch.nn.functional.cross_entropy(logits[train.nodes], train.labels)
            core.require(bool(rt.torch.isfinite(loss)), 'Nonfinite disposable qualification loss')
            loss.backward(); optimizer.step()
            return dict(global_stage=global_stage, train_loss=float(loss.detach()), scientific_result=False)
        ledger.sink('disposable_Adam_population', ledger.measured(rt, 'qualification_state_population', populate))
    frozen = rt.integration.named_optimizer_snapshot(native, optimizer)
    equivalence = ledger.measured(rt, 'disposable_one_step_native_K4_Adam_equivalence', lambda:
        rt.integration.optimizer_equivalence_audit(native, frozen, rt.boundary, graph, train, ledger.sink))
    try:
        core.bind_and_qualify(rt, native, graph, train, ledger)
    except ValueError as error:
        core.require(str(error) == 'Actual predictive-function AD qualification failed', 'Unexpected reproduction failure')
        frame = error.__traceback__
        captured = None
        while frame is not None:
            if frame.tb_frame.f_code.co_name == 'bind_and_qualify':
                captured = frame.tb_frame.f_locals
            frame = frame.tb_next
        core.require(captured is not None and captured['qualification'] == expected_gate and
                     captured['qualification']['passed'] is False, 'Original frozen failed metrics must reproduce exactly')
        result = {key: captured[key] for key in ('theta0', 'closure', 'binding', 'k1', 'k4', 'qualification', 'counter')}
        result.update(graph=graph, train=train, native=native, optimizer=optimizer,
            prefix_rng_sha256=prefix_rng, gate_rng_sha256=rng_hash(rt), equivalence=equivalence)
        core.write_json(out/'EXACT_REPRODUCTION.json', dict(original_failed_gate_reproduced_exactly=True,
            original_gate_stays_failed=True, original_gate=expected_gate, prefix_rng_sha256=prefix_rng,
            gate_rng_sha256=result['gate_rng_sha256'], model_constructed_from_original_native_seed=True,
            state_population_stages=[False, True], state_population_updates=2,
            dropout_off=True, native_K1_K4_identity_checked=True, Adam_equivalence_passed=equivalence['passed'],
            source_labels_read=['train'], final_labels_read=False, no_useful_checkpoint_exported=True))
        return result
    raise ValueError('Original failed gate unexpectedly passed; exact failed-state diagnostic is invalid')


def arithmetic(rt, core, state, ledger):
    """All A/B/C are reported; no mode/direction/epsilon/tolerance selection."""
    torch = rt.torch
    F = torch.nn.functional
    theta = state['theta0']
    train = state['train']
    counts = {'closure_calls': 0, 'loss_vjp_calls': 0, 'logit_jvp_calls': 0, 'loss_jvp_calls': 0}
    def closure(value):
        counts['closure_calls'] += 1
        return state['closure'](value)
    def loss(mode, logits):
        rows = logits[train.nodes]
        if mode == MODES[0]:
            return F.cross_entropy(rows, train.labels)
        if mode == MODES[1]:
            return F.cross_entropy(rows, train.labels, reduction='none').double().mean()
        return F.cross_entropy(rows.double(), train.labels)
    z, pullback = ledger.measured(rt, 'paired_base_FP32_logit_VJP', lambda: torch.func.vjp(closure, theta))
    core.require(z.dtype == torch.float32 and theta.dtype == torch.float32, 'No model/logit dtype change allowed')
    residual = torch.zeros_like(z)
    residual[train.nodes] = (z[train.nodes].softmax(-1).detach()-F.one_hot(train.labels, z.shape[1]).to(z.dtype))/train.nodes.numel()
    original_gradient = ledger.measured(rt, 'original_FP32_residual_VJP', lambda: pullback(residual)[0].detach())
    gradients, cotangents = {}, {}
    baseline = {}
    for mode in MODES:
        baseline[mode] = float(loss(mode, z).detach())
        cotangents[mode] = ledger.measured(rt, mode+'_consistent_CE_output_cotangent', lambda mode=mode:
            torch.func.grad(lambda logits: loss(mode, logits))(z).detach())
        gradients[mode] = ledger.measured(rt, mode+'_consistent_model_VJP', lambda mode=mode:
            pullback(cotangents[mode])[0].detach())
        counts['loss_vjp_calls'] += 1
    generator = torch.Generator(device='cpu').manual_seed(90017)
    records = []
    for index in range(3):
        d = torch.randn(theta.shape, generator=generator, dtype=theta.dtype).to(theta.device)
        d = d/d.norm().clamp_min(1e-20)*(theta.numel()**0.5)
        cotangent = torch.randn(z.shape, generator=generator, dtype=z.dtype).to(z.device)
        _, jvp = ledger.measured(rt, 'direction'+str(index)+'_same_FP32_logit_JVP', lambda:
            torch.func.jvp(closure, (theta,), (d,)))
        counts['logit_jvp_calls'] += 1
        vjp = ledger.measured(rt, 'direction'+str(index)+'_same_FP32_random_cotangent_VJP', lambda:
            pullback(cotangent)[0])
        lhs = float((jvp.double()*cotangent.double()).sum())
        rhs = float((vjp.double()*d.double()).sum())
        dual_error = abs(lhs-rhs)/max(abs(lhs), abs(rhs), 1e-8)
        original_derivative = float((original_gradient.double()*d.double()).sum())
        mode_derivatives = {}
        for mode in MODES:
            derivative = float((gradients[mode].double()*d.double()).sum())
            _, loss_jvp = ledger.measured(rt, mode+'_direction'+str(index)+'_consistent_loss_JVP', lambda mode=mode:
                torch.func.jvp(lambda logits: loss(mode, logits), (z,), (jvp,)))
            counts['loss_jvp_calls'] += 1
            output_chain = float((cotangents[mode].double()*jvp.double()).sum())
            mode_derivatives[mode] = dict(analytic_chain_VJP_derivative=derivative,
                analytic_chain_loss_JVP_derivative=float(loss_jvp), output_cotangent_dot_logit_JVP=output_chain,
                loss_JVP_VJP_relative_error=abs(float(loss_jvp)-derivative)/max(abs(float(loss_jvp)), abs(derivative), 1e-8))
        checks = []
        for epsilon in EPSILONS:
            def paired_forward():
                with torch.no_grad():
                    plus = closure(theta+epsilon*d)
                    minus = closure(theta-epsilon*d)
                    fd = (plus-minus)/(2*epsilon)
                    centered_fd = fd-fd.mean(-1, keepdim=True)
                    centered_jvp = jvp-jvp.mean(-1, keepdim=True)
                    logit_error = float((centered_fd-centered_jvp).double().norm()/centered_jvp.double().norm().clamp_min(1e-8))
                    values = {}
                    for mode in MODES:
                        lp, lm = loss(mode, plus), loss(mode, minus)
                        derivative = mode_derivatives[mode]['analytic_chain_VJP_derivative']
                        ce_fd = float((lp-lm)/(2*epsilon))
                        values[mode] = dict(plus_CE=float(lp), minus_CE=float(lm), CE_difference=float(lp-lm),
                            reduced_scalar_dtype=str(lp.dtype), finite_CE_derivative=ce_fd,
                            CE_error_against_precision_consistent_chain=abs(ce_fd-derivative)/max(abs(ce_fd), abs(derivative), 1e-3),
                            CE_error_against_original_FP32_analytic=abs(ce_fd-original_derivative)/max(abs(ce_fd), abs(original_derivative), 1e-3))
                    return dict(epsilon=epsilon, centered_logit_relative_error=logit_error, modes=values)
            checks.append(ledger.measured(rt, 'direction'+str(index)+'_epsilon'+str(epsilon)+'_paired_A_B_C', paired_forward))
        records.append(dict(direction=index, random_cotangent_dual_relative_error=dual_error,
            original_FP32_analytic_derivative=original_derivative, mode_chain_derivatives=mode_derivatives,
            finite_differences=checks))
    # The recreated A/manual-gradient/logit metrics must equal the unchanged old gate again.
    reconstructed = []
    for row in records:
        reconstructed.append(dict(direction=row['direction'], dual_relative_error=row['random_cotangent_dual_relative_error'],
            finite_differences=[dict(epsilon=c['epsilon'], logits_relative_error=c['centered_logit_relative_error'],
                ce_directional_error=c['modes'][MODES[0]]['CE_error_against_original_FP32_analytic'],
                analytic_ce_derivative=row['original_FP32_analytic_derivative'],
                finite_ce_derivative=c['modes'][MODES[0]]['finite_CE_derivative']) for c in row['finite_differences']]))
    core.require(reconstructed == state['qualification']['records'], 'Paired same-FP32 arithmetic must reproduce old metrics exactly')
    summaries = {}
    for mode in MODES:
        summaries[mode] = dict(diagnostic_consistent_CE_checks_with_original_thresholds=all(
            row['random_cotangent_dual_relative_error'] <= 2e-4 and any(
                c['centered_logit_relative_error'] <= .05 and
                c['modes'][mode]['CE_error_against_precision_consistent_chain'] <= .05
                for c in row['finite_differences']) for row in records),
            maximum_loss_chain_JVP_VJP_relative_error=max(row['mode_chain_derivatives'][mode]['loss_JVP_VJP_relative_error'] for row in records))
    return dict(schema='photo-fd-arithmetic-diagnostic-result-v1', records=records, mode_summaries=summaries,
        baseline_CE=baseline, modes=list(MODES), fixed_direction_seed=90017, directions=3,
        epsilons=list(EPSILONS), thresholds=dict(dual_rtol=2e-4, logits_FD_rtol=.05, CE_FD_rtol=.05, CE_denominator_floor=1e-3),
        model_logits_and_factors_dtype='FP32 unchanged', original_gate_stays_failed=True,
        old_registry_downstream_closure_stays_blocked=True, diagnostic_mode_selection_performed=False,
        native_recipe_or_method_or_tolerances_changed=False, full_FP64_model_test_performed=False,
        analytic_precision_note='Every A/B/C cotangent is differentiated through that exact loss expression; cotangents/VJP vectors retain FP32 interface rounding. Loss JVP also uses the same expression on the same FP32 logit tangent.',
        counts=counts, no_useful_checkpoint_or_fit=True, source_labels_read=['train'], final_labels_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--supervisor', required=True)
    args = parser.parse_args()
    h = helpers()
    h.require(PHASE == REMOTE and Path.cwd().resolve() == REPO and
        Path(os.path.abspath(sys.executable)) == REPO/'.venv/bin/python', 'Exact authorized repository/venv required')
    h.require(os.environ.get('GNNM_PHASE_ROOT') == str(PHASE) and os.environ.get('GNNM_SSH_DESTINATION') == LOGIN and
        os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Exact environment/route required')
    request_path = h.confined(args.request)
    request_hash = h.sha(request_path)
    request = h.read(request_path)
    h.require(request['schema'] == 'photo-fd-arithmetic-root-request-v1' and request['root_admitted'] is True and
        request['action'] == 'photo_fd_arithmetic_diagnostic' and request['full_fit_admitted'] is False and
        request['heldout_scoring_admitted'] is False and request['whole_cap_seconds'] == 3600,
        'One bounded separately admitted diagnostic only')
    supervisor = h.confined(args.supervisor)
    h.require(str(supervisor) == request['supervisor_directory'] and supervisor.is_dir(), 'Exact active supervisor required')
    command = h.read(supervisor/'command.json')
    environment = h.read(supervisor/'environment.json')
    h.require(command['argv'] == [str(REPO/'.venv/bin/python'), str(Path(__file__)), '--request', str(request_path), '--supervisor', str(supervisor)] and
        command['cwd'] == str(REPO) and command['shell'] is False and environment['repository'] == str(REPO) and
        environment['phase'] == str(PHASE), 'Exact declared supervisor invocation required')
    outer = h.read(h.confined(os.environ.get('GNNM_BOUND_START_JSON', '')))
    h.require(outer['whole_cap_seconds'] == 3600 and outer['root_request']['sha256'] == request_hash and
        outer['outer_output'] == request['outer_supervisor_directory'], 'Whole diagnostic bound required')
    route = h.read(h.bound(request['allocation_route']))
    h.require(route['ssh_destination'] == LOGIN and route['repository'] == str(REPO) and route['visible_gpu_count'] == 1, 'Exact route evidence required')
    rows = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader,nounits'],
        text=True, capture_output=True, check=True, timeout=15).stdout.strip().splitlines()
    h.require(len(rows) == 1 and [x.strip() for x in rows[0].split(',')] == ['0', UUID, 'NVIDIA A100-SXM4-80GB', '81920'], 'Exact GPU UUID required')
    decision = h.read(h.bound(request['root_decision']))
    admission = h.read(h.bound(request['diagnostic_admission']))
    h.require(decision['schema'] == 'photo-fd-arithmetic-root-decision-v1' and decision['approved'] is True and
        decision['diagnostic_execution_authorized'] is True and decision['source_review_accepted'] is True and
        decision['approved_by'].strip() and decision['approved_utc'] and
        decision['retry_or_registry_repair_authorized'] is False and decision['fit_or_final_labels_authorized'] is False,
        'Separate signed diagnostic decision required')
    own_guard(h, decision['source_manifest'], decision['source_seal'])
    h.bound(decision['source_review'])
    expected_admission = copy.deepcopy(h.read(h.bound(decision['admission_draft'])))
    h.require(expected_admission['execution_authorized'] is False, 'Original false diagnostic draft required')
    expected_admission['execution_authorized'] = True
    h.require(admission == expected_admission, 'Exact root-approved diagnostic draft required; only execution flag may change')
    h.require(admission['schema'] == 'photo-fd-arithmetic-diagnostic-admission-v1' and admission['execution_authorized'] is True and
        admission['modes'] == list(MODES) and admission['epsilons'] == list(EPSILONS) and admission['directions'] == 3 and
        admission['direction_seed'] == 90017 and admission['full_FP64_model_authorized'] is False and
        admission['original_gate_stays_failed'] is True and admission['retry_authorized'] is False,
        'Exactly frozen A/B/C diagnostic contract required')
    h.verify_sources()
    for record in admission['protected_metadata']:
        h.bound(record)
    failure = h.read(h.bound(admission['prior_failure']))
    h.require(failure['message'] == 'Actual predictive-function AD qualification failed', 'Exact prior failure required')
    prior_terminal = h.read(h.bound(admission['prior_attempt_terminal']))
    h.require(prior_terminal['completed'] is False, 'Prior registered attempt must remain failed')
    original = h.read(h.bound(admission['original_failed_admission']))
    context = original['context']
    h.require(context == admission['context'] and (context['graph'], context['backbone'], context['seed'], context['config']) ==
        ('Photo', 'polynormer_r', 17, 0) and original['dependencies'] == {} and original['arm'] is None,
        'Exact failed Photo17 full-graph TRAIN-only cold source context required')
    prior_claim = h.read(h.bound(prior_terminal['claim']))
    h.require(prior_claim['admission'] == admission['original_failed_admission'] and
        prior_claim['attempt_registry'] == original['attempt_registry'] and prior_claim['attempt']['phase'] == 'qualify' and
        prior_claim['attempt']['context_sha256'] == h.object_hash(context), 'Exact failed canonical registry claim required')
    trace = [json.loads(line) for line in h.text_path(h.bound(admission['prior_interface_trace'])).read_text().splitlines()]
    gates = [row['receipt'] for row in trace if row['event'] == 'gradient_qualification']
    h.require(len(gates) == 1 and gates[0]['passed'] is False, 'One retained original failed gate required')
    out = h.confined(request['output'])
    receipt = h.confined(request['receipt_directory'])
    h.require(out == PHASE/'photo_fd_arithmetic_diagnostic_v1/diagnostic_v1' and
        receipt == PHASE/'photo_fd_arithmetic_diagnostic_v1/root_receipts/diagnostic_v1' and not out.exists() and not receipt.exists(),
        'Fresh fixed diagnostic output/receipt required; no hidden retry')
    receipt.mkdir(parents=True, exist_ok=False); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    h.write(receipt/'START.json', dict(UTC=datetime.now(timezone.utc).isoformat(), request_sha256=request_hash,
        root_decision=request['root_decision'], prior_failure=admission['prior_failure'], old_gate_stays_failed=True, automatic_retry=False))
    ledger = None
    try:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # Only now import unchanged sealed stdlib driver, then its lazy guarded science.
        driver_path = PHASE/h.R17_REL/'prototype/graph_init_driver.py'
        spec = importlib.util.spec_from_file_location('photo_fd_original_core', driver_path)
        core = importlib.util.module_from_spec(spec); sys.modules[spec.name] = core; spec.loader.exec_module(core)
        source = core.context_guard(context)
        rt = core.load_runtime(context, source)
        ledger = core.Ledger(out)
        state = reproduce(core, rt, original, out, ledger, gates[0])
        before = rng_hash(rt)
        result = arithmetic(rt, core, state, ledger)
        after = rng_hash(rt)
        h.require(before == after, 'Diagnostic must preserve exact post-gate global RNG')
        result.update(global_RNG_unchanged=True, global_RNG_sha256=before, operation_costs=ledger.costs,
            elapsed_seconds=time.monotonic()-start)
        h.write(out/'ARITHMETIC_COMPARISON.json', result)
        h.verify_sources(); own_guard(h, decision['source_manifest'], decision['source_seal'])
        h.bound(request['root_decision']); h.bound(request['diagnostic_admission'])
        for record in admission['protected_metadata']:
            h.bound(record)
        h.require(h.sha(request_path) == request_hash, 'Diagnostic request changed')
        h.write(receipt/'TERMINAL.json', dict(completed=True, seconds=time.monotonic()-start,
            request_sha256=request_hash, comparison=h.descriptor(out/'ARITHMETIC_COMPARISON.json'),
            prior_registered_gate_stays_failed=True, registry_downstream_closure_blocked=True,
            fit_or_final_labels_executed=False, useful_checkpoint_exported=False, automatic_retry=False))
        return 0
    except BaseException as error:
        failed = dict(error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
            seconds=time.monotonic()-start, completed_operation_costs=ledger.costs if ledger is not None else [],
            prior_registered_gate_stays_failed=True, automatic_retry=False)
        h.write(out/'FAILED_DIAGNOSTIC.json', failed)
        h.write(receipt/'FAILED_DIAGNOSTIC.json', failed)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
