"""Source-only paired path. No Torch numerical fixture run by the Mac author.

One common identity-factor warm boundary, four arms, one shared alpha grid.
This module neither changes the sealed support-v1 default nor trains a model.
The caller supplies a genuine Pi S Pi^T topology control and homogeneous full
node outputs; shape checks cannot establish either semantic contract.
"""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path

_base_path = Path(__file__).resolve().parents[1] / 'base/graph_band_route_initializer.py'
_spec = importlib.util.spec_from_file_location('paired_bound_support_v1', _base_path)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

ARM_NAMES = ('common_only', 'train_remasked', 'full_node', 'full_node_permuted')


class PairedGeometryError(RuntimeError):
    """Geometry abort with charged operation counters available in .report."""

    def __init__(self, stage, error, report):
        report.update(status='joint_failure', reason='geometry_operation_failed',
                      geometry_error=dict(stage=stage, error_type=type(error).__name__,
                                          error_message=str(error)))
        self.report = report
        super().__init__(f'{stage}: {type(error).__name__}: {error}')


def initialize_paired_four_arms(logits_fn, theta0, S, S_permuted, target_nodes,
                               train_rows, train_labels,
                               homogeneous_full_node_outputs=False):
    """Return (arm_slices or None, report); never silently replace a failed arm.

    S_permuted is caller-supplied Pi S Pi^T in the original feature/label order.
    All four candidate arms use the same theta0, TRAIN labels and alpha. A failed
    joint search returns None, retaining every rejected trial and arm status.
    Full-output/signed diagnostics are never acceptance gates. No optimizer,
    continuation, warm cloning, source qualification or launch is implemented.
    """
    import torch
    import torch.nn.functional as F
    require = base._require
    require(homogeneous_full_node_outputs is True,
            'Explicit homogeneous full-node output contract required')
    require(theta0.ndim == 1 and bool(torch.isfinite(theta0).all()), 'Finite flat factors')
    for graph in (S, S_permuted):
        require(graph.layout == torch.sparse_coo and graph.shape[0] == graph.shape[1],
                'Need normalized sparse S and caller-permuted S')
        require(graph.shape == S.shape and graph.device == theta0.device
                and graph.dtype == theta0.dtype, 'Graph/model shape dtype/device agree')
    for ids in (target_nodes, train_rows):
        require(ids.dtype == torch.int64 and ids.ndim == 1
                and ids.device == theta0.device, 'Index vectors must be device int64')
        require(ids.numel() > 0 and ids.unique().numel() == ids.numel(),
                'Unique nonempty indices')
    require(target_nodes.numel() == S.shape[0]
            and bool(torch.equal(target_nodes.sort().values,
                                 torch.arange(S.shape[0], device=theta0.device))),
            'Exact homogeneous full-node target coverage required')
    require(train_labels.dtype == torch.int64 and train_labels.shape == train_rows.shape
            and train_labels.device == theta0.device, 'Only compact TRAIN labels [T]')

    report = dict(operation='paired_full_node_cotangent_shared_alpha_v1',
                  source_labels='train_only', homogeneous_full_node_outputs=True,
                  caller_supplied_permuted_topology=True, arm_names=list(ARM_NAMES),
                  fallback_enabled=False, status='preparing', accepted_alpha=None,
                  vjp_forwards=0, vjp_calls=0, jvp_calls=0, graph_sparse_products=0,
                  line_search_forward_calls=0, candidate_trial_forward_calls=0,
                  same_alpha_common_forward_calls=0, trial_forward_calls_by_arm={
                      name: 0 for name in ARM_NAMES}, attempts=[], arms={})

    def geometry_call(stage, callback):
        try:
            return callback()
        except Exception as error:
            raise PairedGeometryError(stage, error, report) from error

    def geometry_require(condition, message):
        try:
            base._require(condition, message)
        except Exception as error:
            raise PairedGeometryError('geometry_certificate', error, report) from error

    require = geometry_require
    report['vjp_forwards'] += 1
    z0, pullback = geometry_call('common_VJP_primal', lambda: torch.func.vjp(logits_fn, theta0))
    require(z0.ndim == 2 and z0.shape[0] == target_nodes.numel() and z0.shape[1] >= 2,
            'Need full-node class logits [N,C]')
    require(bool(torch.isfinite(z0).all()), 'Nonfinite common logits')
    require(bool(((train_rows >= 0) & (train_rows < z0.shape[0])).all()), 'Train rows range')
    require(bool(((train_labels >= 0) & (train_labels < z0.shape[1])).all()), 'Class range')
    base_loss = F.cross_entropy(z0[train_rows], train_labels).detach()
    residual = torch.zeros_like(z0)
    residual[train_rows] = (z0[train_rows].softmax(-1).detach()
                           - F.one_hot(train_labels, z0.shape[1]).to(z0.dtype))/train_rows.numel()
    report['vjp_calls'] += 1
    g = geometry_call('TRAIN_CE_pullback', lambda: pullback(residual.detach()))[0].detach()
    require(bool(torch.isfinite(g).all()), 'Nonfinite common gradient')
    g64, g2 = g.double(), g.double().square().sum()
    report.update(baseline_train_ce=float(base_loss), common_gradient_squared_norm=float(g2))
    if float(g2) <= 1e-20:
        report.update(status='joint_failure', reason='zero_common_gradient',
                      arm_failures={name: 'zero_common_gradient' for name in ARM_NAMES})
        return None, report

    full_residual = torch.zeros((S.shape[0], z0.shape[1]),
                                dtype=theta0.dtype, device=theta0.device)
    full_residual[target_nodes] = residual
    banks = {}
    for topology, graph in [('original', S), ('permuted', S_permuted)]:
        report['graph_sparse_products'] += 3
        bands = geometry_call(topology+'_graph_bank', lambda: base.bernstein_cubic_bands(
            graph, full_residual)).detach()
        error = float((bands.sum(0)-full_residual).norm()/full_residual.norm().clamp_min(
            torch.finfo(theta0.dtype).tiny))
        require(error <= base.ALGEBRA_TOLERANCE, 'Graph-band partition identity failed')
        banks[topology] = bands
        report[topology+'_partition_relative_error'] = error

    geometry = {'common_only': dict(h=(g/4).repeat(4, 1), q=None, topology=None,
                                   support='common_only')}
    for name, topology, support in [('train_remasked', 'original', 'train_remasked'),
                                    ('full_node', 'original', 'full_node'),
                                    ('full_node_permuted', 'permuted', 'full_node')]:
        bands = banks[topology]
        hs = []
        for band in bands:
            if support == 'full_node':
                cotangent = band[target_nodes]
            else:
                cotangent = torch.zeros_like(z0)
                cotangent[train_rows] = band[target_nodes[train_rows]]
            report['vjp_calls'] += 1
            hs.append(geometry_call(name+'_band_pullback', lambda: pullback(cotangent))[0].detach())
        geometry[name] = dict(h=torch.stack(hs),
                              q=(bands[:, target_nodes]-residual[None]/4).detach(),
                              topology=topology, support=support)
    del pullback
    z0 = z0.detach()

    def centered(logits):
        return logits-logits.mean(-1, keepdim=True)

    threshold = base.FUNCTION_RMS_TOLERANCE*float(
        centered(z0[train_rows]).square().mean().sqrt().clamp_min(1.0))
    bound_norm = math.sqrt(1+base.CAP**2)*float(g64.norm())
    alpha0 = base.RELATIVE_FACTOR_RADIUS*math.sqrt(theta0.numel())/bound_norm
    report.update(alpha0=alpha0, direction_norm_bound=bound_norm,
                  shared_trial_limit=base.BACKTRACK_ATTEMPTS,
                  functional_rms_threshold=threshold,
                  alpha_policy='RELATIVE_FACTOR_RADIUS*sqrt(d)/(sqrt(1+CAP^2)*norm(g))')
    for name in ARM_NAMES:
        state = geometry[name]
        h64 = state['h'].double()
        require(bool(torch.isfinite(h64).all()), 'Nonfinite band gradients')
        sum_error = float((h64.sum(0)-g64).norm()/g64.norm().clamp_min(1e-20))
        require(sum_error <= base.ALGEBRA_TOLERANCE,
                'Supported band-gradient sum differs from TRAIN common gradient')
        t64 = h64-(h64 @ g64/g2)[:, None]*g64
        t64 = t64-t64.mean(0, keepdim=True)
        t64 = t64-(t64 @ g64/g2)[:, None]*g64
        t64 = t64-t64.mean(0, keepdim=True)
        if name == 'common_only':
            t64 = torch.zeros_like(t64)
        lam = base.CAP*g64.norm()/t64.norm().clamp_min(1e-20)
        tangent = (lam*t64).to(theta0.dtype)
        directions = -g[None]-tangent
        orth_error = float(((directions.double() @ g64)+g2).abs().max()/g2)
        mean_error = float(tangent.double().mean(0).norm()/g64.norm())
        require(max(orth_error, mean_error) <= base.ALGEBRA_TOLERANCE,
                'Orthogonality/zero-mean certificate failed after cast')
        max_norm = float(directions.double().norm(dim=1).max())
        require(max_norm <= bound_norm*(1+base.ALGEBRA_TOLERANCE),
                'Common direction radius bound failed')
        functions = []
        if name != 'common_only' and float(tangent.double().norm()) > 1e-20:
            for row in tangent:
                report['jvp_calls'] += 1
                _, jvp = geometry_call(name+'_JVP', lambda: torch.func.jvp(
                    logits_fn, (theta0,), (row,)))
                functions.append(centered(jvp).detach())
            full_functions = torch.stack(functions)
        else:
            full_functions = torch.zeros_like(z0).repeat(4, 1, 1)
        train_functions = full_functions[:, train_rows]
        train_finite = bool(torch.isfinite(train_functions).all())
        full_finite = bool(torch.isfinite(full_functions).all())
        train_pairs = [float((train_functions[i]-train_functions[j]).square().mean().sqrt())
                       for i in range(4) for j in range(i)] if train_finite else None
        functional_ok = name == 'common_only' or (
            train_finite and min(train_pairs) > threshold)
        slope = None if state['q'] is None or not full_finite else float(
            (state['q'].double()*full_functions.double()).sum())
        state.update(directions=directions, tangent=tangent, slope=slope,
                     functional_ok=functional_ok)
        report['arms'][name] = dict(cotangent_support=state['support'], topology=state['topology'],
            geometry_status='ready' if functional_ok else 'failed_TRAIN_functional_separation',
            band_gradient_sum_relative_error=sum_error, common_scalar=float(lam),
            descent_relative_error=orth_error, tangent_mean_relative_error=mean_error,
            tangent_frobenius_norm=float(tangent.double().norm()), max_direction_norm=max_norm,
            functional_tangent_admitted=functional_ok, source_tangent_pair_rms=train_pairs,
            full_output_tangent_finite=full_finite,
            full_output_tangent_gram=base._full_output_gram(full_functions) if full_finite else None,
            signed_graph_contrast_first_order_slope=slope,
            signed_graph_contrast_functional='shared_full_node_q_for_arm_topology',
            signed_graph_contrast_support_matched=state['support'] == 'full_node')

    for attempt in range(base.BACKTRACK_ATTEMPTS):
        alpha = alpha0/(2**attempt)
        trial_slices, trial_logits, errors = {}, {}, {}
        # Complete every arm/member at this alpha, including rejected work.
        with torch.no_grad():
            for name in ARM_NAMES:
                trial_slices[name] = theta0[None]+alpha*geometry[name]['directions']
                outputs = []
                for row in trial_slices[name]:
                    report['line_search_forward_calls'] += 1
                    report['candidate_trial_forward_calls'] += 1
                    report['trial_forward_calls_by_arm'][name] += 1
                    try:
                        z = logits_fn(row).detach()
                        base._require(tuple(z.shape) == tuple(z0.shape), 'Trial output shape changed')
                        outputs.append(z)
                    except Exception as error:
                        outputs.append(None)
                        errors.setdefault(name, []).append(dict(
                            member=len(outputs)-1, error_type=type(error).__name__,
                            error_message=str(error)))
                trial_logits[name] = None if any(z is None for z in outputs) else torch.stack(outputs)
            arm_records = {}
            common_zs = trial_logits['common_only']
            common_z = None if common_zs is None else common_zs[0]
            bound = float(base_loss)-base.ARMIJO_C*alpha*float(g2)
            for name in ARM_NAMES:
                zs = trial_logits[name]
                losses = pooled_loss = actual_pairs = full_pairs = None
                finite = False
                if zs is not None:
                    losses = torch.stack([F.cross_entropy(z[train_rows], train_labels) for z in zs])
                    pooled_loss = F.cross_entropy(zs.mean(0)[train_rows], train_labels)
                    finite = bool(torch.isfinite(zs).all() and torch.isfinite(losses).all()
                                  and torch.isfinite(pooled_loss))
                    if finite:
                        actual_pairs = [float((centered(zs[i, train_rows])-centered(zs[j, train_rows]))
                                              .square().mean().sqrt())
                                        for i in range(4) for j in range(i)]
                        full_pairs = [float((centered(zs[i])-centered(zs[j])).square().mean().sqrt())
                                      for i in range(4) for j in range(i)]
                quality = finite and bool((losses.double() <= bound).all()) and float(pooled_loss) <= bound
                useful = name == 'common_only' or (finite and geometry[name]['functional_ok']
                                                  and min(actual_pairs)/alpha > threshold)
                accepted = quality and useful
                reasons = []
                if not finite:
                    reasons.append('nonfinite_or_unavailable_candidate')
                if not quality:
                    reasons.append('TRAIN_CE_gate_failed')
                if not useful:
                    reasons.append('TRAIN_functional_separation_failed')
                signed = dict(status='unavailable', diagnostic_only=True)
                if name == 'common_only':
                    signed.update(status='identical_to_same_alpha_common',
                                  signed_value=0.0 if finite else None, signed_positive=False)
                elif zs is not None and common_z is not None:
                    try:
                        signed = base.signed_graph_contrast(geometry[name]['q'], zs, common_z, alpha,
                                                            geometry[name]['slope'])
                    except Exception as error:
                        signed.update(error_type=type(error).__name__, error_message=str(error))
                signed.update(cotangent_functional='shared_full_node_q_for_arm_topology',
                              initializer_cotangent_support=geometry[name]['support'],
                              support_matched=geometry[name]['support'] == 'full_node',
                              prediction_is_predictive_success=False)
                arm_records[name] = dict(accepted=accepted, finite=finite,
                    quality_accepted=quality, functional_accepted=useful, failure_reasons=reasons,
                    route_train_ce=[float(x) for x in losses] if finite else None,
                    pooled_train_ce=float(pooled_loss) if finite else None,
                    actual_centered_pair_rms=actual_pairs, full_output_actual_centered_pair_rms=full_pairs,
                    signed_graph_contrast=signed, closure_errors=errors.get(name, []))
        joint_accepted = all(row['accepted'] for row in arm_records.values())
        report['attempts'].append(dict(attempt=attempt, alpha=alpha, armijo_bound=bound,
                                       arms=arm_records, joint_accepted=joint_accepted))
        if joint_accepted:
            report.update(status='joint_accepted', reason='first_shared_alpha_accepted_by_all',
                          accepted_alpha=alpha)
            return {name: value.detach() for name, value in trial_slices.items()}, report
    report.update(status='joint_failure', reason='shared_alpha_grid_exhausted',
                  arm_failures={name: report['attempts'][-1]['arms'][name]['failure_reasons']
                                for name in ARM_NAMES})
    return None, report
