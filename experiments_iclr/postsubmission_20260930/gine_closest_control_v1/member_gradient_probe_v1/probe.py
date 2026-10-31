"""Small training-target-only gradient probe; no loaders, fits or entrypoint.

The coordinator must admit data/state/runtime and numerically qualify this
source before use. It imports Torch only when called. Nothing runs on import.
"""
import math
import time


PRIVATE = frozenset(('input_r', 'input_s', 'input_b', 'output_r', 'output_s', 'output_b'))


def _groups(names, single):
    shared = [n for n in names if n not in PRIVATE]
    groups = {'all_shared': shared}
    groups['input_map'] = [n for n in shared if n == 'input_weight' or n.startswith('input.')]
    for layer in range(2):
        groups[f'hidden_layer{layer}'] = [n for n in shared if n.startswith(f'blocks.{layer}.') or n.startswith(f'norm.{layer}.')]
    groups['output_map'] = [n for n in shared if n == 'output_weight' or n.startswith('output.')]
    groups['output_norm'] = [n for n in shared if n.startswith('output_norm.')]
    if any(not rows for rows in groups.values()):
        raise ValueError('Model is not the inspected two-layer source layout')
    if not single and not PRIVATE.issubset(names):
        raise ValueError('Source member-factor layout missing')
    return groups


def _summary(torch, gradients, pooled, groups):
    result = {}
    M = len(gradients)
    for label, names in groups.items():
        vectors = [torch.cat([g[n].reshape(-1) for n in names]) for g in gradients]
        stack = torch.stack(vectors)
        G = stack @ stack.T
        q = torch.cat([pooled[n].reshape(-1) for n in names])
        own = stack.mean(0)
        extra = own - q
        diag = G.diag()
        denom = torch.sqrt(diag[:, None] * diag[None, :])
        cosine = [[float(G[m,n]/denom[m,n]) if float(denom[m,n]) > 0 else None
                   for n in range(M)] for m in range(M)]
        pair_neg = sum(min(float(G[m,n]), 0.) for m in range(M) for n in range(m))
        mean_diag = float(diag.mean())
        result[label] = dict(
            parameter_names=names, parameters=sum(gradients[0][n].numel() for n in names),
            member_gradient_gram=G.tolist(), member_gradient_cosines=cosine,
            member_gradient_norms=diag.sqrt().tolist(),
            negative_pair_dot_sum=pair_neg,
            own_mean_norm_squared=float(own@own),
            gradient_coherence_ratio=(float(own@own)/mean_diag if mean_diag > 0 else None),
            member_dot_own_mean=(stack@own).tolist(),
            pooled_gradient_norm_squared=float(q@q),
            own_dot_pooled=float(own@q),
            jensen_extra_dot_pooled=float(extra@q),
            norm_or_cosine_zero_denominator_policy='undefined/null; not replaced by zero',
        )
    return result


def _factor_routes(torch, gradients):
    if not PRIVATE.issubset(gradients[0]):
        return None
    out = {}
    M = len(gradients)
    for side in ('input', 'output'):
        names = [side+'_r', side+'_s', side+'_b']
        rows = []
        for g in gradients:
            rows.append([math.sqrt(sum(float((g[n][j]**2).sum()) for n in names))
                         for j in range(M)])
        out[side+'_factor_gradient_norm_by_loss_and_factor_row'] = rows
    out['interpretation'] = 'Connected-path diagnostic only; norms in different parameterizations are not comparable capacity estimates.'
    return out


def probe(model, family, features, train_ids, train_targets, *, architecture,
          dropout_disabled=True, dropout_key=None, serial_repeat_tolerance=1e-6):
    """Return scalar JSON-safe geometry at one fixed, externally admitted state.

    architecture: exact_member / shared_mean / single.
    No step, data subset, checkpoint access or tensor serialization occurs.
    Shared forwards are recomputed for each VJP to avoid retaining five
    simultaneous/reused activation graphs. Keyed dropout pairs those passes.
    Primary p=0 + train mode preserves the source activation checkpoint gate.
    """
    import torch
    from torch.nn import functional as F
    if architecture not in ('exact_member', 'shared_mean', 'single'):
        raise ValueError('Outside inspected architectures')
    if train_ids.ndim != 1 or train_targets.ndim != 1 or train_ids.numel() != train_targets.numel():
        raise ValueError('Aligned full official training IDs and targets required')
    if train_targets.dtype != torch.long or not train_ids.numel():
        raise ValueError('Nonempty integer training targets required')
    if not dropout_disabled and dropout_key is None:
        raise ValueError('Secondary needs a fixed source DropoutKey')
    if not hasattr(model, 'p'):
        raise ValueError('Source dropout interface missing')
    named = list(model.named_parameters())
    if any(p.grad is not None for _,p in named):
        raise ValueError('Fresh diagnostic-loaded model needs empty .grad fields')
    names, params = zip(*named)
    groups = _groups(set(names), architecture == 'single')
    # A stable parameter order makes scalar dot products reproducible.
    groups = {k: sorted(v) for k,v in groups.items()}
    mode_before = [(m,m.training) for m in model.modules()]
    p_before = model.p
    M = 1 if architecture == 'single' else 4
    sparse_before, dense_before = len(family.ledger.sparse), len(family.ledger.dense)

    def synchronize():
        if features.device.type == 'cuda':
            torch.cuda.synchronize(features.device)

    def cpu_grads(loss=None, output=None, cotangent=None):
        values = torch.autograd.grad(loss if output is None else output, params,
            grad_outputs=cotangent, allow_unused=True, create_graph=False, retain_graph=False)
        result = {n:(torch.zeros_like(p,device='cpu',dtype=torch.float64) if g is None
                      else g.detach().to(device='cpu',dtype=torch.float64))
                  for n,p,g in zip(names,params,values)}
        if any(not bool(torch.isfinite(g).all()) for g in result.values()):
            raise ValueError('Nonfinite diagnostic gradient')
        return result

    def full_forward():
        # CompleteBASE and SharedMeanEnsemble both have a packed default.
        return model(family, features, colors=(0,2), dropout_key=dropout_key)

    try:
        model.train(True)
        if dropout_disabled:
            model.p = 0.
        synchronize(); start = time.perf_counter()
        gradients, losses = [], []
        repeat_error = 0.
        scope = 'gradient_probe_p0' if dropout_disabled else 'gradient_probe_fixed_dropout'
        with family.ledger.scope(scope), torch.enable_grad():
            if architecture == 'exact_member':
                # First pass obtains pooled-logit cotangents with no full graph
                # retained. All temporary outputs are official train slices.
                member_logits = []
                with torch.no_grad():
                    for m in range(M):
                        values = model.forward_member(family,features,m,colors=(0,2),dropout_key=dropout_key)[train_ids]
                        member_logits.append(values.detach().to('cpu'))
                    pooled_values = member_logits[0].to(features.device).clone()
                    for values in member_logits[1:]:
                        pooled_values.add_(values.to(features.device))
                    pooled_values.div_(M)
                    pooled_loss = float(F.cross_entropy(pooled_values,train_targets))
                    cotangent = torch.softmax(pooled_values,dim=-1)
                    cotangent[torch.arange(train_ids.numel(),device=features.device),train_targets] -= 1.
                    cotangent.div_(M*train_ids.numel())
                    del pooled_values
                pool_contributions = []
                for m in range(M):
                    # Separate own and pooled VJPs keep peak memory close to a
                    # qualified serial training forward/backward.
                    values = model.forward_member(family,features,m,colors=(0,2),dropout_key=dropout_key)[train_ids]
                    loss = F.cross_entropy(values,train_targets)
                    losses.append(float(loss.detach()))
                    gradients.append(cpu_grads(loss=loss))
                    del values,loss
                    values = model.forward_member(family,features,m,colors=(0,2),dropout_key=dropout_key)[train_ids]
                    error = float((values.detach().to('cpu')-member_logits[m]).abs().max())
                    repeat_error = max(repeat_error,error)
                    if error > serial_repeat_tolerance:
                        raise ValueError('Serial repeats do not match the pooled-cotangent forward')
                    pool_contributions.append(cpu_grads(output=values,cotangent=cotangent))
                    del values
                pooled = {n:sum(g[n] for g in pool_contributions) for n in names}
                del member_logits,cotangent,pool_contributions
                full_member_forward_count, vjp_count = 12, 8
            else:
                for m in range(M):
                    values = full_forward()[:,train_ids]
                    loss = F.cross_entropy(values[m],train_targets)
                    losses.append(float(loss.detach()))
                    gradients.append(cpu_grads(loss=loss))
                    del values,loss
                if architecture == 'single':
                    pooled, pooled_loss = gradients[0], losses[0]
                else:
                    values = full_forward()[:,train_ids]
                    loss = F.cross_entropy(values.mean(0),train_targets)
                    pooled_loss = float(loss.detach())
                    pooled = cpu_grads(loss=loss)
                    del values,loss
                full_member_forward_count = M if architecture == 'single' else M+1
                vjp_count = full_member_forward_count
        if not all(math.isfinite(v) for v in losses+[pooled_loss]):
            raise ValueError('Nonfinite training loss at diagnostic state')
        synchronize(); elapsed = time.perf_counter()-start
        result = dict(schema='gnnm-member-gradient-probe-v1', architecture=architecture,
            training_nodes=train_ids.numel(), members=M, training_target_only=True,
            dropout_disabled=dropout_disabled, checkpoint_path_preserved=True,
            own_member_ce=losses, pooled_ce=pooled_loss, jensen_gap=sum(losses)/M-pooled_loss,
            shared_blocks=_summary(torch,gradients,pooled,groups),
            private_factor_paths=_factor_routes(torch,gradients),
            serial_repeat_max_abs_error=repeat_error,
            whole_source_forward_calls=full_member_forward_count, vjp_calls=vjp_count,
            synchronized_gradient_probe_seconds=elapsed,
            sparse_ledger_rows=family.ledger.sparse[sparse_before:],
            dense_ledger_rows=family.ledger.dense[dense_before:],
            gradients_are_parameterization_dependent=True, actual_adamw_step_measured=False,
            optimizer_or_model_parameters_updated=False, logits_or_gradients_serialized=False)
        result['probe_end_to_end_seconds_including_scalar_reductions'] = time.perf_counter()-start
        return result
    finally:
        model.p = p_before
        for module, flag in mode_before:
            module.training = flag
