"""Standalone link objectives and one-old-state block-gradient assignment.

Torch is injected. No source session, topology, sampler, source model body,
optimizer construction, selection policy, or frozen smoke port is modified.
"""
from .planning import authenticate, contrast_plan, match_endpoint_support, query_keys, raw_responsibilities


def _family(family):
    family = str(family).upper()
    if family not in ('NCN', 'NCNC', 'BUDDY'):
        raise ValueError('An explicit native link reduction family is required')
    return family


def native_record_losses(torch, logits, groups, family):
    labels = torch.as_tensor(groups, dtype=logits.dtype, device=logits.device)
    if _family(family) == 'BUDDY':
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.expand_as(logits), reduction='none')
    return torch.where(labels.bool().expand_as(logits),
                       -torch.nn.functional.logsigmoid(logits),
                       -torch.nn.functional.logsigmoid(-logits))


def reduce_native(torch, records, groups, family, weights=None, permit_empty=False):
    """NCN: positive mean + negative mean; BUDDY: source joint batch mean.

    Positive weights are renormalized to mean one WITHIN the original source
    reduction group(s). No implicit positive/negative balancing is added to BUDDY.
    """
    family = _family(family)
    labels = torch.as_tensor(groups, dtype=torch.long, device=records.device)
    selectors = [labels == 1, labels == 0] if family != 'BUDDY' else [torch.ones_like(labels, dtype=torch.bool)]
    parts = []
    for mask in selectors:
        if not bool(mask.any()):
            if permit_empty:
                continue
            raise ValueError('Native loss reduction group is empty')
        current = records[..., mask]
        if weights is not None:
            w = weights[..., mask]
            current = current * (w / w.mean(dim=-1, keepdim=True))
        parts.append(current.mean(dim=-1))
    return sum(parts) if parts else records.sum(dim=-1) * 0


def query_auxiliary(torch, preheads, groups, family, plan, temperature=.2, chunk_size=128):
    if preheads.ndim != 3 or preheads.shape[0] != 4 or preheads.shape[1] != len(groups):
        raise ValueError('Aligned original preheads [4,native_queries,width] required')
    if temperature <= 0 or chunk_size < 1:
        raise ValueError('Positive temperature and chunk size required')
    anchors = plan['anchors']
    if not anchors:
        return preheads.sum() * 0, {'active': False, 'reason': 'no endpoint-compatible native alternatives'}
    features = torch.nn.functional.normalize(preheads, p=2, dim=-1, eps=1e-12)
    per_pair = []
    for member in range(4):
        for view in range(4):
            if view == member:
                continue
            chunks = []
            for start in range(0, len(anchors), chunk_size):
                ids = anchors[start:start+chunk_size]
                anchor = features[member, ids]
                positive = (anchor * features[view, ids]).sum(-1)
                if plan['mode'] == 'cosine':
                    chunks.append(1-positive)
                    continue
                rows = plan['alternatives'][start:start+chunk_size]
                width = max(map(len, rows))
                padded = [row + [row[0]] * (width-len(row)) for row in rows]
                valid = torch.arange(width, device=preheads.device)[None, :] < torch.tensor(
                    [len(row) for row in rows], device=preheads.device)[:, None]
                negative = (anchor[:, None, :] * features[view, padded]).sum(-1)
                negative = negative.masked_fill(~valid, float('-inf'))
                scores = torch.cat((positive[:, None], negative), dim=-1) / temperature
                chunks.append(torch.logsumexp(scores, dim=-1) - scores[:, 0])
            per_pair.append(torch.cat(chunks))
    records = torch.stack(per_pair).mean(0)
    value = reduce_native(torch, records, [groups[i] for i in anchors], family, permit_empty=True)
    return value, {'active': True, 'ordered_member_view_pairs': 12, 'used_auxiliary_anchors': len(anchors),
                   'new_source_forwards': 0, 'temperature': temperature,
                   'capture': 'live native top-level task prehead before final boundary factors',
                   'final_boundary_factors_receive_auxiliary_gradient': False}


def build_objectives(torch, *, family, logits, preheads, endpoints, groups,
                     native_member_losses, native_pool_loss=None, seed=0,
                     contrast='endpoint', responsibility='none', aux_credit='private',
                     coefficient=.05, temperature=.2, anchor_limit=1024,
                     negative_limit=16, relation_ids=None, role='TRAIN',
                     route_query_signatures=None, match_endpoint_control_support=True):
    """Return explicit shared/private criteria, never a fictitious global loss.

    native_member_losses are the caller's unchanged live source scalar losses.
    All native query rows remain in F. Auxiliary subsampling changes A only.
    Input axes must be authenticated and equal across all four route forwards.
    """
    family = _family(family)
    groups = [int(g) for g in groups]
    keys = query_keys(endpoints, relation_ids)
    signature = authenticate(keys, groups, role)
    if route_query_signatures is None or list(route_query_signatures) != [signature]*4:
        raise ValueError('Four query signatures from actual native route captures are required')
    if logits.ndim == 3 and logits.shape[-1] == 1:
        logits = logits[..., 0]
    if tuple(logits.shape) != (4, len(keys)) or len(native_member_losses) != 4:
        raise ValueError('Four aligned native logits and four source scalar losses required')
    if contrast not in ('none', 'cosine', 'endpoint', 'global') or responsibility not in ('none', 'endpoint', 'query'):
        raise ValueError('Unknown fixed link hypothesis')
    if aux_credit not in ('private', 'all') or coefficient < 0:
        raise ValueError('Invalid auxiliary gradient credit or coefficient')
    F = torch.stack(native_member_losses).mean()
    W = F
    weight_diagnostics = None
    if responsibility != 'none':
        raw = raw_responsibilities(keys, seed, responsibility)
        weights = torch.tensor(raw, device=logits.device, dtype=logits.dtype)
        records = native_record_losses(torch, logits, groups, family)
        reconstructed = reduce_native(torch, records, groups, family)
        if not torch.allclose(reconstructed.detach(), torch.stack(native_member_losses).detach(), rtol=1e-6, atol=1e-7):
            raise ValueError('Per-query weighted loss does not recover the supplied source reduction')
        W = reduce_native(torch, records, groups, family, weights).mean()
        weight_diagnostics = {'law': responsibility, 'hash': 'BLAKE2b64/link-ext-v1/52bit-midpoint', 'seed': int(seed),
                              'positive_floor': .25, 'raw_weights_fixed_by_query': True,
                              'normalization': 'each original native reduction group in current minibatch',
                              'unweighted_reduction_checked_against_source': True,
                              'minimum_raw_weight': float(weights.min().detach()),
                              'maximum_raw_weight': float(weights.max().detach())}
    A = F * 0
    plan = aux_info = None
    if contrast != 'none':
        if preheads is None:
            raise ValueError('Contrast requires passive live top-level native task preheads')
        plan = contrast_plan(keys, groups, seed, contrast, anchor_limit, negative_limit)
        if contrast in ('global', 'cosine') and match_endpoint_control_support:
            reference = contrast_plan(keys, groups, seed, 'endpoint', anchor_limit, negative_limit)
            plan = match_endpoint_support(reference, plan)
        A, aux_info = query_auxiliary(torch, preheads, groups, family, plan, temperature)
    private = W + coefficient*A if contrast != 'none' else W
    shared = F + coefficient*A if contrast != 'none' and aux_credit == 'all' else F
    if responsibility == 'none' and aux_credit == 'all':
        private = shared
    return {'shared_objective': shared, 'private_objective': private,
            'native_own_loss': F, 'native_pool_loss': native_pool_loss,
            'weighted_private_task_loss': W, 'query_auxiliary_loss': A,
            'diagnostics': {'family': family, 'role': role, 'query_alignment_sha256': signature,
                'native_query_count': len(keys), 'native_positive_count': groups.count(1),
                'native_negative_count': groups.count(0), 'native_task_loss_changed': False,
                'native_reduction': 'joint batch mean' if family == 'BUDDY' else 'positive mean plus negative mean',
                'contrast': contrast, 'responsibility': weight_diagnostics, 'aux_credit': aux_credit,
                'coefficient': coefficient, 'contrast_plan': plan, 'auxiliary': aux_info,
                'sampler_or_topology_changes': False, 'extra_source_forwards': 0}}


def step_blocks(torch, optimizer, shared_parameters, private_parameters, objectives, clip=None):
    """Collect all block gradients at the old state, then perform ONE source step.

    Caller owns the original source optimizer/group extension and zero_grad.
    A block-credit rule may have no global scalar potential; both criteria are
    reported explicitly. Inactive native parameters retain None gradients.
    """
    shared = tuple(p for p in shared_parameters if p.requires_grad)
    private = tuple(p for p in private_parameters if p.requires_grad)
    if not shared or not private:
        raise ValueError('Explicit nonempty shared/private parameter sets required')
    ids = [id(p) for p in shared+private]
    owners = [id(p) for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
    if len(ids) != len(set(ids)) or len(owners) != len(set(owners)) or set(ids) != set(owners):
        raise ValueError('Shared/private blocks must partition source optimizer ownership exactly')
    left, right = objectives['shared_objective'], objectives['private_objective']
    if left is right:
        gradients = torch.autograd.grad(left, shared+private, allow_unused=True)
        collections = 1
    else:
        gs = torch.autograd.grad(left, shared, retain_graph=True, allow_unused=True)
        gp = torch.autograd.grad(right, private, allow_unused=True)
        gradients = gs+gp
        collections = 2
    for parameter, gradient in zip(shared+private, gradients):
        parameter.grad = gradient
    if clip is not None:
        clip()
    optimizer.step()
    return {'reverse_collections': collections, 'optimizer_transitions': 1,
            'all_gradients_from_old_state': True,
            'source_parameter_None_gradients': sum(g is None for g in gradients[:len(shared)]),
            'private_parameter_None_gradients': sum(g is None for g in gradients[len(shared):])}
