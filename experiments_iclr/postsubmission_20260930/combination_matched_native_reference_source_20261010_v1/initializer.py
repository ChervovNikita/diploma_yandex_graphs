"""Exact candidate row law for native M1 vectors; no four-row-bank assumption."""
import hashlib
import math
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def initialize(session, paired_seed, member_index, row_seed):
    torch = session.torch
    began = time.monotonic()
    require(paired_seed in (6101, 6203, 6307) and type(member_index) is int
            and member_index in range(4), 'Fixed paired block/member index')
    require(session.task == 'wikics' and session.arm == 'single'
            and session.model.independent and session.model.members == 1
            and len(session.model.models) == 1 and not session.model.contrastive,
            'One fresh native body, not a shared four-row scorer bank')
    require(not hasattr(session, 'optimizers') and not hasattr(session, 'streams'),
            'Install before original fresh Adam and original dropout streams')
    require(session.seed == paired_seed + 1009 * member_index,
            'Declared private native body reset seed')
    require(not any(isinstance(module, session.core['factors'].FactorLinear)
                    for module in session.model.modules()), 'Ordinary native affine coordinates')
    body = session.model.models[0].body
    require(len(body.local_convs) == 7 and body._global is False,
            'Native seven-local-layer initial stage')
    keys = {'local_convs.%d.%s' % (layer, side) for layer in range(7)
            for side in ('att_src', 'att_dst')}
    common = {name: tensor_sha(value) for name, value in body.state_dict().items()
              if name not in keys}
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state(session.device).clone() if session.device.type == 'cuda' else None
    rows, identities = [], set()
    with torch.no_grad():
        for layer, conv in enumerate(body.local_convs):
            require(conv.heads == 1 and conv.out_channels == 512 and conv.concat
                    and not conv.add_self_loops and conv.bias is None and conv.edge_dim is None,
                    'Exact native GAT scorer sites')
            for side_index, name in enumerate(('att_src', 'att_dst')):
                require(not hasattr(conv, 'parametrizations')
                        or name not in conv.parametrizations, 'Direct native scorer, no installed bank')
                parameter = getattr(conv, name)
                require(isinstance(parameter, torch.nn.Parameter) and parameter.requires_grad
                        and parameter.dtype == torch.float32 and tuple(parameter.shape) == (1, 1, 512)
                        and id(parameter) not in identities, 'Distinct native M1 scorer shape/owner')
                identities.add(id(parameter))
                bound = math.sqrt(6.0 / (parameter.size(-2) + parameter.size(-1)))
                generator_seed = row_seed(paired_seed, layer, side_index, member_index)
                generator = torch.Generator(device='cpu').manual_seed(generator_seed)
                draw = torch.empty((1, 1, 512), dtype=torch.float32, device='cpu')
                draw.uniform_(-bound, bound, generator=generator)
                require(torch.isfinite(draw).all() and (draw.abs() <= bound).all(),
                        'Finite prescribed native-support row, no redraw')
                parameter.copy_(draw.to(device=parameter.device))
                rows.append(dict(layer=layer, side=name, member=member_index,
                    generator_seed=generator_seed, row_shape=[1, 1, 512],
                    row_sha256=tensor_sha(draw), native_final_two_dimension_bound=bound))
    after = {name: tensor_sha(value) for name, value in body.state_dict().items() if name not in keys}
    require(common == after and body._global is False, 'Every non-scorer tensor and stage unchanged')
    require(torch.equal(cpu, torch.get_rng_state()) and
            (cuda is None or torch.equal(cuda, torch.cuda.get_rng_state(session.device))),
            'Native/default CPU/CUDA RNG unchanged; original streams constructed later')
    return dict(schema='matched-native-M1-scorer-start-v1', paired_seed=paired_seed,
        member_index=member_index, native_body_seed=session.seed, rows=rows,
        generated_rows=14, generated_scalar_values=7168, direct_native_parameters=True,
        before_original_Adam=True, before_original_streams=True, non_scorer_bytes_unchanged=True,
        default_CPU_CUDA_RNG_unchanged=True, no_redraw=True, elapsed_seconds=time.monotonic()-began)


def stable(receipt):
    return {key: value for key, value in receipt.items() if key != 'elapsed_seconds'}
