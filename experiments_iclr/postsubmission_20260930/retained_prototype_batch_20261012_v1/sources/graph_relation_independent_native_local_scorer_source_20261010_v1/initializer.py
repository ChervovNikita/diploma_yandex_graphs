"""One fixed native-distribution local-scorer initializer; imports stay lazy."""
import hashlib
import math
import time

SCHEMA = 'independent-native-private-local-scorer-start-v1'
SEEDS = (6101, 6203, 6307)
SEED_OFFSET = 770000000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def row_seed(seed, layer, side, member):
    require(seed in SEEDS and layer in range(7) and side in (0, 1) and member in range(4), 'Fixed initializer indices')
    return 100000 * seed + SEED_OFFSET + 10000 * layer + 100 * side + member


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def initialize(torch, controller, seed):
    """After copied parametrization registration, before original fresh Adam.

    Each [1,heads,channels] row has U(-sqrt(6/(heads+channels)),+bound),
    the PyG Glorot final-two-dimension law. No torch Xavier fan conversion.
    Explicit CPU generators never consume native/default/dropout RNG states.
    """
    started = time.monotonic()
    require(seed in SEEDS and controller.members == 4 and len(controller.sites) == 14,
        'Fixed four-member/seven-layer native local scorer bank')
    expected = {(layer, name) for layer in range(7) for name in ('att_src', 'att_dst')}
    require({(layer, name) for layer, name, _ in controller.sites} == expected, 'Only the exact fourteen scorer sites')
    body = controller.body
    scorer_keys = {'local_convs.%d.parametrizations.%s.original' % (layer, name) for layer, name in expected}
    before_common = {name: tensor_sha(value) for name, value in body.state_dict().items() if name not in scorer_keys}
    cpu_before = torch.get_rng_state().clone()
    device = next(body.parameters()).device
    cuda_before = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
    rows = []
    with torch.no_grad():
        for layer, name, _ in controller.sites:
            conv = body.local_convs[layer]
            bank = getattr(conv.parametrizations, name).original
            require(tuple(bank.shape) == (4, 1, conv.heads, conv.out_channels)
                and tuple(bank.shape) == (4, 1, 1, 512) and bank.dtype == torch.float32,
                'Pinned native row shape/dtype')
            require(all(torch.equal(bank[0], bank[m]) for m in range(1, 4)), 'Fresh copied bank before its sole initializer')
            bound = math.sqrt(6.0 / (bank.size(-2) + bank.size(-1)))
            side = 0 if name == 'att_src' else 1
            digests = []
            for member in range(4):
                chosen = row_seed(seed, layer, side, member)
                generator = torch.Generator(device='cpu')
                generator.manual_seed(chosen)
                draw = torch.empty(tuple(bank.shape[1:]), dtype=torch.float32, device='cpu')
                draw.uniform_(-bound, bound, generator=generator)
                require(torch.isfinite(draw).all() and (draw.abs() <= bound).all(), 'Finite native-support draw')
                bank[member].copy_(draw.to(device=bank.device))
                digests.append(tensor_sha(draw))
                rows.append(dict(layer=layer, side=name, member=member, generator_seed=chosen,
                    row_shape=list(draw.shape), native_final_two_dimension_bound=bound, row_sha256=digests[-1]))
            require(len(set(digests)) == 4, 'Four independently seeded rows; no redraw on collision')
    after_common = {name: tensor_sha(value) for name, value in body.state_dict().items() if name not in scorer_keys}
    require(after_common == before_common, 'Every non-scorer shared/head/factor/QK/buffer byte remains unchanged')
    require(torch.equal(torch.get_rng_state(), cpu_before)
        and (cuda_before is None or torch.equal(torch.cuda.get_rng_state(device), cuda_before)),
        'No native/default CPU/CUDA RNG consumption')
    return dict(schema=SCHEMA, seed=seed, seed_formula='100000*seed+770000000+10000*layer+100*side_index+member',
        side_indices=dict(att_src=0, att_dst=1), generator_device='cpu', members=4, banks=14,
        generated_rows=56, generated_scalar_values=28672, rows=rows,
        unchanged_non_scorer_state_sha256=before_common, default_CPU_CUDA_RNG_unchanged=True,
        original_member_dropout_stream_construction_unchanged=True, before_original_Adam=True,
        no_new_parameters=True, no_redraw=True, elapsed_seconds=time.monotonic() - started)


def stable_receipt(receipt):
    """Remove cost observation only when comparing deterministic start metadata."""
    return {key: value for key, value in receipt.items() if key != 'elapsed_seconds'}
