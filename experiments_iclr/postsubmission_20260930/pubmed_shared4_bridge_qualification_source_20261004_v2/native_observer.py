"""Observe the original native epoch without changing sampler/iterator semantics."""
from hashlib import sha256
import json
from types import FunctionType
from common import require


def tensor_identity(value):
    copied = value.detach().cpu().contiguous()
    return {'shape': list(copied.shape), 'dtype': str(copied.dtype),
            'sha256': sha256(copied.numpy().tobytes(order='C')).hexdigest()}


def state_identity(value, torch):
    def reduce(child):
        if isinstance(child, torch.Tensor):
            return tensor_identity(child)
        if isinstance(child, dict):
            return {key: reduce(item) for key, item in child.items()}
        if isinstance(child, (tuple, list)):
            return [reduce(item) for item in child]
        return child
    return sha256(json.dumps(reduce(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def observed_epoch(native, bodies, unit, train, torch, np, progress, exact, comparisons):
    """One complete unchanged body; wrapped delegates only observe returned draws.

    The observer contains no manual seed, extra sampling, resize, reorder,
    early stop, generator injection, custom tail or alternate negative count.
    Every hash/RNG observation is checked neutral and belongs to the budget.
    """
    model, predictor, optimizer, _, data, _ = unit
    start = native.clone(native.rng(torch, np, True), torch)
    receipt = {'negative_calls': [], 'iterator_calls': [], 'batches': []}
    original = bodies.candidate_ncnc_train
    sampler = original.__globals__['negative_sampling']
    iterator = original.__globals__['PermIterator']
    require(sampler is bodies.negative_sampling and iterator is bodies.PermIterator, 'Native function globals changed')

    def observe(callback):
        before = native.clone(native.rng(torch, np, True), torch)
        value = callback()
        exact(before, native.rng(torch, np, True), 'observer_RNG_neutral', comparisons)
        return value

    def sample(*args, **kwargs):
        require(len(args) == 2 and not kwargs and args[1] == 19717, 'Native sampler invocation changed')
        require(args[0] is data.edge_index, 'Native sampler edge tensor identity changed')
        result = sampler(*args, **kwargs)
        receipt['negative_calls'].append(observe(lambda: tensor_identity(result)))
        return result

    class ObservedIterator:
        def __init__(self, *args, **kwargs):
            require(len(args) == 3 and not kwargs and args[1:] == (37676, 1024), 'Native TRAIN iterator invocation changed')
            self.original = iterator(*args, **kwargs)
            receipt['iterator_calls'].append(observe(lambda: {
                'full_order': tensor_identity(self.original.idx),
                'dropped_tail': tensor_identity(self.original.idx[36864:]),
                'native_length': len(self.original), 'training': self.original.training,
                'batch_size': self.original.bs}))

        def __iter__(self):
            iter(self.original)
            return self

        def __next__(self):
            result = next(self.original)
            require(result.shape == (1024,), 'Native full batch changed')
            receipt['batches'].append(observe(lambda: tensor_identity(result)))
            return result

    scope = dict(original.__globals__)
    scope.update(negative_sampling=sample, PermIterator=ObservedIterator)
    measured = FunctionType(original.__code__, scope, original.__name__, original.__defaults__, original.__closure__)
    require(measured.__code__ is original.__code__, 'Native epoch body changed')
    progress.add(native_epochs_started=1)
    before_updates = progress.snapshot()['Adam_completed']
    loss = measured(model, predictor, data, {'train': {'edge': train}}, optimizer, 1024, True, [], None)
    progress.add(native_epochs_completed=1)
    require(len(receipt['negative_calls']) == len(receipt['iterator_calls']) == 1 and len(receipt['batches']) == 36,
            'Full epoch sampler/iterator/batch coverage differs')
    require(progress.snapshot()['Adam_completed'] - before_updates == 36, 'Full epoch Adam coverage differs')
    order = receipt['iterator_calls'][0]
    require(order['native_length'] == 36 and order['training'] is True and order['full_order']['shape'] == [37676]
            and order['dropped_tail']['shape'] == [812], 'Native full permutation/tail contract differs')
    require(receipt['negative_calls'][0]['shape'][0] == 2 and receipt['negative_calls'][0]['shape'][1] >= 37676,
            'Native sampler returned too few negatives for the original record indexing')
    end = native.clone(native.rng(torch, np, True), torch)
    receipt.update(start_RNG_sha256=state_identity(start, torch), end_RNG_sha256=state_identity(end, torch),
                   original_body_reused=True, native_sampler_defaults_unchanged=True,
                   full_batches=36, trained_positive_rows=36864, dropped_positive_tail=812,
                   original_implicit_requested_negative_rows=int(data.edge_index.shape[1]))
    return float(loss), receipt


def observe_work(unit, prototype, progress):
    """Scalar completion work and exact begun/completed optimizer accounting."""
    encoder, predictor, optimizer = unit[:3]
    handles = [
        encoder.register_forward_pre_hook(lambda module, args: progress.add(encoder_started=1)),
        encoder.register_forward_hook(lambda module, args, result: progress.add(encoder_completed=1)),
        predictor.decoder.register_forward_pre_hook(lambda module, args: progress.add(root_decoder_started=1, root_query_rows_started=int(args[2].shape[0]))),
        predictor.decoder.register_forward_hook(lambda module, args, result: progress.add(root_decoder_completed=1, root_query_rows_completed=int(args[2].shape[0]))),
        optimizer.register_step_pre_hook(lambda optimizer, args, kwargs: progress.add(Adam_started=1)),
        optimizer.register_step_post_hook(lambda optimizer, args, kwargs: progress.add(Adam_completed=1)),
    ]
    original = prototype.enumerate_neighbors
    def neighbors(*args, **kwargs):
        progress.add(neighbor_calls_started=1)
        result = original(*args, **kwargs)
        progress.add(neighbor_calls_completed=1, neighbor_query_rows=int(args[1].shape[0]),
                     common_rows=len(result.common[0]), left_residual_rows=len(result.left[0]),
                     right_residual_rows=len(result.right[0]))
        return result
    prototype.enumerate_neighbors = neighbors
    def remove():
        require(prototype.enumerate_neighbors is neighbors, 'Neighbor observer unexpectedly replaced')
        prototype.enumerate_neighbors = original
        for handle in handles:
            handle.remove()
    return remove
