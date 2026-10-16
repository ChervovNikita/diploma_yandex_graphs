"""Bound actual native callbacks without replacing a training or forward body."""
import math


class SmokeUpdatesComplete(BaseException):
    """Only this runner catches the deliberate boundary before callback three."""


def completed(observer):
    return sum(observer.state()['completed'])


def install_step_cap(observer, optimizers):
    """Keep each runtime's own gradient observer and its actual signature."""
    for optimizer in optimizers:
        observed_step = optimizer.step

        def step(*args, _observed_step=observed_step, **kwargs):
            if completed(observer) >= 2:
                raise ValueError('Third optimizer transition is outside the smoke budget')
            return _observed_step(*args, **kwargs)

        optimizer.step = step


def train_two_updates(adapter, observer, report, progress=lambda: None):
    # Native session wrappers may delegate train_epoch() to base.train_epoch(),
    # whose loop calls base.train_update() directly. Retain those epoch and
    # operation contexts; observe each explicit owned base-chain callback.
    sessions, seen = [], set()
    session = adapter.session
    while session is not None:
        if id(session) in seen:
            raise ValueError('Cyclic explicit session.base chain')
        seen.add(id(session))
        original = getattr(session, 'train_update', None)
        if not callable(original):
            raise ValueError('Native session base chain lacks train_update')
        owned = vars(session)
        sessions.append((session, original, 'train_update' in owned, owned.get('train_update')))
        # Do not follow __getattr__ aliases or traverse unrelated native state.
        session = owned.get('base')
    report['TRAIN_callback_calls'] = 0
    report['TRAIN_nested_callback_calls'] = 0
    report['TRAIN_boundary_probes'] = 0
    report['TRAIN_losses'] = []
    report['TRAIN_epoch_calls'] = 0
    report['TRAIN_update_callback_chain'] = [type(item[0]).__module__ + '.' + type(item[0]).__qualname__
                                             for item in sessions]
    depth = 0

    def bounded(original):
        def invoke(*args, **kwargs):
            nonlocal depth
            before = completed(observer)
            if before >= 2:
                report['TRAIN_boundary_probes'] += 1
                raise SmokeUpdatesComplete()
            outermost = depth == 0
            key = 'TRAIN_callback_calls' if outermost else 'TRAIN_nested_callback_calls'
            report[key] += 1
            depth += 1
            try:
                result = original(*args, **kwargs)
            finally:
                depth -= 1
            if outermost:
                after = completed(observer)
                if after != before + 1:
                    raise ValueError('One real optimizer transition per native update callback required')
                loss = float(result['loss'])
                if not math.isfinite(loss):
                    raise ValueError('Nonfinite native TRAIN loss')
                report['TRAIN_losses'].append(loss)
                report['TRAIN_updates'] = after
                progress()
            return result
        return invoke

    patched = []
    try:
        for item in sessions:
            session, original, _, _ = item
            session.train_update = bounded(original)
            patched.append(item)
        while completed(observer) < 2:
            before = completed(observer)
            report['TRAIN_epoch_calls'] += 1
            if report['TRAIN_epoch_calls'] > 2:
                raise ValueError('Native epoch did not produce a bounded update')
            try:
                adapter.train_epoch()
            except SmokeUpdatesComplete:
                if completed(observer) != 2:
                    raise ValueError('Premature smoke boundary')
                break
            if completed(observer) <= before:
                raise ValueError('Native epoch returned without an optimizer transition')
        if observer.state() != {'attempts': [2], 'completed': [2]} or len(report['TRAIN_losses']) != 2:
            raise ValueError('Exactly two completed native updates and finite losses required')
        report['TRAIN_finite'] = True
        report['completed_adapter_epochs'] = adapter.completed_epochs
    finally:
        for session, _, owned, previous in reversed(patched):
            if owned:
                session.train_update = previous
            else:
                del session.train_update
