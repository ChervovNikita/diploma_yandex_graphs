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
    session = adapter.session
    original = session.train_update
    report['TRAIN_callback_calls'] = 0
    report['TRAIN_boundary_probes'] = 0
    report['TRAIN_losses'] = []
    report['TRAIN_epoch_calls'] = 0

    def bounded(*args, **kwargs):
        before = completed(observer)
        if before >= 2:
            report['TRAIN_boundary_probes'] += 1
            raise SmokeUpdatesComplete()
        report['TRAIN_callback_calls'] += 1
        result = original(*args, **kwargs)
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

    session.train_update = bounded
    try:
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
        session.train_update = original
