"""Reconstruct the exact native VALID selection and first stopping event."""
import math


def audit_selection(history, freeze, config):
    assert config['max_epochs'] == 9999
    assert config['eval_steps'] == 5 and config['kill_cnt'] == 10
    last_epoch = freeze['last_epoch']
    assert isinstance(last_epoch, int) and 1 <= last_epoch <= 9999
    assert [row['epoch'] for row in history] == list(range(5, last_epoch + 1, 5))
    assert history and all(math.isfinite(row['VALID_MRR']) and
                           0 < row['VALID_MRR'] <= 1 and
                           round(row['VALID_MRR'], 4) == row['VALID_MRR']
                           for row in history)
    best, selected_epoch, misses, first_stop = 0.0, None, 0, None
    for row in history:
        if first_stop is not None:
            raise AssertionError('VALID history continues after the first native stopping event')
        if row['VALID_MRR'] > best:
            best, selected_epoch, misses = row['VALID_MRR'], row['epoch'], 0
        else:
            misses += 1
            if misses > config['kill_cnt']:
                first_stop = row['epoch']
    assert selected_epoch == freeze['selected_epoch']
    assert best == freeze['selected_VALID_MRR']
    assert freeze['updates'] == 3 * last_epoch
    if first_stop is None:
        assert last_epoch == config['max_epochs'], 'Native fit ended without stop or maximum epoch'
    else:
        assert first_stop == last_epoch, 'Native fit did not end at its first stopping event'
    return dict(selected_epoch=selected_epoch, first_stop_epoch=first_stop,
                last_epoch=last_epoch, final_misses=misses,
                selection='first maximum rounded4-decimal VALID MRR',
                stopping='first eleven consecutive misses or9999epochs')
