"""Explicit native link epoch accounting, without importing a tensor library."""


def epoch_layout(family, positive_count, query_count=None):
    family = str(family).upper()
    if family in ('NCN', 'NCNC'):
        batch = 65536
        steps, tail = divmod(int(positive_count), batch)
        records = steps * batch
        query_records = 2 * records
    elif family == 'BUDDY':
        batch = 1024
        if query_count is None:
            raise ValueError('Actual filtered positive plus sampled-negative count required')
        steps = (int(query_count)+batch-1)//batch
        tail = int(query_count) % batch
        records = query_records = int(query_count)
    else:
        raise ValueError('Explicit NCN/NCNC/BUDDY source family required')
    if steps < 1:
        raise ValueError('Zero-map native epochs are not admitted')
    return {'family': family, 'batch_size': batch, 'optimizer_maps_per_epoch': steps,
            'source_iterator_records_per_epoch': records, 'task_query_records_per_epoch': query_records,
            'dropped_positive_tail': tail if family != 'BUDDY' else 0,
            'included_partial_tail': tail if family == 'BUDDY' else 0}


def progress(layout, started_epoch, consumed_maps):
    total = layout['optimizer_maps_per_epoch']
    started_epoch, consumed_maps = int(started_epoch), int(consumed_maps)
    if started_epoch < 0 or consumed_maps < 0 or consumed_maps > total:
        raise ValueError('Native sampler epoch/cursor is inconsistent')
    if started_epoch == 0 and consumed_maps != 0:
        raise ValueError('Unstarted native epoch has consumed maps')
    complete = max(0, started_epoch-1) + int(started_epoch > 0 and consumed_maps == total)
    boundary = consumed_maps in (0, total)
    remaining = total if consumed_maps == total else total-consumed_maps
    return {'started_native_epoch': started_epoch, 'consumed_maps_in_current_epoch': consumed_maps,
            'completed_epochs': complete, 'at_epoch_boundary': boundary,
            'maps_to_next_completed_epoch': remaining,
            'resumed_prefix_maps': 0 if boundary else consumed_maps}
