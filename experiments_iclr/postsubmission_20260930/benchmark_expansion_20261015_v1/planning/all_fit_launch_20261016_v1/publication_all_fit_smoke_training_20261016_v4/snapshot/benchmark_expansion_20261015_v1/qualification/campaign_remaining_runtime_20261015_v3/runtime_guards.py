"""Scalar adapters for the actual host descriptor and finite native epoch loop."""
from witness_support import ROOT, GPU, require


def projection_host(config):
    """Add the projection label to the verified actual allocation config."""
    require(config['schema'] == 'allocation-server51-campaign-host-v1'
            and config['worker'] == 'allocation/gpu0' and config['project_root'] == str(ROOT)
            and config['expected_hostname'] == 'anogena-2-0'
            and config['physical_gpu_uuid'] == GPU and config['logical_device'] == 'cuda:0',
            'This witness projects the exact allocation host descriptor')
    host = config['worker'].split('/')[0]
    require(config.get('host', host) == host, 'Conflicting projection host label')
    return {**config, 'host': host}


def guard_training_transition(work, training_open, forbid):
    """No forward/reverse/optimizer transition during the exhaustion-only probe."""
    if not training_open or work['TRAIN_updates'] >= 38:
        forbid('No native training transition beyond the 38 admitted updates')


def native_update_call(function, work, training_open, epoch_complete, save, forbid, *args, **kwargs):
    """Count 38 update calls plus one expected native EpochComplete probe separately.

    The original session is called on the final probe so the original loader and
    adapter remain the authority for epoch completion. Native forward, reverse
    and optimizer guards independently prohibit work while that probe is open.
    """
    if not training_open or work['TRAIN_callback_calls'] >= 39 or work['TRAIN_exhaustion_probes_completed']:
        forbid('No callback outside the one native epoch or after its exhaustion probe')
    probe = work['TRAIN_updates'] == 38
    if probe:
        if work['TRAIN_exhaustion_probe_attempts']:
            forbid('Only one native exhaustion probe admitted')
        work['TRAIN_exhaustion_probe_attempts'] += 1
    else:
        if work['TRAIN_update_attempts'] >= 38:
            forbid('No extra TRAIN update call admitted')
        work['TRAIN_update_attempts'] += 1
    work['TRAIN_callback_calls'] += 1
    protected = ('TRAIN_updates', 'optimizer_attempts', 'optimizer_completed',
                 'native_forwards_attempted', 'native_forwards_completed')
    before = {key: work[key] for key in protected}
    save()
    try:
        value = function(*args, **kwargs)
    except epoch_complete:
        if not probe or any(work[key] != before[key] for key in protected):
            forbid('Native epoch exhausted early or exhaustion probe performed work')
        work['TRAIN_exhaustion_probes_completed'] += 1
        save()
        raise  # The unchanged native train_epoch catches its own exact exception.
    if probe:
        forbid('The final callback must report native EpochComplete, not an extra update')
    work['TRAIN_updates'] += 1
    save()
    return value
