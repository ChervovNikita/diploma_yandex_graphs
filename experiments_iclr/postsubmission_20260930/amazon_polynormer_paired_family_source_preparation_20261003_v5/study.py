"""One physical fit per registered row; persistent selector spans both stages."""
import json
import time
import common as c
import native_training as n


def validate_registry(a):
    r = c.read(c.verify(a['registry']))
    c.require(r['schema'] == 'amazon_polynormer_registry_v2' and r['source'] == a['source'] and
              [v['row'] for v in r['physical_fits']] == c.schedule() and
              len(r['physical_fits']) == 15 and len(r['families']) == 9,
              'Fixed complete fifteen-fit/nine-family registry required')
    master = c.read(c.verify(r['master_source_claim']))
    c.require(master['source'] == a['source'] and master['no_second_registry_or_replacement'] is True,
              'One-cohort-per-source claim differs')
    row = next((v for v in r['physical_fits'] if v['id'] == a['fit_id']), None)
    c.require(row is not None and row['output'] == a['output'] and
              row['release_path'] == a['self_path'], 'Unregistered physical attempt/output/release')
    return r, row['row']


def freeze(row):
    p = c.verify(row)
    f = c.read(p)
    c.require(f['schema'] == 'amazon_polynormer_success_freeze_v2' and f['status'] == 'success',
              'Successful physical whole-process freeze required')
    directory = p.parent
    actual = sorted(str(q.relative_to(directory)) for q in directory.rglob('*')
                    if q.is_file() and q.name != 'FREEZE.json')
    c.require(actual == sorted(v['relative'] for v in f['files']), 'Frozen output inventory changed')
    for v in f['files']:
        c.require(c.record(directory / v['relative']) == v['descriptor'], 'Frozen output custody changed')
    terminal = c.read(directory / 'TERMINAL.json')
    c.require(terminal['physical_exit_code'] == 0 and terminal['status'] == 'success', 'Physical success differs')
    return f, directory


def science_gate(a):
    _, row = validate_registry(a)
    f, path = freeze(a['qualification_freeze'])
    q = c.read(path / 'RESULT.json')
    resource = c.read(c.verify(a['resource_admission']))
    c.require(q['schema'] == 'amazon_polynormer_qualification_v2' and q['status'] == 'passed' and
              q['source'] == a['source'] and q['runtime_receipt'] == a['runtime_receipt'] and
              q['consumer_release'] == a['consumer_release'] and q['report_eligible'] is False,
              'Exact Amazon numerical qualification required')
    c.require(resource['schema'] == 'amazon_polynormer_resource_admission_v2' and
              resource['execution_authorized'] is True and resource['source'] == a['source'] and
              resource['qualification_freeze'] == a['qualification_freeze'] and
              resource['registry'] == a['registry'] and resource['full_15_fit_schedule_authorized'] is True and
              resource['all_selected_checkpoint_replays_costed'] is True and
              resource['caps'] == a['caps'], 'Independent measured whole-study resource admission required')
    forecast = resource['forecast_to_fill_from_actual_probe']
    c.require(forecast['worst_case_selected_candidates_per_fit'] == 2700 and
              all(v is not None for v in forecast.values()), 'Measured full-schedule resource forecast is incomplete')
    return row


def caps(a, device):
    value = n.memory(device)
    c.require(value['host_peak_rss_bytes'] <= a['caps']['rss_bytes'], 'Admitted host memory exceeded')
    if device.startswith('cuda:'):
        c.require(value['cuda_peak_allocated_bytes'] <= a['caps']['cuda_peak_allocated_bytes'] and
                  value['cuda_peak_reserved_bytes'] <= a['caps']['cuda_peak_reserved_bytes'],
                  'Admitted CUDA memory exceeded')
    return value


def run(a, admission_record, output):
    import torch
    row = science_gate(a)
    device = a['device']
    start = time.perf_counter()
    x, edge, blocks, preprocessing = c.load_data(a, device)
    roles = blocks[row['split']]
    model, optimizer, construction = n.build(row, device)
    bindings = {'source': a['source'], 'admission': admission_record, 'registry': a['registry'],
                'fit_id': a['fit_id'], 'row': row, 'recipe': c.read(c.PACKET / 'DESIGN.json')['recipe'],
                'runtime_receipt': a['runtime_receipt'], 'consumer_release': a['consumer_release'],
                'preprocessing': preprocessing, 'role_record': roles['role_record']}
    n.save_image(output / 'CONSTRUCTION.pt', {'rng': construction})
    best = -1
    selected = None
    selected_local = None
    checkpoints = []
    stage_times = {'local': [], 'global': []}
    trace_path = output / 'TRACE.jsonl'
    transition_record = None
    with trace_path.open('x') as trace:
        for update in range(1, 2701):
            if update == 201:
                image = n.load_image(selected_local, bindings)
                transition_record = n.transition(model, optimizer, image, device)
                transition_record.update(actual_local_updates=200, actual_global_updates=0,
                                         selected_checkpoint=selected_local)
                c.write(output / 'TRANSITION.json', transition_record)
                del image
            expected_global = update > 200
            c.require(n.stage(model) == expected_global, 'Actual schedule stage differs')
            n.synchronize(device)
            step_start = time.perf_counter()
            loss = n.train_step(model, optimizer, x, edge, roles)
            logits = n.eval_logits(model, x, edge)
            correct = n.correct_count(logits, roles)
            n.synchronize(device)
            seconds = time.perf_counter() - step_start
            current_stage = 'global' if expected_global else 'local'
            stage_times[current_stage].append(seconds)
            event = {'actual_update': update, 'author_display_epoch': update - 1,
                     'stage': current_stage, 'stage_epoch': update - 200 if expected_global else update,
                     'actual_local_updates': min(update, 200), 'actual_global_updates': max(update - 200, 0),
                     'global': expected_global, 'fit_CE': float(loss), 'val_correct': correct,
                     'val_count': len(roles['val']), 'strict_selected': correct > best,
                     'training_plus_VAL_seconds': seconds, 'checkpoint': None, 'portable_replay': None,
                     'restored_from_actual_update': (transition_record['from_selected']['actual_update']
                                                    if update == 201 else None)}
            if correct > best:
                selection = {k: event[k] for k in ('actual_update', 'author_display_epoch', 'stage', 'stage_epoch',
                             'actual_local_updates', 'actual_global_updates', 'global', 'val_correct', 'val_count')}
                checkpoint, checkpoint_io = n.snapshot_write(output / 'checkpoints' / ('selected_%04d.pt' % update),
                                model, optimizer, bindings, selection, construction, logits, device)
                replay = n.portable_replay(model, optimizer, checkpoint, row, x, edge, roles, device, bindings)
                event.update(checkpoint=checkpoint, checkpoint_io=checkpoint_io, portable_replay=replay)
                selected = checkpoint
                if not expected_global:
                    selected_local = checkpoint
                checkpoints.append({'checkpoint': checkpoint, 'selection': selection, 'replay': replay,
                                    'checkpoint_io': checkpoint_io})
                best = correct
            del logits
            event['memory'] = caps(a, device)
            trace.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
            trace.flush()
        trace.flush()
    c.require(selected is not None and selected_local is not None and transition_record is not None,
              'Missing first-update/local/final selector image')
    final_image = n.load_image(selected, bindings)
    n.restore(model, optimizer, final_image, device)
    final_logits = n.eval_logits(model, x, edge)
    maximum = n.logit_gate(final_image['selected_raw_logits'], final_logits.cpu())
    c.require(n.correct_count(final_logits, roles) == best and
              n.stage(model) == final_image['selection']['global'], 'Final selected-stage/VAL restoration differs')
    final_replay = n.portable_replay(model, optimizer, selected, row, x, edge, roles, device, bindings)
    logits_row = n.save_image(output / 'SELECTED_LOGITS.pt', {'raw_logits': n.cpu_tree(final_logits),
                  'selected_checkpoint': selected, 'bindings': bindings})
    c.preserve(a, admission_record)
    for v in checkpoints:
        c.verify(v['checkpoint'])
    c.verify(logits_row)
    result = {'schema': 'amazon_polynormer_physical_fit_v2', 'status': 'complete', 'report_eligible': True,
              'source': a['source'], 'bindings': bindings, 'selected_checkpoint': selected,
              'selected_logits': logits_row, 'selection': final_image['selection'],
              'checkpoints': checkpoints, 'trace': c.record(trace_path), 'transition': c.record(output / 'TRANSITION.json'),
              'actual_local_updates': 200, 'actual_global_updates': 2500, 'actual_optimizer_updates': 2700,
              'complete_member_trajectory_updates': 2700 * (4 if row['kind'] == 'gnnm_boundary_4' else 1),
              'final_logit_replay_max_absolute': maximum, 'final_portable_replay': final_replay,
              'cost': {'body_wall_seconds': time.perf_counter() - start,
                       'local_train_VAL_seconds': sum(stage_times['local']),
                       'global_train_VAL_seconds': sum(stage_times['global']),
                       'selected_checkpoint_count': len(checkpoints),
                       'charged_probe_updates': 2 * (len(checkpoints) + 1),
                       'state_bytes': n.state_bytes(model, optimizer), 'memory': caps(a, device)}}
    c.write(output / 'RESULT.json', result)
