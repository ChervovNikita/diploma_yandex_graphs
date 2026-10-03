"""Complete-cohort checkpoint/logit replays precede TRAIN-control comparisons."""
import math
import time
import common as c
import native_training as n
import study


def check_trace(result):
    path = c.verify(result['trace'])
    rows = [__import__('json').loads(line) for line in path.read_text().splitlines()]
    c.require(len(rows) == 2700, 'Incomplete physical update trace')
    best = -1
    selected = None
    checkpoint_events = []
    for update, row in enumerate(rows, 1):
        global_stage = update > 200
        c.require(row['actual_update'] == update and row['author_display_epoch'] == update - 1 and
                  row['global'] is global_stage and row['stage'] == ('global' if global_stage else 'local') and
                  row['stage_epoch'] == (update - 200 if global_stage else update) and
                  row['actual_local_updates'] == min(update, 200) and
                  row['actual_global_updates'] == max(update - 200, 0) and row['val_count'] == 6123 and
                  type(row['val_correct']) is int and 0 <= row['val_correct'] <= 6123 and
                  math.isfinite(row['fit_CE']) and row['strict_selected'] is (row['val_correct'] > best),
                  'Schedule/strict cross-stage selector trace differs')
        if row['strict_selected']:
            c.require(row['checkpoint'] is not None and row['portable_replay']['bitwise_full_next_step'] is True,
                      'Selected candidate lacks portable checkpoint/replay')
            checkpoint_events.append(row['checkpoint'])
            best, selected = row['val_correct'], row['checkpoint']
        else:
            c.require(row['checkpoint'] is None and row['portable_replay'] is None, 'Unselected checkpoint event differs')
    c.require(selected == result['selected_checkpoint'] and best == result['selection']['val_correct'] and
              checkpoint_events == [q['checkpoint'] for q in result['checkpoints']] and
              result['actual_local_updates'] == 200 and result['actual_global_updates'] == 2500 and
              result['actual_optimizer_updates'] == 2700 and result['final_portable_replay']['bitwise_full_next_step'] is True,
              'Final selection/full fit closure differs')
    return len(rows)


def cohort(a):
    _, path = study.freeze(a['closure_freeze'])
    closure = c.read(path / 'RESULT.json')
    registry = c.read(c.verify(a['registry']))
    c.require(closure['schema'] == 'amazon_polynormer_cohort_closure_v2' and closure['status'] == 'complete' and
              closure['source'] == a['source'] and closure['registry'] == a['registry'] and
              closure['families'] == registry['families'] and
              [v['id'] for v in closure['physical_fits']] == [c.fit_id(v) for v in c.schedule()],
              'Exact complete nine-family/fifteen-fit closure required')
    results = {}
    for record in closure['physical_fits']:
        _, fit_path = study.freeze(record['freeze'])
        result = c.read(fit_path / 'RESULT.json')
        c.require(result['schema'] == 'amazon_polynormer_physical_fit_v2' and result['status'] == 'complete' and
                  result['report_eligible'] is True and result['bindings']['source'] == a['source'] and
                  result['bindings']['runtime_receipt'] == a['runtime_receipt'] and
                  result['bindings']['consumer_release'] == a['consumer_release'] and
                  result['bindings']['registry'] == a['registry'] and
                  result['bindings']['fit_id'] == record['id'] and record['result'] == c.record(fit_path / 'RESULT.json'),
                  'Physical fit/runtime/data identity differs')
        check_trace(result)
        c.verify(result['bindings']['admission'])
        results[record['id']] = result
    c.require(len(results) == 15 and sum(q['actual_optimizer_updates'] for q in results.values()) == 40500 and
              sum(q['complete_member_trajectory_updates'] for q in results.values()) == 64800,
              'Complete unique physical work counts differ')
    return closure, registry, results


def metrics(logits, ids, labels):
    import torch
    c.require(logits.dtype == torch.float32 and logits.ndim == 3 and
              tuple(logits.shape[1:]) == (24492, 5) and logits.shape[0] in (1, 4), 'Fixed report logit schema required')
    n.finite(logits, 'Report raw logits nonfinite')
    index = torch.tensor(ids, dtype=torch.long, device=logits.device)
    y = torch.tensor(labels, dtype=torch.long, device=logits.device)
    z = logits.index_select(1, index)
    prediction = z[0].argmax(-1) if z.shape[0] == 1 else torch.softmax(z, dim=-1).mean(0).argmax(-1)
    logp = torch.log_softmax(z.to(torch.float64), dim=-1)
    mixture_logp = torch.logsumexp(logp, dim=0) - math.log(z.shape[0])
    row = torch.arange(len(ids), device=z.device)
    pooled_probability = logp.exp().mean(0)
    onehot = torch.nn.functional.one_hot(y, num_classes=5).to(torch.float64)
    values = {'count': len(ids), 'accuracy': float((prediction == y).to(torch.float64).mean()),
              'NLL': float(-mixture_logp[row, y].mean()),
              'Brier': float(((pooled_probability - onehot) ** 2).sum(-1).mean()),
              'member_NLL': [float(-logp[m, row, y].mean()) for m in range(z.shape[0])]}
    c.require(all(math.isfinite(v) for v in (values['accuracy'], values['NLL'], values['Brier'], *values['member_NLL'])),
              'Reporting arithmetic nonfinite')
    return values


def metric_gate(actual, expected):
    c.require(actual.keys() == expected.keys() and actual['count'] == expected['count'], 'Recomputed metric schema differs')
    for name in ('accuracy', 'NLL', 'Brier'):
        c.require(math.isclose(actual[name], expected[name], rel_tol=0.0, abs_tol=1e-6), 'Saved-logit metric replay differs')
    c.require(len(actual['member_NLL']) == len(expected['member_NLL']) and
              all(math.isclose(x, y, rel_tol=0.0, abs_tol=1e-6) for x, y in zip(actual['member_NLL'], expected['member_NLL'])),
              'Saved-logit member NLL replay differs')


def run(a, admission_record, output):
    import torch
    start = time.perf_counter()
    closure, registry, results = cohort(a)
    c.require(a['control_predictive_scoring_authorized_after_complete_closure'] is True,
              'Separate post-closure TRAIN-control evaluation authority required')
    x, edge, blocks, preprocessing = c.load_data(a, a['device'], evaluation=True)
    raw = {}
    replay_receipts = []
    for row in c.schedule():
        ident = c.fit_id(row)
        result = results[ident]
        c.require(result['bindings']['row'] == row and result['bindings']['preprocessing'] == preprocessing,
                  'Exact row/role/preprocessing identity differs')
        image = n.load_image(result['selected_checkpoint'], result['bindings'])
        model, optimizer, _ = n.build(row, a['device'])
        n.restore(model, optimizer, image, a['device'])
        n.synchronize(a['device'])
        before = time.perf_counter()
        actual = n.eval_logits(model, x, edge)
        n.synchronize(a['device'])
        inference_seconds = time.perf_counter() - before
        maximum = n.logit_gate(actual.cpu(), image['selected_raw_logits'])
        saved = torch.load(c.verify(result['selected_logits']), map_location='cpu', weights_only=True)
        c.require(saved['bindings'] == result['bindings'] and saved['selected_checkpoint'] == result['selected_checkpoint'],
                  'Saved raw-logit reference differs')
        n.logit_gate(actual.cpu(), saved['raw_logits'])
        c.require(n.correct_count(actual, blocks[row['split']]) == result['selection']['val_correct'] and
                  n.stage(model) == result['selection']['global'], 'Reopened selected VAL/stage differs')
        replay = n.portable_replay(model, optimizer, result['selected_checkpoint'], row, x, edge,
                  blocks[row['split']], a['device'], result['bindings'])
        raw[ident] = saved['raw_logits']
        replay_receipts.append({'id': ident, 'selected_checkpoint': result['selected_checkpoint'],
                               'inference_seconds': inference_seconds, 'logit_max_absolute': maximum,
                               'portable_replay': replay, 'memory': study.caps(a, a['device'])})
        del actual, model, optimizer, image, saved
    family_reports = []
    for family in registry['families']:
        logits = torch.cat([raw[ident] for ident in family['physical_fit_references']], dim=0)
        c.require(logits.shape[0] == (1 if family['family'] == 'single_author' else 4), 'All four members must be retained')
        roles = blocks[family['split']]
        scores = {name: metrics(logits, roles[key], roles[key + '_labels'])
                  for name, key in (('TRAIN_FIT', 'fit'), ('VAL', 'val'), ('TRAIN_control', 'control'))}
        # Independent second deserialization of frozen raw logits, same fixed arithmetic.
        reopened = []
        for ident in family['physical_fit_references']:
            result = results[ident]
            saved = torch.load(c.verify(result['selected_logits']), map_location='cpu', weights_only=True)
            reopened.append(saved['raw_logits'])
        reopened_logits = torch.cat(reopened, dim=0)
        for name, key in (('TRAIN_FIT', 'fit'), ('VAL', 'val'), ('TRAIN_control', 'control')):
            metric_gate(metrics(reopened_logits, roles[key], roles[key + '_labels']), scores[name])
        family_reports.append({'reference': family, 'metrics': scores, 'saved_logit_metrics_replayed': True,
                'selected_stages': [results[i]['selection'] for i in family['physical_fit_references']],
                'physical_cost_references': [results[i]['cost'] for i in family['physical_fit_references']],
                'complete_family_inference_seconds': sum(r['inference_seconds'] for r in replay_receipts
                                      if r['id'] in family['physical_fit_references'])})
    differences = []
    for split, seed in c.BLOCKS:
        by_family = {r['reference']['family']: r for r in family_reports if r['reference']['split'] == split}
        differences.append({'split': split, 'block_seed': seed,
                'GNNM4_minus_independent4_control_NLL': by_family['gnnm_boundary_4']['metrics']['TRAIN_control']['NLL'] -
                         by_family['independent_author_4_same_width']['metrics']['TRAIN_control']['NLL']})
    c.preserve(a, admission_record)
    cohort(a)  # Revalidate every input/freeze/checkpoint/logit/trace/cost after all reads.
    c.write(output / 'RESULT.json', {'schema': 'amazon_polynormer_comparison_v2', 'status': 'complete',
            'source': a['source'], 'closure_freeze': a['closure_freeze'], 'registry': a['registry'],
            'family_reports': family_reports, 'checkpoint_replays': replay_receipts,
            'all_three_pairs': differences, 'mean_GNNM4_minus_independent4_control_NLL':
                       sum(r['GNNM4_minus_independent4_control_NLL'] for r in differences) / 3,
            'unique_physical_fits': 15, 'family_records': 9,
            'unique_scientific_optimizer_steps': 40500, 'complete_member_training_trajectory_updates': 64800,
            'native_single_semantics': 'same exact independent member0; charged once in unique study total',
            'cost': {'body_seconds': time.perf_counter() - start, 'extra_replay_updates': 30,
                     'extra_replay_member_trajectory_updates': 48, 'memory': study.caps(a, a['device'])},
            'TEST_labels_used': False, 'new_recipe_objective_or_superiority_certificate': False})
