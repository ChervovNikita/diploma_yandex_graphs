"""Matched learning-rate sensitivity for Roman Empire architecture controls.

Every method receives the same three learning rates and five depths. Selection
uses splits 0–2 validation only. Splits 3–9 evaluate the selected settings.
"""

from itertools import product
import hashlib
import math
from pathlib import Path
import statistics

from ablation.p05_hparam_grid.grid import REFERENCE, config_name

LEARNING_RATES = [3e-5, 1e-4, 3e-4]
BACKBONES = ['GAT', 'SAGE']
ALTERNATIVES = ['input_only', 'output_only_identity', 'output_only_signs',
                'all_layer_be', 'heads_single_pass']
PILOT_SPLITS = [0, 1, 2]
DEPTHS = [1, 2, 3, 4, 5]
STAGE = 'architecture_confirmation'


def candidate_configs():
    return {config_name(hp): hp for hp in
            [dict(REFERENCE, lr=lr, weight_decay=0.0) for lr in LEARNING_RATES]}


def add_architecture_search(builder):
    """Add matched pilot, selection and confirmation dependencies to an existing DAG."""
    from .plan import node_spec

    reports = []
    dataset = 'roman-empire'
    grid = candidate_configs()
    for model in BACKBONES:
        cells = []
        for (name, hp), depth, split in product(grid.items(), DEPTHS, PILOT_SPLITS):
            cell, _ = builder.cell(dataset, model, depth, split, hp, name,
                                   'architecture_pilot', 12)
            cells.append(cell)
        control_decision = f'decision/architecture/{model}/controls'
        builder.add(control_decision, 'select',
                    dict(cells=cells, dataset=dataset, model=model,
                         pilot_splits=PILOT_SPLITS), cells, 14)
        builder.selected(control_decision, dataset, model, STAGE)
        scores = [
            f'score/{STAGE}/{dataset}/{model}/split{split}/{setting}'
            for split, setting in product(range(3, 10), ['BASE', 'ENS', 'GNNM'])
        ]
        decisions = [control_decision]
        selections = {setting: control_decision for setting in ('BASE', 'ENS', 'GNNM')}
        for variant in ALTERNATIVES:
            candidates = []
            for (name, hp), depth, split in product(grid.items(), DEPTHS, PILOT_SPLITS):
                fit = builder.fit(node_spec(dataset, model, depth, split, variant, hp),
                                  'architecture_pilot', 12)
                candidates.append(dict(config_name=name, depth=depth, split=split, fit=fit))
            decision = f'decision/architecture/{model}/{variant}'
            builder.add(decision, 'architecture_select',
                        dict(candidates=candidates, dataset=dataset, model=model, variant=variant,
                             candidate_configs=grid, depths=DEPTHS, pilot_splits=PILOT_SPLITS),
                        [item['fit'] for item in candidates], 14)
            decisions.append(decision)
            selections[variant] = decision
            condition = dict(task=decision, path='output/done.json', field='ready', equals=True)
            for split in range(3, 10):
                spec = node_spec(dataset, model, 1, split, variant)
                fit = builder.fit(spec, STAGE, 18, [decision], condition,
                                  dict(selection=decision, setting=variant))
                score = f'score/{STAGE}/{dataset}/{model}/split{split}/{variant}'
                builder.add(score, 'node_score',
                            dict(fits=[fit], setting=variant, dataset=dataset,
                                 model=model, split=split, stage=STAGE),
                            [fit, decision], 19, 'gpu', condition, memory=24)
                scores.append(score)
        profiles = []
        for setting, decision in selections.items():
            score = f'score/{STAGE}/{dataset}/{model}/split3/{setting}'
            identifier = f'profile/architecture/{model}/{setting}'
            ready_field = 'ready_' + setting if setting in ('BASE', 'ENS', 'GNNM') else 'ready'
            condition = dict(task=decision, path='output/done.json', field=ready_field, equals=True)
            builder.add(identifier, 'profile',
                        dict(score=score, selection=decision, setting=setting,
                             dataset=dataset, model=model, split=3),
                        [score, decision], 23, 'gpu', condition, memory=24, exclusive=True)
            profiles.append(dict(task=identifier, score=score, selection=decision, setting=setting))
        report = f'report/architecture/{model}'
        builder.add(report, 'architecture_comparison',
                    dict(scores=scores, decisions=decisions, profiles=profiles,
                         dataset=dataset, model=model,
                         learning_rates=LEARNING_RATES, weight_decay=0.0,
                         pilot_splits=PILOT_SPLITS, confirmation_splits=list(range(3, 10))),
                    [], 40, optional=[*scores, *decisions, *[item['task'] for item in profiles]])
        reports.append(report)
    return reports


def execute_selection(metadata, root):
    """Select one alternative from the complete declared validation candidate set."""
    from .tasks import result

    expected = set(product(metadata['candidate_configs'], metadata['depths'],
                           metadata['pilot_splits']))
    entries = metadata['candidates']
    identities = [(item['config_name'], item['depth'], item['split']) for item in entries]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError('Architecture pilot must contain exactly one fit for every declared candidate/split')
    grouped = {}
    unresolved = []
    for item in entries:
        fit = result(root, item['fit'])
        if not fit.get('defer_test', fit.get('test_deferred', False)):
            raise ValueError('Architecture selection requires deferred held-out scoring')
        expected_identity = {
            'dataset': metadata['dataset'], 'model': metadata['model'],
            'variant': metadata['variant'], 'num_layers': item['depth'], 'split': item['split'],
        }
        for key, value in expected_identity.items():
            if fit.get(key) != value:
                raise ValueError(f'Architecture candidate identity mismatch for {item["fit"]}: {key}')
        hp = metadata['candidate_configs'][item['config_name']]
        for key in ('lr', 'weight_decay', 'dropout', 'attention_dropout', 'num_heads'):
            if fit.get(key) != hp[key]:
                raise ValueError(f'Architecture candidate hyperparameter mismatch for {key}')
        grouped.setdefault((item['config_name'], item['depth']), []).append((item, fit))
        if fit.get('needs_convergence_extension'):
            unresolved.append(item['fit'])
    ranking, infeasible = [], []
    for (name, depth), rows in grouped.items():
        rows.sort(key=lambda pair: pair[0]['split'])
        failures = [item['fit'] for item, fit in rows
                    if fit.get('scientific_status') == 'numerical_failure']
        if failures:
            infeasible.append(dict(config_name=name, depth=depth, failed_fits=failures))
            continue
        values = [fit['val_metric'] for _, fit in rows]
        if any(not math.isfinite(value) for value in values):
            raise ValueError('Nonfinite architecture validation score without a numerical-failure receipt')
        ranking.append(dict(config_name=name, depth=depth, mean=statistics.mean(values),
                            values=values, spec=rows[0][1],
                            fits=[item['fit'] for item, _ in rows]))
    ranking.sort(key=lambda row: (-row['mean'], row['depth'], row['config_name']))
    chosen = ranking[0] if ranking else None
    selected_unresolved = sorted(set(chosen['fits']) & set(unresolved)) if chosen else []
    return dict(
        role='val', dataset=metadata['dataset'], model=metadata['model'],
        selections={metadata['variant']: chosen} if chosen else {},
        rankings={metadata['variant']: ranking}, ready=bool(chosen),
        infeasible_candidates=infeasible, unresolved_convergence=sorted(unresolved),
        selected_unresolved=selected_unresolved, no_test_read=True,
        pilot_splits=metadata['pilot_splits'], candidate_count=len(expected),
        tie_break='lower depth then lexical configuration name',
    )


def bind_selected_profile(binding, profile, root):
    """Link measured costs to the exact selected score and checkpoint receipts."""
    from .tasks import output, result

    score = result(root, binding['score'])
    decision = result(root, binding['selection'])
    setting = binding['setting']
    if decision.get('role') != 'val' or not decision.get('ready_' + setting, decision.get('ready', False)):
        raise ValueError('Architecture cost profile requires a ready validation selection')
    selected = decision['selections'][setting]
    if score['setting'] != setting:
        raise ValueError('Architecture profile setting differs from its score')
    for key in ('dataset', 'model', 'split', 'num_layers'):
        if profile[key] != score[key]:
            raise ValueError(f'Architecture profile differs from its selected score in {key}')
    fits = score['fit_ids']
    expected_dirs = [str(output(root, identifier).resolve()) for identifier in fits]
    if profile['job_dirs'] != expected_dirs:
        raise ValueError('Architecture profile checkpoint directories differ from its selected score')
    checkpoints = {}
    for identifier in fits:
        fit = result(root, identifier)
        for key in ('num_layers', 'lr', 'weight_decay', 'dropout', 'attention_dropout', 'num_heads'):
            if fit[key] != selected['spec'][key]:
                raise ValueError(f'Architecture profile fit differs from its selection in {key}')
        checkpoints[identifier] = fit.get('selected_checkpoint_sha256')
        if not checkpoints[identifier]:
            raise ValueError('Architecture profile requires a recorded selected checkpoint hash')
    return dict(profile,
                setting=setting, selection_task=binding['selection'], score_task=binding['score'],
                profile_task=binding['task'], fit_ids=fits,
                selected_config_name=selected['config_name'], selected_depth=selected['depth'],
                selection_receipt_sha256=hashlib.sha256(
                    (output(root, binding['selection']) / 'done.json').read_bytes()).hexdigest(),
                score_receipt_sha256=hashlib.sha256(
                    (output(root, binding['score']) / 'done.json').read_bytes()).hexdigest(),
                profile_receipt_sha256=hashlib.sha256(
                    (output(root, binding['task']) / 'done.json').read_bytes()).hexdigest(),
                selected_checkpoint_sha256=checkpoints,
                unresolved_convergence=score.get('unresolved_convergence', []))

def execute_comparison(metadata, root):
    """Keep successful comparisons available when another declared arm is infeasible."""
    from .analyses import comparison
    from .tasks import output, read_json, result

    available, missing, skipped, states = [], [], [], {}
    for identifier in metadata['scores']:
        path = output(root, identifier) / 'done.json'
        if path.exists():
            available.append(result(root, identifier))
            states[identifier] = 'complete'
            continue
        state_path = Path(root) / 'tasks' / identifier / 'state.json'
        state = read_json(state_path).get('status', 'missing') if state_path.exists() else 'missing'
        states[identifier] = state
        missing.append(identifier)
        if state == 'skipped':
            skipped.append(identifier)
    decisions = {
        identifier: result(root, identifier) for identifier in metadata['decisions']
        if (output(root, identifier) / 'done.json').exists()
    }
    profiles, missing_profiles, profile_issues = {}, [], []
    for binding in metadata.get('profiles', []):
        identifier = binding['task']
        path = output(root, identifier) / 'done.json'
        if path.exists():
            profile = result(root, identifier)
            states[identifier] = 'complete'
            if profile.get('scientific_status') == 'infeasible':
                profiles[identifier] = dict(profile, **binding)
                profile_issues.append(identifier)
            else:
                profiles[identifier] = bind_selected_profile(binding, profile, root)
            continue
        state_path = Path(root) / 'tasks' / identifier / 'state.json'
        states[identifier] = read_json(state_path).get('status', 'missing') if state_path.exists() else 'missing'
        missing_profiles.append(identifier)
    report = comparison(available)
    complete = (not missing and not missing_profiles and not profile_issues and report['complete']
                and len(decisions) == len(metadata['decisions'])
                and all(item['ready'] and not item.get('selected_unresolved')
                        for item in decisions.values()))
    terminal = {'complete', 'failed', 'skipped', 'blocked'}
    report.update(
        complete=complete, evaluation_complete=all(state in terminal for state in states.values()),
        scientific_status='complete' if complete else 'partial',
        missing_score_tasks=missing, skipped_score_tasks=skipped, task_states=states,
        selection_receipts=decisions, requested_score_cells=len(metadata['scores']),
        cost_profiles=profiles, missing_profile_tasks=missing_profiles,
        infeasible_profile_tasks=profile_issues,
        skipped_profile_tasks=[identifier for identifier in missing_profiles if states[identifier] == 'skipped'],
        dataset=metadata['dataset'], model=metadata['model'],
        matched_search=dict(learning_rates=metadata['learning_rates'], weight_decay=0.0,
                            depths=DEPTHS, pilot_splits=PILOT_SPLITS,
                            confirmation_splits=list(range(3, 10))),
        inference_scope='Descriptive paired comparisons on overlapping graph splits',
    )
    return report
