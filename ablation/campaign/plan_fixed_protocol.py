"""Build the final rebuttal campaign under the frozen protocol.

    python -m ablation.campaign.plan_fixed_protocol --out ablation/campaign/plans/fixed_protocol_20261116.json

Read ablation/fixed_protocol/PROTOCOL.md first. Every node fit uses the paper's
protocol with the step limit raised to 40000 (ablation/fixed_protocol/protocol.py);
the fit refuses to start otherwise. Stages, in queue priority order:

  1  parity            ten archived runs that stopped by patience; they must reproduce
  1  main_rerun        main-table runs that stopped at the old step limit or are missing
  2  gat_pilot ...     the staged hyperparameter search, design unchanged (GAT first,
                       validation-only decisions, gated expansion, confirmations)
  3  collab_gcn/sage   two more Collab backbones (BASE, ENS, GNNM, GNNM + contrast)
  4  roman_ablation    Roman Empire variants, splits 0-4 first
  6  roman_ablation    the same variants, splits 5-9
  9  node_contrast     GNNM + contrastive term at the paper's selected depths
"""

import argparse
import csv
import json
from collections import Counter
from itertools import product
from pathlib import Path

from ablation.fixed_protocol.protocol import CAMPAIGN_ID, MAIN_TABLE, PROTOCOL_ID, node_protocol
from ablation.p05_hparam_grid.grid import REFERENCE, REFERENCE_NAME, configs_for
from .plan import DATASETS, MODELS, Builder
from .schema import SCHEMA, canonical_hash, validate_plan

HERE = Path(__file__).resolve().parent
ABLATION = HERE.parent
RERUN_LIST = ABLATION / 'fixed_protocol' / 'rerun_list.csv'
CONTRAST_CELLS = ABLATION / 'fixed_protocol' / 'contrast_cells.csv'

# Roman Empire variants: module and number of members. BASE, ENS and GNNM come from
# the main table, which uses the same protocol.
ROMAN_VARIANTS = {
    'input_only': ('p01_placement', 4),
    'output_only_identity': ('p01_placement', 4),
    'output_only_signs': ('p01_placement', 4),
    'all_layer_be': ('p01_placement', 4),
    'gnnm_m1': ('p01_placement', 1),
    'm2': ('p02_full_ablations', 2),
    'm8': ('p02_full_ablations', 8),
    'xavier_all': ('p02_full_ablations', 4),
    'ones_all': ('p02_full_ablations', 4),
    'xavier_rms': ('p03_rms_init', 4),
    'heads_single_pass': ('p04_shared_heads', 4),
    'independent_projectors': ('p04_shared_heads', 4),
}
COLLAB_METHODS = ['native0', 'native1', 'native2', 'native3', 'gnnm', 'gnnm_contrast']


def fixed_spec(dataset, model, depth, split, method, hp=None):
    """One fit. Seeds follow the paper: BASE and GNNM use the split, ensemble
    member k uses 1000 * split + k, and every Roman variant uses the split."""
    hp = dict(hp or REFERENCE)
    spec = dict(node_protocol(), dataset=dataset, model=model, num_layers=depth, split=split,
                hidden_dim=MAIN_TABLE['hidden_dim'],
                weight_decay=hp.pop('weight_decay', MAIN_TABLE['weight_decay']), **hp)
    if method.startswith('native'):
        spec.update(seed=1000 * split + int(method[-1]), members=1, method='base',
                    model_module='ablation.p05_hparam_grid.models')
    elif method == 'base':
        spec.update(seed=split, members=1, method='base', model_module='ablation.p05_hparam_grid.models')
    elif method == 'gnnm':
        spec.update(seed=split, members=4, method='gnnm', model_module='ablation.p05_hparam_grid.models')
    elif method == 'gnnm_contrast':
        spec.update(seed=split, members=4, method='gnnm', variant='gnnm_contrast',
                    model_module='ablation.p13_node_contrast.models')
    elif method in ROMAN_VARIANTS:
        module, members = ROMAN_VARIANTS[method]
        spec.update(seed=split, members=members, method=method, variant=method,
                    model_module=f'ablation.{module}.models')
    else:
        raise ValueError(f'Unknown method {method}')
    return spec


class FixedBuilder(Builder):
    """The colleague's builder with every fit specification taken from fixed_spec."""

    def cell(self, dataset, model, depth, split, hp, label, stage, priority,
             deps=(), when=None, resolve=None):
        fits = [self.fit(fixed_spec(dataset, model, depth, split, method, hp),
                         stage, priority, deps, when, resolve)
                for method in ('native0', 'native1', 'native2', 'native3', 'gnnm')]
        task_id = 'cell/' + canonical_hash(fits)[:20]
        self.add(task_id, 'node_cell', dict(fits=fits, dataset=dataset, model=model,
                 depth=depth, split=split, config_name=label, stage=stage),
                 [*deps, *fits], priority + 1, when=when)
        if split == 0 and stage in ('gat_pilot', 'expanded_pilot'):
            self.add('budget_profile/' + canonical_hash(fits)[:20], 'budget_profile',
                     dict(fits=fits, dataset=dataset, model=model, depth=depth,
                          config_name=label, stage=stage),
                     [*deps, *fits], priority + 2, 'gpu', when, memory=24, exclusive=True)
        return task_id, fits

    def selected(self, decision, dataset, model, stage, when=None, gate=None, priority=12):
        all_scores = []
        for split in range(3, 10):
            for setting, methods in (('BASE', ['native0']), ('ENS', [f'native{i}' for i in range(4)]),
                                     ('GNNM', ['gnnm'])):
                condition = dict(task=decision, path='output/done.json', field='ready_' + setting, equals=True)
                dependencies = [decision] + ([gate] if gate else [])
                fits = [self.fit(fixed_spec(dataset, model, 1, split, method), stage, priority,
                                 dependencies, condition, dict(selection=decision, setting=setting))
                        for method in methods]
                score_id = f'score/{stage}/{dataset}/{model}/split{split}/{setting}'
                self.add(score_id, 'node_score', dict(fits=fits, setting=setting, dataset=dataset,
                         model=model, split=split, stage=stage),
                         [*fits, *dependencies], priority + 3, 'gpu', condition, memory=24)
                all_scores.append(score_id)
        report_id = f'report/{stage}/{dataset}/{model}'
        self.add(report_id, 'comparison', dict(scores=all_scores, stage=stage),
                 [gate] if gate else [], priority + 4, when=when, optional=all_scores + [decision])
        return report_id

    def score(self, task_id, fits, setting, dataset, model, depth, split, stage, priority):
        return self.add(task_id, 'node_score', dict(fits=fits, setting=setting, dataset=dataset,
                        model=model, depth=depth, split=split, stage=stage),
                        fits, priority + 1, 'gpu', memory=24)


def add_main_rerun(b):
    """Stage 1: only the main-table runs whose result can change."""
    with RERUN_LIST.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    scores = []
    for row in rows:
        dataset, model, method = row['dataset'], row['model'], row['method']
        depth, split = int(row['num_layers']), int(row['split'])
        if method == 'ens':
            members = [f'native{k}' for k in range(4)]
        else:
            members = [method]
        fits = [b.fit(fixed_spec(dataset, model, depth, split, member), 'main_rerun', 1)
                for member in members]
        setting = {'base': 'BASE', 'ens': 'ENS', 'gnnm': 'GNNM'}[method]
        scores.append(b.score(f'main_rerun/{dataset}/{model}/L{depth}/split{split}/{setting}', fits,
                              setting, dataset, model, depth, split, 'main_rerun', 1))
    return scores


PARITY_CELLS = [('roman-empire', 'GCN', 2, split, method)
                for split in range(5) for method in ('base', 'gnnm')]


def add_parity(b):
    """Stage 1: ten archived runs that stopped by patience. Under the frozen
    protocol they must stop at about the same step with about the same score,
    which is what justifies retraining only the runs on the rerun list."""
    with RERUN_LIST.open(newline='') as stream:
        rerun = {(row['dataset'], row['model'], int(row['num_layers']), int(row['split']), row['method'])
                 for row in csv.DictReader(stream)}
    scores = []
    for dataset, model, depth, split, method in PARITY_CELLS:
        if (dataset, model, depth, split, method) in rerun:
            raise ValueError('A parity cell must have stopped by patience in the archive')
        fit = b.fit(fixed_spec(dataset, model, depth, split, method), 'parity', 1)
        setting = method.upper() if method == 'base' else 'GNNM'
        scores.append(b.score(f'parity/{dataset}/{model}/L{depth}/split{split}/{setting}', [fit], setting,
                              dataset, model, depth, split, 'parity', 1))
    return scores


def add_grid(b):
    """Stage 2: the colleague's staged search, unchanged except for the protocol."""
    reports, decisions = [], []
    grid = configs_for('combined')
    for dataset in DATASETS:
        cells = []
        for name, hp in grid.items():
            for depth, split in product(range(1, 6), range(3)):
                cell, _ = b.cell(dataset, 'GAT', depth, split, hp, name, 'gat_pilot', 2)
                cells.append(cell)
        decision = f'decision/gat/{dataset}'
        b.add(decision, 'select', dict(cells=cells, dataset=dataset, model='GAT', pilot_splits=[0, 1, 2]),
              cells, 2)
        profiles = [task['id'] for task in b.tasks.values()
                    if task['metadata']['action'] == 'budget_profile'
                    and task['metadata']['dataset'] == dataset and task['metadata']['model'] == 'GAT']
        budget = f'budget/gat/{dataset}'
        b.add(budget, 'budget_report', dict(cells=cells, profiles=profiles), [*cells, *profiles], 2)
        reports.append(budget)
        reports.append(b.selected(budget, dataset, 'GAT', 'gat_budget_confirmation', priority=2))
        reports.append(b.selected(decision, dataset, 'GAT', 'gat_confirmation', priority=2))
        decisions.append(decision)
    gate = 'decision/expand'
    b.add(gate, 'expand', dict(decisions=decisions,
          rule='GNNM best VALID exceeds independently tuned ENS on at least 3 of 5 datasets',
          common_grid_rule='reference plus two LR/WD configurations with best mean within-dataset within-method VALID rank',
          candidate_names=list(configs_for('lrwd'))), decisions, 2)
    when = dict(task=gate, path='output/done.json', field='expand', equals=True)
    for model in [name for name in MODELS if name != 'GAT']:
        for dataset in DATASETS:
            cells = []
            for slot, depth, split in product(range(3), range(1, 6), range(3)):
                cell, _ = b.cell(dataset, model, depth, split, REFERENCE, f'slot{slot}', 'expanded_pilot',
                                 5, [gate], when, dict(gate=gate, grid_slot=slot))
                cells.append(cell)
            decision = f'decision/expanded/{dataset}/{model}'
            b.add(decision, 'select', dict(cells=cells, dataset=dataset, model=model, pilot_splits=[0, 1, 2]),
                  [*cells, gate], 5, when=when)
            profiles = [task['id'] for task in b.tasks.values()
                        if task['metadata']['action'] == 'budget_profile'
                        and task['metadata']['dataset'] == dataset and task['metadata']['model'] == model]
            budget = f'budget/expanded/{dataset}/{model}'
            b.add(budget, 'budget_report', dict(cells=cells, profiles=profiles), [*cells, *profiles, gate], 5,
                  when=when)
            reports.append(budget)
            reports.append(b.selected(budget, dataset, model, 'expanded_budget_confirmation', when, gate, 5))
            reports.append(b.selected(decision, dataset, model, 'expanded_confirmation', when, gate, 5))
    return reports


def add_collab(b):
    """Stage 3: GCN and GraphSAGE on Collab with the OGB example recipe."""
    scores = []
    for pair in ('collab_gcn', 'collab_sage'):
        for cohort in range(5):
            method_scores = []
            for method in COLLAB_METHODS:
                fit = b.add(f'transfer/{pair}/cohort{cohort}/{method}', 'transfer_fit',
                            dict(pair=pair, module='p14_collab_ogb', cohort=cohort, method=method,
                                 protocol_id=PROTOCOL_ID), priority=3, kind='gpu', memory=40)
                score = b.add(f'transfer_method_score/{pair}/cohort{cohort}/{method}', 'transfer_score',
                              dict(fits=[fit], methods=[method], pair=pair, module='p14_collab_ogb',
                                   cohort=cohort), [fit], 3, 'gpu', memory=40)
                method_scores.append(dict(method=method, task=score, fit=fit))
            scores.append(b.add(f'transfer_score/{pair}/cohort{cohort}', 'transfer_score',
                                dict(method_scores=method_scores, methods=COLLAB_METHODS, pair=pair,
                                     module='p14_collab_ogb', cohort=cohort),
                                [], 4, 'cpu', memory=40, optional=[entry['task'] for entry in method_scores]))
    b.add('report/collab', 'transfer_report', dict(scores=scores), [], 4, optional=scores)
    return ['report/collab']


def add_roman_ablation(b):
    """Stages 4 and 6: every variant on 7 backbones x 5 depths x 10 splits."""
    scores = []
    for model, depth, split, variant in product(MODELS, range(1, 6), range(10), ROMAN_VARIANTS):
        priority = 4 if split < 5 else 6
        fit = b.fit(fixed_spec('roman-empire', model, depth, split, variant), 'roman_ablation', priority)
        scores.append(b.score(f'score/roman/{model}/L{depth}/split{split}/{variant}', [fit], variant,
                              'roman-empire', model, depth, split, 'roman_ablation', priority))
    return scores


def add_node_contrast(b):
    """Stage 9: GNNM + contrastive term at the paper's selected depths."""
    with CONTRAST_CELLS.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    scores = []
    for row in rows:
        dataset, model = row['dataset'], row['model']
        depth, split = int(row['num_layers']), int(row['split'])
        fit = b.fit(fixed_spec(dataset, model, depth, split, 'gnnm_contrast'), 'node_contrast', 9)
        scores.append(b.score(f'node_contrast_score/{dataset}/{model}/split{split}', [fit], 'gnnm_contrast',
                              dataset, model, depth, split, 'node_contrast', 9))
    return scores


def build_plan():
    b = FixedBuilder()
    add_parity(b)
    add_main_rerun(b)
    reports = add_grid(b)
    reports += add_collab(b)
    add_roman_ablation(b)
    add_node_contrast(b)
    b.add('report/final', 'final_report', dict(reports=reports), [], 100, optional=reports)
    plan = dict(schema=SCHEMA, campaign_id=CAMPAIGN_ID,
                protocol=dict(protocol_id=PROTOCOL_ID, node=node_protocol(), main_table=MAIN_TABLE,
                              pilot_splits=[0, 1, 2], confirmation_splits=list(range(3, 10)),
                              grid=configs_for('combined'),
                              freeze='Decisions use VALID only; all outcomes and failures retained'),
                tasks=list(b.tasks.values()))
    validate_plan(plan)
    for task in plan['tasks']:
        if task['metadata']['action'] == 'node_fit':
            from ablation.fixed_protocol.protocol import check_spec
            check_spec(task['metadata']['spec'])
    return plan


def summary(plan):
    fits = [task for task in plan['tasks'] if task['metadata']['action'] in ('node_fit', 'transfer_fit')]
    by_stage = Counter(task['metadata'].get('stage', task['metadata'].get('pair')) for task in fits)
    return dict(campaign_id=plan['campaign_id'], protocol_id=PROTOCOL_ID, tasks=len(plan['tasks']),
                fits=len(fits), fits_by_stage=dict(sorted(by_stage.items())),
                conditional_fits=sum(bool(task.get('when')) for task in fits))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if 'fixed_protocol' not in args.out.name:
        parser.error('The plan file name must contain fixed_protocol')
    plan = build_plan()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + '\n')
    counts = summary(plan)
    args.out.with_suffix('.counts.json').write_text(json.dumps(counts, indent=2) + '\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
