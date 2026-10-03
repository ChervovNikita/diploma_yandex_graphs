#!/usr/bin/env python3
"""Summarize only pinned closed JSON scores; standard library, no model scoring."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SEEDS = [131, 137, 139, 149, 151]
HGT = ['global_BE', 'native_HGT', 'untied_HGT']
NATIVE = ['native_GAT', 'native_Simple_HGN', 'native_SeHGNN']
ARMS = HGT + NATIVE
METRICS = ['NLL', 'micro_F1', 'macro_F1']
T95_DF4 = 2.7764451051977987


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fingerprint(path):
    data = path.read_bytes()
    return dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def matches(actual, descriptor):
    return all(actual[key] == descriptor[key] for key in ('sha256', 'bytes'))


def read_inputs():
    binding = json.loads((HERE/'INPUTS.json').read_text())
    require(binding['schema'] == 'closed_DBLP_native_HGT_summary_inputs_v1', 'Input binding schema differs')
    values = {}
    # Only enumerated LOCAL metadata/source files are opened. Remote archive,
    # label, split, checkpoint and logits descriptors are never dereferenced.
    for role, record in binding['inputs'].items():
        relative = Path(record['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Confined phase-relative input required')
        path = PHASE/relative
        require(path.resolve().is_relative_to(PHASE), 'Input escapes project phase')
        require(matches(fingerprint(path), record), 'Changed pinned input: '+role)
        if role in ('native_study', 'native_audit', 'hgt_decision', 'native_freeze', 'hgt_freeze', 'hgt_execution_binding', 'hgt_provenance'):
            values[role] = json.loads(path.read_text())
    for record in binding['prior_reviews']:
        path = PHASE/record['path']
        require(matches(fingerprint(path), record), 'Prior review changed: '+record['path'])
    require(not any(name.split('.')[0] in ('torch', 'numpy', 'scipy', 'dgl', 'torch_geometric') for name in sys.modules), 'Unexpected numerical runtime')
    return binding, values


def uncertainty(values):
    require(len(values) == 5 and all(math.isfinite(v) for v in values), 'Five finite seed values required')
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    se = sd/math.sqrt(5)
    return dict(values=values, mean=mean, SD=sd, SE=se,
                illustrative_t95=[mean-T95_DF4*se, mean+T95_DF4*se])


def assemble():
    binding, data = read_inputs()
    ns, na, hd, nf, hf = [data[key] for key in ('native_study', 'native_audit', 'hgt_decision', 'native_freeze', 'hgt_freeze')]
    ha = hd['root_audit']
    require(na['schema'] == 'complete15_native_development_audit_v1' and na['status'] == 'complete', 'Closed native audit required')
    for key in ('root_observed', 'all15_closed', 'training_child_reaped', 'all15_checkpoint_replays_audited', 'originals_preserved', 'heldout_labels_closed'):
        require(na[key] is True, 'Native audit closure differs: '+key)
    require(na['TEST_label_reads'] == na['TEST_diagnostics'] == 0 and na['successful_subset_scored'] is False, 'Closed native label boundary required')
    require(matches(binding['inputs']['native_study'], na['study']), 'Copied original native study differs from audit')
    require(matches(binding['inputs']['native_freeze'], na['freeze']), 'Native freeze differs from audit')
    require(matches(binding['inputs']['hgt_freeze'], ha['study_freeze']), 'HGT freeze differs from closed35 evidence')
    require(hd['heldout_labels_closed'] is True and hd['original_scores_unchanged'] is True
            and ha['status'] == 'audited_complete_development_family' and ha['heldout_opened'] is False
            and ha['exact_selected_trace_audits'] == 35, 'Existing root-audited closed35 evidence required')
    require(hf['seeds'] == nf['seeds'] == SEEDS and nf['arms'] == NATIVE, 'Five frozen paired seeds required')
    require(nf['test_labels_closed'] is True and hf['test_labels_closed'] is True, 'Closed heldout freezes required')
    shared_keys = ['archive', 'development_labels', 'splits', 'source_label_member_sha256', 'members', 'member_sha256', 'expected_node_counts']
    require(all(nf[key] == hf[key] for key in shared_keys), 'Native/HGT shared input binding differs')
    require(nf['paired_HGT_freeze'] == ha['study_freeze'], 'Native paired HGT freeze differs from closed35 evidence')
    require(all(record in na['source_inputs'] for record in [nf['archive'], nf['development_labels']]+[row['descriptor'] for row in nf['splits']]), 'Shared inputs absent from native audit custody')
    hb = data['hgt_execution_binding']
    require(hb['freeze']['sha256'] == ha['study_freeze']['sha256'] and hb['arms'] == hf['arms'] and hb['seeds'] == SEEDS, 'Original HGT execution binding differs')
    require(matches(binding['inputs']['hgt_driver'], hb['driver_source']), 'HGT diagnostic driver differs from original execution source')
    schema = data['hgt_provenance']['actual_schema']
    require((schema['development_pool'], schema['train_per_seed'], schema['validation_per_seed'], schema['target_nodes']) == (1217, 974, 243, 4057), 'Fixed development schema differs')
    require(ns['summary']['status'] == 'complete_development_challengers' and ns['original_inputs_unchanged'] is True
            and ns['preservation_error'] is None and ns['summary']['all_frozen_terminals'] is True
            and ns['summary']['successful_subset_scored'] is False, 'Complete original15 study required')
    expected_native = [(seed, arm) for seed in SEEDS for arm in NATIVE]
    require([(r['seed'], r['arm']) for r in ns['rows']] == expected_native
            and [(r['seed'], r['arm']) for r in na['cases']] == expected_native, 'All15 exact ordered native cases required')
    require(all(r['status'] == 'selected' and r['selected_state_replay'] is True and r['TEST_label_reads'] == r['TEST_diagnostics'] == 0 for r in ns['rows']), 'Unclosed native terminal')
    require(all(r['status'] == 'audited' and r['original_selected_scores_verified'] is True and r['independent_selected_inference_replay'] is True for r in na['cases']), 'Unaudited native case')
    require(ns['admission'] == dict(TEST_labels_closed=True, prepared_manifest_sha256=na['source_manifest']['sha256'],
            root_admission_sha256=na['release']['sha256'], study_freeze_sha256=na['freeze']['sha256']), 'Native original admission differs')
    require(set(ha['scores']) == set(hf['arms']) and sum(len(r) for r in ha['scores'].values()) == 35, 'Complete35 source scores required')
    for arm, rows in ha['scores'].items():
        require([r['seed'] for r in rows] == SEEDS and all(math.isfinite(r[m]) for r in rows for m in METRICS), 'Malformed complete35 score records: '+arm)
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            if arm in HGT:
                source = next(r for r in ha['scores'][arm] if r['seed'] == seed)
                row = dict(arm=arm, seed=seed, selected_epoch=source['epoch'], **{m: source[m] for m in METRICS})
            else:
                source = next(r for r in ns['rows'] if r['seed'] == seed and r['arm'] == arm)
                selection = source['selection']
                require(selection['heldout'] is False and selection['calibrated'] is False, 'Native raw development scores required')
                row = dict(arm=arm, seed=seed, selected_epoch=selection['epoch'], epochs_paid=source['epochs_paid'],
                    parameters=source['parameters'], **{m: selection['validation_'+m] for m in METRICS})
            require(all(math.isfinite(row[m]) and (row[m] >= 0 if m == 'NLL' else 0 <= row[m] <= 1) for m in METRICS), 'Invalid recorded metric')
            rows.append(row)
    index = {(r['arm'], r['seed']): r for r in rows}
    summaries = {arm: {metric: uncertainty([index[arm, seed][metric] for seed in SEEDS]) for metric in METRICS} for arm in ARMS}
    pairs = []
    for hgt in HGT:
        for native in NATIVE:
            metrics = {m: uncertainty([index[hgt, s][m]-index[native, s][m] for s in SEEDS]) for m in METRICS}
            pairs.append(dict(hgt_arm=hgt, native_arm=native, direction='HGT minus native recipe', metrics=metrics))
    max_margin = math.sqrt(4/3)
    simple_bound = dict(classes=4, L2_norm_at_most=1, maximum_softmax_class_probability=1/(1+3*math.exp(-max_margin)),
                        minimum_raw_per_row_NLL=math.log1p(3*math.exp(-max_margin)),
                        source='native_models.py:128-129', exact_arithmetic_bound=True,
                        derivation='For target a and mean other b: a^2+3b^2<=1 gives a-b<=sqrt(4/3). Convexity gives sum_other exp(z_j-a)>=3 exp(-sqrt(4/3)).')
    return dict(schema='closed_DBLP_native_and_HGT_descriptive_development_comparison_v1',
        status='complete_recorded_score_summary', dataset='HGB-DBLP', seeds=SEEDS, arms=ARMS,
        metrics=METRICS, per_seed=rows, per_arm_summary=summaries, paired_HGT_minus_native=pairs,
        input_hashes=binding['inputs'], input_binding=fingerprint(HERE/'INPUTS.json'),
        shared_input_binding=dict(exact_equality_keys=shared_keys, all_equal=True, paired_HGT_freeze=nf['paired_HGT_freeze'],
            archive=nf['archive'], development_labels=nf['development_labels'], splits=nf['splits'],
            development_pool=1217, TRAIN_per_split=974, VAL_per_split=243, target_nodes=4057,
            shared_descriptors_present_in_native_audit=True, underlying_archive_labels_splits_opened=False),
        original_HGT_family_terminal=ha['family_terminal'], original_native_study=na['study'],
        native_inference_replay_provenance='Verified existing root audit metadata; no inference performed by summary tool',
        simple_HGN_normalization_bound=simple_bound, t95_multiplier_df4=T95_DF4,
        uncertainty_scope='Illustrative t95 summaries of five paired overlapping splits of one graph; independence/normality are not established; no confirmatory confidence or significance interpretation.',
        selection_scope='All displayed scores are from checkpoints selected using the same243 VAL nodes subsequently summarized. F1 is measured at the NLL-selected checkpoint. Different architectures, features, propagated labels, capacities, optimizer schedules and selectors remain as recorded.',
        prior_review_hashes=binding['prior_reviews'], all_prior_reviews_preserved=True,
        model_scores_recalculated=False, tensors_or_labels_deserialized=False, numerical_libraries_imported=False,
        server_access=False, current_mixed40_or_initializer_or_BUDDY_outcomes_read=False,
        calibration_or_tuning_performed=False, new_significance_gate=False, superiority_claim=False, manuscript_acceptance_review=False)


def report(value):
    out = ['# Closed DBLP native and HGT development comparison', '',
        'Descriptive comparison of recorded, checkpoint-selected validation scores. All 35 HGT and all 15 native fits were closed in the supplied root evidence; the existing native15 audit reports complete selected-state restoration/replay and original custody. This tool does not run a model or recalculate a model score.', '',
        '## Shared input custody', '',
        'The frozen studies have exactly equal archive, development-label and all five split descriptors, including paths, SHA256 and byte counts. Member names/hashes, source label-member hash and node counts also agree. The native paired-HGT freeze equals the closed35 audit freeze, and all shared input descriptors occur in native audit custody. Only those descriptors were inspected; archive, label and split payloads remained closed.', '',
        'Five seeds (131, 137, 139, 149, 151) reuse one DBLP graph and its 1217-node development pool: 974 TRAIN and 243 VAL per split. The splits overlap; this is not five independent datasets.', '',
        '## Means of selected raw scores', '',
        'NLL is in nats per VAL node; F1 values are fractions. Means average the five recorded split scores.', '',
        '| Recipe | Raw NLL | Micro-F1 | Macro-F1 |', '|---|---:|---:|---:|']
    for arm in ARMS:
        scores = value['per_arm_summary'][arm]
        out.append('| '+arm+' | '+' | '.join(f"{scores[m]['mean']:.6f}" for m in METRICS)+' |')
    out += ['', '## Full per-seed recorded scores', '',
        'Display values are rounded to nine decimal places; SUMMARY.json preserves every parsed source value. Epoch is the selected post-update checkpoint, not the paid training budget.', '',
        '| Recipe | Seed | Selected epoch | Raw NLL | Micro-F1 | Macro-F1 |', '|---|---:|---:|---:|---:|---:|']
    for r in value['per_seed']:
        out.append(f"| {r['arm']} | {r['seed']} | {r['selected_epoch']} | "+' | '.join(f'{r[m]:.9f}' for m in METRICS)+' |')
    out += ['', '## Paired differences and illustrative t95 intervals', '',
        'Each cell is mean HGT minus native recipe, followed by [illustrative t95 lower, upper]. Negative NLL means smaller recorded HGT raw NLL; positive F1 means larger recorded HGT F1. All 27 metric contrasts are descriptive. No interval is used as a significance or continuation gate.', '',
        'For each metric, differences are matched by seed; sample SD uses denominator 4, SE=SD/sqrt(5), and the interval is mean +/-2.7764451051977987*SE (t quantile 0.975, df=4). These intervals assume independent approximately normal differences, which overlapping splits do not establish. They cannot quantify graph-population generalization or remove checkpoint-selection optimism.', '',
        '| HGT recipe | Native recipe | Raw NLL difference | Micro-F1 difference | Macro-F1 difference |', '|---|---|---:|---:|---:|']
    for pair in value['paired_HGT_minus_native']:
        cells = []
        for m in METRICS:
            s = pair['metrics'][m]; lo, hi = s['illustrative_t95']
            cells.append(f"{s['mean']:+.6f} [{lo:+.6f}, {hi:+.6f}]")
        out.append('| '+pair['hgt_arm']+' | '+pair['native_arm']+' | '+' | '.join(cells)+' |')
    bound = value['simple_HGN_normalization_bound']
    out += ['', '## Recipe and output differences', '',
        '| Recipe | Inputs and prediction | Optimizer and selection |', '|---|---|---|',
        '| global_BE | HGT feature2: provided author attributes, identity for other types; four members share a core with private BE factors; raw member logits are averaged | AdamW defaults/decay 1e-4 + OneCycle 300/max_lr 1e-3; TRAIN mean member CE, select mean-logit VAL CE; latest ties, patience 30/max 300 |',
        '| native_HGT | Same HGT features/backbone; one factor-off member with an unconstrained final linear classifier | Same AdamW/OneCycle; latest ties, patience 30/max 300 |',
        '| untied_HGT | Same HGT features; four separate cores, raw logits averaged; joint mean member CE with one selected ensemble checkpoint | Same AdamW/OneCycle; latest ties, patience 30/max 300; not four separately selected native fits |',
        '| native_GAT | Identity inputs for every type, including authors; homogeneous support plus self loops; no final per-node L2 normalization | Adam lr 5e-4/decay 1e-4; latest ties, patience 30/max 300 |',
        '| native_Simple_HGN | Same identity/homogeneous inputs; typed edges, feature/attention residuals; final four-class vector divided by its L2 norm (clamp 1e-12) | Adam lr 5e-4/decay 1e-4; latest ties, patience 30/max 300 |',
        '| native_SeHGNN | All provided A/P/T attributes, venue identity, normalized metapath feature products; TRAIN-only one-hot labels propagated with diagonal removed after complete products; final class BatchNorm with affine=False/track_running_stats=False uses all 4057 authors | Adam lr 1e-3/decay 0; earliest tied minimum via strict improvement; 51 nonimproving epochs/max 200 |', '',
        'Source anchors: `native_models.py:128-129` (Simple-HGN normalization), `native_models.py:200-201` (SeHGNN BatchNorm), `native_inputs.py:84-173` (metapaths, TRAIN-only labels and batch membership), `train_native.py:45-57,124-165` (selectors and original predictor), `train_dblp.py:97-135` (HGT logits, objective and selector), `families.py:22-67` (one/shared/untied members), and `hgt_private.py:144-163` (final HGT classifier). Their exact source hashes appear below.', '',
        f"**Simple-HGN's raw probability scale is bounded by its output rule.** In exact arithmetic, for four logits with L2 norm at most 1, any softmax class probability is at most {bound['maximum_softmax_class_probability']:.9f}, so each raw NLL is at least {bound['minimum_raw_per_row_NLL']:.9f} nats (up to floating-point rounding). Write the target logit as a and the mean of the other three as b. The norm gives a^2+3b^2<=1, hence a-b<=sqrt(4/3); convexity gives sum(exp(other-a))>=3exp(-sqrt(4/3)). This is an analytic source invariant, not a calculation from predictions. Positive per-row scaling preserves argmax while limiting confidence. Its large raw NLL gap therefore cannot support a pure classification-quality inferiority claim.", '',
        'SeHGNN also serves logits on a different scale: final class BatchNorm standardizes using the whole 4057-author evaluation batch, with no learned affine scale. The exact TRAIN/VAL/topology-complement composition was retained; no complement labels are inputs. Its label propagation, supplied attributes, model size (10,842,378 parameters versus 1,971,464 GAT and 2,308,872 Simple-HGN), optimizer and budget all differ. HGT hidden LayerNorm does not impose Simple-HGN sample-unit-norm or SeHGNN final-class-BatchNorm constraints on its final classifier.', '',
        'Raw NLL evaluates the probabilities served by each recorded recipe and mixes calibration/scale with discrimination. F1 describes argmax classification at the NLL-selected checkpoint, yet remains affected by feature, architecture and selection differences. No temperature calibration, shared feature ablation, optimizer matching, reselection or causal isolation is performed here.', '',
        '## Conditional reading', '',
        'The recorded means show untied_HGT with the smallest raw NLL among these six recipes; the three native recipes have larger mean micro/macro-F1 than the three HGT recipes in this development comparison. Those observations summarize this fixed graph, these overlapping splits and their selected checkpoints. The paired tables retain the per-seed variation and do not establish broad superiority. In particular, Simple-HGN can have high F1 and high raw NLL because its normalization caps confidence.', '',
        'The 243 VAL nodes both select the checkpoint and provide the reported scores, so this is development reuse rather than an independent final test. Macro-F1 uses the fixed four-class schema; micro-F1 is single-label accuracy. Five seeds cannot identify dataset variation or provide strong uncertainty calibration. All nine recipe pairs and three metrics are displayed without selecting a favorable subset; no multiplicity-adjusted testing or new gate is introduced. There is no heldout or ACM confirmation here, and no manuscript acceptance verdict.', '',
        '## Reproducibility and exact inputs', '',
        'Run `python3 summarize.py` from this directory to create fresh SUMMARY.json and REPORT.md; existing outputs are protected. Run `python3 summarize.py --check` to regenerate in memory and verify both existing outputs. INPUTS.json pins only the authorized closed score JSON, frozen/admission metadata and inspected source text. The script imports only the Python standard library and never opens data/labels/tensors or follows remote descriptors.', '',
        '| Input role | Phase-relative path | SHA256 | Bytes |', '|---|---|---|---:|']
    for role, record in value['input_hashes'].items():
        out.append(f"| {role} | {record['path']} | {record['sha256']} | {record['bytes']} |")
    out += ['', f"All {len(value['prior_review_hashes'])} earlier review files retained their exact pinned hashes. Only the new comparison directory was written.", '']
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Regenerate in memory and verify existing outputs')
    args = parser.parse_args()
    value = assemble()
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    text = report(value)
    if args.check:
        require((HERE/'SUMMARY.json').read_text() == serialized and (HERE/'REPORT.md').read_text() == text, 'Saved summary/report differ from pinned inputs/tool')
    else:
        require(not (HERE/'SUMMARY.json').exists() and not (HERE/'REPORT.md').exists(), 'Fresh outputs required; use --check for reproduction')
        with (HERE/'SUMMARY.json').open('x') as stream:
            stream.write(serialized)
        with (HERE/'REPORT.md').open('x') as stream:
            stream.write(text)
    print(json.dumps(dict(status='reproduction_verified' if args.check else 'summary_written', rows=len(value['per_seed']), paired_recipe_comparisons=len(value['paired_HGT_minus_native']), model_scores_recalculated=False)))


if __name__ == '__main__':
    main()
