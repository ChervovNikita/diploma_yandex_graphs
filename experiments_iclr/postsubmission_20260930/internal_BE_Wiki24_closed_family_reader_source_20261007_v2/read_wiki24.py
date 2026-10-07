"""Scalar report from the separately charged export; no tensor/framework import."""
import argparse
import csv
import math
import os
from pathlib import Path
import resource
import statistics
import time
from gate import ARMS, SEEDS, binding, bound, fresh_output, preflight, read, require, write

CONTRASTS = [('primary', 'be_init_contrastive', 'be_init'),
             ('declared_secondary', 'be_init', 'be_unit'),
             ('declared_secondary', 'be_unit_contrastive', 'be_unit'),
             ('DESCRIPTIVE', 'be_init_contrastive', 'single'),
             ('DESCRIPTIVE', 'be_init_contrastive', 'independent4')]
CAVEATS = [
    'VALID is repeatedly used for strict first-best checkpoint selection over1100 epochs, including the100-epoch local restoration. Selected VALID accuracy is optimistic for generalization.',
    'Three paired optimizer seeds share official split0 and the same graph. Seed dispersion and df2 t intervals are exploratory; they do not quantify uncertainty over graphs, splits, or independent query populations.',
    'No TEST, confirmation-seed results, checkpoint reselection, iid-node bootstrap, multiplicity-adjusted inference, or quality/novelty conclusion is supplied.',
    'Ordinary independent4 serves an evaluation-only bank from four independently selected epochs; other arms use a coherent joint selected checkpoint. Member metrics have those same selection identities.',
    'For ordinary independent4, final CLOSED_OWN_BEST_BANK member metrics are table authority. Earlier own-checkpoint member metrics, exact deltas, epochs and modes are retained as diagnostics from different original evaluation events; discrepancies do not discard final-bank scores.',
    'be_init_contrastive minus single and minus ordinary independent4 are DESCRIPTIVE baseline comparisons, not new declared primary or secondary contrasts.',
    'Any missing required seed makes the full three-seed summary unavailable. Observed seed values remain explicit; no complete-case aggregate or imputed score is calculated.',
    'Pool minus mean member accuracy is a descriptive score difference, not evidence about shared error support or a causal diversity mechanism.'
]


def describe(seed_values, paired=False):
    observed = [x['value'] for x in seed_values if x['value'] is not None]
    result = dict(per_seed=seed_values, observed_count=len(observed), required_count=3,
                  complete_three_seed_summary=len(observed) == 3)
    if len(observed) != 3:
        result.update(mean=None, sample_SD=None, range=None, positive_seeds=None, negative_seeds=None,
                      zero_seeds=None, exploratory_t95_interval_df2=None,
                      unavailable_reason='At least one required seed cell is failed or not launched; observed values are not aggregated')
        return result
    mean, sd = statistics.mean(observed), statistics.stdev(observed)
    result.update(mean=mean, sample_SD=sd, range=[min(observed), max(observed)],
                  positive_seeds=sum(x > 0 for x in observed), negative_seeds=sum(x < 0 for x in observed),
                  zero_seeds=sum(x == 0 for x in observed))
    result['exploratory_t95_interval_df2'] = ([mean - 4.302652729911275 * sd / math.sqrt(3),
                                              mean + 4.302652729911275 * sd / math.sqrt(3)] if paired else None)
    return result


def summaries(records):
    keyed = {(r['arm'], r['seed']): r for r in records}; arms, contrasts = {}, {}
    for arm in ARMS:
        rows = [keyed[arm, seed] for seed in SEEDS]
        funcs = dict(selected_VALID=lambda r: r['selected_VALID'],
                     mean_member_VALID=lambda r: statistics.mean(r['member_VALID']),
                     minimum_member_VALID=lambda r: min(r['member_VALID']),
                     pool_minus_mean_member_VALID=lambda r: r['selected_VALID'] - statistics.mean(r['member_VALID']))
        arms[arm] = {name: describe([dict(seed=r['seed'], status=r['status'],
                      value=function(r) if r['status'] == 'complete' else None) for r in rows]) for name, function in funcs.items()}
        members = 1 if arm.startswith('single') else 4
        arms[arm]['member_VALID'] = [describe([dict(seed=r['seed'], status=r['status'],
                                  value=r['member_VALID'][m] if r['status'] == 'complete' else None) for r in rows]) for m in range(members)]
        arms[arm]['selected_epochs'] = [dict(seed=r['seed'], value=r['selected_epochs']) for r in rows]
        arms[arm]['selection'] = 'independently_selected_evaluation_only_bank' if arm == 'independent4' else 'joint_strict_first_VALID_maximum'
    for role, candidate, baseline in CONTRASTS:
        values = []
        for seed in SEEDS:
            a, b = keyed[candidate, seed], keyed[baseline, seed]
            value = a['selected_VALID'] - b['selected_VALID'] if a['status'] == b['status'] == 'complete' else None
            values.append(dict(seed=seed, candidate_status=a['status'], baseline_status=b['status'],
                               candidate_VALID=a['selected_VALID'], baseline_VALID=b['selected_VALID'], value=value,
                               percentage_point_delta=None if value is None else 100 * value))
        contrasts[candidate + ' minus ' + baseline] = dict(role=role, candidate=candidate, baseline=baseline,
                                                          declared_primary_or_secondary=role != 'DESCRIPTIVE',
                                                          selected_VALID_delta=describe(values, paired=True))
    return dict(arms=arms, paired_contrasts=contrasts)


def costs(gate, extraction_cost):
    history = gate['resource_history']; known = [h['metadata'].get('inclusive_seconds') for h in history]
    return dict(inclusive_family_driver_seconds=gate['inclusive_family_driver_seconds'],
                family_driver_scope='Includes current family waits, fresh resource attempts and fit supervisors; do not add these nested costs again',
                per_cell=[dict(cell=s['cell'], status=s['status'], inclusive_cell_driver_seconds=s['closure_row'].get('inclusive_cell_driver_seconds'),
                    scientific_supervisor=s['closure_row'].get('scientific_supervisor'),
                    new_resource_attempt=s['closure_row'].get('new_resource_attempt'),
                    fit_memory=s['closure_row'].get('fit_memory'), source_fit_seconds=s.get('freeze', {}).get('seconds'),
                    parameter_counts=s.get('freeze', {}).get('parameter_counts'),
                    reused_resource_evidence=s['closure_row'].get('resource_evidence') if 'new_resource_attempt' not in s['closure_row'] else None,
                    absent_cost_fields='Missing values are unavailable, not zero') for s in gate['cells']],
                preserved_resource_attempts=history,
                recorded_resource_attempt_seconds=sum(x for x in known if type(x) in (int, float) and math.isfinite(x) and x >= 0),
                resource_attempt_seconds_unavailable_count=sum(not (type(x) in (int, float) and math.isfinite(x) and x >= 0) for x in known),
                resource_cost_scope='Unique saved receipts across prior origins and this family; includes failed attempts when receipts exist. Nested fresh attempts and previously paid reuse are distinguished, not added to family elapsed.',
                resource_queue_supersession_metadata=gate['resource_queue_supersession_metadata'],
                missing_cost_limits='Resource waits not retained in cell/supersession metadata and earlier attempts without a saved receipt remain unknown. Scientific source FREEZE has no measured GPU/RSS peaks; no estimates are invented.',
                metadata_extraction=extraction_cost)


def main():
    started = time.monotonic(); cpu_started = time.process_time(); os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activation', type=Path, required=True); parser.add_argument('--activation-sha256', required=True)
    args = parser.parse_args()
    cfg, gate = preflight(args.activation, args.activation_sha256, 'read_selected_metadata')
    export_path = bound(cfg['metadata_export']); cost_path = bound(cfg['metadata_extraction_cost'])
    # First scalar metric access follows the fresh whole24 engineering gate.
    export = read(export_path); extraction_cost = read(cost_path)
    require(export.get('schema') == 'internal-be-Wiki24-selected-metadata-export-v1' and export.get('whole24_accounted') is True
            and export.get('TEST_access') is False and export.get('reselected') is False, 'Completed scalar metadata export')
    prior_gate = export['gate']
    for key in ('closure', 'family_release', 'owner', 'reader_manifest_sha256', 'source_manifest_sha256',
                'config_binding', 'data_manifest_binding', 'cells', 'resource_history'):
        require(prior_gate[key] == gate[key], 'Export exact unchanged closed-cell identity: ' + key)
    require(extraction_cost.get('status') == 'complete' and extraction_cost.get('metadata_export') == binding(export_path)
            and extraction_cost.get('closure') == gate['closure'] and extraction_cost.get('activation') == prior_gate['activation']
            and extraction_cost.get('completed_cells') == gate['complete_cells']
            and extraction_cost.get('model_constructed') is False and extraction_cost.get('model_state_loaded') is False
            and extraction_cost.get('forwards') == 0 and extraction_cost.get('dataset_payloads_opened') == 0,
            'Separate successful charged extraction')
    expected_attempts = [checkpoint for state in gate['cells'] if state['status'] == 'complete'
                         for checkpoint in [state['selected_checkpoint']] + state.get('own_checkpoints', [])]
    require(extraction_cost.get('checkpoint_deserialization_attempts') == expected_attempts
            and extraction_cost.get('checkpoint_file_bytes_submitted_to_deserializer') == sum(r['bytes'] for r in expected_attempts),
            'Exact separately charged source-selected checkpoint list')
    records = export['cells']
    require(len(records) == 24 and [(r['arm'], r['seed'], r['cell'], r['status']) for r in records]
            == [(s['arm'], s['seed'], s['cell'], s['status']) for s in gate['cells']], 'Every selected/failure row retained')
    output = fresh_output(cfg)
    report = dict(schema='internal-be-Wiki24-selected-VALID-report-v1', gate=gate, cells=records,
                  metric=export['metric'], summary=summaries(records), costs=costs(gate, extraction_cost),
                  caveats=CAVEATS, TEST_access=False)
    write(output / 'REPORT.json', report)
    with (output / 'CELLS.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['arm', 'seed', 'cell', 'status', 'selected_VALID', 'member_VALID', 'selected_epochs', 'selection', 'member_VALID_authority', 'own_selected_member_diagnostics'])
        writer.writeheader()
        for row in records: writer.writerow({k: row.get(k) for k in writer.fieldnames})
    lines = ['# WikiCS closed24 selected VALID readout', '',
             'Official split0; full5274 VALID accuracy. All8 arms and3 seeds are retained. Accuracy differences below are in percentage points.', '',
             '| Arm |6101|6203|6307| Mean | Sample SD | Range |', '|---|---:|---:|---:|---:|---:|---|']
    fmt = lambda x: 'unavailable' if x is None else f'{100*x:.4f}'
    for arm, summary in report['summary']['arms'].items():
        stat = summary['selected_VALID']; values = [fmt(r['value']) for r in stat['per_seed']]
        extent = 'unavailable' if stat['range'] is None else '–'.join(fmt(x) for x in stat['range'])
        lines.append('| ' + ' | '.join([arm] + values + [fmt(stat['mean']), fmt(stat['sample_SD']), extent]) + ' |')
    lines += ['', '## Paired contrasts (declared and DESCRIPTIVE)', '']
    for name, contrast in report['summary']['paired_contrasts'].items():
        stat = contrast['selected_VALID_delta']; interval = stat['exploratory_t95_interval_df2']
        lines.append(f"- {name} ({contrast['role']}): seeds " + ', '.join(str(r['seed']) + '=' + fmt(r['value']) for r in stat['per_seed'])
                     + '; mean ' + fmt(stat['mean']) + '; sample SD ' + fmt(stat['sample_SD'])
                     + '; signs +/−/0=' + '/'.join(str(stat.get(k)) for k in ('positive_seeds', 'negative_seeds', 'zero_seeds'))
                     + '; exploratory paired95% df2 interval ' + ('unavailable' if interval is None else '[' + ', '.join(fmt(x) for x in interval) + ']') + '.')
    lines += ['', '## Failures and costs', '',
              f"Complete cells: {gate['complete_cells']}/24. Failed or unlaunched: {gate['failed_or_not_launched_cells']}/24.", '',
              f"Actual family driver elapsed: {gate['inclusive_family_driver_seconds']:.3f}s. Separate metadata extraction: {extraction_cost['inclusive_wall_seconds']:.3f}s wall, {extraction_cost['process_CPU_seconds']:.3f}s process CPU.", '',
              'REPORT.json retains every failure reason, official selected member metric, selected epoch, original terminal custody, reused/fresh resource cost and unavailable cost field.', '',
              ]
    lines += ['- ' + s['cell'] + ': ' + s['status'] + '; error type '
              + str(s['closure_row'].get('error_type', 'not supplied by source; inspect retained terminal'))
              for s in gate['cells'] if s['status'] != 'complete']
    lines += ['', '## Interpretation limits', ''] + ['- ' + caveat for caveat in CAVEATS]
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    write(output / 'READER_COST.json', dict(inclusive_wall_seconds=time.monotonic() - started,
          process_CPU_seconds=time.process_time() - cpu_started, process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
          output_storage_bytes_before_cost_receipt=sum(p.stat().st_size for p in output.iterdir() if p.is_file()),
          metadata_extraction_charged_separately=True, model_constructed=False, forwards=0, TEST_access=False))
    print(str(output / 'REPORT.md'))


if __name__ == '__main__':
    main()
