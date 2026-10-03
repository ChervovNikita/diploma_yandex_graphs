"""Audit the closed, unchanged DBLP family before applying its frozen gate.

Run on the authorized server repository. An unfinished family exposes no scores
and writes no decision. This does not open labels, states, or tensor payloads.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
RUN = PHASE / 'graph_heterogeneous_dblp_parallel_cpu_preparation_20261003_v2/runs/root_parallel_cpu_run02'
FREEZE = PHASE / 'graph_heterogeneous_dblp_execution_root_v1/FROZEN_STUDY.json'
FREEZE_SHA = '29885a100527226e9d182c54567e748f17f384254d23e009d9089fb18352d352'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def verify(record):
    path = Path(record['path'])
    require(path.is_absolute() and path.resolve() == path and path.is_relative_to(REPO),
            'Evidence outside the authorized repository')
    value = fingerprint(path)
    require(value['sha256'] == record['sha256']
            and ('bytes' not in record or value['bytes'] == record['bytes']),
            'Changed evidence: ' + str(path))
    return path


def uncertainty(values):
    require(len(values) == 5 and all(math.isfinite(x) for x in values), 'Five finite paired values required')
    mean = sum(values) / 5
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / 4)
    se = sd / math.sqrt(5)
    return dict(values=values, mean=mean, SD=sd, SE=se,
                illustrative_t95=[mean - 2.7764451051977987 * se, mean + 2.7764451051977987 * se],
                leave_one_block_out_means=[(sum(values) - x) / 4 for x in values],
                interpretation='Descriptive overlapping graph splits; not independent graphs, power, or significance')


def audit():
    require(Path(subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout.strip()) == REPO,
            'Wrong Git repository')
    require(subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                           capture_output=True, text=True, check=True).stdout.splitlines() == [UUID],
            'Wrong allocation')
    require(fingerprint(FREEZE)['sha256'] == FREEZE_SHA, 'Scientific design changed')
    terminal = RUN / 'PARALLEL_STUDY.json'
    if not terminal.is_file():
        return dict(status='waiting_for_complete35', scores_disclosed=False, decision_written=False)
    frozen = json.loads(FREEZE.read_text())
    result = json.loads(terminal.read_text())
    if result.get('status') != 'complete_development_summary':
        return dict(status='closed_family_not_valid_for_comparison', terminal_status=result.get('status'),
                    gate_passed=False, scores_disclosed=False, successful_subset_scored=False)
    require(result.get('originals_preserved') is True
            and result.get('worker_preservation_and_completion_passed') is True
            and result.get('final_labels_closed') is True, 'Family preservation/completion failed')
    closure = result['closure']
    require(closure['all35_terminals'] is True and closure['all35_selected_fits'] is True
            and closure['successful_subset_scored'] is False, 'Complete valid family required')
    seeds, arms = frozen['seeds'], frozen['arms']
    rows = result['rows']
    require(len(rows) == 35 and len({(r['seed'], r['arm']) for r in rows}) == 35
            and {(r['seed'], r['arm']) for r in rows} == {(s, a) for s in seeds for a in arms},
            'Duplicate, missing, or extra model/split terminal')
    bindings_path = PHASE / 'graph_heterogeneous_dblp_parallel_cpu_preparation_20261003_v2/BINDINGS.json'
    bindings = json.loads(bindings_path.read_text())
    require(bindings['freeze']['sha256'] == FREEZE_SHA, 'Execution bound to another design')
    for record in bindings['source_records'] + [frozen['archive'], frozen['development_labels']] + [r['descriptor'] for r in frozen['splits']]:
        verify(record)
    scores = {arm: [] for arm in arms}
    for seed in seeds:
        for arm in arms:
            row = next(r for r in rows if (r['seed'], r['arm']) == (seed, arm))
            require(row['status'] == 'selected' and row['selected_state_replay'] is True
                    and row['checkpoint_bindings_verified'] is True
                    and row['checkpoint_binding_sha256'] == bindings['study_checkpoint_binding_sha256'],
                    'Unverified selected-state replay/custody')
            case = RUN / f'seed{seed}' / arm
            for key, filename in [('selected_checkpoint', 'selected.pt'), ('selected_logits', 'selected_member_logits.pt'),
                                  ('selection_receipt', 'SELECTION.json'), ('training_trace', 'TRACE.jsonl')]:
                require(Path(row[key]['path']) == case / filename, 'Artifact outside exact output slot')
                verify(row[key])
            selection = json.loads((case / 'SELECTION.json').read_text())
            require(all(row.get(k) == v for k, v in selection.items()), 'Selection receipt and terminal disagree')
            trace = [json.loads(line) for line in (case / 'TRACE.jsonl').read_text().splitlines()]
            require(0 < len(trace) <= 300 and [r['epoch'] for r in trace] == list(range(1, len(trace) + 1)),
                    'Noncontiguous or ineligible checkpoint trace')
            best, counter, chosen = None, 0, None
            for event in trace:
                value = event['validation_NLL']
                require(math.isfinite(value), 'Nonfinite validation trace')
                replace = best is None or value <= best
                if replace:
                    best, counter, chosen = value, 0, event
                else:
                    counter += 1
                require(event['checkpoint_replaced'] == replace and event['patience_counter'] == counter,
                        'Native latest-tie/patience semantics differ')
            require(selection['updates'] == len(trace) and selection['selection']['epoch'] == chosen['epoch'],
                    'Selected checkpoint differs from complete native trace')
            require(len(trace) == 300 or counter == 30, 'Fit ended outside native stopping rule')
            for name in ['validation_NLL', 'validation_micro_F1', 'validation_macro_F1']:
                require(selection['selection'][name] == chosen[name] and math.isfinite(chosen[name]),
                        'Selected metric differs from selected trace')
            scores[arm].append(dict(seed=seed, epoch=chosen['epoch'],
                NLL=chosen['validation_NLL'], micro_F1=chosen['validation_micro_F1'], macro_F1=chosen['validation_macro_F1']))
    contrasts = {}
    gate = frozen['continuation_gate']
    for arm in arms:
        if arm == 'CP':
            continue
        nll = [a['NLL'] - b['NLL'] for a, b in zip(scores['CP'], scores[arm])]
        f1 = [a['macro_F1'] - b['macro_F1'] for a, b in zip(scores['CP'], scores[arm])]
        contrasts[arm] = dict(NLL=uncertainty(nll), macro_F1=uncertainty(f1), strict_NLL_wins=sum(x < 0 for x in nll))
    decisions = {arm: contrasts[arm]['NLL']['mean'] <= gate['per_control_mean_NLL_difference_at_most']
                 and contrasts[arm]['strict_NLL_wins'] >= gate['per_control_strict_paired_NLL_wins_at_least']
                 and contrasts[arm]['macro_F1']['mean'] >= gate['per_control_mean_macro_F1_difference_at_least']
                 for arm in gate['primary_controls']}
    return dict(status='audited_complete_development_family', UTC=datetime.now(timezone.utc).isoformat(),
                gate_passed=all(decisions.values()), gate_by_primary_control=decisions,
                frozen_gate=gate, study_freeze=fingerprint(FREEZE), family_terminal=fingerprint(terminal),
                exact_selected_trace_audits=35, scores=scores, contrasts=contrasts,
                heldout_opened=False, original_paper_scores_changed=False, baseline_artifacts_changed=False,
                interpretation='Development continuation decision only; no methodological novelty, heldout utility, or acceptance claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='New result file under this root study directory; optional observation only')
    args = parser.parse_args()
    result = audit()
    if args.output and result['status'] == 'audited_complete_development_family':
        output = Path(args.output)
        require(output.is_absolute() and output.resolve().is_relative_to(FREEZE.parent)
                and not output.exists(), 'New confined decision output required')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
