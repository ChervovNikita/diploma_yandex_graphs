"""Complete-nine metadata comparison only, after separate root closure release."""
import argparse
import json
import math
from pathlib import Path
import socket
from source import HERE, PHASE, bind, inside, sha, verify_manifest, write
from stage_plan import CONDITIONS, SEEDS, SERVER_HOST, SERVER_PHASE, SELECTOR, roster


def summarize(records):
    expected = {(seed,name) for seed in SEEDS for name in CONDITIONS}
    if set(records) != expected: raise ValueError('No partial-family comparison: exactly all nine records required')
    for record in records.values():
        if record.get('complete') is not True: raise ValueError('A failed/incomplete record prevents comparison')
        validate_metrics(record['selected_VALID'])
    mean = lambda values: sum(values)/len(values)
    pooled = lambda record: record['selected_VALID']['pooled']
    by_condition = {}
    for name in CONDITIONS:
        values = [records[seed,name]['selected_VALID'] for seed in SEEDS]
        by_condition[name] = dict(accuracy=mean([v['pooled']['accuracy'] for v in values]),
            NLL=mean([v['pooled']['NLL'] for v in values]),
            mean_member_accuracy=mean([v['mean_member_accuracy'] for v in values]),
            worst_member_accuracy=mean([v['worst_member_accuracy'] for v in values]),
            macro_accuracy=mean([v['macro_accuracy'] for v in values]))
    paired = []
    for seed in SEEDS:
        core, own, independent = [records[seed,name]['selected_VALID'] for name in ('shared4_core','shared4_own','independent4_native')]
        row = dict(seed=seed, core_minus_own_accuracy_pp=100*(core['pooled']['accuracy']-own['pooled']['accuracy']),
            independent_minus_own_accuracy_pp=100*(independent['pooled']['accuracy']-own['pooled']['accuracy']),
            core_minus_independent_accuracy_pp=100*(core['pooled']['accuracy']-independent['pooled']['accuracy']),
            core_minus_own_NLL=core['pooled']['NLL']-own['pooled']['NLL'],
            core_minus_own_mean_member_pp=100*(core['mean_member_accuracy']-own['mean_member_accuracy']),
            core_minus_own_worst_member_pp=100*(core['worst_member_accuracy']-own['worst_member_accuracy']),
            core_minus_own_macro_pp=100*(core['macro_accuracy']-own['macro_accuracy']),
            selected_epochs={name:records[seed,name]['selected_epoch'] for name in CONDITIONS})
        row['classes'] = [dict(class_id=c, **{
            label:dict(accuracy_pp=100*(core['classes'][c]['accuracy']-other['classes'][c]['accuracy']),
                       NLL=core['classes'][c]['NLL']-other['classes'][c]['NLL'])
            for label,other in (('core_minus_own', own), ('core_minus_independent', independent))}) for c in range(3)]
        paired.append(row)
    gain = mean([r['core_minus_own_accuracy_pp'] for r in paired])
    gap = mean([r['independent_minus_own_accuracy_pp'] for r in paired])
    gates = dict(mean_accuracy_gain_at_least_0_20pp=gain >= .2,
        accuracy_gain_positive_all_three_seeds=all(r['core_minus_own_accuracy_pp'] > 0 for r in paired),
        quarter_positive_independent_gap_or_at_least_independent_mean=(gain >= .25*gap if gap > 0 else by_condition['shared4_core']['accuracy'] >= by_condition['independent4_native']['accuracy']),
        mean_pooled_NLL_no_worse_than_own=by_condition['shared4_core']['NLL'] <= by_condition['shared4_own']['NLL'],
        mean_member_degradation_no_more_than_0_10pp=mean([r['core_minus_own_mean_member_pp'] for r in paired]) >= -.1,
        worst_member_degradation_no_more_than_0_20pp=mean([r['core_minus_own_worst_member_pp'] for r in paired]) >= -.2,
        mean_macro_accuracy_no_worse_than_own=by_condition['shared4_core']['macro_accuracy'] >= by_condition['shared4_own']['macro_accuracy'])
    return dict(schema='masked-context-PubMed-stage1-complete-nine-comparison-v1', complete_nine=True,
        condition_means=by_condition, paired_seeds=paired, gates=gates,
        continuation_eligible=all(gates.values()), further18_activated=False,
        mean_core_minus_own_accuracy_pp=gain, mean_independent_minus_own_accuracy_pp=gap,
        claim_limit='One previously encountered graph, one split, three optimizer seeds, primary common independent-bank selector. Passing admits consideration of the other18 under separate root authorization. It does not establish superiority, capable-single adequacy, causal sharing, novelty or unused confirmation.',
        required_stronger_references='Capable native single and individually selected ordinary independent ensemble, plus unused splits/tasks and uncertainty assessment.')


def validate_metrics(value):
    def checked(row, count):
        if row.get('count') != count or type(row.get('correct')) is not int or not 0 <= row['correct'] <= count:
            raise ValueError('Complete fixed-class correct counts required')
        if row.get('accuracy') != row['correct']/count:
            raise ValueError('Accuracy must equal exact correct count/population')
        if not isinstance(row.get('NLL'), (int,float)) or not math.isfinite(row['NLL']) or row['NLL'] < 0:
            raise ValueError('Finite nonnegative NLL required')
    checked(value['pooled'],3942)
    if len(value['members']) != 4 or len(value['classes']) != 3: raise ValueError('Complete member/class readouts required')
    for member in value['members']: checked(member,3942)
    for c,count in enumerate((820,1547,1575)):
        row=value['classes'][c]
        if row['class_id'] != c or len(row['members']) != 4: raise ValueError('Fixed class order and all member values required')
        checked(row,count)
        for member in row['members']: checked(member,count)
    if sum(row['correct'] for row in value['classes']) != value['pooled']['correct']:
        raise ValueError('Pooled correct count disagrees with classes')
    for m in range(4):
        if sum(row['members'][m]['correct'] for row in value['classes']) != value['members'][m]['correct']:
            raise ValueError('Member correct count disagrees with classes')
    if value['mean_member_accuracy'] != sum(row['accuracy'] for row in value['members'])/4 or value['worst_member_accuracy'] != min(row['accuracy'] for row in value['members']):
        raise ValueError('Member mean/worst readouts disagree')
    if value['macro_accuracy'] != sum(row['accuracy'] for row in value['classes'])/3:
        raise ValueError('Macro accuracy disagrees with class readouts')


def admit_comparison(path, digest):
    path = inside(path)
    if sha(path) != digest: raise ValueError('Exact separate comparison release digest required')
    spec = json.loads(path.read_text())
    if spec.get('schema') != 'masked-context-stage1-complete-nine-comparison-release-v1': raise ValueError('Complete-nine comparison release required')
    for key in ('enabled', 'root_comparison_authorized', 'whole_nine_owned_completion_verified'):
        if spec.get(key) is not True: raise ValueError('Comparison remains disabled: '+key)
    if spec.get('TEST_access') is not False or spec.get('automatic_retry') is not False: raise ValueError('No TEST or retries')
    verify_manifest(spec['source_manifest_sha256'])
    expected = {row['record_id']:(row['seed'],row['condition']) for row in roster()}
    if set(spec.get('records', {})) != set(expected): raise ValueError('Exactly the whole nine roster is required')
    if socket.gethostname() != SERVER_HOST or str(PHASE) != SERVER_PHASE: raise ValueError('Server-only comparison custody')
    terminals = {}
    # Validate every terminal before opening any completion report with metrics.
    for identity, rows in spec['records'].items():
        terminal = json.loads(bind(rows['terminal_custody']).read_text())
        if terminal.get('schema') != 'masked-context-stage1-owned-terminal-custody-v1' or terminal.get('record_id') != identity:
            raise ValueError('Exact owned terminal identity required')
        if terminal.get('directly_waited') is not True or terminal.get('child_exit_code') != 0 or terminal.get('cap_or_owner_failure') is not None:
            raise ValueError('All nine actual successful owner terminals required')
        if terminal.get('owned_process_absence_verified') is not True or terminal.get('owned_CUDA_absence_verified') is not True:
            raise ValueError('Actual owned absence evidence required')
        raw_owner=json.loads(bind(terminal['raw_owner_terminal']).read_text())
        if raw_owner.get('directly_waited') is not True or raw_owner.get('child_exit_code') != 0 or raw_owner.get('cap_or_owner_failure') is not None:
            raise ValueError('Raw owner evidence must itself establish actual successful direct wait')
        if terminal.get('release_sha256') not in raw_owner.get('argv', []):
            raise ValueError('Raw owner must identify the exact scientific release')
        bind(terminal['owned_absence_evidence'])
        if terminal.get('source_manifest_sha256') != spec['source_manifest_sha256'] or terminal.get('complete_sha256') != rows['complete']['sha256']:
            raise ValueError('Terminal custody must bind this source and completion')
        terminals[identity] = terminal
    records = {}
    for identity, rows in spec['records'].items():
        result_path = bind(rows['complete']); result = json.loads(result_path.read_text())
        seed, name = expected[identity]
        if result.get('schema') != 'masked-context-PubMed-stage1-complete-v1' or result.get('complete') is not True:
            raise ValueError('All nine complete records required')
        if result.get('record_id') != identity or result.get('seed') != seed or result.get('condition') != name:
            raise ValueError('Roster identity mismatch')
        if result.get('source_manifest_sha256') != spec['source_manifest_sha256'] or result.get('release_sha256') != terminals[identity]['release_sha256']:
            raise ValueError('Completed source/release differs')
        if result.get('selector') != SELECTOR or result.get('split_seed') != 190111 or result.get('split_identity') != 'PubMed-class-stratified-floor60-20-20':
            raise ValueError('Complete fixed selector/split required')
        if result.get('max_epochs') != 2000 or result.get('patience') != 250 or not 1 <= result.get('selected_epoch',0) <= result.get('epochs_executed',0) <= 2000:
            raise ValueError('Full frozen trajectory and selected state required')
        epochs,selected=result['epochs_executed'],result['selected_epoch']
        if result.get('stopped_by') == 'patience250':
            if epochs-selected != 250: raise ValueError('Exact250 consecutive non-improvements required')
        elif result.get('stopped_by') != 'max2000' or epochs != 2000:
            raise ValueError('No shortened run can be admitted as scientific completion')
        if result.get('selected_mask_diagnostics_epoch') != result['selected_epoch'] or result.get('TEST_scored') is not False or result.get('TEST_labels_or_id_inputs') is not False:
            raise ValueError('Same-selected-epoch diagnostics and TEST closure required')
        for key in ('complete_saved_member_logits', 'selected_predictor'):
            row = result[key]; target = (result_path.parent/row['path']).resolve(strict=True)
            if not target.is_relative_to(result_path.parent) or row.get('server_only') is not True or sha(target) != row['sha256'] or target.stat().st_size != row['bytes']:
                raise ValueError('Complete server-only selected payload custody required')
        saved=result['complete_saved_member_logits']
        expected_keys=['factual_member_logits']+(['owned_masked_member_logits'] if name == 'shared4_core' else [])
        if saved.get('keys') != expected_keys or saved.get('factual_shape') != [4,19717,3] or saved.get('masked_shape') != ([4,19717,3] if name == 'shared4_core' else None) or saved.get('contains_labels_or_role_ids') is not False:
            raise ValueError('Every selected member needs complete full-graph logits without labels/role IDs')
        records[seed,name] = result
    data_pairs = {(r['train_bundle_sha256'], r['valid_bundle_sha256']) for r in records.values()}
    if len(data_pairs) != 1: raise ValueError('All nine must share the exact same TRAIN/VALID arrays')
    output = inside(spec['output'], existing=False)
    if output.exists() or output.is_relative_to(HERE): raise ValueError('Fresh separate comparison output required')
    return spec, records, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('describe','compare'), default='describe')
    parser.add_argument('--release', type=Path); parser.add_argument('--release-sha256')
    args = parser.parse_args()
    if args.mode == 'describe':
        print(json.dumps(dict(enabled=False, required_records=[r['record_id'] for r in roster()], TEST_access=False))); return
    if args.release is None or args.release_sha256 is None: parser.error('Separate exact root complete-nine release required')
    spec, records, output = admit_comparison(args.release, args.release_sha256)
    output.mkdir(parents=True, exist_ok=False)
    result = summarize(records)
    result['source_manifest_sha256'] = spec['source_manifest_sha256']
    result['comparison_release_sha256'] = args.release_sha256
    result['record_custody'] = spec['records']
    write(output/'COMPARISON.json', result)
    print(json.dumps(dict(complete_nine=True, output=str(output), further18_activated=False)))


if __name__ == '__main__': main()
