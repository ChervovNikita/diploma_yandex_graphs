"""Require all12 completed cells; verify nine F4 streams and apply fixed rule."""
import argparse
import itertools
import json
from math import isfinite, sqrt
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
from pilot_contract import ARMS, SEEDS, EPOCHS, file_sha256, verify_pins


def describe(first, second, cells, role, pairing):
    differences = [cells[seed,first]['best_valid_hits20'] - cells[seed,second]['best_valid_hits20'] for seed in SEEDS]
    mean, sd = statistics.mean(differences), statistics.stdev(differences)
    radius = 4.302652729911275 * sd / sqrt(3)
    signs = [statistics.mean(sign*value for sign,value in zip(signs,differences))
             for signs in itertools.product((-1,1),repeat=3)]
    return {'contrast':f'{first}-{second}', 'role':role, 'pairing':pairing,
            'seed_differences':differences, 'mean':mean, 'sample_sd':sd,
            'range':[min(differences),max(differences)], 'descriptive_t95':[mean-radius,mean+radius],
            'exact_two_sided_sign_flip_p':sum(abs(value)>=abs(mean) for value in signs)/8,
            'units':'Hits@20_fraction'}


def compare(root, packet):
    binding = verify_pins(packet)
    sys.path.insert(1,str(packet.parent / binding['sealed_F4_packet']))
    from paired_comparison import compare_family
    cells = {}
    config_digest, manifest_digest = file_sha256(packet/'config.json'), file_sha256(packet/'MANIFEST.json')
    for seed,arm in itertools.product(SEEDS,ARMS):
        directory = root / arm / f'seed_{seed}'
        row = json.loads((directory/'cell_summary.json').read_text())
        metric = row['best_valid_hits20']
        if (row['schema'] != 'hlgnn-ddi-development-pilot-cell-v1' or row['status'] != 'complete'
                or type(row['seed']) is not int or row['seed'] != seed or row['arm'] != arm
                or type(row['completed_epochs']) is not int or row['completed_epochs'] != EPOCHS
                or row['actual_loss_branch'] != 'AUC' or row['selected_replay'] != 'PASS'
                or type(row['best_epoch']) is not int or row['best_epoch'] not in range(5,EPOCHS+1,5)
                or type(metric) not in (int,float) or not isfinite(metric) or not 0<=metric<=1
                or row['selected_replay_metrics']['Hits@20'] != metric
                or row['pilot_manifest_sha256'] != manifest_digest or row['scientific_config_sha256'] != config_digest
                or row['artifact_sha256'] != binding['artifact']['sha256']
                or row['VALID_traversals_including_replay'] != 21
                or row['full500_qualification'] is not False or row['author_budget_parity'] is not False
                or row['TEST_access'] is not False or row['donor_state'] is not False
                or file_sha256(directory/'valid_best.pt') != row['selected_checkpoint_sha256']):
            raise RuntimeError(f'Incomplete/invalid declared pilot cell: {arm}/seed_{seed}')
        cells[seed,arm] = row
    # No primary/secondary mean is computed until every declared cell is complete.
    f4 = compare_family(root,[cells[seed,arm] for seed in SEEDS for arm in ('target_only','joint','separate')],
                        epochs=EPOCHS,train_records=1067911,batch_size=65536)
    pairing = 'nine F4 streams verified' if f4['paired_eligible'] else 'F4 stream verification failed; paired interpretation ineligible'
    contrasts = [describe('joint','separate',cells,'primary',pairing),
                 describe('joint','target_only',cells,'secondary',pairing),
                 describe('separate','target_only',cells,'secondary',pairing)]
    contrasts += [describe(arm,'native_m1',cells,'secondary','same seed labels; constructor/native RNG streams differ')
                  for arm in ('target_only','joint','separate')]
    primary, target_reference = contrasts[0], contrasts[1]
    rule = {'positive_primary_mean':primary['mean']>0,
            'at_least_two_of_three_positive_primary_seeds':sum(value>0 for value in primary['seed_differences'])>=2,
            'no_pooled_joint_regression_vs_F4_target_only':target_reference['mean']>=0,
            'complete_12_cells':True, 'nine_F4_streams_paired_eligible':f4['paired_eligible']}
    decision = 'GO_RECOMMEND_SEPARATELY_ADMITTED_500EPOCH_CONFIRMATION' if all(rule.values()) else 'NO_GO_FOR_CURRENT_COUPLING_CANDIDATE'
    if not f4['paired_eligible']:
        decision = 'INELIGIBLE_RECEIPT_MISMATCH'
    return {'schema':'hlgnn-ddi-development-pilot-comparison-v1', 'cells':list(cells.values()),
            'F4_paired_comparison':f4, 'contrasts':contrasts, 'continuation_rule_checks':rule,
            'development_decision':decision, 'training_budget':{'development_epochs':100,'author_epochs':500,'author_budget_parity':False},
            'limits':'Development on one graph with three seed blocks; descriptive inference only. Native M1 contrasts are not F4 stream/RNG-paired. A go recommends separately prereleased/admitted fresh500-epoch competitive confirmation; no automatic launch, significance, generalization or acceptance claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    packet = Path(__file__).resolve().parent
    binding = verify_pins(packet)
    sys.path.insert(1,str(packet.parent/binding['sealed_F4_packet']))
    from runtime_paths import require_project_output
    root = args.family_root.resolve()
    if (not args.family_root.is_absolute() or not root.is_dir()
            or not root.is_relative_to(packet.parent) or root.is_relative_to(packet)):
        raise ValueError('Family root must be an absolute project execution directory.')
    output = require_project_output(args.output_dir,packet)
    output.mkdir(parents=True,exist_ok=False)
    try:
        result = compare(root,packet)
    except (KeyError,ValueError,RuntimeError,OSError) as error:
        (output/'incomplete_family.json').write_text(json.dumps({
            'status':'INCOMPLETE_OR_INVALID_FAMILY', 'error_type':type(error).__name__,
            'error':str(error),'no_success_only_mean_or_go_decision':True},indent=2)+'\n')
        raise
    (output/'pilot_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'development_decision':result['development_decision'],
                      'continuation_rule_checks':result['continuation_rule_checks']},allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
