"""Bind a prospective accuracy comparison, without admitting scientific execution."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def descriptor(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(PHASE)), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


def main():
    input_path = PHASE/'graph_view_paired_input_binding_root_20261004_v1/INPUT_IDENTITIES.json'
    inputs = json.loads(input_path.read_text())
    assert inputs['training_updates'] == 0 and not inputs['predictive_values_read']
    assert not inputs['TEST_labels_read'] and not inputs['scientific_comparison_frozen']
    sources = {
        'bank': 'accuracy_first_graph_view_source_preparation_20261004_v2',
        'reference': 'accuracy_first_graph_view_reference_source_preparation_20261004_v2',
        'native_gnnm': 'accuracy_first_native_gnnm_reference_source_preparation_20261004_v2',
    }
    manifests = {key: descriptor(PHASE/name/'MANIFEST.json') for key,name in sources.items()}
    for key,row in manifests.items():
        assert inputs[key+'_manifest_sha256'] == row['sha256']
    blocks = [(0,17),(1,29),(2,43)]
    cells = []
    for split,seed in blocks:
        for condition in ('tied_persistent','untied_persistent','tied_shuffled','tied_random_null'):
            cells.append(dict(id=f'split{split}_{condition}_seed{seed}',family='bank',condition=condition,
                              split=split,block_seed=seed,schedule_seed=seed,view_seed=seed))
        cells.append(dict(id=f'split{split}_view_augmented_single_seed{seed}',family='reference',
                          condition='view_augmented_single',split=split,member=None,seed=seed))
        for member in range(4):
            cells.append(dict(id=f'split{split}_native_member{member}_seed{seed+1009*member}',
                              family='reference',condition='native_member',split=split,
                              member=member,seed=seed+1009*member))
        cells.append(dict(id=f'split{split}_native_gnnm4_seed{seed}',family='native_gnnm',
                          condition='native_gnnm4',split=split,block_seed=seed))
    assert len(cells)==30 and len({r['id'] for r in cells})==30
    plan = dict(
        schema='accuracy-first-graph-view-paired-plan-v1',UTC=datetime.now(timezone.utc).isoformat(),
        status='DECISION_AND_INPUTS_FIXED_EXECUTION_DISABLED_PENDING_RUNTIME_AND_RESOURCE_ADMISSION',
        execution_authorized=False,scientific_comparison_frozen=False,
        primary_objective='Better served ensemble accuracy over competent single and independent-ensemble controls',
        **{k:inputs[k] for k in ('bank_manifest_sha256','reference_manifest_sha256',
                               'native_gnnm_manifest_sha256','input_bindings_by_split')},
        input_identity_receipt=descriptor(input_path),source_manifests=manifests,
        source_protocols={key:descriptor(PHASE/name/'PROTOCOL.json') for key,name in sources.items()},
        official_TRAIN_coverage_by_split=inputs['official_TRAIN_coverage_by_split'],
        blocks=[dict(split=split,block_seed=seed,view_seed=seed,schedule_seed=seed) for split,seed in blocks],
        physical_cells=cells,physical_fits=30,
        aliases=dict(native_single='Predeclared native member0 only, no best-member selection',
                     independent4='Fixed mean of four independently selected native class-probability vectors'),
        label_visibility=dict(fitting='Complete official TRAIN only',
                              views='Class categories use TRAIN-TRAIN endpoints only',
                              VALIDATION='Native accuracy selector and exploratory development screen',
                              TEST='Mask IDs only, no TEST labels or score reader'),
        recipe=dict(native_local_updates=200,native_global_updates=2500,
                    selector='Strict native-graph VALIDATION correct-count best across updates1..2700, earliest ties',
                    transition='Restore own selected-local model/Adam, keep live RNG, then enable global stage',
                    pooling='Arithmetic mean of class-softmax probabilities on native graph',
                    sources='Exact immutable per-family source protocols',HPO=False,retries=False),
        runtime_policy=dict(deterministic_algorithms=True,warn_only=False,
                            CUBLAS_WORKSPACE_CONFIG=':4096:8',cuda_matmul_TF32=False,cudnn_TF32=False,
                            common_interpreter_and_device_family='Must be bound by subsequent execution admission'),
        primary_comparison='tied_persistent minus untied_persistent accuracy',
        decision_rule=dict(
            endpoint='Exploratory selected VALIDATION accuracy, fraction units',
            mechanism_gate=dict(mean_gain_minimum=0.005,positive_paired_blocks_required=3,
                                comparator='untied_persistent'),
            accuracy_utility_gate=dict(mean_gain_minimum=0.005,positive_paired_blocks_required=3,
                                      comparators=['native_gnnm4','native_independent4',
                                                   'view_augmented_single','native_single_member0']),
            interpretation_gate='No graph/view-specialization claim without the shuffled and random-null controls',
            mechanism_and_utility_both_required_for_broader_followup=True,
            thresholds_are_prospective_development_choices_not_significance_tests=True,
            check_all_cells_before_any_comparative_outcome_read=True,
            missing_failed_cells='Preserve costs and cause. No outcome-driven replacement, subset report or donor reuse.'),
        reporting=['all paired accuracy differences and signs','selected-checkpoint NLL',
                   'macro-F1','each member accuracy/NLL','physical fits and graph passes',
                   'wall time and observed peak memory','all failures and null outcomes'],
        statistical_scope='Three paired split/seed blocks on one graph. Selected VALIDATION evidence is development only. No independent graph population or acceptance claim.',
        confirmation='Separate prospective competent-baseline comparison on an audited heldout endpoint after a useful screen. Existing Amazon split/TEST exposure must be disclosed.',
        attribution='Sharing, graph augmentation and diversified ensembles are established prior. No novelty clearance or methodological advantage assumed.',
        pending=['Exact common-runtime nativeGNNM CUDA replay/cost qualification',
                 'Whole-family runtime/resource admission and explicit device schedule',
                 'Root-reviewed caller and per-family context admission',
                 'Actual history audit before any independent heldout confirmation'],
        original_paper_scores_unchanged=True,predictive_values_read=False)
    with (HERE/'PAIRED_PROTOCOL.json').open('x') as stream:
        json.dump(plan,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(physical_fits=30,paired_blocks=3,execution_authorized=False,
                         sha256=descriptor(HERE/'PAIRED_PROTOCOL.json')['sha256'])))


if __name__=='__main__':
    main()
