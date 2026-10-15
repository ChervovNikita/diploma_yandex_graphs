"""Metadata contracts for one existing supervisor command per stage."""
import copy
from pathlib import Path
import re
from final_scoring_integration_v2.common import digest, inside, require, verify

WORKERS = {'allocation/gpu0', 'server77/gpu0', 'server77/gpu1'}


def validate_template(template):
    require(template['schema'] == 'prepared-combined-worker-execution-template-v1'
            and template['worker'] in WORKERS, 'Frozen worker execution template required')
    require(template['training_entry']['path'].endswith('/portable_dispatcher_v3/run_gpu.py'), 'Reuse exact existing training entrypoint')
    final_path = Path(template['final_entry']['path'])
    require(final_path.name == 'run_gpu.py' and final_path.parent.name == 'final_scoring_integration_v2', 'Use the deliberately rebound final integrationv2 entrypoint')
    require(set(template['final_static_fields']) in ({'release', 'readiness'}, {'release', 'readiness', 'continuation'}),
            'Only predeclared final release/readiness/optional existing STOP continuation fields allowed')
    require(template['scorer_source_manifest'] is not None and template['final_integration_manifest'] is not None,
            'Scorer and integration dependencies are unbound; preparation is not readiness')
    require(template['scorer_source_manifest']['path'].endswith('/final_selected_scoring_v4/MANIFEST.json'),
            'Use the coherent repaired scorer4source, not a stale source candidate')
    require(template['original_paper_reruns'] is False and template['new_human_action_at_barrier'] is False,
            'Offline barrier derives evidence, never a new experiment or approval')
    require(template['maximum_native_children_per_GPU'] == 2, 'Original two-slot ceiling required')
    return template


def validate_actual(spec, template, go, training_spec, root):
    validate_template(template)
    require(spec['schema'] == 'combined-portable-campaign-run-v1' and spec['worker'] == template['worker'], 'Exact combined worker spec required')
    require(go['schema'] == 'portable-human-GO-v2' and go['user_explicit_GO'] is True
            and spec['worker'] in go['allowed_workers'], 'Actual existing campaign GO required')
    combined = go['combined_execution']
    require(set(combined) == {'source_manifest', 'execution_templates'}
            and combined['source_manifest'] == spec['source_manifest']
            and combined['execution_templates'][spec['worker']] == spec['execution_template']
            and all(worker in WORKERS and worker.split('/')[0] == spec['worker'].split('/')[0]
                    for worker in combined['execution_templates']),
            'Same host GO binds all its worker templates; both77workers share one unchanged GO view')
    require(go['final_scoring']['source_manifest'] == template['scorer_source_manifest']
            and go['final_execution']['source_manifest'] == template['final_integration_manifest']
            and go['final_execution']['release'] == template['final_static_fields']['release']
            and go['final_execution']['readiness'] == template['final_static_fields']['readiness'], 'Final scientific/source authority differs from original GO')
    for field in ('orchestration_output_root', 'offline_delivery_path', 'host_sync_root'):
        inside(template[field], root)
    require(spec['training_go'] == training_spec['go']
            and spec['training_release'] == training_spec['release']
            and go['host_releases'][spec['worker'].split('/')[0]] == spec['training_release'],
            'Training and final stages must use identical local GO/release authority')
    return True


def validate_host_templates(template, host_templates):
    """At most two metadata views; prevent mismatched local barrier paths."""
    require(template['worker'] in host_templates, 'Own host template missing')
    for worker, peer in host_templates.items():
        require(peer['worker'] == worker and worker.split('/')[0] == template['worker'].split('/')[0], 'Foreign host template')
        for field in ('host_sync_root', 'offline_delivery_path', 'scoring_roster', 'scorer_source_manifest', 'final_integration_manifest'):
            require(peer[field] == template[field], 'Local host workers disagree on ' + field)
        for field in ('release', 'readiness'):
            require(peer['final_static_fields'][field] == template['final_static_fields'][field], 'Local workers disagree on final authority')
    return True


def derive_final_spec(template, spec, delivery):
    """Only the globally derived seal and prebound parent authorities are added."""
    validate_template(template)
    require(delivery['schema'] == 'offline-final-host-delivery-v1'
            and delivery['host'] == spec['worker'].split('/')[0]
            and delivery['global_contract'] == spec['global_contract']
            and delivery['new_human_action_claimed'] is False, 'Exact deterministic offline delivery required')
    value = copy.deepcopy(template['final_static_fields'])
    value.update(schema='portable-final-reviewed-run-v1', worker=spec['worker'],
                 training_run_spec=spec['training_run_spec'], training_go=spec['training_go'],
                 global_decision=spec['global_decision'], global_contract=spec['global_contract'],
                 training_seal=delivery['training_seal'])
    return value


def validate_delivery(delivery, spec, template, root):
    require(delivery['schema'] == 'offline-final-host-delivery-v1'
            and delivery['host'] == spec['worker'].split('/')[0]
            and delivery['global_contract'] == spec['global_contract']
            and delivery['new_human_action_claimed'] is False, 'Foreign or changed offline final delivery')
    seal = verify(delivery['training_seal'], root, json_metadata=True)
    require(seal['schema'] == 'all-campaign-outcomes-and-selections-seal-v2'
            and seal['all_6730_training_tasks_terminal'] is True and len(seal['outcomes']) == 6730
            and seal['training_release'] == spec['training_release']
            and seal['scoring_roster'] == template['scoring_roster']
            and seal['TEST_accessed'] is False and seal['retry_or_arm_changes_after_seal'] is False,
            'All global TRAIN outcomes and own original selections must be sealed')
    # The actual final supervisor performs the complete shard/role/source
    # authentication once. This sequencer does not rescan every task per poll.
    return seal
