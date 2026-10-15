"""Explicit later offline JSON synchronization, never new authorization.

Copy all immutable host shards into the receiving project first. This helper
then calls the existing authenticated merge once and writes delivery last.
No connection to another host, checkpoint/logit copy or native work exists.
"""
from pathlib import Path
from final_scoring_integration_v2.common import inside, pin, verify
from final_scoring_integration_v2.offline_seal import merge_host_shards
from .stages import same_or_create


def publish_delivery(shard_pins, ownership_pin, global_contract_pin, local_host, output_dir, root):
    ownership = verify(ownership_pin, root, json_metadata=True)
    contract = verify(global_contract_pin, root, json_metadata=True)
    seal, evidence = merge_host_shards(shard_pins, ownership, contract, root, local_host)
    output = inside(output_dir, root)
    output.mkdir(parents=True, exist_ok=True)
    seal_pin = same_or_create(output / 'GLOBAL_TRAINING_SEAL.json', seal, root)
    evidence_pin = same_or_create(output / 'OFFLINE_MERGE_EVIDENCE.json', evidence, root)
    delivery = {'schema': 'offline-final-host-delivery-v1', 'host': local_host,
                'global_contract': global_contract_pin, 'training_seal': seal_pin,
                'merge_evidence': evidence_pin, 'new_human_action_claimed': False,
                'TEST_accessed': False, 'native_work': 0, 'checkpoint_or_logit_transfer': False}
    delivery_pin = same_or_create(output / 'DELIVERY.json', delivery, root)
    return {'training_seal': seal_pin, 'merge_evidence': evidence_pin, 'delivery': delivery_pin}
