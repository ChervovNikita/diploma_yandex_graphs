"""Explicit root release checks before numerical imports or array reads."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from source import HERE, PHASE, bind, inside, sha, verify_manifest
from stage_plan import CONDITIONS, SEEDS, SPLIT, MAX_EPOCHS, PATIENCE, SELECTOR, LIMITS, QUALIFICATION_UPDATES, SERVER_HOST, SERVER_GPU, SERVER_PHASE


def admit(path, digest):
    path = inside(path)
    if sha(path) != digest: raise ValueError('Exact separate root release digest required')
    spec = json.loads(path.read_text())
    if spec.get('schema') != 'masked-context-PubMed-stage1-scientific-release-v1':
        raise ValueError('Separate Stage1 scientific release required')
    for key in ('enabled', 'root_science_authorized', 'source_review_approved',
                'source_delta_assessment_approved', 'runtime_qualified',
                'complete_graph_data_custody_verified', 'VALID_custody_verified',
                'external_hard_bound_confirmed', 'fresh_resource_readiness_confirmed',
                'ordinary_runtime_confirmed', 'complete_nine_roster_frozen'):
        if spec.get(key) is not True: raise ValueError('Inactive root admission: '+key)
    for key in ('TEST_access', 'automatic_retry', 'resume', 'further18_activated'):
        if spec.get(key) is not False: raise ValueError('TEST, retries, resume and later stages are closed')
    if any(key in spec for key in ('test_bundle', 'test_ids', 'test_y', 'split_ids_bundle')):
        raise ValueError('No TEST artifact or full role bundle admitted')
    if spec.get('condition') not in CONDITIONS or spec.get('seed') not in SEEDS:
        raise ValueError('Exact declared Stage1 condition/seed required')
    if spec.get('record_id') != f"seed{spec['seed']}__{spec['condition']}":
        raise ValueError('Declared record identity required')
    for key, value in SPLIT.items():
        if spec.get(key) != value: raise ValueError('Frozen split metadata differs: '+key)
    if spec.get('max_epochs') != MAX_EPOCHS or spec.get('patience') != PATIENCE or spec.get('selector') != SELECTOR:
        raise ValueError('Fixed2000/250 strict-first pooled selector required')
    bindings = verify_manifest(spec.get('source_manifest_sha256'))
    root = json.loads(bind(spec['root_admission']).read_text())
    if root.get('schema') != 'masked-context-stage1-root-adoption-v1' or root.get('enabled') is not True:
        raise ValueError('Separately reviewed actual root adoption required')
    for key in ('source_review_approved', 'source_delta_assessment_approved', 'complete_nine_roster_frozen'):
        if root.get(key) is not True: raise ValueError('Root has not approved '+key)
    if root.get('source_manifest_sha256') != spec['source_manifest_sha256'] or root.get('prototype_manifest_sha256') != bindings['prototype_manifest']['sha256']:
        raise ValueError('Root must adopt exact new runner and exact V3')
    if root.get('roster_sha256') != sha(HERE/'ROSTER.json') or root.get('protocol_sha256') != sha(HERE/'PROTOCOL.json'):
        raise ValueError('Root roster/protocol adoption mismatch')
    bind(root['source_review']); bind(root['source_delta_assessment'])
    train = bind(spec['train_bundle']); valid = bind(spec['valid_bundle'])
    if train == valid: raise ValueError('Separate TRAIN-only and VALID-only projections required')
    custody = json.loads(bind(spec['split_custody']).read_text())
    if spec['split_custody']['sha256'] != bindings['actual_TRAIN_custody']['sha256'] or spec['train_bundle']['sha256'] != bindings['TRAIN_bundle_server_only']['sha256']:
        raise ValueError('Exact actual absolute-exporter V4 custody and TRAIN bundle required')
    if spec.get('frozen_providers') != bindings['frozen_qualified_providers']:
        raise ValueError('Bind the actual core-qualified provider versions')
    if custody.get('schema') != 'masked-context-PubMed-TRAIN-role-custody-v1':
        raise ValueError('Exact data-custodian TRAIN role custody required')
    for key in ('split_protocol', 'split_seed', 'split_identity', 'TRAIN_count', 'VALID_count'):
        if custody.get(key) != spec[key]: raise ValueError('TRAIN custody disagrees: '+key)
    if custody.get('train_bundle_sha256') != spec['train_bundle']['sha256'] or custody.get('feature_normalization') != 'PyG.NormalizeFeatures':
        raise ValueError('Exact normalized factual graph and TRAIN bundle required')
    if custody.get('class_role_counts', {}).get('TRAIN') != SPLIT['TRAIN_class_counts'] or custody.get('class_role_counts', {}).get('VALID') != SPLIT['VALID_class_counts']:
        raise ValueError('Frozen representative class populations required')
    exporter = bind(custody['exporter_source'])
    if exporter != bind(bindings['exporter']): raise ValueError('Exact unchanged V4 exporter custody required')
    if custody.get('fresh_processed_cache') is not True or custody.get('factual_x_and_edges_match_native_custody') is not True:
        raise ValueError('Fresh public raw-data processing and factual graph parity required')
    if set(custody.get('array_fingerprints', {})) != {'x', 'edge_index', 'train_ids', 'train_y'}:
        raise ValueError('Exact four TRAIN-array fingerprints required')
    if spec.get('edge_shape') != custody.get('edge_shape'): raise ValueError('Full ordered edge dimensions differ')
    validation = json.loads(bind(spec['validation_custody']).read_text())
    if validation.get('schema') != 'masked-context-PubMed-VALID-role-custody-v1':
        raise ValueError('Separate VALID projection custody required')
    for key in ('split_protocol', 'split_seed', 'split_identity', 'VALID_count', 'VALID_class_counts'):
        if validation.get(key) != spec[key]: raise ValueError('VALID custody disagrees: '+key)
    if validation.get('train_custody_sha256') != spec['split_custody']['sha256'] or validation.get('valid_bundle_sha256') != spec['valid_bundle']['sha256']:
        raise ValueError('VALID must belong to this exact frozen TRAIN/split custody')
    if validation.get('TEST_labels_loaded') is not False or validation.get('model_constructed') is not False:
        raise ValueError('Only a data-custodian projection, without TEST labels/model, is admitted')
    if set(validation.get('array_fingerprints', {})) != {'valid_ids', 'valid_y'}:
        raise ValueError('Both VALID-array fingerprints required')
    bind(validation['custodian_source'])
    qualifications = root.get('qualifications', {})
    if set(qualifications) != set(CONDITIONS): raise ValueError('All Stage1 runtime surfaces need explicit root coverage')
    terminals = root.get('qualification_terminals', {})
    if set(terminals) != set(CONDITIONS): raise ValueError('Actual qualification owner terminals required')
    for name, row in qualifications.items():
        result = json.loads(bind(row).read_text())
        if result.get('schema') != 'masked-context-PubMed-TRAIN-only-engineering-result-v1' or result.get('complete') is not True:
            raise ValueError('Actual complete TRAIN-only V3 qualification required')
        if result.get('condition') != name or result.get('seed') != 9101 or result.get('source_manifest_sha256') != bindings['prototype_manifest']['sha256']:
            raise ValueError('Qualification source/condition/seed differs')
        if any(result.get(k) is not False for k in ('VALID_access', 'TEST_access', 'science_enabled')):
            raise ValueError('Qualification must remain TRAIN-only')
        expected_updates=QUALIFICATION_UPDATES[name]
        if result.get('counters', {}).get('updates') != expected_updates or len(result.get('trace', [])) != expected_updates:
            raise ValueError('Frozen full-TRAIN engineering coverage count differs')
        if row['sha256'] != bindings['actual_qualifications'][name]['sha256'] or terminals[name]['sha256'] != bindings['actual_qualification_terminals'][name]['sha256']:
            raise ValueError('Exact actual V3/source/runtime engineering custody required')
        if any(t.get('TRAIN', {}).get('TRAIN_label_count') != 11829 or t.get('TRAIN', {}).get('complete_graph_nodes') != 19717 for t in result['trace']):
            raise ValueError('Qualification must use every TRAIN label on the complete graph')
        if result.get('data_SHA') != spec['train_bundle']['sha256'] or result.get('providers') != spec['frozen_providers']:
            raise ValueError('Qualification data/runtime differs')
        for key in ('split_protocol', 'split_seed', 'split_identity'):
            if result.get(key) != spec[key]: raise ValueError('Qualification split differs')
        terminal = json.loads(bind(terminals[name]).read_text())
        if terminal.get('directly_waited') is not True or terminal.get('child_exit_code') != 0 or terminal.get('cap_or_owner_failure') is not None:
            raise ValueError('Qualification requires an actually waited successful owner terminal')
        if result.get('engineering_release_sha256') not in terminal.get('argv', []):
            raise ValueError('Qualification owner must bind the actual engineering release')
    limits = spec.get('limits', {})
    for key, value in LIMITS.items():
        if value is not None and limits.get(key) != value: raise ValueError('Fixed resource cap differs: '+key)
    for key in ('external_active_seconds', 'external_cleanup_seconds'):
        if type(limits.get(key)) is not int or limits[key] <= 0: raise ValueError('Finite root-owned active and cleanup caps required')
    owner = json.loads(bind(spec['external_owner_release']).read_text())
    if owner.get('enabled') is not True or owner.get('record_id') != spec['record_id'] or owner.get('limits') != limits:
        raise ValueError('Exact separately reviewed finite owner release required')
    for key in ('separate_process_group', 'direct_wait_required', 'resource_caps_enforced', 'output_and_log_caps_enforced'):
        if owner.get(key) is not True: raise ValueError('Owner fact missing: '+key)
    if owner.get('automatic_retry') is not False: raise ValueError('No owner retries')
    bind(owner['owner_source']); bind(spec['resource_readiness_evidence'])
    runtime = spec['runtime']
    if socket.gethostname() != SERVER_HOST or str(PHASE) != SERVER_PHASE:
        raise ValueError('Science runs only on the exact allocation project route')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines()
    if inventory != [SERVER_GPU] or os.environ.get('CUDA_VISIBLE_DEVICES') != SERVER_GPU or spec.get('device') != 'cuda:0':
        raise ValueError('Exact sole allocation GPU and ordinary cuda:0 required')
    python_path = Path(os.path.abspath(runtime['python']['path']))
    if not python_path.is_relative_to(PHASE) or not python_path.is_file() or sha(python_path) != runtime['python']['sha256']:
        raise ValueError('Existing project runtime launcher/hash required; its ordinary interpreter symlink is permitted')
    if Path(sys.executable).resolve() != python_path.resolve() or os.environ.get('PYTHONPATH', '') != runtime['PYTHONPATH']:
        raise ValueError('Exact root-bound existing runtime executable/PYTHONPATH required')
    output = inside(spec['output'], existing=False)
    if output.is_relative_to(HERE) or output.exists(): raise ValueError('Fresh separate server execution output required')
    spec['_release_sha256'], spec['_train_custody'], spec['_validation_custody'] = digest, custody, validation
    return spec, output
