"""Separate reviewed root releases precede any numeric import or array read."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from source import HERE, PHASE, bind, inside, sha, verify_manifest
from reference_plan import CONDITIONS, SEEDS, SPLIT, SELECTOR, LIMITS, ENGINEERING_LIMITS, SERVING, body_roster, SERVER_HOST, SERVER_GPU, SERVER_PHASE


def common(path,digest,engineering=False):
    path=inside(path)
    if sha(path) != digest: raise ValueError('Exact separate root release digest required')
    spec=json.loads(path.read_text())
    schema='PubMed-strong-reference-engineering-release-v1' if engineering else 'PubMed-strong-reference-scientific-release-v1'
    if spec.get('schema') != schema: raise ValueError('New strong-reference release required')
    flags=('enabled','root_authorized','source_review_approved','source_delta_assessment_approved',
           'external_hard_bound_confirmed','fresh_resource_readiness_confirmed','ordinary_runtime_confirmed',
           'complete_graph_data_custody_verified','complete_reference_roster_frozen','post_screen_exploration')
    if any(spec.get(k) is not True for k in flags): raise ValueError('Inactive new root admission')
    if any(spec.get(k) is not False for k in ('TEST_access','CORE','masking','HPO','automatic_retry','resume','further18_activated','paper_score_recalculation')):
        raise ValueError('Closed TEST, CORE, masking, HPO, retry, resume, Stage2 and paper scores')
    if any(k in spec for k in ('test_bundle','test_ids','test_y','split_ids_bundle')):
        raise ValueError('No TEST or full-role artifact admitted')
    if spec.get('condition') not in CONDITIONS or spec.get('seed') not in SEEDS:
        raise ValueError('Frozen seed/condition required')
    if spec.get('record_id') != f"seed{spec['seed']}__{spec['condition']}": raise ValueError('Frozen group identity required')
    for k,v in SPLIT.items():
        if spec.get(k) != v: raise ValueError('Frozen split differs: '+k)
    if spec.get('body_roster') != body_roster(spec['seed'],spec['condition']): raise ValueError('Every fresh private body/stream/selector must be frozen')
    bindings=verify_manifest(spec.get('source_manifest_sha256'))
    data=json.loads((HERE/'DATA_AND_RUNTIME.json').read_text())
    for k in ('train_bundle','split_custody','runtime','frozen_providers'):
        if spec.get(k) != data[k]: raise ValueError('Exact actual data/runtime binding differs: '+k)
    train=bind(spec['train_bundle'])
    custody=json.loads(bind(spec['split_custody']).read_text())
    if custody.get('schema') != 'masked-context-PubMed-TRAIN-role-custody-v1' or custody.get('train_bundle_sha256') != spec['train_bundle']['sha256']:
        raise ValueError('Exact actual TRAIN role custody required')
    if custody.get('feature_normalization') != 'PyG.NormalizeFeatures' or custody.get('fresh_processed_cache') is not True or custody.get('factual_x_and_edges_match_native_custody') is not True:
        raise ValueError('Exact normalized full factual graph required')
    for k in ('split_protocol','split_seed','split_identity','TRAIN_count','VALID_count','edge_shape'):
        if custody.get(k) != spec[k]: raise ValueError('Frozen TRAIN/split custody differs')
    if custody.get('class_role_counts',{}).get('TRAIN') != SPLIT['TRAIN_class_counts'] or custody.get('class_role_counts',{}).get('VALID') != SPLIT['VALID_class_counts']:
        raise ValueError('Frozen representative populations required')
    if bind(custody['exporter_source']) != bind(bindings['exporter']): raise ValueError('Exact absolute V4 exporter required')
    if set(custody.get('array_fingerprints',{})) != {'x','edge_index','train_ids','train_y'}:
        raise ValueError('Four exact TRAIN array fingerprints required')
    limits=ENGINEERING_LIMITS if engineering else LIMITS
    if spec.get('limits') != limits: raise ValueError('Frozen finite caps required')
    owner=json.loads(bind(spec['external_owner_release']).read_text())
    if owner.get('enabled') is not True or owner.get('record_id') != spec['record_id'] or owner.get('limits') != limits:
        raise ValueError('New separate finite root owner required')
    for k in ('separate_process_group','direct_wait_required','resource_caps_enforced','output_and_log_caps_enforced'):
        if owner.get(k) is not True: raise ValueError('Finite owner fact missing: '+k)
    if owner.get('automatic_retry') is not False: raise ValueError('No automatic owner retry')
    bind(owner['owner_source']); bind(spec['resource_readiness_evidence'])
    bind(spec['source_review']); bind(spec['source_delta_assessment'])
    if socket.gethostname() != SERVER_HOST or str(PHASE) != SERVER_PHASE:
        raise ValueError('Exact allocation project route required')
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()
    if inventory != [SERVER_GPU] or os.environ.get('CUDA_VISIBLE_DEVICES') != SERVER_GPU or spec.get('device') != 'cuda:0':
        raise ValueError('Exact sole allocation GPU required')
    runtime=spec['runtime']; python_path=Path(os.path.abspath(runtime['python']['path']))
    if not python_path.is_relative_to(PHASE) or not python_path.is_file() or sha(python_path) != runtime['python']['sha256']:
        raise ValueError('Pinned existing runtime required')
    if Path(sys.executable).resolve() != python_path.resolve() or os.environ.get('PYTHONPATH','') != runtime['PYTHONPATH']:
        raise ValueError('Exact runtime executable and PYTHONPATH required')
    output=inside(spec['output'],existing=False)
    if output.exists() or output.is_relative_to(HERE): raise ValueError('Fresh separate execution output required')
    spec['_release_sha256'],spec['_train_custody']=digest,custody
    return spec,output,bindings


def admit(path,digest):
    spec,output,bindings=common(path,digest)
    if spec.get('max_epochs') != 2000 or spec.get('patience') != 250 or spec.get('selector') != SELECTOR or spec.get('serving') != SERVING:
        raise ValueError('Every own2000/patience250 strict-first selector and serving protocol required')
    if spec.get('VALID_custody_verified') is not True: raise ValueError('Actual VALID projection required')
    data=json.loads((HERE/'DATA_AND_RUNTIME.json').read_text())
    for k in ('valid_bundle','validation_custody'):
        if spec.get(k) != data[k]: raise ValueError('Exact actual VALID binding required')
    valid=bind(spec['valid_bundle'])
    if valid == bind(spec['train_bundle']): raise ValueError('Separate projected roles required')
    validation=json.loads(bind(spec['validation_custody']).read_text())
    if validation.get('schema') != 'masked-context-PubMed-VALID-role-custody-v1': raise ValueError('Actual VALID custody required')
    for k in ('split_protocol','split_seed','split_identity','VALID_count','VALID_class_counts'):
        if validation.get(k) != spec[k]: raise ValueError('Frozen VALID custody differs')
    if validation.get('train_custody_sha256') != spec['split_custody']['sha256'] or validation.get('valid_bundle_sha256') != spec['valid_bundle']['sha256']:
        raise ValueError('Exact TRAIN/VALID association required')
    if validation.get('TEST_labels_loaded') is not False or validation.get('model_constructed') is not False or set(validation.get('array_fingerprints',{})) != {'valid_ids','valid_y'}:
        raise ValueError('Custodian-only VALID without TEST/model required')
    bind(validation['custodian_source'])
    root=json.loads(bind(spec['root_admission']).read_text())
    if root.get('schema') != 'PubMed-strong-reference-root-adoption-v1' or root.get('enabled') is not True:
        raise ValueError('New separately reviewed root adoption required')
    for k in ('source_review_approved','source_delta_assessment_approved','complete_reference_roster_frozen','post_screen_exploration'):
        if root.get(k) is not True: raise ValueError('Root fact missing: '+k)
    if root.get('CORE_continuation_eligible') is not False or root.get('further18_activated') is not False:
        raise ValueError('Failed CORE and Stage2 remain closed')
    for k,name in (('source_manifest_sha256','SOURCE_MANIFEST.json'),('roster_sha256','ROSTER.json'),('protocol_sha256','PROTOCOL.json'),('anchors_sha256','ANCHORS.json')):
        if root.get(k) != sha(HERE/name): raise ValueError('New exact root adoption binding differs')
    bind(root['source_review']);bind(root['source_delta_assessment'])
    if set(root.get('qualifications',{})) != set(CONDITIONS) or set(root.get('qualification_terminals',{})) != set(CONDITIONS):
        raise ValueError('New complete TRAIN-only qualification custody for all three interfaces required')
    for condition in CONDITIONS:
        result=json.loads(bind(root['qualifications'][condition]).read_text())
        terminal=json.loads(bind(root['qualification_terminals'][condition]).read_text())
        if result.get('schema') != 'PubMed-strong-reference-TRAIN-only-engineering-v1' or result.get('complete') is not True or result.get('condition') != condition or result.get('seed') != 9101:
            raise ValueError('New exact condition qualification required')
        if result.get('source_manifest_sha256') != spec['source_manifest_sha256'] or result.get('data_SHA') != spec['train_bundle']['sha256'] or result.get('providers') != spec['frozen_providers']:
            raise ValueError('New source/data/provider qualification required')
        if any(result.get(k) is not False for k in ('VALID_access','TEST_access','science_enabled')):
            raise ValueError('Qualification must remain TRAIN-only')
        expected=4 if condition == 'independent4_own' else 1
        if len(result.get('bodies',[])) != expected or any(r.get('updates') != 1 or r.get('TRAIN_label_count') != 11829 or r.get('complete_graph_nodes') != 19717 for r in result['bodies']):
            raise ValueError('Full TRAIN engineering coverage differs')
        if terminal.get('directly_waited') is not True or terminal.get('child_exit_code') != 0 or terminal.get('cap_or_owner_failure') is not None or result.get('engineering_release_sha256') not in terminal.get('argv',[]):
            raise ValueError('Actual successful waited qualification owner required')
    spec['_validation_custody']=validation
    return spec,output


def admit_engineering(path,digest):
    spec,output,_=common(path,digest,engineering=True)
    if spec['seed'] != 9101 or spec.get('updates_per_body') != 1 or spec.get('science_enabled') is not False or spec.get('VALID_access') is not False:
        raise ValueError('Frozen one-update TRAIN-only engineering release required')
    if any(k in spec for k in ('valid_bundle','validation_custody','test_bundle')):
        raise ValueError('Engineering cannot receive VALID or TEST artifacts')
    return spec,output
