"""Explicit finite NCNC lifecycle policy; never a production matched schedule."""
from __future__ import annotations
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
EXPANSION=next(p for p in HERE.parents if p.name=='benchmark_expansion_20261015_v1')
BASE_MANIFEST_SHA='2cc662050bd964b4fcc0b46923e3bf00cf31e1ee4f48e02107db7b79bbc28936'
BASE_POLICY_SHA='732eaf774347b0c1eb48d30f436a7315af4d8cc32bd3b7b43685facfcb818fbc'
FINITE_POLICY={'family':'link','track':'bounded_engineering','maximum_epochs':2,'minimum_epochs':2,
    'patience':1,'metric':'hits@50','direction':'max','ties':'first','evaluate_TRAIN_each_epoch':False,
    'evaluation_mode':'VALID_only_disclosed_adaptation'}


def base_policy(root):
    root=Path(root).resolve();folder=EXPANSION/'link_policy_v1'
    if not folder.is_relative_to(root):raise ValueError('Original link policy must remain in actual repository')
    manifest=folder/'MANIFEST.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=BASE_MANIFEST_SHA:
        raise ValueError('Frozen original link policy manifest changed')
    for row in json.loads(manifest.read_text())['files']:
        p=(folder/row['path']).resolve()
        if not p.is_relative_to(folder):raise ValueError('Original policy manifest path escapes')
        b=p.read_bytes()
        if len(b)!=row['bytes'] or hashlib.sha256(b).hexdigest()!=row['sha256']:
            raise ValueError('Frozen original policy source changed')
    source=folder/'policy.py'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=BASE_POLICY_SHA:raise ValueError('Original policy bytes differ')
    name='_runtime_link_v5_engineering_original_policy_'+BASE_POLICY_SHA
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def create_policy(job,root=None):
    if job.get('family')!='link' or job.get('engineering_only') is not True or job.get('TEST_access') is not False \
            or job.get('policy')!=FINITE_POLICY or job['recipe'].get('architecture')!='NCNC' \
            or job.get('procedure')!='interaction_input_conditionalJ':
        raise ValueError('Only explicit two-epoch NCNC conditional-J engineering lifecycle policy is admitted')
    root=Path(root or job['recipe']['_repository_root']).resolve();module=base_policy(root)
    binding=module.source_bindings(root)
    cell=next(row for row in binding['cells'] if row['native_family']=='NCNC')
    track=cell['tracks']['long_convergence'];stopping=track['stopping']
    parent_job=copy.deepcopy(job)
    parent_job['policy']={**FINITE_POLICY,'track':'matched_extended','maximum_epochs':track['maximum'],
        'minimum_epochs':stopping['minimum_units'],'patience':stopping['patience_validation_checks']}
    parent=module.create_policy(parent_job,root)
    source_binding=copy.deepcopy(parent.source_binding)
    source_binding['engineering_parent_policy']={'manifest_sha256':BASE_MANIFEST_SHA,'policy_sha256':BASE_POLICY_SHA}
    return replace(parent,schema='bounded-native-NCNC-engineering-policy-v1',track='bounded_engineering',
        maximum_epochs=2,minimum_epochs=2,patience=1,source_role='finite_native_lifecycle_engineering',
        source_protocol_equivalent=False,source_RNG_trajectory_equivalent=False,
        fidelity_limitations=[*parent.fidelity_limitations,
            'Finite two-epoch engineering horizon; no convergence, production matched-schedule or accuracy inference'],
        source_binding=source_binding)


def create_selector(policy):
    return base_policy(EXPANSION.parent.parent).create_selector(policy)
