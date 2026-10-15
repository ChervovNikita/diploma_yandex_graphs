"""Two exact finite engineering cases; native policy/configuration stays pinned."""
from __future__ import annotations
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
EXPANSION=HERE.parents[1]
BASE_MANIFEST_SHA='2cc662050bd964b4fcc0b46923e3bf00cf31e1ee4f48e02107db7b79bbc28936'
BASE_POLICY_SHA='732eaf774347b0c1eb48d30f436a7315af4d8cc32bd3b7b43685facfcb818fbc'
FINITE_POLICY={'family':'link','track':'bounded_engineering','maximum_epochs':2,'minimum_epochs':2,
    'patience':1,'metric':'hits@50','direction':'max','ties':'first','evaluate_TRAIN_each_epoch':False,
    'evaluation_mode':'VALID_only_disclosed_adaptation'}


def base_policy(root):
    root=Path(root).resolve();folder=EXPANSION/'link_policy_v1'
    if not folder.is_relative_to(root):raise ValueError('Bind native policy inside actual repository')
    manifest=folder/'MANIFEST.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=BASE_MANIFEST_SHA:raise ValueError('Native policy manifest changed')
    for row in json.loads(manifest.read_text())['files']:
        p=(folder/row['path']).resolve()
        if not p.is_relative_to(folder):raise ValueError('Native policy source escapes package')
        b=p.read_bytes()
        if len(b)!=row['bytes']or hashlib.sha256(b).hexdigest()!=row['sha256']:raise ValueError('Native policy source changed')
    source=folder/'policy.py'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=BASE_POLICY_SHA:raise ValueError('Native policy bytes changed')
    name='_runtime_v8_combined_original_link_policy_'+BASE_POLICY_SHA
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def create_policy(job,root=None):
    family=job.get('recipe',{}).get('architecture')
    specs=json.loads((HERE/'CASES.json').read_text())['cases']
    case=specs.get(family)
    if case is None or job.get('family')!='link'or job.get('engineering_only')is not True or job.get('TEST_access')is not False \
            or job.get('policy')!=FINITE_POLICY or job.get('procedure')!=case['method']or job.get('seed')!=case['seed'] \
            or job.get('device')!='cuda:0'or job.get('split')!=0 or job.get('body')!=case['engineering_body'] \
            or job.get('recipe',{}).get('native_fixed_config')!=case['native_fixed_config']:
        raise ValueError('Only the exact frozen NCN/BUDDY two-epoch case is admitted')
    root=Path(root or job['recipe']['_repository_root']).resolve();module=base_policy(root)
    binding=module.source_bindings(root)
    cell=next(row for row in binding['cells']if row['native_family']==family)
    track=cell['tracks']['long_convergence'];stopping=track['stopping']
    parent_job=copy.deepcopy(job)
    parent_job['policy']={**FINITE_POLICY,'track':'matched_extended','maximum_epochs':track['maximum'],
        'minimum_epochs':stopping['minimum_units'],'patience':stopping['patience_validation_checks']}
    parent=module.create_policy(parent_job,root)
    source_binding=copy.deepcopy(parent.source_binding)
    source_binding['engineering_parent_policy']={'manifest_sha256':BASE_MANIFEST_SHA,'policy_sha256':BASE_POLICY_SHA}
    source_binding['engineering_case_spec_sha256']=hashlib.sha256((HERE/'CASES.json').read_bytes()).hexdigest()
    return replace(parent,schema='bounded-native-link-v8-combined-policy-v1',track='bounded_engineering',
        maximum_epochs=2,minimum_epochs=2,patience=1,source_role='finite_native_lifecycle_engineering',
        source_protocol_equivalent=False,source_RNG_trajectory_equivalent=False,
        fidelity_limitations=[*parent.fidelity_limitations,'Two-epoch engineering horizon; no convergence or accuracy inference'],
        source_binding=source_binding)


def create_selector(policy):return base_policy(EXPANSION.parent.parent).create_selector(policy)
