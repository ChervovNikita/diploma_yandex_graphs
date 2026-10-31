"""Exact SeHGNN source custody and private module loading; imports stdlib only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PINS={
 'model':('sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2/model.py','0948239c4c4d06dd258cd106fd2fa7b6aaa2a62688fe70dd662413fa1eae532a'),
 'native_helpers':('sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2/native_helpers.py','c60e9de1b38b314c93c3ed33cbd6ba79125fcaf7cc6c792af3ceac78b41b54cd'),
 'engine':('sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2/engine.py','716aacf8ba70832f47a35854757e9a4b97d5eadad0095a33dcfa02c851f66a36'),
 'state_helpers':('sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2/state_helpers.py','00059ffe54571f59bb17084c64e0dcd3f50d3b4f2b31de785e4306beb989b512'),
 'role_loader':('imdb_role_isolated_public_schema_backbone_preparation_20261009_v1/role_loader.py','3b7a739df6fdf39b96c66984c5924a471adc19642573e3620523f3d7ccbbd8ba'),
 'views':('sehgnn_literal_raw_family_source_views_source_20261009_v1/native_family_views.py','e97bde3e8d1be96d4f864e723ef46dc5718ccde5c9552d3f0b76fd8f6df0f9e2'),
 'bank':('sehgnn_grouped_bank_training_callback_source_20261009_v2/bank_training.py','7bfd302e8ef4858f714f6db56d78c0aab6f1634ca5fc6c0d1da0f472abb5a443'),
 'factors':('sehgnn_grouped_member_factor_adapter_source_20261009_v1/grouped_member_factors.py','7d5f899645361212a64fe70ab0b06fd74cc5cfcf8d2ca361e833b884cadd5e47'),
 'supply':('source_supply_private_gradient_helper_20261009_v2/source_supply.py','302d8b4fa102ffd4ca89166f65f411302590d643e4e75bf732f5f8397178edb8'),
 'policy':('sehgnn_paired_six_arm_source_supply_pilot_source_20261009_v2/control_policy.py','defe2360e3375c11a7b84a120e04779651104bc5a58a16be4150b2c25a423353'),
 'joint_contracts':('joint_source_additive_sehgnn_training_source_20261009_v2/joint_contracts.py','0581f4abe141fe3db2aeee8f8f57b18d11ceca771066ea5a07ddd629f798ec28'),
 'joint_additive':('joint_source_additive_sehgnn_training_source_20261009_v2/joint_additive_adapter.py','6e2ad01d0d10b52a8d36e642ae9cfa02d5d082af71a8f9e962a9a58e22eb6799'),
 'joint_training':('joint_source_additive_sehgnn_training_source_20261009_v2/joint_training.py','727e20ccc27eb366291807cd34bd61c3ff06ae6f5896749a6c3764285a071083'),
 'diagnostics':('sehgnn_paired_six_arm_source_supply_pilot_source_20261009_v2/selected_diagnostics.py','b76c6a319cb21c71ffa7ef4a3e39b1ea6ef79ab580e8e188bb02505fc2acca5d'),
 'original_rsb':('common_wrapper_native_source_20261010_v1/models.py','07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303')}
METADATA_PATHS={
 'protocol':'sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2/PROTOCOL.json',
 'schema':'imdb_role_isolated_public_schema_backbone_preparation_20261009_v1/SOURCE_EXPECTATIONS.json',
 'bank_sources':'sehgnn_grouped_bank_training_callback_source_20261009_v2/SOURCE_BINDINGS.json',
 'joint_sources':'joint_source_additive_sehgnn_training_source_20261009_v2/SOURCE_BINDINGS.json'}
RAW_FILES={
 'node.dat':(185065301,'863bfb675c978ba6caaf208a747eb4cdc72ecefa992bd198d3baf267728fb726'),
 'link.dat':(1256872,'ee417344ed4486b65c416c12a0ca8848fda3b99f90d372df61fcbe0f881cba3c'),
 'label.dat':(36277,'51c78600da7d1f8152b707b2e000939d1deb814d99efc7c05dec5bb88924ba9c')}


def receipt(root,path):
    root=Path(root).resolve();path=Path(path)
    path=path if path.is_absolute() else root/path
    if path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('Confined exact source/input file required')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return {'path':str(path.resolve().relative_to(root)),'bytes':path.stat().st_size,'sha256':h.hexdigest()}


def bound(root,row):
    path=Path(root)/row['path']
    if receipt(root,path)!=row:raise ValueError('Exact SeHGNN source/input bytes changed: '+row['path'])
    return path


def load(root,row,tag):
    path=bound(root,row);name='gnnm_sehgnn_newdata_'+tag+'_'+row['sha256'][:16]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
        try:spec.loader.exec_module(m)
        except BaseException:del sys.modules[name];raise
    return sys.modules[name]


def bind_source(root):
    rows={k:receipt(root,p) for k,(p,h) in PINS.items()}
    if any(h is not None and rows[k]['sha256']!=h for k,(p,h) in PINS.items()):raise ValueError('Pinned original SeHGNN source changed')
    metadata={k:receipt(root,p) for k,p in METADATA_PATHS.items()}
    provider=receipt(root,'sehgnn_typed_CPU_provider_check_20261012_v1/PROVIDER_MANIFEST.json')
    if provider['sha256']!='201da757261ef04f4d1f70771ba6d42a613bab824bce04642f667db9d0b7a509':raise ValueError('Passed isolated CPU provider metadata changed')
    return {'modules':rows,'metadata':metadata,'provider_manifest':provider,'old_fitted_state_or_outcomes_bound':False,'historical_authorization':'unresolved; not inherited','new_scientific_GO':False}


def native_runtime(root,sources,device):
    manifest=json.loads(bound(root,sources['provider_manifest']).read_text())
    if manifest['provider']!='dgl' or manifest['version']!='1.1.3':raise ValueError('Exact qualified CPU operator provider required')
    provider=Path(root)/manifest['provider_root']
    for row in manifest['files']:bound(root,row)
    if not provider.is_dir():raise ValueError('Pinned project-local CPU DGL provider must be synchronized')
    sys.path.insert(0,str(provider))
    import torch
    import numpy as np
    import dgl
    if dgl.__version__!='1.1.3' or not Path(dgl.__file__).resolve().is_relative_to(provider):raise ValueError('Source helper imported a different DGL provider')
    from torch_sparse import SparseTensor,remove_diag
    states=load(root,sources['modules']['state_helpers'],'states')
    # The original engine imports its state helpers by this exact alias.
    prior=sys.modules.get('state_helpers');sys.modules['state_helpers']=states
    try:engine=load(root,sources['modules']['engine'],'engine')
    finally:
        if prior is None:sys.modules.pop('state_helpers',None)
        else:sys.modules['state_helpers']=prior
    model=load(root,sources['modules']['model'],'model');native=load(root,sources['modules']['native_helpers'],'native')
    protocol=json.loads(bound(root,sources['metadata']['protocol']).read_text())
    return {'torch':torch,'numpy':np,'dgl':dgl,'SparseTensor':SparseTensor,'remove_diag':remove_diag,'device':torch.device(device),
        'model_module':model,'model_class':model.SeHGNN,'native':native,'protocol':protocol},engine
