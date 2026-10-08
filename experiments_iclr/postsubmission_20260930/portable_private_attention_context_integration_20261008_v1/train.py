"""Public full-recipe private-local-attention context CLI; author study inactive.

Run exactly one caller-requested COMMON or ROUTE cell. Reuse the existing context
CLI/replay/selector; consume caller targets without regeneration. This callable
public source does not admit or launch the proposed six-fit author study.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from constructor_adapter import adapted_public,descriptor

HERE = Path(__file__).resolve().parent
CONTEXT_MANIFEST_SHA = '2222beb59a2279602cc72723a6cd924ec6978fbd6a02e46a73db781b31736ed5'
METHODS = ('shared_common','shared_route')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_driver(root):
    root = Path(root).resolve()
    if sha(root/'MANIFEST.json') != CONTEXT_MANIFEST_SHA:
        raise ValueError('Pinned public context training dependency required')
    for row in json.loads((root/'MANIFEST.json').read_text())['files']:
        p = root/row['path']
        if p.stat().st_size != row['bytes'] or sha(p)!=row['sha256']:
            raise ValueError('Context dependency changed: '+row['path'])
    for name in ('context_dispatch','context_recompute','train_targets',
                 'context_positive_masks','context_alignment_objective','session_objectives_adapter'):
        cached = sys.modules.get(name)
        if cached and Path(cached.__file__).resolve().parent != root:
            raise ValueError('Run in a fresh process; a different context module is cached')
    sys.path.insert(0,str(root))
    spec = importlib.util.spec_from_file_location('_original_context_CLI',root/'train.py')
    driver = importlib.util.module_from_spec(spec);sys.modules[spec.name]=driver
    spec.loader.exec_module(driver)
    return driver


def consume_targets(driver,archive,digest,train,torch,device):
    """Caller-supplied logical targets only; no author-only path/hash or remake."""
    import numpy as np
    from train_targets import PreparedTargets,KEYS,content_sha
    from context_positive_masks import sampled_weights
    archive = Path(archive)
    if len(digest)!=64 or sha(archive)!=digest:
        raise ValueError('Explicit caller target archive SHA256 differs')
    with np.load(archive,allow_pickle=False) as source:
        if set(source.files)!=set(KEYS):
            raise ValueError('Exact public prepared-target fields required')
        arrays = {key:source[key].copy() for key in KEYS}
    hashes = {key:content_sha(value) for key,value in arrays.items()}
    for value in arrays.values():value.setflags(write=False)
    metadata = {'source':'Explicit caller immutable TRAIN target archive',
        'consumed_target_archive_sha256':digest,'target_relations_regenerated':False,
        'frozen_array_content_sha256':hashes,
        'VALID_labels_or_model_predictions_accessed_by_consumer':False,
        'caller_target_generation_custody_certified':False,
        'official_source_certified_from_arbitrary_NPZ':False,'no_target_or_seed_retry':True}
    bundle = PreparedTargets(arrays,metadata,hashes)
    bundle.validate_current(train,torch,device)
    route = sampled_weights(arrays['masks'],arrays['panel_rows'],'route')
    common = sampled_weights(arrays['masks'],arrays['panel_rows'],'common')
    permuted = sampled_weights(arrays['permuted_masks'],arrays['panel_rows'],'route')
    tvc = float(.5*np.abs(route-common).sum(-1).mean())
    tvp = float(.5*np.abs(route-permuted).sum(-1).mean())
    if tvc<.05 or tvp<.05:
        raise ValueError('Original fixed target-differentiation rule failed; no retry')
    metadata.update(route_common_mean_target_TV=tvc,route_permuted_mean_target_TV=tvp)
    return bundle


def configure_driver(driver,hook_root,archive,digest):
    """Override only constructor, target consumption and provenance wrappers."""
    original_make,original_snapshot = driver.make_session,driver.joint_snapshot
    original_write = driver.json_write
    live = {}
    driver.METHODS = {method:driver.METHODS[method] for method in METHODS}
    driver.public_identity = lambda method:'public_private_attention_context__'+method
    driver.prepare_targets = lambda train,torch,device:consume_targets(
        driver,archive,digest,train,torch,device)

    def make(public,method,*args,**kwargs):
        if method not in METHODS:raise ValueError('Only the fixed two target policies')
        result = original_make(adapted_public(public,hook_root),method,*args,**kwargs)
        live['session'] = result[0]
        return result

    def snapshot(session,*args,**kwargs):
        return dict(original_snapshot(session,*args,**kwargs),
                    private_local_attention=descriptor(session))

    def write(path,value):
        if Path(path).name in ('RUN.json','COMPLETE.json','PUBLIC_COST.json') and 'session' in live:
            value = dict(value,private_local_attention=descriptor(live['session']),
                private_context_successor_sha256=sha(__file__),
                original_context_manifest_sha256=CONTEXT_MANIFEST_SHA,
                proposed_author_six_fit_family_admitted=False)
        original_write(path,value)

    driver.make_session,driver.joint_snapshot,driver.json_write = make,snapshot,write
    return driver


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    help_requested = '--help' in argv or '-h' in argv
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--context-interface',type=Path,
        default=HERE.parent/'portable_context_steering_public_interface_20261008_v1')
    extra.add_argument('--attention-interface',type=Path,
        default=HERE.parent/'portable_private_local_attention_20261008_v1')
    extra.add_argument('--frozen-targets',type=Path,required=not help_requested)
    extra.add_argument('--frozen-targets-sha256',required=not help_requested)
    options,rest = extra.parse_known_args(argv)
    driver = load_driver(options.context_interface)
    configure_driver(driver,options.attention_interface,options.frozen_targets,
                     options.frozen_targets_sha256)
    if help_requested:
        print(__doc__)
        extra.print_help()
    previous = sys.argv
    try:
        sys.argv = [str(HERE/'train.py')]+rest
        driver.main()
    finally:
        sys.argv = previous


if __name__=='__main__':main()
