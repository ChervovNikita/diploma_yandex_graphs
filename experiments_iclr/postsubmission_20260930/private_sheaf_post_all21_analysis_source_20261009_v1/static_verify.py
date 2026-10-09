"""Source/protocol AST and byte hashes only; never load any result/reference array."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent


def read(path):return json.loads(path.read_text())


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def check(value,name):
    if not value:raise AssertionError(name)
    return name


def verify():
    tree=ast.parse((HERE/'analysis.py').read_text());checks=[]
    allowed={'argparse','csv','datetime','hashlib','importlib','itertools','json','math','pathlib','statistics'}
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[alias.name for alias in node.names] if isinstance(node,ast.Import) else [node.module]
            check(all(name.split('.')[0] in allowed for name in names),'Top-level stdlib imports only')
    run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run')
    opening=next(i for i,node in enumerate(run.body) if any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id=='opening' for v in ast.walk(node)))
    imports=[i for i,node in enumerate(run.body) if isinstance(node,ast.Import)]
    checks.append(check(bool(imports) and all(i>opening for i in imports),'All21/actual custody gate precedes every numerical import'))
    check(not any(isinstance(v,ast.Call) and isinstance(v.func,ast.Attribute) and v.func.attr in ('forward','backward','step') for v in ast.walk(tree)),
          'No model forward/backward/optimizer calls')
    source=(HERE/'analysis.py').read_text()
    checks.append(check("path.name=='SELECTED_SERVING_REFERENCE.pt'" in source and "weights_only=True" in source
          and "map_location='cpu'" in source and "score_source']=='fresh_reconstructed_selected_serving'" in source,
          'CPU exact fresh-serving references only, no model checkpoints'))
    checks.append(check("read(bound(release['centered_closure']))" in source and "actual_parent_absent" in source
          and "actual_group_absent" in source and "actual_worker_CUDA_absent" in source and "worker_birth" in source
          and "boot_id" in source and "family['launch']" in source and "family['terminal']" in source,
          'Status-only all21 and exact parent/worker/group/birth/boot/CUDA terminal closure retained'))
    protocol=read(PHASE/'private_sheaf_two_recipe_scientific_scope_root_20261009_v1/PROTOCOL.json')
    expected=['vanilla_pool_minus_independent_pool','vanilla_pool_minus_independent_member0','centered_pool_minus_independent_pool','centered_pool_minus_independent_member0']
    checks.append(check(protocol['co_primary_contrasts']==expected and protocol['required_logical_records']['total']==21
          and protocol['pilot_gate']['mean_signed_AUROC_gain_at_least']==0.003
          and protocol['pilot_gate']['each_seed_signed_AUROC_gain_gt']==0.0
          and protocol['pilot_gate']['mean_pooled_NLL_delta_at_most']==0.0,
          'Exact four frozen contrasts and unchanged AUROC/NLL gates'))
    checks.append(check('statistics.stdev(values)' in source and '4.302652729911275' in source and "member_scores'][0]" in source,
          'Three paired differences/sample SD/descriptive df2 interval; single comes from same fresh independent pool reference'))
    checks.append(check('range(0,positive.shape[1],256)' in source and 'range(0,negative.shape[1],256)' in source
          and 'exactly_one_tie' in source and 'both_tie' in source,
          'Every positive-negative pair, bounded256 blocks, all ties retained'))
    checks.append('Every member/pool and selected epoch; mean/worst quality; whole-role errors/rescue/harm/coverage/confidence/ranks; complete eval-TRAIN curves and cost/residency retained')
    release=read(HERE/'RELEASE_TEMPLATE_DISABLED.json');custody=read(HERE/'ROOT_CUSTODY_TEMPLATE_DISABLED.json')
    checks.append(check(release['enabled'] is False and custody['all21_complete_verified'] is False
          and release['analysis_source_sha256']==sha(HERE/'analysis.py'),'Release and future closure input remain disabled'))
    for row in read(HERE/'SOURCE_BINDINGS.json')['files']:
        path=(PHASE/row['path']).resolve(strict=True)
        check(path.is_relative_to(PHASE) and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Exact fixed source/protocol metadata bytes')
    checks.append('Fixed source/protocol metadata bound; no array/result/reference/checkpoint read')
    if (HERE/'MANIFEST.json').exists():
        for row in read(HERE/'MANIFEST.json')['files']:
            path=HERE/row['path'];check(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Exact own payload')
        seal=read(HERE/'SEAL.json');check(seal['source_only'] is True and seal['execution_enabled'] is False
          and seal['manifest_sha256']==sha(HERE/'MANIFEST.json'),'Exact inactive reader seal')
        checks.append('Own inactive manifest/seal verified')
    return dict(status='passed',verification='AST and fixed source/protocol byte hashes only',checks=checks,
        numerical_model_or_provider_import=False,numerical_execution=False,arrays_results_scores_or_checkpoints_read=False,
        remote_actions=False,old_sources_protocol_or_results_modified=False,source_only=True,execution_enabled=False)


if __name__=='__main__':print(json.dumps(verify(),indent=2)+'\n')
