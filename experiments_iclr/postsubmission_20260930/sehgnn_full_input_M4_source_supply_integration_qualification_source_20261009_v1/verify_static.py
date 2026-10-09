"""Stdlib AST/protocol/hash checks; no numerical model/graph/data execution."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def check():
    sources = read(HERE / 'SOURCE_BINDINGS.json')
    for row in sources['files']:
        path = HERE.parent / row['path']
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Unchanged actual reviewed dependency ' + row['path'])
    text = (HERE / 'qualify_integration.py').read_text()
    tree = ast.parse(text)
    numerical = {'torch', 'numpy', 'dgl', 'torch_sparse', 'sklearn'}
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            require(not any(alias.name.split('.')[0] in numerical for alias in node.names), 'No numerical import in public qualifier')
        elif isinstance(node, ast.ImportFrom):
            require((node.module or '').split('.')[0] not in numerical, 'No numerical provider import')
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
            require(name not in {'fit_bank', 'train', 'evaluate', 'evaluator', 'f1_score', 'reconstruct_selected',
                                 'load_dataset', 'check_acc', 'DataLoader', 'autograd_grad', 'grad',
                                 'masked_fill', 'gen_file_for_evaluate'}, 'No extra fit/quality score/VJP/input/source shortcut: ' + name)
            calls.append((name, node))
    require(sum(name == 'train_epoch' for name, _ in calls) == 1
            and sum(name == 'correct' for name, _ in calls) == 1
            and sum(name == 'build_family_views' for name, _ in calls) == 1
            and sum(name == 'restore_selected' for name, _ in calls) == 1
            and sum(name == 'snapshot' for name, _ in calls) == 1, 'One actual own/correction/native-view/selected opportunity')
    loads = [node for name, node in calls if name == 'load' and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch']
    require(len(loads) == 1 and {item.arg: ast.literal_eval(item.value) for item in loads[0].keywords}
            == {'map_location': 'cpu', 'weights_only': True}, 'Safe owned selected checkpoint load')
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    observer = ast.get_source_segment(text, functions['observe_real_correction'])
    require('plan = make(*args, **kwargs)' in observer and 'value = replay(plan, key, output)' in observer
            and 'helper.make_credit_plan, helper.accumulate_replay = make, replay' in observer,
            'Observational helper delegates once and restores original functions')
    require("'delta of actual FP32 helper J accumulation" in text and 'full_Jacobian_inferred=False' in text,
            'Actual weighted-direction scope; no Jacobian claim')
    require('Every actual native prototype slow Parameter' in text and 'Every registered buffer separate across M4' in text
            and 'Actual slow values unchanged by private correction' in text and 'BN objects/values, modes and caller streams' in text,
            'Actual slow identity/value and native state witness')
    require('quality_scoring=False' in text and 'fixed single post-opportunity state; no quality scoring' in text
            and 'output_bitwise_gate=False' in text and 'qualifier_complete_factual_known_role_serving' in text,
            'Factual complete known serving with no quality selector or drift gate')
    protocol = read(HERE / 'PROTOCOL.json')
    release = read(HERE / 'RELEASE.disabled.json')
    require(protocol['assignments'] == release['assignments'] == ['actor', 'director', 'keyword', None]
            and protocol['member_rng_seeds'] == release['member_rng_seeds'] == [1001, 1002, 1003, 1004]
            and protocol['native_seed'] == release['native_seed'] == 1, 'Prospective registered assignment/member streams')
    require(protocol['own']['epochs'] == protocol['own']['actual_deduplicated_Adam_steps'] == protocol['source']['opportunities'] == 1
            and protocol['own']['full_TRAIN_rows'] == 1097 and protocol['source']['references'] == 20
            and protocol['source']['replay_paths'] == 11 and protocol['source']['qualification_retries'] == 0,
            'Real native fixed full opportunity')
    require(protocol['resource_budget'] == release['resource_budget']
            == {'host_RSS_bytes': 34359738368, 'device_bytes': 25769803776, 'seconds': 3600}
            and release['execution_host'] == 'anogena-2', 'Fixed root allocation scope and caps')
    require(release['enabled'] is False and release['root_source_review_approved'] is False
            and release['bank_view_source_review_approved'] is False and release['source_seal_sha256'] is None,
            'Default disabled root-owned template')
    review = read(HERE / 'SOURCE_REVIEW.json')
    for row in review['files']:
        path = HERE.parent / row['path']
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Immutable independent audit binding')
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text())
    for path in HERE.glob('*.json'):
        read(path)
    return dict(status='passed', source_AST_hash_and_protocol_only=True,
        frozen_dependencies_and_actual_qualifier_API_verified=True, no_extra_forward_VJP_or_quality_score=True,
        prospective_one_full_opportunity_and_disabled_release=True, safe_checkpoint_and_native_state_source_checks=True,
        numerical_model_graph_data_provider_or_server_execution=False, actual_integration_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check()
    if args.output:
        require(args.output.resolve().parent == HERE and not args.output.exists(), 'Fresh source-only check receipt')
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
