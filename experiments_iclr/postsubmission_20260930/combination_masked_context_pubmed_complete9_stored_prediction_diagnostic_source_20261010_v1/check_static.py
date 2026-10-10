"""Source/disabled-entry checks only. No numerical function, array or server."""
import ast
import json
import sys
from common import HERE,PHASE,sha,sources
from diagnose import admit


def main():
    binding=sources()
    parsed={f.name:ast.parse(f.read_text(),filename=str(f)) for f in HERE.glob('*.py')}
    for tree in parsed.values():
        for node in tree.body:
            if isinstance(node,ast.Import):assert not any(a.name.split('.')[0] in ('torch','numpy') for a in node.names)
            if isinstance(node,ast.ImportFrom):assert (node.module or '').split('.')[0] not in ('torch','numpy')
    original=ast.parse((PHASE/binding['stage1_manifest']['path']).parent.joinpath('source.py').read_text())
    fp=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fingerprint')
    assert ast.dump(fp(original),include_attributes=False)==ast.dump(fp(parsed['common.py']),include_attributes=False)
    release=HERE/'RELEASE_TEMPLATE_DISABLED.json'
    try:admit(release,sha(release))
    except ValueError as error:assert str(error).startswith('Stored diagnostic remains disabled:')
    else:raise AssertionError('Disabled diagnostic entered hardware/data path')
    source=(HERE/'diagnose.py').read_text()
    assert source.index('terminals={}')<source.index('records={}')<source.index("valid=bind(spec['valid_bundle'])")
    assert 'probabilities=bank.softmax(-1).mean(0)' in source
    assert 'member=bank[:,ids].argmax(-1);pooled=probabilities[ids].argmax(-1)' in source
    assert 'strict_common_rival=(role_logits>target).all(0)' in source
    assert 'if derived!=expected:raise ValueError' in source
    assert source.index('if derived!=expected:raise ValueError')<source.index('banks={identity:')<source.index("write(output/'COUNTS.json'")
    assert 'fixed_shared_route_member_repairs' in source
    for forbidden in ('cross_entropy','log_softmax','float64','Session(','model.forward','optimizer.step','factual_probabilities('):assert forbidden not in source
    assert all(name not in sys.modules for name in ('torch','numpy','scipy','torch_geometric'))
    print(json.dumps(dict(schema='complete9-stored-prediction-source-static-checks-v1',complete=True,
        parsed_python_files=sorted(parsed),original_fingerprint_AST_identical=True,
        disabled_release_rejected_before_hardware_or_data=True,
        all_nine_terminal_first_and_VALID_after_bindings=True,original_full_bank_FP32_pool_source_verified=True,
        exact_signatures_before_any_count_publication=True,strict_common_rival_definition_source_verified=True,
        no_accuracy_NLL_or_model_forward_source=True,numerical_modules_imported=False,
        numerical_functions_arrays_models_or_server_executed=False,current_outcomes_or_TEST_read=False,
        actual_prediction_parity_and_resource_cost_unmeasured=True),indent=2,sort_keys=True))


if __name__=='__main__':main()
