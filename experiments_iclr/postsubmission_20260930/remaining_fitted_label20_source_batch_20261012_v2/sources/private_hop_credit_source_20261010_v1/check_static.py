"""AST/hash checks only. Never import the adapter, Torch or scientific modules."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def tree(relative):
    return ast.parse((ROOT / relative).read_text())


def function(module, name):
    matches = [n for n in ast.walk(module) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    assert len(matches) == 1, name
    return matches[0]


def calls(node, attribute):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attribute]


adapter = ast.parse((HERE / 'adapter.py').read_text())
checks = {}
checks['adapter_AST_parses'] = True
flag = next(n for n in adapter.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ENABLED' for t in n.targets))
assert ast.literal_eval(flag.value) is False
checks['numerical_admission_disabled'] = True
assert not any(isinstance(n, ast.Import) for n in adapter.body)
assert {n.module for n in adapter.body if isinstance(n, ast.ImportFrom)} <= {'contextlib', 'types'}
checks['module_import_is_stdlib_only'] = True
gradient = function(adapter, '_gradient')
grad_call = calls(gradient, 'grad')
assert len(grad_call) == 1
settings = {k.arg: ast.literal_eval(k.value) for k in grad_call[0].keywords}
assert settings == dict(create_graph=False, retain_graph=False, allow_unused=False)
checks['first_order_autograd_only'] = True
gather = function(adapter, 'gather')
commit = function(adapter, 'commit')
assert not calls(gather, 'step') and len(calls(commit, 'step')) == 1
assert len(calls(function(adapter, 'step'), 'gather')) == 1
checks['gather_precedes_one_native_transition'] = True
independent = function(adapter, 'independent_factor1_step')
assert calls(independent, 'gather')[0].lineno < calls(independent, 'commit')[0].lineno
assert 'identities[a] & identities[b]' in ast.unparse(independent)
checks['I4_disjoint_and_unscaled_M1_path_present'] = True
assert 'scale = 1 / self.members' in ast.unparse(gather)
assert 'scale = (1 + LAMBDA / 4) / (1 + LAMBDA)' in ast.unparse(gather)
checks['M1_allview_full_loss_derivative_reused_unscaled'] = True
assert '1 + self.counters[\'updates\'] % 3' in ast.unparse(gather)
assert 'expected_aux = 3' in ast.unparse(gather)
checks['corrected_nonfull_common_cycle_and_exact_path_guard'] = True
assert 'factual_states' in ast.unparse(gather)
assert 'FactorLinear' in ast.unparse(function(adapter, '_guard_plain_state'))
assert 'if value_changed:' in ast.unparse(function(adapter, '_guard_plain_state'))
checks['RNG_and_plain_state_guard_present_no_unchanged_bias_copy'] = True
source = ROOT / 'coordinate_source_independent_review_v1/strong_backbones_v1/sources/polyformer_code/node_classification'
model = ast.parse((source / 'mymodels.py').read_text())
block = ast.parse((source / 'layers/PolyFormerBlock.py').read_text())
utils = ast.parse((source / 'utils.py').read_text())
mono = function(utils, 'mono_base')
assert any(ast.unparse(n) == 'list_mat.append(x)' for n in calls(mono, 'append'))
assert len(calls(mono, 'spmm')) == 1 and len(calls(mono, 'append')) == 2
model_forward = next(n for n in ast.walk(model) if isinstance(n, ast.FunctionDef) and n.name == 'forward')
text = ast.unparse(model_forward)
assert 'data.list_mat' in text and 'torch.stack(input_mat, dim=1)' in text
assert 'self.lin1(input_mat)' in text and 'torch.sum(input_mat, dim=1)' in text
assert not any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'data' and n.attr == 'x' for n in ast.walk(model_forward))
checks['native_mono_ordered_three_slot_interface_no_rawX_bypass'] = True
assert not calls(model, 'register_buffer') and not calls(block, 'register_buffer')
assert len(calls(block, 'LayerNorm')) == 2
assert 'self.bias = torch.tensor(' in ast.unparse(block)
assert 'self.bias_scale = nn.Parameter(' in ast.unparse(block)
checks['native_LayerNorm_no_registered_buffers_plain_PolyAttn_bias'] = True
factor_tree = tree('learnable_internal_be_contrastive_multitask_suite_20261007_v1/factors.py')
assert function(factor_tree, 'member_context').body[-1].finalbody
checks['existing_factor_context_restores_member'] = True
protocol = json.loads((HERE / 'PROTOCOL.json').read_text())
assert protocol['enabled'] is False
for condition in ('private_missinghop', 'full_aux', 'common_nonfull', 'allblock_missinghop'):
    row = protocol['conditions'][condition]
    assert row['paths_per_update'] == row['gradient_calls'] == 7 and row['Adam_transitions'] == 1
checks['paired_M4_protocol_counters_exact7'] = True
checks['no_numerical_import_model_data_or_checkpoint_execution'] = True
bindings = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
for row in bindings['files']:
    path = ROOT / row['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], row['path']
checks['selected_pinned_source_hashes_match'] = True
result = dict(schema='private-hop-source-static-checks-v1', passed=all(checks.values()), checks=checks,
              qualification='AST/hash only; numerical semantics, runtime, gradients and quality untested')
(HERE / 'STATIC_CHECKS.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, indent=2, sort_keys=True))
