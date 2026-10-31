"""Stdlib AST/hash verification only. Never imports a prepared model/helper."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def require(value, message):
    if not value: raise ValueError(message)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(path): return ast.parse(path.read_text(), filename=str(path))


def function(module, name):
    return next(value for value in module.body if isinstance(value, ast.FunctionDef) and value.name == name)


def main():
    policy = json.loads((HERE/'INITIALIZATION_TEMPLATE_DISABLED.json').read_text())
    release = json.loads((HERE/'RELEASE_TEMPLATE_DISABLED.json').read_text())
    pins = json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    require(policy['execution_enabled'] is False and release['enabled'] is False, 'Inactive template/release')
    require(policy['candidate_direction_count'] == 8 and policy['signs_per_direction'] == 2
            and policy['candidate_trial_native_forwards'] == 16 and policy['screen_native_forwards'] == 17, 'Exact8 pairs/17 screen paths')
    require(policy['warmup_epochs'] == 100 and policy['continuation_epochs'] == 400 and policy['final_epoch'] == 500,
            'One100 plus four400')
    require(policy['step'] == .01 and policy['relative_nll_budget'] == .001 and policy['nll_reference_floor'] == .01
            and policy['visibility_floor'] == 1e-6 and policy['vector_reference_floor'] == 1., 'Sealed screening constants')
    require(policy['arms'] == ['identity','random_admissible','live_action_rank','classifier_response_rank'], 'Four controlled choices')
    for row in pins['files']:
        path = PHASE/row['path']
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Pinned source changed: '+row['path'])
    core = PHASE/policy['base_core']
    original, derived = tree(core/'shared_fit.py'), tree(HERE/'shared_fit_initialized.py')
    for name in ('optimizer','train_update','evaluate'):
        require(ast.dump(function(original,name),include_attributes=False) == ast.dump(function(derived,name),include_attributes=False),
                'Unchanged original V2 function: '+name)
    helper = tree(HERE/'live_action_init.py')
    for name in ('warm_once','candidate_screen','restore_branch','authorize'):
        fn = function(helper,name)
        defaults = dict(zip([value.arg for value in fn.args.args][-len(fn.args.defaults):],fn.args.defaults))
        require(isinstance(defaults['execute'],ast.Constant) and defaults['execute'].value is False, 'Default inactive: '+name)
    imports = [alias.name for node in ast.walk(helper) if isinstance(node,(ast.Import,ast.ImportFrom)) for alias in node.names]
    require(not any(name.startswith(('torch','numpy','scipy','sklearn')) for name in imports), 'No numerical imports')
    require(not any(isinstance(node,ast.Attribute) and node.attr == 'L' for node in ast.walk(helper)), 'No detached .L read')
    screen = function(helper,'candidate_screen')
    data_keys = [node.slice.value for node in ast.walk(screen) if isinstance(node,ast.Subscript)
                 and isinstance(node.value,ast.Name) and node.value.id == 'data' and isinstance(node.slice,ast.Constant)]
    require(set(data_keys) <= {'x','train_index','train_y'}, 'Screen uses only x/all TRAIN, no VALID/TEST')
    text = (HERE/'live_action_init.py').read_text()
    require("register_forward_hook(self._builder)" in text and "register_forward_hook(self._value)" in text
            and "self.first_sparse = output[0]" in text and "forward.__globals__['torch_sparse']" in text,
            'Exact normalized builder/actual first-right-value hooks/provider')
    require("value.requires_grad_(False)" in text and "value not in opt.state" in text
            and "core_fit.train_update" in text and "core_fit.evaluate" in text, 'Identity-frozen original warmup and no private moments')
    fit = function(derived,'fit')
    calls = [node for node in ast.walk(fit) if isinstance(node,ast.Call)]
    require(any(isinstance(node.func,ast.Name) and node.func.id == 'range'
                and [getattr(value,'value',None) for value in node.args] == [101,501] for node in calls), 'Exactly400 continuation loop')
    require(not any(isinstance(node,ast.Break) for node in ast.walk(fit)), 'No patience truncation')
    require("initializer_effect_selected" in (HERE/'shared_fit_initialized.py').read_text(), 'Preinit-selected limitation retained')
    native = PHASE/'private_sheaf_native_source_design_20261009_v1/source_custody/nsd/models'
    disc, builders = tree(native/'disc_models.py'), tree(native/'laplacian_builders.py')
    general = next(node for node in disc.body if isinstance(node,ast.ClassDef) and node.name == 'DiscreteGeneralSheafDiffusion')
    forward = next(node for node in general.body if isinstance(node,ast.FunctionDef) and node.name == 'forward')
    require(any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr == 'laplacian_builder'
                for node in ast.walk(forward)), 'Author obtains exact builder output')
    require(any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr == 'spmm'
                and isinstance(node.args[-1],ast.Name) and node.args[-1].id == 'x' for node in ast.walk(forward)),
            'Author passes actual transformed x to sparse multiplication')
    builder_base = next(node for node in builders.body if isinstance(node,ast.ClassDef) and node.name == 'LaplacianBuilder')
    require(any(isinstance(node,ast.Attribute) and node.attr == 'Module' for node in builder_base.bases), 'Original builder supports module hooks')
    for path in HERE.glob('*.py'): ast.parse(path.read_text(),filename=str(path))
    result = dict(status='static_source_only_pass', unchanged_core_functions=['optimizer','train_update','evaluate'],
                  pinned_sources_unchanged=True, exact8_direction_count=True, inactive=True,
                  exact_operator_interception_source_feasible=True, numerical_capture_qualification=False,
                  model_array_metric_heldout_or_research_host_access=False, execution_release=False)
    (HERE/'STATIC_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
