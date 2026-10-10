"""Stdlib-only source/JSON/hash checks; never imports any candidate module."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def verify():
    checked = []
    def check(value, name):
        if not value:
            raise AssertionError(name)
        checked.append(name)
    trees = {}
    for path in sorted(HERE.glob('*.py')):
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        compile(tree, str(path), 'exec')  # Compile only; no exec or module import.
        trees[path.name] = tree
        forbidden = ('torch', 'numpy', 'scipy', 'torch_geometric')
        for node in tree.body:
            if isinstance(node, ast.Import):
                check(all(n.name.split('.')[0] not in forbidden for n in node.names), path.name+': no numerical top-level import')
            if isinstance(node, ast.ImportFrom):
                check((node.module or '').split('.')[0] not in forbidden, path.name+': no numerical top-level import')
    check(len(trees) == 12, 'twelve source Python files including this stdlib checker')
    caps = next(n for n in trees['caps.py'].body if isinstance(n, ast.ClassDef) and n.name == 'Caps')
    defaults = {n.target.id:ast.literal_eval(n.value) for n in caps.body if isinstance(n, ast.AnnAssign)}
    check(all(defaults[k] is False for k in ('source_bound','model','data','runtime','scientific')), 'all five execution capabilities default false')
    check(defaults['root_review_sha256'] == '', 'no default root approval digest')
    values = {}
    for node in trees['plan.py'].body:
        if isinstance(node, ast.Assign) and not isinstance(node.value, ast.Call):
            if isinstance(node.targets[0], ast.Tuple):
                for target,value in zip(node.targets[0].elts, ast.literal_eval(node.value)):
                    values[target.id] = value
            elif isinstance(node.targets[0], ast.Name):
                values[node.targets[0].id] = ast.literal_eval(node.value)
    conditions = ('own_floor','private_cmcl','all_block_cmcl','vanilla_cmcl','private_uniform','private_constant_credit')
    check(values['CONDITIONS'] == conditions, 'six fixed attribution conditions')
    check(values['SEEDS'] == (9101,9203,9307) and values['SPLIT_SEED'] == 190111, 'three fixed seeds and predeclared split')
    check((values['MEMBERS'],values['CLASSES'],values['OWNER_K'],values['BETA'],values['LAMBDA']) == (4,3,3,.75,1.), 'M4 categorical C3 K3 beta.75 lambda1')
    check((values['MAXIMUM_EPOCHS'],values['PATIENCE']) == (2000,250), 'full2000/250 horizon')
    # Literal affine dimension arithmetic for native recipe; no model execution.
    stem = [(500,256,True),(256,256,True),(256,3,True)]
    one_block = 3*[(256,512,True),(512,256,True)]+[(256,256,False)]*2+[(256,32,True),(32,256,True)]
    sites = stem+2*one_block
    dense = sum(a*b+(b if bias else 0) for a,b,bias in sites)
    original = dense+2*(2*2*256+8*3)  # two LayerNorms and bias_scale per block.
    private = 4*sum(a+b for a,b,_ in sites)
    check((len(sites),original,private,original+private) == (23,2069875,55772,2125647), 'native all23 sites and symbolic full parameter arithmetic')
    source = {p.name:p.read_text() for p in HERE.glob('*.py')}
    objective = source['objective.py']
    check('kl = -math.log(CLASSES)-logp.mean(dim=-1)' in objective and 'ces-BETA*kls' in objective, 'exact uniform-to-predictive KL and complete CMCL score')
    check('stable=True' in objective and 'mean()/MEMBERS' in objective and 'torch.where(owner, ce, BETA*kl).mean()' in objective, 'stable stopped K3 assignment, own1/M and member-sum auxiliary')
    check('OWNER_K/MEMBERS*ce.mean()' in objective and '(MEMBERS-OWNER_K)/MEMBERS*kl.mean()' in objective, 'gradient-mass and uniform controls present')
    session = source['session.py']
    check('source_gate(identity, caps)' in session and session.index('source_gate(identity, caps)') < session.index('import torch'), 'source/review gate precedes numerical session import')
    check('torch.autograd.grad(floor, self.parameters, retain_graph=True, allow_unused=False)' in session and 'torch.autograd.grad(aux, self.private, retain_graph=False, allow_unused=False)' in session, 'private extra VJP and complete own gradient')
    check('gradient_stamp(self.parameters) == before' in session and '!= member].count_nonzero().item() == 0' in session, 'extra VJP grad-field and cross-member-zero guards')
    check(session.count('self.optimizer.step()') == 1 and session.index('self.optimizer.step()') > session.index('for member in range(MEMBERS):', session.index('losses = []')), 'one simultaneous Adam commit after member gradient collection')
    check('atol=1e-5, rtol=1e-5' in session and 'outgoing, after[member]' in session and 'before_global' in session and 'self.cache_guard' in session, 'same-state tolerant dropout replay and exact RNG/input guards')
    check('torch.logsumexp(logp, dim=0)-math.log(MEMBERS)' in session, 'stable complete arithmetic probability pool')
    fit = source['fit.py']
    check("scores['VALID']['accuracy'] > best" in fit and 'best, bad, selected_epoch, history = -1.' in fit and 'if bad == PATIENCE:' in fit, 'strict complete VALID selector and first finite checkpoint')
    check('weights_only=True' in fit and "prediction_changes'] == 0" in fit and "max_abs_member_logit'] <= .001" in fit, 'weights-only fresh selected reconstruction and output tolerance')
    check('19717, 500' in source['data_interface.py'] and '11828 <= len(train.ids) <= 11830' in source['data_interface.py'] and '3941 <= len(validation.ids) <= 3943' in source['data_interface.py'], 'full factual public input and complete isolated role bounds')
    check('sum(p.numel() for p in private) == 55772' in source['native.py'] and 'sites == 23' in source['native.py'], 'unshrunk runtime parameter/site guards')
    bindings = json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    for row in [*bindings['dependencies'].values(),*bindings['ancestry']]:
        path = (PHASE/row['path']).resolve(strict=True)
        check(path.is_relative_to(PHASE) and path.stat().st_size == row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], 'pinned phase file: '+row['path'])
    for path in sorted(HERE.glob('*.json')):
        json.loads(path.read_text())
    protocol = json.loads((HERE/'PROTOCOL.json').read_text())
    check(protocol['status'] == 'DISABLED_SOURCE_ONLY' and protocol['scientific_admission'] is False, 'no source scientific admission')
    check(protocol['split']['native_split_amendment'] is True and protocol['selector']['initial_vs_author_zero_amendment'] is True, 'native recipe amendments explicitly disclosed')
    runtime = json.loads((HERE/'RUNTIME_TEMPLATE_DISABLED.json').read_text())
    check(runtime['execution_enabled'] is False and all(v is False for v in runtime['all_capabilities'].values()), 'runtime template closed')
    correction = json.loads((HERE/'CANONICAL_ALIAS_CORRECTION.json').read_text())
    check(correction['new_distinct_paper_credit'] == correction['new_full_paper_credit'] == 0 and correction['original_scout_seal_preserved'] is True, 'correct repeated CMCL paper credit without altering predecessor')
    record = dict(schema='private-CMCL-static-checks-v1',status='PASS',checks=checked,python_sources=len(trees),
                  checker='stdlib AST/source/JSON/hash only',source_code_compiled_but_not_executed=True,
                  candidate_modules_imported=False,model_provider_tensor_calls=False,
                  numerical_correctness_or_execution_authorization=False)
    (HERE/'STATIC_CHECKS.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    return record


if __name__ == '__main__':
    value = verify()
    print(json.dumps(dict(status=value['status'],checks=len(value['checks']),python_sources=value['python_sources'],model_provider_tensor_calls=False)))
