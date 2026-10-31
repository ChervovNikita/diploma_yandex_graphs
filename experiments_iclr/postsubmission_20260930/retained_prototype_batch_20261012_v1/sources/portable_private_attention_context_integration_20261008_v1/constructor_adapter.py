"""One insertion into the pinned fresh Session constructor; no optimizer retrofit.

The original constructor is compiled in a new namespace/subclass. Its recipe,
native factory, Adam arguments, streams and inherited methods remain unchanged.
The original public module/class and every existing Session remain untouched.
"""
import ast
import copy
from functools import partial
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

PORTABLE_SHA = '29beab567eeb4c070c432e2ce31990c88ec2d8c8618aa5dfb85ebedcf4f0ca14'
HOOK_MANIFEST_SHA = '5cce87e4cad70942fe9fdcf0784acfb40a509abe01907aa3062a2983bbcc0ec1'
HOOK_SHA = 'ffec3e53b59ad43c1343419f5c957a6a89207d08f76d5c0714c8f61d8b9babcb'
SCHEMA = 'private-local-attention-context-integration-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_hook(root):
    root = Path(root).resolve()
    if sha(root/'MANIFEST.json') != HOOK_MANIFEST_SHA:
        raise ValueError('Pinned public private-attention hook required')
    for row in json.loads((root/'MANIFEST.json').read_text())['files']:
        p = root/row['path']
        if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError('Private-attention dependency changed: '+row['path'])
    if sha(root/'private_local_attention.py') != HOOK_SHA:
        raise ValueError('Private-attention code changed')
    spec = importlib.util.spec_from_file_location('_context_private_attention_hook',
                                                root/'private_local_attention.py')
    hook = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = hook
    spec.loader.exec_module(hook)
    return hook


def descriptor(session):
    value = session.private_local_attention.metadata()
    return {'schema':SCHEMA,'constructor_adapter_sha256':sha(__file__),
            'original_portable_sha256':PORTABLE_SHA,'hook_source_sha256':HOOK_SHA,
            'hook_manifest_sha256':HOOK_MANIFEST_SHA,
            'constructor_AST_sha256':session.private_constructor_AST_sha256,
            'initialization':'Copied native local att_src/att_dst; no additional RNG draw',
            'installation':'Fresh constructor before original Adam',
            'global_QKV_BE_changed':False,'native_GAT_forward_changed':False,
            'all_shadow_replay_and_serving_calls_use_member_wrapper':True,
            'old_optimizer_retrofit_supported':False,'model_recipe':session.config['model'],
            'counts':value}


def adapted_public(public, hook_root):
    """Provide a fresh Session class to existing dispatch; imports remain lazy.

    The sole constructor edit is `_install_private_local_attention(self)` before
    its original nested optimizer factory. No original module/class is patched.
    """
    source = Path(public.__file__).resolve()
    if sha(source) != PORTABLE_SHA:
        raise ValueError('Exact original public Session source required')
    tree = ast.parse(source.read_text())
    classes = [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Session']
    if len(classes) != 1:
        raise ValueError('Unique native Session constructor required')
    constructors = [n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name=='__init__']
    if len(constructors) != 1:
        raise ValueError('Unique native Session.__init__ required')
    init = copy.deepcopy(constructors[0])
    positions = [i for i,n in enumerate(init.body)
                 if isinstance(n,ast.FunctionDef) and n.name=='optimizer']
    if len(positions) != 1:
        raise ValueError('Original pre-Adam insertion location changed')
    insertion = ast.parse('_install_private_local_attention(self)').body[0]
    init.body.insert(positions[0],insertion)
    module = ast.fix_missing_locations(ast.Module(body=[init],type_ignores=[]))
    ast_sha = hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest()

    def install(session):
        if session.task != 'wikics' or session.arm != 'be_unit_contrastive':
            raise ValueError('Only original shared unit-factor WikiCS context constructor')
        if session.model.independent or session.model.members != 4 or len(session.model.models)!=1:
            raise ValueError('One shared body with four complete member routes required')
        if hasattr(session,'optimizers'):
            raise ValueError('Private scorers must be installed before Adam construction')
        hook = verified_hook(hook_root)
        session.private_local_attention = hook.install_private_local_attention(
            session.model.models[0].body,4)
        original = session.model.member_forward
        session.model.member_forward = partial(session.private_local_attention.forward_member,original)
        session.private_constructor_AST_sha256 = ast_sha

    namespace = dict(vars(public))
    namespace['_install_private_local_attention'] = install
    exec(compile(module,str(source)+':private-local-attention','exec'),namespace)
    private_session = type('PrivateLocalAttentionSession',(public.Session,),
                           {'__init__':namespace['__init__'],'__module__':__name__})
    return SimpleNamespace(Session=private_session,recipe=public.recipe)


def reconstruct_selected(public, hook_root, state, device, polynormer):
    """Rebuild a trusted selected-state dictionary for serving, not training resume.

    The private banks and member wrapper exist before state loading. Python local/
    global stage is restored explicitly. No target data or optimizer/RNG histories
    are loaded, and no checkpoint is reselected.
    """
    run = state['run']
    if run.get('arm') not in ('public_private_attention_context__shared_common',
                              'public_private_attention_context__shared_route'):
        raise ValueError('This private-context selected-state identity is required')
    if run.get('underlying_session_arm') != 'be_unit_contrastive':
        raise ValueError('Original shared unit-factor context arm required')
    if type(state.get('global')) is not bool:
        raise ValueError('Explicit selected local/global flag required')
    facade = adapted_public(public,hook_root)
    session = facade.Session('wikics','be_unit_contrastive',run['seed'],device,polynormer)
    if state.get('private_local_attention') != descriptor(session):
        raise ValueError('Selected state scorer/source/constructor/count contract differs')
    if state['config']['model'] != session.config['model']:
        raise ValueError('Selected model recipe differs')
    session.model.load_state_dict(state['model'],strict=True)
    session.config = copy.deepcopy(state['config'])
    session.model.set_global(state['global'])
    session.model.eval()
    def serving_only(*args,**kwargs):
        raise RuntimeError('Selected reconstruction is serving-only; no target facade or training resume')
    session.train_step = serving_only
    session.private_selected_reconstruction = {'serving_only':True,
        'selected_global_restored':state['global'],'checkpoint_reselected':False,
        'optimizer_or_RNG_history_loaded':False,'training_resume_supported':False}
    return session
