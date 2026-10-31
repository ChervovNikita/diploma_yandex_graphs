"""Inactive Q/K operator adapter. Default factory refuses before numerical imports.

Attach only to fresh original Sessions; no data, checkpoint, loop or launch owner.
Native source bytes and FactorLinear remain unchanged. Bias stays outside scales.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import types

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
OPERATORS = ('native_tied', 'active_reversible_exp', 'pre_sigmoid_split', 'full_qk')
KINDS = ('single', 'independent4', 'be_init')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sources():
    seal = json.loads((HERE / 'SEAL.json').read_text())
    require(seal['source_only'] is True and seal['execution_enabled'] is False
            and sha(HERE / 'MANIFEST.json') == seal['manifest_sha256'], 'Exact inactive adapter seal')
    for row in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
        path = (HERE / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes']
                and sha(path) == row['sha256'], 'Changed adapter source payload')
    pins = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
    paths = {}
    for key, row in pins['runtime_sources'].items():
        path = (PHASE / row['path']).resolve(strict=True)
        require(path.is_relative_to(PHASE) and path.stat().st_size == row['bytes']
                and sha(path) == row['sha256'], 'Changed pinned source: ' + key)
        paths[key] = path
    public_root = paths['public_manifest'].parent
    for row in json.loads(paths['public_manifest'].read_text())['files']:
        path = (public_root / row['path']).resolve(strict=True)
        require(path.is_relative_to(public_root) and path.stat().st_size == row['bytes']
                and sha(path) == row['sha256'], 'Changed original public source')
    return paths, public_root, seal['manifest_sha256']


def _global_forward(native_path, torch):
    """Reuse the exact native forward AST; replace only the three Q/K statements."""
    tree = ast.parse(native_path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GlobalAttn')
    forward = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward'))
    loop = next(node for node in forward.body if isinstance(node, ast.For))
    original = ast.parse('''
k = F.sigmoid(self.k_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
if self.qk_shared:
    q = k
else:
    q = F.sigmoid(self.q_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
''').body
    require([ast.dump(node, include_attributes=False) for node in loop.body[1:3]]
            == [ast.dump(node, include_attributes=False) for node in original], 'Exact native Q/K site required')
    replacement = ast.parse('''
q, k = self._operator_qk_pair(x, i)
q = q.view(seq_len, self.hidden_channels, self.heads)
k = k.view(seq_len, self.hidden_channels, self.heads)
''').body
    loop.body[1:3] = replacement
    module = ast.fix_missing_locations(ast.Module(body=[forward], type_ignores=[]))
    namespace = {'F': torch.nn.functional, 'torch': torch}
    exec(compile(module, str(native_path) + ':qk-site-only', 'exec'), namespace)
    return namespace['forward']


def _attach(session, operator, native_path):
    torch = session.torch
    FactorLinear = session.core['factors'].FactorLinear
    require(session.steps == 0 and not session.model.contrastive
            and all(not opt.state for opt in session.optimizers), 'Fresh F-only model and empty native Adams required')
    require(all(p.dtype == torch.float32 and p.grad is None for p in session.model.parameters()), 'Fresh float32 parameters')
    patched = _global_forward(native_path, torch) if operator in ('active_reversible_exp', 'pre_sigmoid_split') else None
    added_scales = 0
    for model in session.model.models:
        global_attn = model.body.global_attn
        require(global_attn.qk_shared is True and not hasattr(global_attn, '_operator_mode')
                and global_attn.hidden_channels == 512 and global_attn.heads == 1
                and global_attn.num_layers == 2 and not hasattr(global_attn, 'q_lins'), 'Fresh pinned tied two-layer global core')
        # Ordinary references receive the same trainable global r/s coordinates.
        # Their W/b are private per model; M=1 adds no inter-member parameter sharing.
        members = 1 if session.model.independent else session.model.members
        for index, lin in enumerate(global_attn.k_lins):
            if isinstance(lin, torch.nn.Linear):
                lin = FactorLinear(lin, members).to(session.device, dtype=torch.float32)
                global_attn.k_lins[index] = lin
                added_scales += lin.r.numel() + lin.s.numel()
            require(isinstance(lin, FactorLinear) and lin.r.shape == (members, 512)
                    and lin.s.shape == (members, 512), 'Pinned FactorLinear affine convention')
            if operator in ('active_reversible_exp', 'pre_sigmoid_split'):
                name = 'gamma' if operator == 'active_reversible_exp' else 'delta'
                lin.register_parameter(name, torch.nn.Parameter(lin.weight.new_zeros((members, 512))))
        if operator == 'full_qk':
            # No constructor/reset draw; each copied query parameter is independent.
            global_attn.q_lins = copy.deepcopy(global_attn.k_lins)
            global_attn.qk_shared = False
            require(all(not {id(p) for p in q.parameters()} & {id(p) for p in k.parameters()}
                        for q, k in zip(global_attn.q_lins, global_attn.k_lins)), 'Full Q/K parameters must be untied')
        if patched is not None:
            def pair(self, x, index):
                lin = self.k_lins[index]
                member = lin.member  # Original member_context sets/restores this.
                a = torch.nn.functional.linear(x * lin.r[member], lin.weight)  # One large projection; no bias.
                s = lin.s[member]
                if self._operator_mode == 'pre_sigmoid_split':
                    q, k = a * (s + lin.delta[member]), a * (s - lin.delta[member])
                    if lin.bias is not None:
                        q, k = q + lin.bias, k + lin.bias
                    return torch.nn.functional.sigmoid(q), torch.nn.functional.sigmoid(k)
                z = a * s
                if lin.bias is not None:
                    z = z + lin.bias
                u = torch.nn.functional.sigmoid(z) * torch.exp(.5 * lin.gamma[member])
                return u, u  # Positive weighted Gram core; active zero-start gamma tangent.
            global_attn._operator_qk_pair = types.MethodType(pair, global_attn)
            global_attn.forward = types.MethodType(patched, global_attn)
        global_attn._operator_mode = operator
    # Register every added coordinate before creating the only usable Adam bank.
    session.optimizers = [torch.optim.Adam(model.parameters(), lr=.001, eps=1e-8, weight_decay=0.)
                          for model in session.model.models]
    owner_ids = [id(p) for opt in session.optimizers for group in opt.param_groups for p in group['params']]
    require(len(owner_ids) == len(set(owner_ids)) and set(owner_ids) == {id(p) for p in session.model.parameters()},
            'Every learned coordinate has exactly one fresh optimizer owner')
    require(len(session.optimizers) == (4 if session.arm == 'independent4' else 1), 'Original ordinary/shared ownership')
    if session.model.independent:
        sets = [{id(p) for p in model.parameters()} for model in session.model.models]
        require(all(not left & right for i, left in enumerate(sets) for right in sets[i+1:]), 'Genuinely private native models')
    return added_scales


def _streamed_f_step(self, batch, labels):
    """Two own-CE views in original view/member order; every reverse precedes Adam."""
    torch = self.torch
    require(not self._operator_failed and labels.device == self.device and len(labels) == 580
            and batch['ids'].numel() == 580, 'Complete TRAIN F update; no failed-step retry')
    self.model.train()
    for optimizer in self.optimizers:
        optimizer.zero_grad(set_to_none=True)
    parameters = tuple(self.model.parameters())
    versions = tuple(parameter._version for parameter in parameters)
    own_mean = labels.new_zeros((), dtype=torch.float32)
    self.operator_work['update_attempts'] += 1
    try:
        for view in range(2):
            for member, stream in enumerate(self.streams):
                # This is the original Session.forward stream boundary and order.
                with torch.random.fork_rng(devices=self.cuda_devices):
                    torch.set_rng_state(stream['cpu'])
                    if self.cuda_index is not None:
                        torch.cuda.set_rng_state(stream['cuda'], self.cuda_index)
                    logits, representation = self.model.member_forward(batch, member)
                    stream['cpu'] = torch.get_rng_state()
                    if self.cuda_index is not None:
                        stream['cuda'] = torch.cuda.get_rng_state(self.cuda_index)
                self.operator_work['member_view_forwards'] += 1
                require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(representation).all()), 'Finite full own view')
                own = torch.nn.functional.cross_entropy(logits, labels)
                require(bool(torch.isfinite(own)), 'Finite own CE')
                own_mean += own.detach() * (.5 / self.model.members)
                scale = .5 if self.model.independent else .5 / self.model.members
                (own * scale).backward()
                self.operator_work['member_view_backwards'] += 1
                del own, logits, representation
        require(tuple(parameter._version for parameter in parameters) == versions, 'All own gradients at one old parameter state')
        active = [parameter for parameter in parameters if parameter.grad is not None]
        require(active and all(bool(torch.isfinite(parameter.grad).all()) for parameter in active), 'Finite accumulated F gradients')
        for optimizer in self.optimizers:
            optimizer.step()
            self.operator_work['Adam_steps'] += 1
        self.core['selection'].finite_state(self.model, self.optimizers)
        self.steps += 1
        self.operator_work['completed_updates'] += 1
        return dict(loss=own_mean * (self.model.members if self.model.independent else 1),
                    own_mean=own_mean, auxiliary=own_mean * 0)
    except BaseException:
        self._operator_failed = True
        raise


def _route(paths, session=None):
    runtime = json.loads(paths['runtime'].read_text())
    require(PHASE == Path(runtime['phase']).resolve(strict=True) and socket.gethostname() == runtime['hostname']
            and Path(sys.executable).resolve() == Path(runtime['python']).resolve()
            and os.environ.get('PYTHONPATH', '').split(os.pathsep) == runtime['PYTHONPATH']
            and os.environ.get('CUDA_VISIBLE_DEVICES') == runtime['GPU_uuid'], 'Existing qualified runtime only')
    if session is not None:
        require(str(session.device) == 'cuda:0' and str(session.torch.__version__) == runtime['torch']
                and session.np.__version__ == runtime['numpy'] and session.torch.cuda.device_count() == 1
                and session.torch.cuda.get_device_name(0) == runtime['GPU_name'], 'Pinned numerical runtime')
    return runtime


def install(session, *, operator, later_execution_authorized=False):
    """The sole installer: fresh original F Session, then all modules and new Adams.

    Keep the original complete driver, local/best restoration and selector. Record
    operator_binding/work in its receipts; install this same mode before loading
    a selected state. This function supplies no loop, launch or source mutation.
    """
    require(later_execution_authorized is True, 'Inactive installer; separate root release required')
    require(operator in OPERATORS and session.task == 'wikics' and session.arm in KINDS, 'Declared F-only operator family')
    paths, public_root, manifest_sha = _sources()
    _route(paths, session)
    rows = {row['path']: row['sha256'] for row in json.loads((public_root / 'MANIFEST.json').read_text())['files']}
    require(all(session.core_provenance[name] == rows['core/' + name]
                for name in ('factors.py', 'models.py', 'objectives.py', 'selection.py'))
            and session.native_provenance['polynormer_model_sha256'] == sha(paths['native'])
            and session.config == json.loads((public_root / 'recipes/wikics.json').read_text()), 'Exact original Session sources/recipe')
    added_scales = _attach(session, operator, paths['native'])
    session._operator_failed = False
    session.operator_work = dict(update_attempts=0, completed_updates=0, member_view_forwards=0,
                                 member_view_backwards=0, Adam_steps=0)
    session.train_step = types.MethodType(_streamed_f_step, session)
    session.operator_binding = dict(operator=operator, kind=session.arm, seed=session.seed, adapter_manifest_sha256=manifest_sha,
        adapter_program_sha256=sha(__file__), native_sha256=sha(paths['native']),
        factor_sha256=session.core_provenance['factors.py'], bias_outside_scale=True,
        ordinary_global_factor_coordinates_added=added_scales,
        operator_delta_or_gamma_scalars=sum(p.numel() for name, p in session.model.named_parameters()
                                           if name.endswith(('.delta', '.gamma'))),
        parameter_count=sum(p.numel() for p in session.model.parameters()), optimizer_count=len(session.optimizers),
        large_qk_projections_per_global_layer_member=2 if operator == 'full_qk' else 1,
        objective='F only: original two own-CE views; shared mean, independent unscaled own gradients',
        F_engine='streamed view then member; no shadow/replay; all old-state backwards before Adam',
        copied_full_qk_start=operator == 'full_qk', numerical_qualification=False, scientific_launch_admitted=False)
    return session


def make_session(*, operator, kind, seed, device='cuda:0', later_execution_authorized=False):
    """Source integration factory, disabled until a separate root release.

    Reuse original RNG streams, F objective and selection; stream the own reverse.
    The caller must record operator_binding in every run/snapshot/endpoint and
    reconstruct this same operator before any state load. No loop/owner is supplied.
    """
    require(later_execution_authorized is True, 'Inactive operator source; separate root release required')
    require(operator in OPERATORS and kind in KINDS and type(seed) is int and 0 <= seed < 2**63
            and device == 'cuda:0', 'Declared ordinary/shared operator cell and qualified CUDA route')
    paths, public_root, _ = _sources()
    _route(paths)
    spec = importlib.util.spec_from_file_location('_inactive_qk_original_public', public_root / 'portable.py')
    public = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = public
    spec.loader.exec_module(public)
    session = public.Session('wikics', kind, seed, device, paths['native'])
    return install(session, operator=operator, later_execution_authorized=True)
