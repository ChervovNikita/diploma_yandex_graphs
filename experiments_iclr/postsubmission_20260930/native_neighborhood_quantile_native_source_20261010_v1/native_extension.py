"""Inactive five-arm distribution correction atop the pinned native Family.

Stdlib-only import, ordinary Torch/autograd at root-authorized construction.
Bounded fit/run grafts reuse the existing lifecycle; no new model wrapper,
owner, launcher, sampling rule or retry loop. CLI refuses scientific launch.
"""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'common_wrapper_native_backbone_full_family_source_20261010_v1/run_family.py'
BASE_SHA256 = 'def9e67b42e618f3b14fb98e616c145706e367d4347d3ce9d6251e390c9402c4'
CONTRACT = HERE.parent / 'native_neighborhood_distribution_source_contract_20261010_v1/CONTRACT.json'
CONTRACT_SHA256 = '8df938530129183a89bcfbc647da4d7fc1e2abef0a01c1e90aebd387b6e3fe88'
ARMS = ('shared4_private_quartile', 'shared4_private_moments',
        'shared4_common_direction_quartile', 'factorized_M1_all_signature_quartile',
        'factorized_genuine_I4_all_signature_quartile')


def _statement(source):
    return ast.parse(source).body[0]


def _edit(function, replacements):
    targets = {ast.dump(_statement(old), include_attributes=False): ast.parse(new).body
               for old, new in replacements}
    counts = dict.fromkeys(targets, 0)
    class Edit(ast.NodeTransformer):
        def visit(self, node):
            if isinstance(node, ast.stmt):
                key = ast.dump(node, include_attributes=False)
                if key in targets:
                    counts[key] += 1
                    return copy.deepcopy(targets[key])
            return super().visit(node)
    Edit().visit(function)
    if any(value != 1 for value in counts.values()):
        raise ValueError('Exact unique native lifecycle integration sites required')


def adapted_tree(source):
    tree = ast.parse(source)
    family = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Family')
    methods = {n.name: n for n in family.body if isinstance(n, ast.FunctionDef)}
    construct = copy.deepcopy(methods['make'])
    construct.name = 'construct_native'
    stop = next(i for i, n in enumerate(construct.body) if ast.dump(n, include_attributes=False)
                == ast.dump(_statement('model = wrapper if members > 1 else body'), include_attributes=False))
    construct.body = construct.body[:stop] + ast.parse('return body, wrapper').body
    family.body.append(construct)
    fit = methods['fit']
    branch = next(n for n in ast.walk(fit) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Name) and n.test.id == 'bank')
    _edit(fit, [(ast.unparse(branch), 'train_loss = self.train_update(model, optimizer, streams, bank, members)'),
        ('selected_logits = selected_logits.cpu()', '''
native_restored = self.metrics(self._last_native_logits)
selected_native_logits = self._last_native_logits.cpu()
selected_logits = selected_logits.cpu()
'''), ('write_json(folder / "RESULT.json", result)', '''
result.update(self.fit_metadata(model, members, step))
result['native_at_corrected_selected'] = native_restored
write_json(folder / 'RESULT.json', result)
'''), ('return selected_logits, result', 'return selected_logits, result, selected_native_logits')])
    for n in ast.walk(fit):
        if isinstance(n, ast.Constant) and n.value == 'train_own_ce':
            n.value = 'train_native_corrected_ce'
    run = methods['run']
    _edit(run, [('bank = arm in ARMS[4:]', 'bank = arm in ARMS[:3]'),
        ('factorized = arm.startswith("factorized") or bank', 'factorized = True'),
        ('kind = "exchange" if arm == "exchange" else "separable" if arm == "separable_equal_size" else "baseline"', 'kind = "baseline"'),
        ('pool_seconds = time.perf_counter() - pool_start', '''
native_logits = torch.cat([x[2] for x in fits])
native_metrics = self.metrics(native_logits)
native_probs = native_logits.softmax(-1)
native_pool = native_probs.mean(0)
native_member_errors = native_probs.argmax(-1).ne(y)
native_pooled_errors = native_pool.argmax(-1).ne(y)
pool_seconds = time.perf_counter() - pool_start
'''), ('write_json(folder / "RESULT.json", result)', '''
result['native_at_corrected_selected'] = native_metrics
result['semantics'] = self.arm_metadata(arm)
write_json(folder / 'RESULT.json', result)
''')])
    for n in ast.walk(run):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'savez':
            n.keywords.extend(ast.parse('f(member_probabilities=probs.numpy(), native_logits=native_logits.numpy(), native_member_probabilities=native_probs.numpy(), native_probability_mean=native_pool.numpy(), native_member_errors=native_member_errors.numpy(), native_pooled_errors=native_pooled_errors.numpy())').body[0].value.keywords)
        if isinstance(n, ast.Subscript) and ast.dump(n, include_attributes=False) == ast.dump(ast.parse('ARMS[:6]').body[0].value, include_attributes=False):
            n.slice = ast.Slice()
        if isinstance(n, ast.Dict):
            keys = [k.value if isinstance(k, ast.Constant) else None for k in n.keys]
            if 'expected_fit_units' in keys:
                n.values[keys.index('expected_groups')] = ast.Constant(15)
                n.values[keys.index('expected_fit_units')] = ast.Constant(24)
                n.keys.extend([ast.Constant('native_body_fits'), ast.Constant('declared_acquisition_arms'),
                               ast.Constant('distribution_operation_counts'), ast.Constant('descriptor_support_preparation_once')])
                n.values.extend([ast.parse('body_fits').body[0].value, ast.parse('list(ARMS)').body[0].value,
                                 ast.parse('distribution_counts').body[0].value, ast.parse('self.support_record').body[0].value])
    last_write = next(n for n in run.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                     and any(isinstance(a, ast.BinOp) and isinstance(a.right, ast.Constant)
                             and a.right.value == 'COMPLETE_FAMILY.json' for a in n.value.args))
    index = run.body.index(last_write)
    run.body[index:index] = ast.parse("""
body_fits = sum(u['native_bodies_fitted'] for r in results for u in r['fits'])
distribution_counts = {key: sum(u['distribution_operation_counts'][key] for r in results for u in r['fits'])
                       for key in results[0]['fits'][0]['distribution_operation_counts']}
if len(results) != 15 or self.completed_units != 24 or body_fits != 24:
    raise RuntimeError('Full five-arm three-seed roster did not complete')
""").body
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == 'ARMS' for t in n.targets))
                 and not (isinstance(n, ast.FunctionDef) and n.name == 'main')
                 and not (isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'")]
    return ast.fix_missing_locations(tree)


def family_class():
    """Return existing native Family with source-defined five-arm extension."""
    if hashlib.sha256(CONTRACT.read_bytes()).hexdigest() != CONTRACT_SHA256:
        raise ValueError('Sealed method/control semantics changed')
    raw = BASE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError('Pinned common native Family source changed')
    namespace = {'__file__': str(BASE), '__name__': '_quartile_existing_native_family', 'ARMS': ARMS}
    exec(compile(adapted_tree(raw.decode()), str(BASE) + ':distribution-extension', 'exec'), namespace)
    BaseFamily, write_json = namespace['Family'], namespace['write_json']
    spec = importlib.util.spec_from_file_location('_native_projected_quartiles', HERE / 'quartiles.py')
    descriptor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(descriptor)

    class Family(BaseFamily):
        def __init__(self, cfg, output, config_sha256):
            if cfg['backbone'] != 'SAGE' or cfg['seeds'] != [7301, 7403, 7507]:
                raise ValueError('Prospective representative SAGE and all three paired seeds')
            if any(cfg.get(k, v) != v for k, v in {'depth': 2, 'hidden': 128,
                   'max_updates': 1000, 'patience_updates': 300, 'learning_rate': .001,
                   'dropout': .2, 'member_seed_stride': 1000003,
                   'factor_seed_offset': 2000003, 'dropout_seed_offset': 3000007}.items()):
                raise ValueError('Preserve fixed original native recipe and route seed lineage')
            super().__init__(cfg, output, config_sha256)
            torch, edges, nodes = self.torch, self.data.edge_index, len(self.data.x)
            self.synchronize()
            start = time.perf_counter()
            if edges.numel() and (edges.min() < 0 or edges.max() >= nodes):
                raise ValueError('Factual graph IDs outside full native node order')
            nonself = edges[0] != edges[1]
            source, target = edges[0, nonself], edges[1, nonself]
            order = torch.argsort(target, stable=True)
            source = source[order]
            degree = torch.bincount(target, minlength=nodes)
            first = degree.cumsum(0) - degree
            self.degree_buckets = []
            for count in torch.unique(degree[degree > 0], sorted=True).tolist():
                recipients = (degree == count).nonzero(as_tuple=False).flatten()
                positions = first[recipients, None] + torch.arange(count, device=self.device)[None]
                self.degree_buckets.append((count, recipients, source[positions]))
            self.synchronize()
            self.support_record = {'support': 'all factual incoming nonself records', 'duplicates': 'retained',
                'native_graph_unchanged': True, 'tie_order': 'stable original factual edge record order',
                'nonself_records': len(source), 'nonempty_recipients': int((degree > 0).sum().item()),
                'degree_buckets': len(self.degree_buckets), 'maximum_degree': int(degree.max().item()),
                'preparation_seconds': time.perf_counter() - start,
                'bucket_tensor_bytes': sum(t.numel() * t.element_size() for _, ids, src in self.degree_buckets for t in (ids, src))}
            self.source_hashes.update({str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (Path(__file__), HERE / 'quartiles.py', CONTRACT)})
            write_json(output / 'SOURCE_HASHES.json', {'config_sha256': config_sha256, 'files': self.source_hashes})
            write_json(output / 'DESCRIPTOR_SUPPORT.json', self.support_record)

        def make(self, seed, member, kind, factorized, members):
            torch, arm = self.torch, self.current['arm']
            body, wrapper = self.construct_native(seed, member, 'baseline', True, members)
            model = wrapper if members > 1 else body
            all_signature = arm in ARMS[3:]
            directions = 4 if all_signature or arm != ARMS[2] else 1
            body_index = member if arm == ARMS[4] else 0
            rows, direction_seeds = [], []
            for index in range(directions):
                draw_seed = seed + 5000011 + 1000003 * (4 * body_index + index)
                generator = torch.Generator(device='cpu').manual_seed(draw_seed)
                rows.append(torch.randn(128, dtype=torch.float32, device='cpu', generator=generator))
                direction_seeds.append(draw_seed)
            model.register_parameter('dist_U', torch.nn.Parameter(torch.stack(rows)))
            if all_signature:
                head_seed = seed + 6000017 + 1000003 * body_index
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(head_seed)
                    model.dist_residual = torch.nn.Sequential(torch.nn.Linear(141, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))
                    with torch.no_grad():
                        model.dist_residual[2].weight.zero_()
                        model.dist_residual[2].bias.zero_()
            else:
                head_seed = None
                width = 5 if arm == ARMS[1] else 4
                model.register_parameter('dist_V', torch.nn.Parameter(torch.zeros((members, width, 10), dtype=torch.float32, device='cpu')))
            model._distribution_seeds = {'directions': direction_seeds, 'joint_residual_head': head_seed}
            model.to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.get('learning_rate', .001),
                                         betas=(.9, .999), eps=1e-8, weight_decay=0)
            expected = {id(p) for p in model.parameters() if p.requires_grad}
            actual = [id(p) for group in optimizer.param_groups for p in group['params']]
            if len(actual) != len(set(actual)) or set(actual) != expected:
                raise ValueError('Register all native/factor/distribution parameters exactly once before AdamW')
            streams = []
            for row in range(members):
                _, _, dropout_seed = self.seeds(seed, member + row)
                self.set_seed(dropout_seed)
                stream = {'cpu': torch.get_rng_state()}
                if self.device.type == 'cuda':
                    stream['cuda'] = torch.cuda.get_rng_state(self.device)
                streams.append(stream)
            return model, optimizer, streams

        def predictions(self, model, streams, bank):
            torch, arm = self.torch, self.current['arm']
            if bank:
                native, hidden = model(self.batch, route_scope=lambda row: self.scope(streams, row))
            else:
                adapter = self.routes.NativeModelAdapter(torch, model)
                with self.scope(streams, 0):
                    logits, hidden = adapter.native(self.batch)
                native = logits.unsqueeze(0)
            projected = descriptor.project(torch, hidden, model.dist_U)
            fields = descriptor.describe(torch, projected, self.degree_buckets,
                                         kind='moments' if arm == ARMS[1] else 'quartiles')
            if arm in ARMS[3:]:
                quartiles = fields[..., :3].permute(1, 0, 2).reshape(len(self.data.x), 12)
                features = torch.cat((hidden, quartiles, fields[0, :, 3:4]), dim=-1)
                corrected = native + model.dist_residual(features).unsqueeze(0)
            else:
                corrected = descriptor.correct(native, fields, model.dist_V)
            return native, corrected

        def train_update(self, model, optimizer, streams, bank, members):
            model.train()
            optimizer.zero_grad()
            native, corrected = self.predictions(model, streams, bank)
            labels = self.data.y[self.train_ids].repeat(members)
            loss = .5 * self.F.cross_entropy(native[:, self.train_ids].flatten(0, 1), labels) + \
                   .5 * self.F.cross_entropy(corrected[:, self.train_ids].flatten(0, 1), labels)
            if not self.torch.isfinite(loss).item():
                raise FloatingPointError('Nonfinite native/corrected TRAIN loss')
            loss.backward()
            optimizer.step()
            return loss.item()

        def logits(self, model, streams, bank, ids):
            native, corrected = self.predictions(model, streams, bank)
            self._last_native_logits = native[:, ids]
            return corrected[:, ids]

        def metrics(self, logits):
            if not self.torch.isfinite(logits).all().item():
                raise FloatingPointError('Nonfinite selected/evaluation scores')
            return super().metrics(logits)

        def arm_metadata(self, arm):
            return {'training': 'four separately fitted/selected complete all-signature models' if arm == ARMS[4]
                    else 'one complete factorized all-signature M1' if arm == ARMS[3] else 'shared native body, four private routes',
                    'genuine_independent': arm == ARMS[4], 'native_factorization': 'unchanged all-map factors/Rademacher stem',
                    'correction': 'all-signature H+four directions/12Q/degree joint nonlinear residual' if arm in ARMS[3:]
                    else 'private mean/std/min/max/log-degree maps' if arm == ARMS[1] else 'private centered quartile/log-degree maps',
                    'direction_ownership': 'one common raw direction for differing route H' if arm == ARMS[2] else
                    'four own directions per native body' if arm in ARMS[3:] else 'one private raw direction per route',
                    'selection': 'own corrected VALID accuracy per independent member' if arm == ARMS[4] else 'whole-role corrected VALID pool accuracy',
                    'archive_raw_logits': 'corrected scores; native_logits separate at same corrected-selected checkpoint',
                    'stronger_I4_capacity': 'each of four members receives the complete capable four-direction M1 corrector' if arm == ARMS[4] else None}

        def fit_metadata(self, model, members, updates):
            forwards = 2 * updates + 1
            arm = self.current['arm']
            extra = sum(p.numel() for name, p in model.named_parameters() if name.startswith('dist_'))
            head_MACs = (141 * 128 + 128 * 10) if arm in ARMS[3:] else members * (5 if arm == ARMS[1] else 4) * 10
            return {'semantics': self.arm_metadata(arm), 'native_bodies_fitted': 1,
                    'distribution_parameters': extra, 'distribution_seeds': model._distribution_seeds,
                    'optimizer_policy': 'one AdamW group/native+factors+correction; .001 lr, zero decay, no extra multiplier',
                    'descriptor_support': self.support_record,
                    'distribution_operation_counts': {'native_route_descriptor_passes': members * forwards,
                    'projected_fields': 4 * forwards, 'projection_MACs': 4 * len(self.data.x) * 128 * forwards,
                    'edge_scalar_gathers': 4 * self.support_record['nonself_records'] * forwards,
                    'complete_recipient_sort_rows': 4 * self.support_record['nonempty_recipients'] * forwards,
                    'residual_dense_MACs': len(self.data.x) * head_MACs * forwards}}

    return Family


if __name__ == '__main__':
    raise SystemExit('Inactive source packet: root has not admitted scientific execution')
