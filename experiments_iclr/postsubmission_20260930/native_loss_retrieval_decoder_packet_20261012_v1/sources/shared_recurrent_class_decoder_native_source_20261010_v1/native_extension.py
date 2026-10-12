"""Thin extension of the existing native Family; stdlib-only module import.

No nn.Module wrapper, owner, launcher, retry loop or second fit lifecycle.
family_class() derives bounded edits from the pinned original source. Runtime
construction belongs to root after admission; this module's CLI is inactive.
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
ARMS = ('ordinary_M1_graphP_commonB', 'factorized_allmap_M1_graphP_commonB',
        'factorized_allmap_genuine_I4_graphP_ownB', 'joint_untied4_graphP_commonB',
        'shared4_graphP_commonB', 'shared4_graphP_privateB4', 'shared4_Pidentity_commonB')
BANK_ARMS = ARMS[3:]


def _statement(text):
    return ast.parse(text).body[0]


def _edit_statements(function, replacements):
    """Exact statement edits; refuse a changed native integration site."""
    pending = {ast.dump(_statement(old), include_attributes=False): ast.parse(new).body
               for old, new in replacements}
    counts = dict.fromkeys(pending, 0)
    class Edit(ast.NodeTransformer):
        def visit(self, node):
            if isinstance(node, ast.stmt):
                key = ast.dump(node, include_attributes=False)
                if key in pending:
                    counts[key] += 1
                    return copy.deepcopy(pending[key])
            return super().visit(node)
    Edit().visit(function)
    if any(count != 1 for count in counts.values()):
        raise ValueError('Exact unique native source integration statements required')


def adapted_tree(source):
    """Preserve native fit/restore/selection/run mechanics with scoped grafts."""
    tree = ast.parse(source)
    family = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Family')
    methods = {n.name: n for n in family.body if isinstance(n, ast.FunctionDef)}
    construction = copy.deepcopy(methods['make'])
    construction.name = 'construct_native'
    stop = next(i for i, n in enumerate(construction.body)
                if ast.dump(n, include_attributes=False) == ast.dump(
                    _statement('model = wrapper if members > 1 else body'), include_attributes=False))
    construction.body = construction.body[:stop] + ast.parse('return body, wrapper').body
    family.body.append(construction)
    fit = methods['fit']
    train_branch = next(n for n in ast.walk(fit) if isinstance(n, ast.If)
                        and isinstance(n.test, ast.Name) and n.test.id == 'bank')
    _edit_statements(fit, [(ast.unparse(train_branch),
        'train_loss = self.train_update(model, optimizer, streams, bank, members)'),
        ('selected_logits = selected_logits.cpu()', '''
native_restored = self.metrics(self._last_native_logits)
selected_native_logits = self._last_native_logits.cpu()
selected_logits = selected_logits.cpu()
'''), ('write_json(folder / "RESULT.json", result)', '''
result.update(self.fit_metadata(model, members, step))
result['native_at_decoded_selected'] = native_restored
write_json(folder / 'RESULT.json', result)
'''), ('return selected_logits, result', 'return selected_logits, result, selected_native_logits')])
    for node in ast.walk(fit):
        if isinstance(node, ast.Constant) and node.value == 'train_own_ce':
            node.value = 'train_native_decoded_ce'
    run = methods['run']
    _edit_statements(run, [
        ('bank = arm in ARMS[4:]', 'bank = arm in BANK_ARMS'),
        ('factorized = arm.startswith("factorized") or bank', 'factorized = arm != ARMS[0]'),
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
result['native_at_decoded_selected'] = native_metrics
result['semantics'] = self.arm_metadata(arm)
write_json(folder / 'RESULT.json', result)
''')])
    for node in ast.walk(run):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'savez':
            node.keywords.extend(ast.parse('f(native_logits=native_logits.numpy(), native_probability_mean=native_pool.numpy(), native_member_errors=native_member_errors.numpy(), native_pooled_errors=native_pooled_errors.numpy())').body[0].value.keywords)
        if isinstance(node, ast.Subscript) and ast.dump(node, include_attributes=False) == ast.dump(ast.parse('ARMS[:6]').body[0].value, include_attributes=False):
            node.value = ast.parse('ARMS').body[0].value
            node.slice = ast.Slice()
        if isinstance(node, ast.Dict):
            keys = [k.value if isinstance(k, ast.Constant) else None for k in node.keys]
            if 'expected_fit_units' in keys:
                node.values[keys.index('expected_fit_units')] = ast.Constant(30)
                node.keys.extend([ast.Constant('native_body_fits'), ast.Constant('declared_acquisition_arms'),
                                  ast.Constant('decoder_operation_counts'), ast.Constant('decoder_graph_preparation_once')])
                node.values.extend([ast.parse('body_fits').body[0].value, ast.parse('list(ARMS)').body[0].value,
                                    ast.parse('decoder_counts').body[0].value, ast.parse('self.decoder_graph').body[0].value])
    complete_write = next(n for n in run.body if isinstance(n, ast.Expr) and
                         isinstance(n.value, ast.Call) and any(isinstance(a, ast.BinOp) and
                         isinstance(a.right, ast.Constant) and a.right.value == 'COMPLETE_FAMILY.json'
                         for a in n.value.args))
    index = run.body.index(complete_write)
    run.body[index:index] = ast.parse("""
body_fits = sum(u['native_bodies_fitted'] for r in results for u in r['fits'])
decoder_counts = {key: sum(u['decoder_operation_counts'][key] for r in results for u in r['fits'])
                  for key in results[0]['fits'][0]['decoder_operation_counts']}
if len(results) != 21 or self.completed_units != 30 or body_fits != 39:
    raise RuntimeError('Full seven-arm three-seed roster did not complete')
""").body
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == 'ARMS' for t in n.targets))
                 and not (isinstance(n, ast.FunctionDef) and n.name == 'main')
                 and not (isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'")]
    return ast.fix_missing_locations(tree)


def family_class():
    """Return the extended existing Family; no model/data construction here."""
    raw = BASE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError('Pinned existing native Family source changed')
    namespace = {'__file__': str(BASE), '__name__': '_recurrent_existing_native_family',
                 'ARMS': ARMS, 'BANK_ARMS': BANK_ARMS}
    exec(compile(adapted_tree(raw.decode()), str(BASE) + ':recurrent-extension', 'exec'), namespace)
    BaseFamily = namespace['Family']
    spec = importlib.util.spec_from_file_location('_stateless_class_decoder', HERE / 'decoder.py')
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    write_json = namespace['write_json']

    class Family(BaseFamily):
        def __init__(self, cfg, output, config_sha256):
            if any(cfg.get(k, v) != v for k, v in {'depth': 2, 'hidden': 128,
                   'max_updates': 1000, 'patience_updates': 300, 'learning_rate': .001,
                   'dropout': .2, 'member_seed_stride': 1000003,
                   'factor_seed_offset': 2000003, 'dropout_seed_offset': 3000007}.items()):
                raise ValueError('Fixed native recipe and existing route seed conventions')
            super().__init__(cfg, output, config_sha256)
            torch = self.torch
            self.synchronize()
            start = time.perf_counter()
            edges, nodes = self.data.edge_index, len(self.data.x)
            if edges.numel() and (edges.min() < 0 or edges.max() >= nodes):
                raise ValueError('Decoder graph IDs outside full native node order')
            nonself = edges[0] != edges[1]
            identity = torch.arange(nodes, device=self.device)
            source = torch.cat((edges[0, nonself], identity))
            target = torch.cat((edges[1, nonself], identity))
            degree = torch.bincount(target, minlength=nodes).to(self.data.x.dtype)
            self.decoder_support = (source, target, degree.reciprocal())
            self.synchronize()
            self.decoder_graph = {'normalization': 'incoming D^-1(A_off+I)',
                'factual_offdiagonal_multiplicity': 'retained', 'identity_records': nodes,
                'replaced_factual_self_records': int((~nonself).sum().item()),
                'decoder_records': len(source), 'native_graph_unchanged': True,
                'preparation_seconds': time.perf_counter() - start,
                'support_tensor_bytes': sum(t.numel() * t.element_size() for t in self.decoder_support)}
            self.source_hashes.update({str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (Path(__file__), HERE / 'decoder.py')})
            write_json(output / 'SOURCE_HASHES.json', {'config_sha256': config_sha256, 'files': self.source_hashes})
            write_json(output / 'DECODER_GRAPH.json', self.decoder_graph)

        def make(self, seed, member, kind, factorized, members):
            torch, arm = self.torch, self.current['arm']
            if arm == 'joint_untied4_graphP_commonB':
                bodies = [self.construct_native(seed, row, 'baseline', True, 1)[0]
                          for row in range(4)]
                model = torch.nn.ModuleList(bodies)
            else:
                body, wrapper = self.construct_native(seed, member, 'baseline', factorized, members)
                model = wrapper if members > 1 else body
            shape = (members, 10, 10) if arm == 'shared4_graphP_privateB4' else (10, 10)
            model.register_parameter('class_decoder_B', torch.nn.Parameter(
                torch.zeros(shape, dtype=torch.float32, device='cpu')))
            model.to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.get('learning_rate', .001), weight_decay=0)
            expected = {id(p) for p in model.parameters() if p.requires_grad}
            actual = [id(p) for group in optimizer.param_groups for p in group['params']]
            if len(actual) != len(set(actual)) or set(actual) != expected:
                raise ValueError('Every native/factor/decoder parameter exactly once before AdamW')
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
            torch = self.torch
            if self.current['arm'] == 'joint_untied4_graphP_commonB':
                rows = []
                for row, body in enumerate(model):
                    with self.scope(streams, row):
                        rows.append(body(self.data, self.data.x))
                native = torch.stack(rows)
            elif bank:
                native = model(self.batch, route_scope=lambda row: self.scope(streams, row))[0]
            else:
                with self.scope(streams, 0):
                    native = model(self.data, self.data.x).unsqueeze(0)
            support = None if self.current['arm'] == 'shared4_Pidentity_commonB' else self.decoder_support
            scores, _ = decoder.decode(torch, native, model.class_decoder_B, support=support)
            return native, scores

        def train_update(self, model, optimizer, streams, bank, members):
            model.train()
            optimizer.zero_grad()
            native, scores = self.predictions(model, streams, bank)
            labels = self.data.y[self.train_ids].repeat(members)
            loss = .5 * self.F.cross_entropy(native[:, self.train_ids].flatten(0, 1), labels) + \
                   .5 * self.F.cross_entropy(scores[:, self.train_ids].flatten(0, 1), labels)
            if not self.torch.isfinite(loss).item():
                raise FloatingPointError('Nonfinite native/decoded TRAIN loss')
            loss.backward()
            optimizer.step()
            return loss.item()

        def logits(self, model, streams, bank, ids):
            native, scores = self.predictions(model, streams, bank)
            self._last_native_logits = native[:, ids]
            return scores[:, ids]

        def metrics(self, logits):
            if not self.torch.isfinite(logits).all().item():
                raise FloatingPointError('Nonfinite evaluation scores')
            return super().metrics(logits)

        def arm_metadata(self, arm):
            return {'training': 'joint untied four bodies/common B' if arm == ARMS[3] else
                    'four separately initialized/fitted/selected native+own-B models' if arm == ARMS[2] else
                    'joint shared-native-weight bank' if arm in ARMS[4:] else 'single complete native model',
                    'genuine_independent': arm == ARMS[2],
                    'factorized_native_maps': arm != ARMS[0],
                    'decoder_B': 'four private matrices' if arm == ARMS[5] else
                    'own matrix per independent fit' if arm == ARMS[2] else 'one common matrix per fit',
                    'decoder_P': 'I' if arm == ARMS[6] else 'incoming D^-1(A_off+I)',
                    'selection': 'each member own decoded VALID accuracy' if arm == ARMS[2] else 'whole-role decoded VALID pool accuracy',
                    'raw_logits_archive_authority': 'five-step decoded scores; native_logits is separate at same selected state'}

        def fit_metadata(self, model, members, updates):
            passes = 5 * members * (2 * updates + 1)
            graph = self.current['arm'] != ARMS[6]
            return {'semantics': self.arm_metadata(self.current['arm']),
                    'native_bodies_fitted': 4 if self.current['arm'] == ARMS[3] else 1,
                    'decoder_parameters': model.class_decoder_B.numel(),
                    'optimizer_policy': 'one AdamW group, lr=.001, weight_decay=0, native+factors+B; joint own-route loss mean',
                    'decoder_graph': self.decoder_graph,
                    'decoder_operation_counts': {'route_updates': passes, 'spatial_row_mean_passes': passes if graph else 0,
                    'edge_class_values': passes * self.decoder_graph['decoder_records'] * 10 if graph else 0,
                    'dense_class_MACs': passes * len(self.data.x) * 100}}

    return Family


if __name__ == '__main__':
    raise SystemExit('Inactive source packet: root has not admitted a scientific launch')
