"""Prospective loaded-bank API. Never called during preparation; default closed.

The future controller owns checkpoint loading, family-closure custody, runtime
qualification, deadlines and server-only writes. This module does not train,
select checkpoints, change serving modes or edit any native/provider function.
"""
import hashlib
import json
import time
from pathlib import Path
from interventions import variants, array_digest, NODES

SOURCE_SHA = '76de82781e7fd496a5a3382b3a71a5781ea023dfe3777469cc005a2b19afbfce'
GAT_SHA = 'a7b2353003394ab433f909c0dbd7e5a06b7ae1289939857245ae3062f1eb5901'
MP_SHA = '66e4ef4afa1d1b2d46c805b8987b6ea0c52ec5c95a9beee8a62758358e481e6f'
TEMPLATE_SHA = 'b0201032bf3c5a1978a8ba8f2769465670458e595abdb1f1e4dd36227f3599f9'
ARMS = ('single', 'single_contrastive', 'independent4', 'independent4_contrastive',
        'be_unit', 'be_init', 'be_unit_contrastive', 'be_init_contrastive')
SEEDS = (6101, 6203, 6307)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def bound(phase, row):
    path = (Path(phase) / row['path']).resolve(strict=True)
    if not path.is_relative_to(Path(phase).resolve()) or sha(path) != row['sha256']:
        raise ValueError('Exact phase file custody required')
    return path


def admit(release, plan, phase):
    # No package/model imports or prediction reads precede these release checks.
    for key in ('root_evaluation_only_release', 'runtime_hook_qualification_approved',
                'resource_qualification_approved', 'owned_external_deadline_verified'):
        if release.get(key) is not True: raise ValueError('Panel remains disabled: ' + key)
    if release.get('source_manifest_sha256') != SOURCE_SHA or release.get('TEST_access') is not False:
        raise ValueError('Frozen v4 source; TRAIN-only panel')
    if release.get('automatic_retry') is not False or release.get('arm') not in ARMS or release.get('seed') not in SEEDS:
        raise ValueError('Exact pilot cell with no automatic retry')
    closure = json.loads(bound(phase, release['family_closure']).read_text())
    cells = closure.get('cells', [])
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if closure.get('task') != 'wikics' or len(cells) != 24 or {(x['arm'], x['seed']) for x in cells} != expected:
        raise ValueError('All 24 fixed WikiCS family cells must close first')
    if any(x.get('status') not in ('complete', 'preserved_failure') for x in cells):
        raise ValueError('Complete or preserved-failure family closure')
    cell = next(x for x in cells if (x['arm'], x['seed']) == (release['arm'], release['seed']))
    if cell['status'] != 'complete' or cell.get('checkpoint') != release.get('checkpoint'):
        raise ValueError('Only completed frozen served checkpoint; failed cells are reported without a bank')
    bound(phase, release['checkpoint'])
    if release.get('data_manifest') != plan['data_manifest']:
        raise ValueError('Original exact TRAIN projection authority')
    bound(phase, release['data_manifest'])
    if release.get('config') != plan['config']: raise ValueError('Exact unchanged WikiCS model/training configuration')
    bound(phase, release['config'])
    identity = {'source_manifest_sha256': SOURCE_SHA, 'task': 'wikics', 'arm': release['arm'],
        'seed': release['seed'], 'checkpoint': release['checkpoint'], 'data_manifest': release['data_manifest'],
        'config': release['config'], 'intervention_freeze_sha256': plan['intervention_freeze_sha256'],
        'hostname': 'anogena-2-0', 'physical_GPU_UUID': 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'}
    for key in ('runtime_hook_evidence', 'resource_evidence', 'live_deadline_evidence'):
        receipt = json.loads(bound(phase, release[key]).read_text())
        if receipt.get('identity') != identity or receipt.get('passed') is not True:
            raise ValueError('Exact future panel qualification/supervisor identity: ' + key)
        if key == 'runtime_hook_evidence' and (receipt.get('nonmutating_hook_transparency') is not True or receipt.get('full_first_last_all_member_attention') is not True):
            raise ValueError('Actual hook qualification is still required')
        if key == 'resource_evidence' and (receipt.get('all_six_full_graph_member_passes') is not True or receipt.get('resource_costs_measured') is not True):
            raise ValueError('Actual representative inference workload/cost qualification is still required')
        if key == 'live_deadline_evidence':
            import os
            fields = Path('/proc/' + str(os.getppid()) + '/stat').read_text().rsplit(') ', 1)[1].split()
            if receipt.get('supervisor_pid') != os.getppid() or receipt.get('supervisor_start_ticks') != int(fields[19]) or receipt.get('hard_seconds') != 1800 or receipt.get('active_seconds') != 1790 or receipt.get('cleanup_seconds') != 10:
                raise ValueError('Owned live external hard deadline with cleanup inside cap')
    return closure


def distribution(value):
    import numpy as np
    value = np.asarray(value, dtype=np.float64)
    return {'mean': float(value.mean()), 'median': float(np.median(value)),
            'q10': float(np.quantile(value, .1)), 'q90': float(np.quantile(value, .9)),
            'min': float(value.min()), 'max': float(value.max())}


class AttentionCapture:
    """Read-only full edge-update output; returning None preserves PyG output."""
    def __init__(self, model, edge_index, cpu_edges, torch):
        self.model, self.edge_index, self.edges, self.torch = model, edge_index, cpu_edges, torch
        self.active = None; self.rows = {}; self.handles = []

    def __enter__(self):
        try:
            for body_id, wrapped in enumerate(self.model.models):
                convs = wrapped.body.local_convs
                if len(convs) != 7: raise ValueError('Exact seven-layer native local path')
                for name, layer in (('first', convs[0]), ('last', convs[-1])):
                    if layer.heads != 1 or layer.add_self_loops or layer.flow != 'source_to_target':
                        raise ValueError('Pinned native attention/input-edge semantics')
                    if layer._edge_update_forward_hooks or layer._edge_update_forward_pre_hooks:
                        raise ValueError('Fresh frozen model: no unqualified attention-changing hooks')
                    def capture(module, inputs, output, layer_name=name, bank_body=body_id):
                        if self.active is None or (self.model.independent and self.active != bank_body):
                            raise ValueError('Exact member attribution required')
                        edge = inputs[0]
                        if not self.torch.is_tensor(edge) or not self.torch.equal(edge, self.edge_index):
                            raise ValueError('Full unchanged input edge order at attention hook')
                        if output.shape != (edge.shape[1], 1) or not self.torch.isfinite(output).all():
                            raise ValueError('Full finite single-head attention')
                        key = (self.active, layer_name)
                        if key in self.rows: raise ValueError('One edge-update capture per native local layer/member')
                        a = output.detach().to('cpu', dtype=self.torch.float64).numpy().copy()[:, 0]
                        import numpy as np
                        if (a < 0).any() or (a > 1 + 2e-5).any(): raise ValueError('Attention probability domain')
                        sums = np.bincount(self.edges[1], weights=a, minlength=NODES)
                        if not np.allclose(sums, 1, atol=2e-5, rtol=2e-5):
                            raise ValueError('All public receivers normalized; selfloops retained')
                        self.rows[key] = a
                        return None
                    self.handles.append(layer.register_edge_update_forward_hook(capture))
        except BaseException:
            for handle in self.handles: handle.remove()
            self.handles.clear()
            raise
        return self

    def __exit__(self, *unused):
        for handle in self.handles: handle.remove()
        self.active = None


def forward_bank(model, batch, torch, capture=None):
    rows = []
    for member in range(model.members):
        if capture is not None: capture.active = member
        logits, ignored_representation = model.member_forward(batch, member)
        rows.append(logits)
        del ignored_representation
    logits = torch.stack(rows)
    probabilities = logits.softmax(-1)  # Same float32/device operation as v4 serving.
    pooled = probabilities.mean(0)
    if not torch.isfinite(logits).all() or not torch.isfinite(pooled).all():
        raise ValueError('Finite frozen predictions')
    return {'logits': logits.detach().cpu().numpy().astype('float64'),
            'probabilities': probabilities.detach().cpu().numpy().astype('float64'),
            'pooled': pooled.detach().cpu().numpy().astype('float64')}


def margin(scores, labels):
    import numpy as np
    own = scores[..., np.arange(len(labels)), labels]
    competitors = scores.copy()
    competitors[..., np.arange(len(labels)), labels] = -np.inf
    return own - competitors.max(-1)


def pair_errors(predictions, labels):
    import numpy as np
    rows = []
    wrong = predictions != labels
    for a in range(len(predictions)):
        for b in range(a + 1, len(predictions)):
            union = wrong[a] | wrong[b]; both = wrong[a] & wrong[b]
            rows.append({'members': [a, b], 'disagreement_count': int((predictions[a] != predictions[b]).sum()),
                'both_wrong_count': int(both.sum()), 'both_wrong_same_class_count': int((both & (predictions[a] == predictions[b])).sum()),
                'wrong_union_count': int(union.sum()), 'error_jaccard': float(both.sum() / union.sum()) if union.any() else None})
    return rows


def attention_vectors(capture, baseline_keys):
    import numpy as np
    # Collapse duplicate edge entries only for summaries, never for model input.
    keys, inverse = np.unique(capture.edges[1] * NODES + capture.edges[0], return_inverse=True)
    positions = np.searchsorted(baseline_keys, keys)
    if (positions >= len(baseline_keys)).any() or not np.array_equal(baseline_keys[positions], keys):
        raise ValueError('Each intervention graph is a subset of the frozen baseline graph')
    result = {}
    for key, alpha in capture.rows.items():
        mass = np.bincount(inverse, weights=alpha, minlength=len(keys))
        # Float64 receiver normalization is diagnostic only, after forward.
        sums = np.bincount(keys // NODES, weights=mass, minlength=NODES)
        mass = mass / sums[keys // NODES]
        aligned = np.zeros(len(baseline_keys), dtype=np.float64); aligned[positions] = mass
        result[key] = aligned
    if len(result) != capture.model.members * 2: raise ValueError('All first/last full member attention captures')
    return result


def neighbor_divergence(a, b, receivers, ids):
    import numpy as np
    mean = .5 * (a + b)
    def term(p):
        out = np.zeros_like(p); nz = p > 0
        out[nz] = p[nz] * np.log(p[nz] / mean[nz])
        return out
    js = np.bincount(receivers, weights=.5 * (term(a) + term(b)), minlength=NODES)[ids]
    tv = np.bincount(receivers, weights=.5 * np.abs(a - b), minlength=NODES)[ids]
    return {'JS_nats': js.tolist(), 'TV': tv.tolist()}, {'JS_nats': distribution(js), 'TV': distribution(tv)}


def run_loaded_panel(model, train, release, plan, phase):
    """Future evaluation only. Return compact summaries/rows; perform no I/O writes."""
    closure = admit(release, plan, phase)
    import sys, inspect
    source = Path(phase) / 'learnable_internal_be_contrastive_multitask_suite_20261007_v4'
    sys.path.insert(0, str(source))
    from runtime import allocation, verify_manifest, runtime_versions
    allocation(); runtime_versions()
    if verify_manifest() != SOURCE_SHA or sha(inspect.getfile(type(model))) != plan['model_source_sha256']:
        raise ValueError('Exact frozen v4 ensemble implementation')
    import numpy as np
    import torch
    from torch_geometric.nn import GATConv
    from torch_geometric.nn.conv import MessagePassing
    template = Path(inspect.getfile(GATConv)).parent / 'edge_updater.jinja'
    if sha(inspect.getfile(GATConv)) != GAT_SHA or sha(inspect.getfile(MessagePassing)) != MP_SHA or sha(template) != TEMPLATE_SHA:
        raise ValueError('Bound provider/hook implementation')
    if model.task != 'wikics' or model.arm != release['arm'] or model.members != (1 if model.arm.startswith('single') else 4):
        raise ValueError('Exact frozen served bank')
    if any(module.training for module in model.modules()): raise ValueError('Caller must supply an evaluation-mode frozen bank')
    if any(p.dtype != torch.float32 for p in model.parameters()): raise ValueError('Frozen float32 parameters; no AMP/half conversion')
    for wrapped in model.models:
        body = wrapped.body
        if tuple(body.lin_in.weight.shape) != (512, 300) or body.global_attn.hidden_channels != 512 or len(body.global_attn.h_lins) != 2 or any(conv.out_channels != 512 for conv in body.local_convs):
            raise ValueError('Full native width512, seven local/two global stages')
    modes = [bool(wrapped.body._global) for wrapped in model.models]
    if modes != release['body_global']: raise ValueError('Preserve actual checkpoint serving mode per body')
    ids = train['ids'].detach().cpu().numpy(); labels = train['y'].detach().cpu().numpy()
    x = train['x'].detach().cpu().numpy(); edges = train['edge_index'].detach().cpu().numpy()
    if ids.shape != (580,) or array_digest(ids) != plan['TRAIN_target_ids']['sha256'] or labels.shape != (580,):
        raise ValueError('All original 580 TRAIN targets in original order')
    if labels.dtype != np.int64 or labels.min() < 0 or labels.max() > 9: raise ValueError('TRAIN class domain')
    device = next(model.parameters()).device
    if device.type != 'cuda': raise ValueError('Separately qualified exact serving device')
    versions_before = [(id(t), t._version) for t in list(model.parameters()) + list(model.buffers())]
    baseline_keys = np.unique(edges[1] * NODES + edges[0]); receivers = baseline_keys // NODES
    baseline = None; baseline_attention = None; summaries = {}; server_rows = {}
    started = time.monotonic()
    with torch.inference_mode():
        for name, features, graph, description in variants(x, edges):
            frozen = plan['variants'][name]
            if array_digest(features) != frozen['x']['sha256'] or array_digest(graph) != frozen['edge_index']['sha256']:
                raise ValueError('Common intervention materialization differs')
            batch = {'x': torch.from_numpy(features).to(device), 'edge_index': torch.from_numpy(graph).to(device),
                     'ids': train['ids'].to(device)}
            unhooked = forward_bank(model, batch, torch) if name == 'baseline' else None
            with AttentionCapture(model, batch['edge_index'], graph, torch) as capture:
                result = forward_bank(model, batch, torch, capture)
            if name == 'baseline':
                if not np.allclose(result['logits'], unhooked['logits'], atol=1e-5, rtol=1e-5) or not np.array_equal(result['pooled'].argmax(-1), unhooked['pooled'].argmax(-1)):
                    raise ValueError('Hook transparency check failed; do not interpret panel')
                baseline = result
            attn = attention_vectors(capture, baseline_keys)
            if name == 'baseline': baseline_attention = attn
            zmargin = margin(result['logits'], labels); pmargin = margin(result['probabilities'], labels)
            poolmargin = margin(result['pooled'], labels)
            predictions = result['probabilities'].argmax(-1); poolpred = result['pooled'].argmax(-1)
            basepred = baseline['probabilities'].argmax(-1); basepool = baseline['pooled'].argmax(-1)
            delta_z = zmargin - margin(baseline['logits'], labels)
            delta_p = pmargin - margin(baseline['probabilities'], labels)
            delta_pool = poolmargin - margin(baseline['pooled'], labels)
            row = {'member_logit_margin': zmargin.tolist(), 'member_probability_margin': pmargin.tolist(),
                'member_logit_margin_change': delta_z.tolist(), 'member_probability_margin_change': delta_p.tolist(),
                'pooled_probability_margin': poolmargin.tolist(), 'pooled_probability_margin_change': delta_pool.tolist(),
                'member_top1': predictions.tolist(), 'pooled_top1': poolpred.tolist(),
                'neighbor_attention_between_members': {}, 'neighbor_attention_response_to_baseline': {}}
            summary = {'description': description, 'TRAIN_targets': 580,
                'member_correct_counts': [int((v == labels).sum()) for v in predictions],
                'pooled_correct_count': int((poolpred == labels).sum()),
                'common_all_member_errors_count': int((predictions != labels).all(0).sum()),
                'pool_rescues_some_member_error_count': int(((poolpred == labels) & (predictions != labels).any(0)).sum()),
                'pool_wrong_despite_some_correct_member_count': int(((poolpred != labels) & (predictions == labels).any(0)).sum()),
                'member_flip_counts': [int((v != b).sum()) for v, b in zip(predictions, basepred)],
                'pooled_flip_count': int((poolpred != basepool).sum()), 'pair_errors': pair_errors(predictions, labels),
                'member_margin_change': [distribution(v) for v in delta_p], 'pool_margin_change': distribution(delta_pool),
                'attention_between_members': {}, 'attention_response_to_baseline': {}}
            for layer in ('first', 'last'):
                for a in range(model.members):
                    key = layer + ':' + str(a)
                    compact, stats = neighbor_divergence(baseline_attention[a, layer], attn[a, layer], receivers, ids)
                    row['neighbor_attention_response_to_baseline'][key] = compact
                    summary['attention_response_to_baseline'][key] = stats
                    for b in range(a + 1, model.members):
                        key = layer + ':' + str(a) + ':' + str(b)
                        compact, stats = neighbor_divergence(attn[a, layer], attn[b, layer], receivers, ids)
                        row['neighbor_attention_between_members'][key] = compact
                        summary['attention_between_members'][key] = stats
            summary['pair_functional_response'] = []
            for a in range(model.members):
                for b in range(a + 1, model.members):
                    correlation = None
                    if delta_p[a].std() > 1e-12 and delta_p[b].std() > 1e-12:
                        correlation = float(np.corrcoef(delta_p[a], delta_p[b])[0, 1])
                    summary['pair_functional_response'].append({'members': [a, b],
                        'absolute_probability_margin_response_difference': distribution(np.abs(delta_p[a] - delta_p[b])),
                        'probability_margin_response_correlation': correlation,
                        'intervention_flip_mask_disagreement_count': int(((predictions[a] != basepred[a]) != (predictions[b] != basepred[b])).sum())})
            summary['fixed_TRUE_class_rows'] = []
            for cls in range(10):
                mask = labels == cls
                summary['fixed_TRUE_class_rows'].append({'class': cls, 'count': int(mask.sum()),
                    'member_correct_counts': [int(((v == labels) & mask).sum()) for v in predictions],
                    'pooled_correct_count': int(((poolpred == labels) & mask).sum()),
                    'common_all_member_error_count': int(((predictions != labels).all(0) & mask).sum())})
            row['common_all_member_error_mask'] = (predictions != labels).all(0).tolist()
            server_rows[name] = row; summaries[name] = summary
    if versions_before != [(id(t), t._version) for t in list(model.parameters()) + list(model.buffers())] or modes != [bool(w.body._global) for w in model.models]:
        raise ValueError('Frozen parameter/buffer versions or serving mode changed')
    payload = {'TRAIN_target_ids': ids.tolist(), 'TRAIN_labels': labels.tolist(), 'variants': server_rows}
    if len(json.dumps(payload, allow_nan=False).encode()) > 4 * 1024**2: raise ValueError('Server compact row output bound')
    summary = {'schema': 'wikics-functional-diversity-panel-v1', 'task': 'wikics', 'arm': model.arm,
        'seed': release['seed'], 'checkpoint': release['checkpoint'], 'body_global': modes,
        'data_manifest': release['data_manifest'], 'source_manifest_sha256': SOURCE_SHA,
        'all580_TRAIN_targets': True, 'prediction_selected_subset': False, 'raw_attention_or_logits_written': False,
        'family_closed_cells': len(closure['cells']), 'member_full_graph_forwards': model.members * 6,
        'hooked_member_full_graph_forwards': model.members * 5, 'wall_seconds': time.monotonic() - started,
        'hook_transparency_passed': True, 'parameter_buffer_versions_and_modes_preserved': True,
        'scientific_fits': 0, 'VALID_TEST_read': False, 'variants': summaries}
    return summary, payload  # Caller keeps row payload server-only; mirrors small aggregates after release.
