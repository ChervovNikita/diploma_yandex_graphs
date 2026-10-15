"""Selected-state inference using existing native routes and restored readouts.

No training callbacks, role aliases, model rewriting or label-conditioned
aggregation. Imports no tensor library until the caller supplies torch.
"""
import csv
import gzip
import hashlib
import math
from pathlib import Path
from .contracts import inside, readout_policy, require, verify


def tensor_sha(value):
    return hashlib.sha256(memoryview(value.detach().cpu().contiguous().numpy()).cast('B')).hexdigest()


def pool(torch, adapter, job, members):
    policy = readout_policy(job['procedure'], job['family'])
    require(len(members) == policy['members'], 'Wrong number of actual native route outputs')
    mean = torch.stack(members).mean(0)
    if policy['kind'] != 'restored_gate':
        return mean, mean, {'kind': policy['kind'], 'gate': None}
    leaf = policy['gate_leaf'].format(architecture=job['recipe'].get('architecture', '').upper())
    roots = [adapter.model]
    native = getattr(adapter.session, 'native', None)
    if native is not None:
        roots += [getattr(native, 'model', None), getattr(native, 'core', None)]
    found = {}
    for model in roots:
        if model is not None:
            for name, module in model.named_modules():
                if name.rsplit('.', 1)[-1] == leaf:
                    found[id(module)] = module
    require(len(found) == 1, 'Exactly one restored declared aggregator required; no mean fallback: ' + leaf)
    gate = next(iter(found.values()))
    # Original link/molecular readout functions convert to the native head's
    # dtype/device. Gates with no parameters retain the route's native type.
    parameter = next(gate.parameters(), None)
    reference = parameter if parameter is not None else next(adapter.model.parameters())
    output = gate([row.to(reference) for row in members])['served_raw_logits']
    return output, mean.to(output), {'kind': 'restored_gate', 'gate': leaf}


def metrics(torch, family, dataset, logits, labels, *, positive=None):
    def checked(values, loss):
        require(all(math.isfinite(float(v)) for v in values.values()) and math.isfinite(float(loss)),
                'Native complete-role evaluator/loss returned nonfinite values')
        return {k: float(v) for k, v in values.items()}, float(loss)
    if family == 'node':
        return checked({'accuracy': float(logits.argmax(-1).eq(labels).float().mean())},
                       torch.nn.functional.cross_entropy(logits, labels))
    if family == 'ogb':
        from ogb.graphproppred import Evaluator
        result = Evaluator(name=dataset).eval({'y_true': labels.numpy(), 'y_pred': logits.numpy()})
        observed = labels == labels
        require(observed.any().item(), 'No observed molecular target support')
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float()[observed], labels.float()[observed])
        return checked(result, loss)
    from ogb.linkproppred import Evaluator
    require(type(positive) is int and 0 < positive < len(logits), 'Complete positive/negative link population required')
    evaluator = Evaluator(name='ogbl-collab'); evaluator.K = 50
    result = evaluator.eval({'y_pred_pos': logits[:positive], 'y_pred_neg': logits[positive:]})
    loss = -torch.nn.functional.logsigmoid(logits[:positive]).mean() \
        - torch.nn.functional.logsigmoid(-logits[positive:]).mean()
    return checked(result, loss)


def node(torch, adapter, job, role):
    session = adapter.session
    mask = session.data.val_mask if role == 'VALID' else session.data.test_mask
    require(mask.ndim == 1 and mask.dtype == torch.bool, 'One exact source role mask required')
    ids = torch.nonzero(mask, as_tuple=False).flatten()
    require(len(ids) > 0 and not (mask & session.data.train_mask).any().item(), 'Native heldout mask invalid')
    if job['procedure'] == 'native_single':
        members = [session.model(session.data)[ids]]
    else:
        # Existing _routes keeps boundary/input adapters and route-private
        # contexts. No new backbone or direct native-forward bypass.
        members = [route.raw_logits[ids] for route in session._routes(False)]
    served, mean, readout = pool(torch, adapter, job, members)
    return {'logits': served.detach().cpu(), 'mean': mean.detach().cpu(),
            'members': torch.stack(members).detach().cpu(), 'labels': session.data.y[ids].detach().cpu(),
            'ids': ids.detach().cpu(), 'readout': readout,
            'work': {'native_route_forwards': len(members), 'transductive_graph_changed': False}}


class MoleculeRoleView:
    """The original _RoleView access/collation law, with explicit final role."""
    def __init__(self, dataset, ids, torch):
        self.dataset, self.ids, self.torch = dataset, ids.detach().cpu().clone(), torch

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, position):
        graph_id = int(self.ids[int(position)])
        data = self.dataset[graph_id].clone()
        data.smoke_graph_id = self.torch.tensor([graph_id], dtype=self.torch.long)
        return data


def molecular(torch, adapter, job, role, contract, root):
    from torch_geometric.loader import DataLoader
    session = adapter.session
    if role == 'VALID':
        loader = session.valid_loader
    else:
        path = verify(contract['test_indices'], root)
        opener = gzip.open if path.name.endswith('.gz') else open
        with opener(path, 'rt', newline='') as stream:
            values = [int(row[0]) for row in csv.reader(stream)]
        require(len(values) == len(set(values)) > 0, 'Official TEST graph IDs must be unique')
        ids = torch.tensor(values, dtype=torch.long)
        require(not torch.isin(ids, session.train_ids.cpu()).any().item()
                and not torch.isin(ids, session.valid_ids.cpu()).any().item(), 'Official scaffold TEST overlaps TRAIN/VALID')
        view = MoleculeRoleView(session.valid_view.dataset, ids, torch)
        loader = DataLoader(view, batch_size=32, shuffle=False, num_workers=0)
    outputs, means, members_out, labels, identities = [], [], [], [], []
    skipped, batches, readout = [], 0, None
    for batch in loader:
        batch = batch.to(session.device)
        if batch.x.shape[0] == 1:  # Exact native main_pyg.eval skip law.
            skipped.extend(batch.smoke_graph_id.detach().cpu().tolist())
            continue
        rows = [session.model(batch)] if job['procedure'] == 'native_single' else \
            [session.model.run_member(member, batch)[1] for member in range(4)]
        served, mean, readout = pool(torch, adapter, job, rows)
        outputs.append(served.detach().cpu()); means.append(mean.detach().cpu())
        members_out.append(torch.stack(rows).detach().cpu())
        labels.append(batch.y.view(served.shape).detach().cpu())
        identities.append(batch.smoke_graph_id.detach().cpu()); batches += 1
    require(outputs, 'No complete eligible native molecular inference output')
    return {'logits': torch.cat(outputs), 'mean': torch.cat(means), 'members': torch.cat(members_out, dim=1),
            'labels': torch.cat(labels), 'ids': torch.cat(identities), 'readout': readout,
            'work': {'native_route_forwards': batches * len(rows), 'native_batches': batches,
                     'native_single_node_skipped_graph_ids': skipped}}


def link(torch, adapter, job, role, contract, root, helper, cache_root):
    architecture = job['recipe']['architecture'].upper()
    session = adapter.session
    # Current exact factory restores nested session extensions. Existing helper
    # calls native _route/predictor, preserving input/private-BN/NCNC task scope.
    require(architecture == contract['architecture'], 'Link role belongs to another native architecture')
    data = contract['data_files']
    role_pin = data['valid' if role == 'VALID' else 'test']
    split = torch.load(verify(role_pin, root), map_location='cpu', weights_only=False)
    positive = 60084 if role == 'VALID' else contract['population']['positive']
    negative = 100000 if role == 'VALID' else contract['population']['negative']
    for key, count in (('edge', positive), ('edge_neg', negative)):
        require(torch.is_tensor(split[key]) and split[key].dtype == torch.long
                and tuple(split[key].shape) == (count, 2)
                and ((split[key] >= 0) & (split[key] < 235868)).all().item(), 'Whole official query schema/order required')
    if architecture != 'BUDDY':
        native = session.native
        before = native.task_call_scope.capture() if native.task_call_scope else None
        _, rows, work = helper._ncn_predict(torch, session, split, role)
        if native.task_call_scope:
            work['NCNC_output_routing'] = native.task_call_scope.routing_since(before)
    else:
        integration = helper._load(helper.PHASE / 'benchmark_expansion_20261015_v1/link_native_integration_v1/session.py',
                                   '_final_scoring_native_link_helpers')
        request = {'role': role, 'topology': contract['topology'] if role == 'TEST' else 'weighted_year2007_TRAIN',
                   'data_files': data, 'query_population': {'positive': positive, 'negative': negative},
                   'cache_directory': str(inside(Path(cache_root) / role, root))}
        context = helper._buddy_dataset(torch, integration, session, job, split, request, root)
        _, rows, work = helper._buddy_predict(torch, session, context[0], context[1])
        work['source_role_context'] = context[2]
    served, mean, readout = pool(torch, adapter, job, list(rows))
    return {'logits': served.detach().cpu(), 'mean': mean.detach().cpu(), 'members': rows.detach().cpu(),
            'labels': torch.cat((torch.ones(positive, dtype=torch.long), torch.zeros(negative, dtype=torch.long))),
            'ids': torch.arange(positive + negative), 'query_endpoints': torch.cat((split['edge'], split['edge_neg'])),
            'positive_records': positive, 'readout': readout, 'work': work}


def predict(torch, adapter, job, role, contract, root, *, link_helper=None, cache_root=None):
    require(role in ('VALID', 'TEST'), 'Explicit inference role required')
    adapter.model.eval()
    with torch.no_grad():
        if job['family'] == 'node':
            return node(torch, adapter, job, role)
        if job['family'] == 'ogb':
            return molecular(torch, adapter, job, role, contract, root)
        require(link_helper is not None and cache_root is not None, 'Native link role helper/cache authority missing')
        return link(torch, adapter, job, role, contract, root, link_helper, cache_root)
