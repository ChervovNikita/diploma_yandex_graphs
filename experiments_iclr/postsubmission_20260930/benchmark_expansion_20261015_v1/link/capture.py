"""Passive, query-authenticated native TASK prehead capture for root integration."""
from contextlib import contextmanager
from .planning import authenticate, query_keys


class TaskPreheadCapture:
    """One context per original positive/negative/mixed task predictor call.

    NCNC requires its explicit depth-one task gate, including no-grad serving.
    This helper adds no source forward and changes neither input nor output.
    """
    def __init__(self, native_final_leaf, top_level_task_gate):
        if top_level_task_gate is None:
            raise ValueError('Explicit source task-call scope required')
        self.gate = top_level_task_gate
        self.current = None
        self.rows = {member: [] for member in range(4)}
        def capture(module, args):
            if self.current is not None and self.gate():
                hidden = args[0]
                if hidden.ndim != 2 or hidden.shape[0] != len(self.current['keys']):
                    raise ValueError('Native task prehead and query row counts differ')
                self.current['hidden'].append(hidden)
            return None
        self.handle = native_final_leaf.register_forward_pre_hook(capture, prepend=True)

    @contextmanager
    def task(self, member, endpoints, groups, relation_ids=None, role='TRAIN'):
        if self.current is not None or int(member) not in self.rows:
            raise ValueError('Nonoverlapping explicit source task call required')
        keys = query_keys(endpoints, relation_ids)
        group_list = [int(g) for g in groups]
        authenticate(keys, group_list, role)
        row = {'keys': keys, 'groups': group_list, 'hidden': []}
        self.current = row
        try:
            yield
            if len(row['hidden']) != 1:
                raise ValueError('Exactly one top-level final native affine capture required')
            self.rows[int(member)].append(row)
        finally:
            self.current = None

    def aligned(self, torch):
        if self.current is not None:
            raise ValueError('Cannot assemble preheads during a task call')
        features, signatures, reference = [], [], None
        for member in range(4):
            rows = self.rows[member]
            if not rows:
                raise ValueError('Every route must preserve original source task-call order')
            keys = [key for row in rows for key in row['keys']]
            groups = [group for row in rows for group in row['groups']]
            signature = authenticate(keys, groups, 'TRAIN')
            if reference is None:
                reference = (keys, groups)
            elif (keys, groups) != reference:
                raise ValueError('Native task query identities/order differ across routes')
            signatures.append(signature)
            features.append(torch.cat([row['hidden'][0] for row in rows], dim=0))
        return {'preheads': torch.stack(features),
                'endpoints': [key[:2] for key in reference[0]],
                'relation_ids': [key[2] for key in reference[0]],
                'groups': reference[1], 'route_query_signatures': signatures}

    def clear(self):
        if self.current is not None:
            raise ValueError('Cannot clear captures during a task call')
        self.rows = {member: [] for member in range(4)}

    def close(self):
        self.handle.remove()
        self.clear()
