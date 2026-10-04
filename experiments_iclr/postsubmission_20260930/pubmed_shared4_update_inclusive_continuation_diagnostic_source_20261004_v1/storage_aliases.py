"""Pure metadata storage alias classes; local addresses never leave this helper."""


def canonical_storage_aliases(rows):
    """Canonicalize inspectable backing storage by class member paths.

    StorageImpl identity detects distinct tensor views, including empty views.
    Overlapping nonempty byte spans also detect separate StorageImpl wrappers
    over shared backing memory. Same-device connected components are named by
    their lexicographically smallest member path. Compared output contains
    only canonical labels, sizes and relative byte offsets, never local
    pointer/handle identities. Invalid/opaque descriptors cannot qualify.
    """
    valid, unsupported, seen = [], [], set()
    for row in rows:
        path = row.get('path')
        if not isinstance(path, str) or not path or path in seen:
            raise ValueError('Unique nonempty tensor paths required')
        seen.add(path)
        device, handle, start, size = (row.get(k) for k in ('device', 'handle', 'start', 'bytes'))
        if (not isinstance(device, str) or not device or type(handle) is not int or handle <= 0
                or type(start) is not int or start < 0 or type(size) is not int or size < 0
                or (size > 0 and start == 0)):
            unsupported.append({'path': path, 'issue': 'Backing storage identity/range uninspectable'})
            continue
        valid.append(row)
    # An inconsistent StorageImpl descriptor is an observer/control failure,
    # rather than a silently accepted class with ambiguous backing memory.
    by_handle = {}
    for row in valid:
        key = (row['device'], row['handle'])
        by_handle.setdefault(key, []).append(row)
    rejected = set()
    for group in by_handle.values():
        if len({(r['start'], r['bytes']) for r in group}) != 1:
            for row in group:
                rejected.add(row['path'])
                unsupported.append({'path': row['path'], 'issue': 'Storage descriptor changed during snapshot'})
    valid = [row for row in valid if row['path'] not in rejected]
    parent = list(range(len(valid)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    for i, a in enumerate(valid):
        for j in range(i):
            b = valid[j]
            overlap = (a['bytes'] > 0 and b['bytes'] > 0
                       and max(a['start'], b['start']) < min(a['start'] + a['bytes'], b['start'] + b['bytes']))
            if a['device'] == b['device'] and (a['handle'] == b['handle'] or overlap):
                parent[find(i)] = find(j)
    groups = {}
    for i, row in enumerate(valid):
        groups.setdefault(find(i), []).append(row)
    classes = {}
    for group in groups.values():
        label = min(row['path'] for row in group)
        nonempty = [row for row in group if row['bytes'] > 0]
        origin = min(row['start'] for row in nonempty) if nonempty else 0
        span = max(row['start'] + row['bytes'] for row in nonempty) - origin if nonempty else 0
        for row in group:
            classes[row['path']] = {
                'storage_alias_class': label, 'storage_bytes': row['bytes'],
                'storage_class_span_bytes': span,
                'storage_start_relative_to_class_bytes': row['start'] - origin if row['bytes'] else None}
    return {'classes': classes, 'fully_inspected': not unsupported,
            'unsupported': sorted(unsupported, key=lambda row: row['path'])}
