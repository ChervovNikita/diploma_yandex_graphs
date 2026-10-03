"""ACM feats0 node/link reader; development and split readers are immutable reuse."""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parent.parent / 'graph_heterogeneous_dblp_training_preparation_20261003_v2/dblp_inputs.py'
spec = importlib.util.spec_from_file_location('immutable_HGB_development_inputs_for_ACM', SOURCE)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
require = common.require
verified = common.verified
read_development_labels = common.read_development_labels
verify_split = common.verify_split


def stream_schema(archive_record, members):
    """Open only node/link members, keeping support separately for every raw ID.

    Native per-relation COO->CSR duplicate coalescing and ignored edge weights
    are unchanged. Existing raw self records survive; no edges are synthesized.
    Feats0 retains all provided type attributes and uses identity when absent.
    """
    require(members == {'nodes': 'ACM/node.dat', 'links': 'ACM/link.dat'},
            'Exact ACM node/link members required')
    archive = verified(archive_record)
    counts, feature_widths, attribute_rows = Counter(), {}, Counter()
    attributes, node_types = defaultdict(list), []
    opened, member_sha256 = [], {}
    with zipfile.ZipFile(archive) as zipped:
        require(len(zipped.namelist()) == len(set(zipped.namelist())), 'Duplicate ZIP member names refused')
        opened.append(members['nodes'])
        digest = hashlib.sha256()
        with zipped.open(members['nodes']) as member:
            for raw_line in member:
                digest.update(raw_line)
                columns = raw_line.decode('utf-8').rstrip('\r\n').split('\t')
                require(len(columns) in (3, 4), 'Node record needs3/4 columns')
                node, type_id = int(columns[0]), int(columns[2])
                require(node == len(node_types) and type_id >= 0, 'Native contiguous ordered node IDs required')
                t = str(type_id)
                node_types.append(type_id)
                counts[t] += 1
                if len(columns) == 4:
                    values = [float(number) for number in columns[3].split(',')]
                    require(all(math.isfinite(x) for x in values), 'Nonfinite provided features')
                    require(feature_widths.get(t, len(values)) == len(values), 'Inconsistent attribute dimension')
                    feature_widths[t] = len(values)
                    attribute_rows[t] += 1
                    attributes[t].append(values)
        member_sha256['nodes'] = digest.hexdigest()
        types = sorted(counts, key=int)
        require(types == [str(i) for i in range(len(types))], 'Native contiguous node types required')
        shifts, offset = {}, 0
        for t in types:
            shifts[t] = offset
            require(node_types[offset:offset + counts[t]] == [int(t)] * counts[t], 'Native contiguous type blocks required')
            require(attribute_rows[t] in (0, counts[t]), 'Partial type attributes refused')
            offset += counts[t]
        edges, raw_counts, endpoints, self_counts = defaultdict(dict), Counter(), {}, Counter()
        opened.append(members['links'])
        digest = hashlib.sha256()
        with zipped.open(members['links']) as member:
            for raw_line in member:
                digest.update(raw_line)
                columns = raw_line.decode('utf-8').rstrip('\r\n').split('\t')
                require(len(columns) == 4, 'Link record needs4 columns')
                source, target, raw = map(int, columns[:3])
                weight = float(columns[3])
                require(0 <= source < len(node_types) and 0 <= target < len(node_types)
                        and raw >= 0 and math.isfinite(weight), 'Invalid directed link record')
                endpoint = (str(node_types[source]), str(node_types[target]))
                require(endpoints.get(raw, endpoint) == endpoint, 'Raw relation has inconsistent endpoints')
                endpoints[raw] = endpoint
                raw_counts[raw] += 1
                self_counts[raw] += source == target
                pair = (source - shifts[endpoint[0]], target - shifts[endpoint[1]])
                edges[raw][pair] = edges[raw].get(pair, 0.0) + weight
        member_sha256['links'] = digest.hexdigest()
    raw_rows = []
    for raw in sorted(edges):
        src_type, dst_type = endpoints[raw]
        raw_rows.append(dict(raw_id=raw, source=src_type, target=dst_type,
            name=f'raw{raw}__{src_type}__{dst_type}', raw_records=raw_counts[raw],
            support_edges=len(edges[raw]), duplicates_coalesced=raw_counts[raw] - len(edges[raw]),
            raw_self_records=self_counts[raw],
            zero_sum_stored_edges=sum(weight == 0 for weight in edges[raw].values())))
    schema = dict(node_counts=dict(counts), node_shifts=shifts, provided_attribute_widths=feature_widths,
        feature_type=0, target='0', target_given_attributes=bool(attributes['0']),
        input_dims={t: feature_widths[t] if attribute_rows[t] else counts[t] for t in types},
        relations=raw_rows, raw_link_records=sum(raw_counts.values()),
        support_edges=sum(len(value) for value in edges.values()), raw_self_records=sum(self_counts.values()),
        node_link_members_opened=opened, label_members_opened=[], member_sha256=member_sha256,
        duplicate_policy='native per-raw-relation COO->CSR support; weights ignored after coalescing',
        synthetic_reverse_or_self_added=False)
    return schema, attributes, edges


def materialize(schema, attributes, edges, implementation, row_order, device):
    import torch
    counts, features = schema['node_counts'], {}
    for t in sorted(counts):
        if attributes.get(t):
            features[t] = torch.tensor(attributes[t], dtype=torch.float32, device=device)
        else:
            ids = torch.arange(counts[t], device=device)
            features[t] = torch.sparse_coo_tensor(torch.stack([ids, ids]), torch.ones(counts[t], device=device),
                (counts[t], counts[t]), device=device).coalesce()
    relations = []
    for row in schema['relations']:
        pairs = sorted(edges[row['raw_id']])
        src = torch.tensor([pair[0] for pair in pairs], dtype=torch.long, device=device)
        dst = torch.tensor([pair[1] for pair in pairs], dtype=torch.long, device=device)
        relations.append(implementation.Relation(row['raw_id'], row['source'], row['target'], src, dst))
    graph = implementation.HeteroGraph(counts, relations, row_order)
    return graph, features, dict(schema, relation_row_order=list(graph.edge_dict),
                                raw_relation_to_row=graph.raw_relation_to_row)
