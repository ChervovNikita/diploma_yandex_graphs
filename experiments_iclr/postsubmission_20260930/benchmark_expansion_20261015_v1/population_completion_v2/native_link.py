"""Original link population function plus native OGB NumPy conversion.

No cast, sorting or data replacement. The unchanged OGB provider performs this
same conversion after loading its native split PT files.
"""
from pathlib import Path
import inspect


def bind(parent):
    for name in ('native_pins','require','verify','inside','access','receipt','definitions','ordered_array','DESIGN'):
        globals()[name] = getattr(parent, name)
    return link_population


def link_population(dataset, recipe, cell, output, torch, np):
    pins = native_pins(cell)
    require(set(pins) == {'graph', 'train', 'valid'}, 'Exact admitted native link files required')
    for pin in pins.values():
        verify(pin)
    folder = inside(cell['data_root']) / 'split/time'
    require(Path(pins['train']['path']) == folder / 'train.pt'
            and Path(pins['valid']['path']) == folder / 'valid.pt', 'Native time-split paths differ')
    access(output, 'before_official_TEST_query_hash_and_read', dataset=dataset, TEST_accessed=True,
           scope='all official positive/negative query endpoints and role membership; no predictions or scores')
    pins['test'] = receipt(folder / 'test.pt')
    from ogb.utils.torch_util import replace_numpy_with_torchtensor
    pins['native_OGB_split_conversion_source'] = receipt(inspect.getfile(replace_numpy_with_torchtensor))
    split = replace_numpy_with_torchtensor(torch.load(verify(pins['test']), map_location='cpu', weights_only=False))
    require(isinstance(split, dict) and {'edge', 'edge_neg'} <= set(split), 'Official Collab query schema required')
    for key in ('edge', 'edge_neg'):
        value = split[key]
        require(torch.is_tensor(value) and value.device.type == 'cpu' and value.dtype == torch.long
                and value.ndim == 2 and value.shape[1] == 2 and value.shape[0] > 0
                and ((value >= 0) & (value < 235868)).all().item(), 'Complete native Collab endpoint tensor required')
    positive, negative = len(split['edge']), len(split['edge_neg'])
    # Use original author concatenation, never sorting, sampling, deduplicating,
    # year-filtering TEST queries or constructing HashDataset/adjacency/model.
    defs = definitions(DESIGN / 'native_sources/buddy/buddy_src_data.py',
                       ['get_ogb_pos_edges', 'make_obg_supervision_edges'], {'torch': torch})
    labels, edge_index = defs['make_obg_supervision_edges']({'test': split}, 'test')
    endpoints = edge_index.t().contiguous()
    require(torch.equal(endpoints[:positive], split['edge'])
            and torch.equal(endpoints[positive:], split['edge_neg']), 'Native positive-then-negative order changed')
    require(len(labels) == positive + negative, 'Native supervision population mismatch')
    ids = ordered_array(output, 'TEST_ids', torch.arange(positive + negative, dtype=torch.long), torch)
    query = ordered_array(output, 'TEST_query_endpoints', endpoints, torch)
    return {'populations': {'0': {'records': positive + negative, 'ids_sha256': ids['tensor_sha256'],
                                  'positive': positive, 'negative': negative,
                                  'ordered_ids': ids, 'query_endpoints': query,
                                  'query_order': 'all edge rows, then all edge_neg rows; original file order'}},
            'source_receipts': list(pins.values()) + [receipt(__file__)], 'data_files': {k: v for k, v in pins.items() if k != 'native_OGB_split_conversion_source'},
            'provider': 'original admitted OGB temporal TEST PT; exact author supervision concatenation',
            'graph_container_deserialized': False, 'serving_topology_constructed': False,
            'BUDDY_hash_features_constructed': False,
            'topology_note': 'BUDDY year>=2007 applies to TRAIN only. Frozen contracts retain architecture-specific TEST topology.'}
