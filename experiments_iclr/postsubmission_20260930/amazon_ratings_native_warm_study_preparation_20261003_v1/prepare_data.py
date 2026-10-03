"""Normal source-faithful PyG preprocessing and compact TRAIN/VAL projection.

The official container's all-node label payload is physically decoded. No TEST
label is used for training, grouping, selection or scoring, and no TEST label
channel is emitted. Role visibility is per official split, whose nodes overlap.
"""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
sys.dont_write_bytecode = True
import common as c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--admission', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    release, release_record = c.admission(args.admission, 'data_projection', args.output)
    c.require(release['runtime_device'] == 'cpu' and release['raw_label_payload_decode_disclosed'] is True,
              'CPU data projection and physical all-label decode disclosure required')
    raw = c.verify(release['raw_release'])
    c.official_reference(release['official_release_reference'])
    c.require(release['class_schema'] == [0, 1, 2, 3, 4],
              'Root-bound official release and public class schema required')
    _, actual = c.runtime('cpu', release)
    import numpy as np
    import torch
    from torch_geometric.datasets import HeterophilousGraphDataset
    from torch_geometric.utils import is_undirected
    out = c.fresh_directory(args.output)
    c.write(out/'STARTED.json', {'UTC': c.utc(), 'release': release_record, 'raw': release['raw_release']})
    captured = []
    # Invoke the pinned normal PyG processor body. Its save callback retains the
    # processed Data in memory rather than creating a second all-label data.pt.
    holder = SimpleNamespace(raw_paths=[str(raw)], pre_transform=None,
                             processed_paths=[str(out/'not_written.pt')],
                             save=lambda data, unused: captured.append(data[0]))
    HeterophilousGraphDataset.process(holder)
    c.require(len(captured) == 1, 'One official graph expected')
    data = captured.pop()
    c.require(data.x.dtype == torch.float32 and data.x.ndim == 2 and
              torch.isfinite(data.x).all().item() and data.y.dtype == torch.int64 and
              tuple(data.y.shape) == (data.x.shape[0],), 'Official native feature/label schema invalid')
    c.require(is_undirected(data.edge_index) and data.edge_attr is None, 'Native unweighted undirected graph required')
    n = data.x.shape[0]
    c.require(all(getattr(data, k).dtype == torch.bool and tuple(getattr(data, k).shape) == (n, 10)
                  for k in ('train_mask', 'val_mask', 'test_mask')), 'Native official mask layout invalid')
    public = out/'public_graph.npz'
    with public.open('xb') as f:
        np.savez(f, features=data.x.numpy(), edge_index=data.edge_index.numpy(),
                 train_mask=data.train_mask.numpy(), val_mask=data.val_mask.numpy(), test_mask=data.test_mask.numpy())
    trains, validations, blocks = {}, {}, []
    for j, seed in c.BLOCKS:
        c.require((data.train_mask[:, j].to(torch.int8)+data.val_mask[:, j].to(torch.int8)+
                   data.test_mask[:, j].to(torch.int8) == 1).all().item(), 'Official split is not a partition')
        info = {'split': j, 'optimizer_seed': seed}
        for role, masks, records in (('train', data.train_mask, trains), ('validation', data.val_mask, validations)):
            ids = masks[:, j].nonzero(as_tuple=False).flatten()
            labels = data.y.index_select(0, ids)
            c.require(ids.numel() > 0 and ((labels >= 0) & (labels < 5)).all().item(), 'Compact role labels invalid')
            name = out/f'split{j}_{role}.npz'
            with name.open('xb') as f:
                np.savez(f, ids=ids.numpy(), labels=labels.numpy())
            records[str(j)] = c.record(name)
            info[role+'_count'] = ids.numel()
        blocks.append(info)
    del data, captured
    c.verify_sources(); c.verify(release['raw_release']); c.verify(release_record)
    result = {'schema': 'amazon_native_train_validation_projection_v1', 'UTC': c.utc(),
              'dataset': 'amazon-ratings', 'class_schema': [0, 1, 2, 3, 4], 'split_count': 10,
              'packet_manifest': c.record(c.PACKET/'MANIFEST.json'),
              'raw_release': release['raw_release'], 'official_release_reference': release['official_release_reference'],
              'producer_release': release_record, 'public_graph': c.record(public),
              'train_labels': trains, 'validation_labels': validations, 'blocks': blocks,
              'raw_all_node_label_payload_decoded': True, 'test_label_artifacts': [],
              'test_labels_used_for_fitting_grouping_selection_or_scoring': False,
              'source_feature_normalization': 'unchanged raw Amazon features; author loader has no NormalizeFeatures',
              'public_processing': 'pinned installed PyG2.7 HeterophilousGraphDataset.process body',
              'split_semantics': 'native [nodes,10]; unchanged author heter_fixed_splits called by readers',
              'role_visibility': 'per block; a node may be TRAIN/VAL in one block and TEST in another',
              'runtime': actual, 'predictive_scoring_performed': False}
    c.write(out/'DATA_MANIFEST.json', result)
    print(str(out/'DATA_MANIFEST.json'))


if __name__ == '__main__':
    main()
