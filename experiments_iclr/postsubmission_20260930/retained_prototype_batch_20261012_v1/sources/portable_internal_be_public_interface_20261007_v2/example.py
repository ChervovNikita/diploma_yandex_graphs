"""Synthetic interface example; no benchmark downloads, data or study results."""
import argparse
import json
from pathlib import Path
from portable import Session, recipe


def synthetic_batch(session):
    import torch
    from torch_geometric.data import Data, Batch
    device = session.device
    if session.task == 'molhiv':
        graphs = [Data(x=torch.zeros(3, 9, dtype=torch.long),
                       edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                       edge_attr=torch.zeros(4, 3, dtype=torch.long)) for _ in range(4)]
        return {'graph': Batch.from_data_list(graphs).to(device)}, torch.tensor([0., 0., 1., 1.], device=device)
    if session.task == 'wikics':
        ids = torch.arange(12, device=device)
        nodes = torch.arange(32, device=device)
        edges = torch.stack((torch.cat((nodes, (nodes+1) % 32, nodes)),
                             torch.cat(((nodes+1) % 32, nodes, nodes))))
        return {'x': torch.randn(32, 300, device=device), 'edge_index': edges, 'ids': ids}, ids % 10
    from torch_sparse import SparseTensor
    nodes = torch.arange(16, device=device)
    row = torch.cat((nodes, (nodes+1) % 16)); column = torch.cat(((nodes+1) % 16, nodes))
    queries = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [0, 4], [2, 6], [3, 7], [5, 9]], device=device)
    # Every supervised positive is removed in BOTH directions from BOTH views.
    pairs = torch.minimum(row, column)*16 + torch.maximum(row, column)
    excluded = queries[:4].min(1).values*16 + queries[:4].max(1).values
    keep = ~torch.isin(pairs, excluded)
    adjacency = SparseTensor(row=row[keep], col=column[keep], sparse_sizes=(16, 16)).coalesce()
    return {'x': torch.randn(16, 128, device=device), 'adj': adjacency, 'query': queries}, torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.], device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('molhiv', 'wikics', 'collab'), default='molhiv')
    parser.add_argument('--arm', default='be_init_contrastive')
    parser.add_argument('--seed', type=int, default=6101)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--polynormer', type=Path)
    parser.add_argument('--ncn-model', type=Path)
    parser.add_argument('--ncn-utils', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='Optional fresh continuation snapshot')
    parser.add_argument('--show-recipe', action='store_true', help='Print recipe without numerical imports')
    args = parser.parse_args()
    if args.show_recipe:
        print(json.dumps(recipe(args.task), indent=2, sort_keys=True)); return
    session = Session(args.task, args.arm, args.seed, args.device,
                      args.polynormer, args.ncn_model, args.ncn_utils)
    batch, labels = synthetic_batch(session)
    measured = session.train_step(batch, labels)
    if args.checkpoint:
        session.save_training_state(args.checkpoint, epoch=0)
    print(json.dumps({'example': 'synthetic one-update interface exercise', 'task': args.task,
                      'arm': args.arm, 'seed': args.seed, 'members': session.model.members,
                      'own_views': 2, 'updates': session.steps, 'synthetic_loss': float(measured['loss']),
                      'parameter_counts': session.core['factors'].factor_counts(session.model),
                      'checkpoint_selected_on_VALID': False, 'benchmark_claim': False}))


if __name__ == '__main__':
    main()
