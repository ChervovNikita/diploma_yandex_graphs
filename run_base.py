import argparse
import os
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from dgl.data import RomanEmpireDataset
from torch_geometric.utils import from_dgl, add_self_loops
from models import Model


ALL_MODELS = ['ResNet', 'GCN', 'SAGE', 'GAT', 'GAT-sep', 'GT', 'GT-sep', 'TAG']
NUM_LAYERS_RANGE = range(1, 6)
NUM_SPLITS = 10
NUM_STEPS = 1000
PATIENCE = 3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(device):
    dataset = RomanEmpireDataset()
    g = dataset[0]
    num_classes = dataset.num_classes

    pyg_data = from_dgl(g)
    pyg_data.x = g.ndata['feat']
    pyg_data.y = g.ndata['label']

    train_masks = g.ndata['train_mask']
    val_masks = g.ndata['val_mask']
    test_masks = g.ndata['test_mask']

    pyg_data.edge_index, _ = add_self_loops(pyg_data.edge_index, num_nodes=pyg_data.num_nodes)
    pyg_data = pyg_data.to(device)

    return pyg_data, train_masks, val_masks, test_masks, num_classes


def train_step(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data, data.x)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data, data.x)
    pred = out[mask].argmax(dim=1)
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    acc = (pred == data.y[mask]).float().mean().item()
    loss = F.cross_entropy(out[mask], data.y[mask]).item()
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    return {
        'acc': acc, 'loss': loss,
        'f1_macro': f1_macro, 'f1_weighted': f1_weighted,
        'prec_macro': prec_macro, 'rec_macro': rec_macro,
    }


def train_single(model_name, num_layers, split_idx, data, train_masks, val_masks, test_masks,
                 num_classes, device, save_dir):
    data.train_mask = train_masks[:, split_idx].to(device)
    data.val_mask = val_masks[:, split_idx].to(device)
    data.test_mask = test_masks[:, split_idx].to(device)

    set_seed(split_idx)

    model = Model(
        model_name=model_name,
        num_layers=num_layers,
        input_dim=data.x.size(1),
        hidden_dim=512,
        output_dim=num_classes,
        hidden_dim_multiplier=1,
        num_heads=8,
        normalization='LayerNorm',
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0)

    best_val_acc = 0.0
    best_step = 0
    steps_without_improvement = 0
    ckpt_path = os.path.join(save_dir, f'{model_name}_{num_layers}L_split{split_idx}.pt')

    for step in range(1, NUM_STEPS + 1):
        train_loss = train_step(model, optimizer, data)
        val_metrics = evaluate(model, data, data.val_mask)

        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_step = step
            steps_without_improvement = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            steps_without_improvement += 1

        if step % 50 == 0 or step == 1:
            print(f'Step {step:4d} | loss {train_loss:.4f} '
                  f'| val_acc {val_metrics["acc"]:.4f} | val_f1m {val_metrics["f1_macro"]:.4f} '
                  f'| val_f1w {val_metrics["f1_weighted"]:.4f} | val_prec {val_metrics["prec_macro"]:.4f} '
                  f'| val_rec {val_metrics["rec_macro"]:.4f}')

        if steps_without_improvement >= PATIENCE * 100:
            print(f'Early stopping at step {step}')
            break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_m = evaluate(model, data, data.test_mask)
    train_m = evaluate(model, data, data.train_mask)

    print(f'\n--- Final (best ckpt @ step {best_step}) ---')
    print(f'  train_acc={train_m["acc"]:.4f}  val_acc={best_val_acc:.4f}')
    print(f'  test_acc={test_m["acc"]:.4f}  test_f1m={test_m["f1_macro"]:.4f}  '
          f'test_f1w={test_m["f1_weighted"]:.4f}  test_prec={test_m["prec_macro"]:.4f}  '
          f'test_rec={test_m["rec_macro"]:.4f}  test_loss={test_m["loss"]:.4f}')

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        'model': model_name,
        'num_layers': num_layers,
        'split': split_idx,
        'best_step': best_step,
        'train_acc': train_m['acc'],
        'val_acc': best_val_acc,
        'test_acc': test_m['acc'],
        'test_loss': test_m['loss'],
        'test_f1_macro': test_m['f1_macro'],
        'test_f1_weighted': test_m['f1_weighted'],
        'test_prec_macro': test_m['prec_macro'],
        'test_rec_macro': test_m['rec_macro'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--models', nargs='+', default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument('--layers', nargs='+', type=int, default=None,
                        help='Layer counts to run (e.g. --layers 1 3 5). Default: 1-5.')
    parser.add_argument('--split', type=int, default=None, choices=range(NUM_SPLITS),
                        help='Run only this split (0-9). If not set, runs all 10.')
    args = parser.parse_args()

    device = torch.device(args.device)

    save_dir = 'checkpoints_base'
    os.makedirs(save_dir, exist_ok=True)

    log_path = 'results_base.csv'
    fieldnames = ['model', 'num_layers', 'split', 'best_step', 'train_acc', 'val_acc',
                   'test_acc', 'test_loss', 'test_f1_macro', 'test_f1_weighted',
                   'test_prec_macro', 'test_rec_macro']

    write_header = not os.path.exists(log_path)
    log_file = open(log_path, 'a', newline='')
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    print('Loading data...')
    pyg_data, train_masks, val_masks, test_masks, num_classes = load_data(device)

    splits = [args.split] if args.split is not None else range(NUM_SPLITS)
    layers = args.layers if args.layers is not None else list(NUM_LAYERS_RANGE)

    for split_idx in splits:
        for model_name in args.models:
            for num_layers in layers:
                tag = f'{model_name} {num_layers}L split={split_idx}'
                print(f'\n{"="*60}\nTraining {tag}\n{"="*60}')

                result = train_single(
                    model_name=model_name,
                    num_layers=num_layers,
                    split_idx=split_idx,
                    data=pyg_data,
                    train_masks=train_masks,
                    val_masks=val_masks,
                    test_masks=test_masks,
                    num_classes=num_classes,
                    device=device,
                    save_dir=save_dir,
                )

                writer.writerow(result)
                log_file.flush()

                print(f'  >> best_step={result["best_step"]}  '
                      f'val_acc={result["val_acc"]:.4f}  '
                      f'test_acc={result["test_acc"]:.4f}  '
                      f'test_f1m={result["test_f1_macro"]:.4f}')

    log_file.close()
    print(f'\nDone. Results saved to {log_path}, checkpoints in {save_dir}/')


if __name__ == '__main__':
    main()
