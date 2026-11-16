"""Load an existing selected checkpoint and score explicitly requested data roles.

No fitting is performed. Campaign tasks call this only after their validation
selection dependencies finish. Results are written separately from training receipts.
"""

import argparse
import importlib
import hashlib
import json
from pathlib import Path

import torch

from datasets import load_dataset

from ablation.common.resumable import atomic_torch_save, atomic_write_json
from ablation.common.tabm_trainer import evaluate, member_logits


MODEL_MODULES = {
    'ablation.p01_placement.models',
    'ablation.p02_full_ablations.models',
    'ablation.p03_rms_init.models',
    'ablation.p04_shared_heads.models',
    'ablation.p05_hparam_grid.models',
    'ablation.p13_node_contrast.models',
}


def read_config(job_dir):
    path = Path(job_dir)
    config = json.loads((path / 'config.json').read_text())
    done = json.loads((path / 'done.json').read_text())
    conflicts = [key for key in config.keys() & done.keys() if config[key] != done[key]]
    if conflicts:
        raise ValueError(f'Config/result mismatch in {path}: {sorted(conflicts)}')
    return {**config, **done}


def load_checkpoint(job_dir, config):
    path = Path(job_dir) / 'model.pt'
    expected = config.get('selected_checkpoint_sha256')
    if expected is not None:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f'Selected checkpoint checksum mismatch in {job_dir}')
    config['checkpoint_sha256_verified'] = expected is not None
    return torch.load(path, map_location='cpu', weights_only=True)


def build_selected(config, data, targets, device, calibrate=False):
    module_name = config.get('model_module', 'ablation.p05_hparam_grid.models')
    if module_name not in MODEL_MODULES:
        raise ValueError(f'Unsupported checkpoint model module {module_name}')
    module = importlib.import_module(module_name)
    method = config.get('variant') or config.get('method')
    members = int(config.get('members', config.get('m', 4)))
    arguments = [method, config['model'], config['num_layers'], data.x.shape[1],
                 config['hidden_dim'], targets, members, device]
    if module_name == 'ablation.p05_hparam_grid.models':
        arguments.extend([config['dropout'], config['attention_dropout'], config['num_heads']])
    if method in {'xavier_same_weights', 'xavier_activation_rms'}:
        from ablation.p03_rms_init.activation import build_model, calibrate_model
        model = build_model(*arguments)
        if method == 'xavier_activation_rms' and calibrate:
            calibrate_model(model, data, members)
    else:
        model = module.build_model(*arguments)
    return model, members


def load_selected(job_dir, data_dir='data', device='cpu'):
    device = torch.device(device)
    config = read_config(job_dir)
    data, train, val, test, _, targets, binary = load_dataset(
        config['dataset'], add_self_loops=True, device=device, data_dir=data_dir)
    model, members = build_selected(config, data, targets, device)
    state = load_checkpoint(job_dir, config)
    model.load_state_dict(state)
    masks = {'train': train[:, config['split']].to(device),
             'val': val[:, config['split']].to(device),
             'test': test[:, config['split']].to(device)}
    return config, model, members, data, masks, binary


def load_selected_group(job_dirs, data_dir='data', device='cpu'):
    """Load one shared fit or four native fits against one shared graph object."""
    config, model, members, data, masks, binary = load_selected(job_dirs[0], data_dir, device)
    units = [(model, members)]
    seeds = [config['seed']]
    targets = int(data.y.max().item()) + 1 if not binary else 1
    for directory in job_dirs[1:]:
        other = read_config(directory)
        for key in ('dataset', 'split', 'model', 'num_layers', 'lr', 'weight_decay',
                    'min_steps', 'num_steps', 'patience', 'selection_interval',
                    'dropout', 'attention_dropout', 'num_heads', 'hidden_dim', 'model_module'):
            if config.get(key) != other.get(key):
                raise ValueError(f'Independent members differ in {key}')
        other_model, other_members = build_selected(other, data, targets, torch.device(device))
        if other_members != 1:
            raise ValueError('Independent ensemble members must be native single models')
        other_model.load_state_dict(load_checkpoint(directory, other))
        units.append((other_model, other_members))
        seeds.append(other['seed'])
    if len(units) > 1:
        if len(units) != 4 or members != 1 or len(set(seeds)) != 4:
            raise ValueError('Independent ensemble requires four distinct native single-model seeds')
    return config, units, data, masks, binary


def score_selected(job_dirs, out, roles=('val',), data_dir='data', device='cpu'):
    """Score one shared model or four separately selected independent members."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not roles or not set(roles).issubset({'train', 'val', 'test'}):
        raise ValueError('Roles must be an explicit subset of train, val, test')
    config, units, data, masks, binary = load_selected_group(job_dirs, data_dir, device)
    logits = torch.cat([member_logits(model, data, members) for model, members in units], dim=0)
    result = {
        'schema': 1, 'job_dirs': [str(Path(path).resolve()) for path in job_dirs],
        'dataset': config['dataset'], 'model': config['model'],
        'split': config['split'], 'num_layers': config['num_layers'],
        'roles': list(roles), 'members': logits.shape[0],
        'checkpoint_hashes_verified': all(read_config(path).get('selected_checkpoint_sha256')
                                        for path in job_dirs),
        'legacy_hash_policy': 'Artifacts without recorded checksums remain readable and are marked unverified',
    }
    predictions = {'schema': 1, 'dataset': config['dataset'], 'split': config['split'],
                   'members': logits.shape[0], 'is_binary': binary, 'roles': {}}
    for role in roles:
        metrics = evaluate(logits, data, masks[role], binary, config['dataset'])
        result.update({f'{role}_{key}': value for key, value in metrics.items()})
        predictions['roles'][role] = {
            'indices': masks[role].nonzero(as_tuple=False).flatten().cpu(),
            'labels': data.y[masks[role]].detach().cpu(),
            'logits': logits[:, masks[role]].detach().cpu(),
        }
    atomic_torch_save(predictions, out / 'predictions.pt')
    atomic_write_json(result, out / 'scored.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--roles', nargs='+', choices=['train', 'val', 'test'], default=['val'])
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    print(json.dumps(score_selected(args.jobs, args.out, args.roles, args.data_dir, args.device),
                     indent=2))


if __name__ == '__main__':
    main()
