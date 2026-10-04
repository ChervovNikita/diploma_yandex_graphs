"""Disabled native-budget three-arm DDI driver; no TEST loader or score path."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from runtime_paths import require_project_output


def verify_dependencies(packet, binding):
    for source in binding['runtime_source_pins']:
        path = (packet.parent / source['path']).resolve()
        if path.stat().st_size != source['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError(f'Dependency source changed: {source["path"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New absolute writable project run directory outside sealed source.')
    args = parser.parse_args()
    packet = Path(__file__).resolve().parent
    config = json.loads(args.config.read_text())
    # Exit before any model/numerical module import or artifact read.
    if config['release_enabled'] is not True:
        raise SystemExit('Release disabled: integration source only; no training started.')
    output = require_project_output(args.output_dir, packet)
    binding = json.loads((packet / 'INPUT_BINDINGS.json').read_text())
    verify_dependencies(packet, binding)
    f4_packet = (packet.parent / binding['f4_packet']).resolve()
    sys.path.insert(1, str(f4_packet))
    import baseline_train_ddi as baseline
    config, artifact_path = baseline.read_release_config(args.config.resolve())
    if config['arms'] != ['target_only', 'joint', 'separate'] or config['seeds'] != [0, 1, 2]:
        raise ValueError('Exactly the prereleased three arms and three seeds are required.')
    if config['artifact_contract']['train_weight_present'] is not False:
        raise ValueError('Native missing-weight AUC must be retained.')
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from torch_geometric.transforms import ToSparseTensor
    from ogb.linkproppred import Evaluator
    from f4_cb_model import F4ConditionalPatternModel, AUXILIARY
    if config['auxiliary'] != AUXILIARY:
        raise ValueError('The reviewed masking/query/scaling recipe changed.')
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Declared CUDA runtime is unavailable.')
    data, split_edge, num_nodes = baseline.load_train_valid(
        artifact_path, config['artifact_contract'], torch, Data, ToSparseTensor)
    data = data.to(device)
    # CLI output is explicit; the sealed config has no implicit run destination.
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for seed in config['seeds']:
        for arm in config['arms']:
            arm_output = output / arm
            arm_output.mkdir(exist_ok=True)
            holder = []
            def factory(*model_args, **model_kwargs):
                model = F4ConditionalPatternModel(
                    *model_args, arm=arm, training_seed=seed,
                    stream_path=arm_output / f'seed_{seed}' / 'paired_stream.jsonl', **model_kwargs)
                holder.append(model)
                return model
            result = baseline.run_seed(seed, config, data, split_edge, num_nodes, device,
                                       arm_output, torch, np, factory, Evaluator)
            model = holder.pop()
            # Restore only this completed cell's own VALID-selected weights.
            selected = torch.load(arm_output / f'seed_{seed}' / 'valid_best.pt', map_location=device, weights_only=True)
            if selected['seed'] != seed or selected['actual_loss_branch'] != 'AUC':
                raise RuntimeError('Selected state ownership/native loss changed.')
            model.encoder.load_state_dict(selected['encoder'])
            model.predictor.load_state_dict(selected['predictor'])
            model.emb.load_state_dict(selected['embedding'])
            replay = model.validate(data, split_edge, config['recipe']['batch_size'], Evaluator(name='ogbl-ddi'))
            if float(replay['Hits@20']) != result['best_valid_hits20']:
                raise RuntimeError('Complete own selected VALID replay failed.')
            summaries.append({'arm': arm, **result, 'selected_replay': 'PASS'})
            del model, selected
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    from paired_comparison import compare_family
    comparison = compare_family(output, summaries, config['recipe']['epochs'])
    (output / 'paired_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')


if __name__ == '__main__':
    main()
