"""One fixed DDI development cell:100 complete epochs and own VALID replay."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
from pilot_contract import ARMS, SEEDS, EPOCHS, file_sha256, read_release, verify_pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New absolute project cell path ending ARM/seed_SEED.')
    args = parser.parse_args()
    packet = Path(__file__).resolve().parent
    # Root admission is separate from empirical runtime or full500 qualification.
    config = read_release(args.config, packet)
    binding = verify_pins(packet)
    source = packet.parent / binding['sealed_F4_packet']
    native = packet.parent / binding['native_packet']
    sys.path[1:1] = [str(source), str(native)]
    from runtime_paths import require_project_output
    output = require_project_output(args.output_dir, packet)
    if output.name != f'seed_{args.seed}' or output.parent.name != args.arm:
        raise ValueError('Use the fixed family layout: FAMILY_ROOT/ARM/seed_SEED.')
    import baseline_train_ddi as baseline
    expected_recipe = {**baseline.AUTHOR_RECIPE, 'epochs': EPOCHS}
    if config['recipe'] != expected_recipe or config['author_commit'] != baseline.PIN:
        raise ValueError('Only the prospective100-epoch budget amendment is allowed.')
    contract = config['artifact_contract']
    artifact = Path(contract['path'])
    if (not artifact.is_absolute() or contract['sha256'] != binding['artifact']['sha256']
            or file_sha256(artifact) != contract['sha256']):
        raise ValueError('The exact qualified TRAIN/VALID artifact is required.')
    report = Path(binding['qualified_artifact_report_path'])
    if file_sha256(report) != binding['qualification_report_sha256']:
        raise ValueError('Qualified TRAIN/VALID report digest changed.')
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from torch_geometric.transforms import ToSparseTensor
    from ogb.linkproppred import Evaluator
    from native.model import BaseModel
    from f4_cb_model import F4ConditionalPatternModel, AUXILIARY
    if config['auxiliary'] != AUXILIARY:
        raise ValueError('The exact sealed auxiliary recipe is required.')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('An admitted CUDA device is required.')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(config['root_admission']['cuda_allocator_fraction'], device)
    torch.cuda.reset_peak_memory_stats(device)
    data, split_edge, num_nodes = baseline.load_train_valid(artifact, contract, torch, Data, ToSparseTensor)
    if num_nodes != 4267 or len(split_edge['train']['edge']) != 1067911:
        raise ValueError('Actual DDI TRAIN dimensions changed.')
    data = data.to(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    holder = []
    owns_output = False

    def factory(*model_args, **model_kwargs):
        nonlocal owns_output
        # run_seed creates the new seed directory atomically before this call.
        owns_output = True
        if args.arm == 'native_m1':
            model = BaseModel(*model_args, **model_kwargs)
        else:
            model = F4ConditionalPatternModel(
                *model_args, arm=args.arm, training_seed=args.seed,
                stream_path=output / 'paired_stream.jsonl', **model_kwargs)
        holder.append(model)
        return model

    started = time.perf_counter()
    try:
        result = baseline.run_seed(args.seed, config, data, split_edge, num_nodes, device,
                                   output.parent, torch, np, factory, Evaluator)
        model = holder.pop()
        selected_path = output / 'valid_best.pt'
        selected = torch.load(selected_path, map_location=device, weights_only=True)
        if (selected['seed'] != args.seed or selected['epoch'] != result['best_epoch']
                or selected['actual_loss_branch'] != 'AUC' or selected['recipe'] != expected_recipe
                or selected['author_commit'] != baseline.PIN
                or selected['artifact_sha256'] != contract['sha256']):
            raise RuntimeError('Own VALID-selected checkpoint contract changed.')
        model.encoder.load_state_dict(selected['encoder'])
        model.predictor.load_state_dict(selected['predictor'])
        model.emb.load_state_dict(selected['embedding'])
        replay = model.validate(data, split_edge, expected_recipe['batch_size'], Evaluator(name='ogbl-ddi'))
        replay = {name:float(value) for name, value in replay.items()}
        if replay['Hits@20'] != result['best_valid_hits20']:
            raise RuntimeError('Complete own selected VALID replay failed.')
        torch.cuda.synchronize(device)
        summary = {
            'schema':'hlgnn-ddi-development-pilot-cell-v1', 'status':'complete', 'arm':args.arm,
            **result, 'selected_replay':'PASS', 'selected_replay_metrics':replay,
            'selected_checkpoint_sha256':file_sha256(selected_path),
            'pilot_manifest_sha256':file_sha256(packet / 'MANIFEST.json'),
            'scientific_config_sha256':file_sha256(packet / 'config.json'),
            'release_config_sha256':file_sha256(args.config), 'artifact_sha256':contract['sha256'],
            'native_records_per_epoch':1067911, 'native_updates_per_epoch':17, 'last_batch_records':19335,
            'VALID_traversals_including_replay':21, 'training_epochs':100, 'author_epochs':500,
            'author_budget_parity':False, 'TEST_access':False, 'donor_state':False,
            'wall_seconds':time.perf_counter()-started,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device),
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device),
            'root_admission':config['root_admission'], 'full500_qualification':False,
        }
        (output / 'cell_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        print(json.dumps(summary,allow_nan=False),flush=True)
    except BaseException as error:
        if owns_output and output.exists():
            (output / 'CELL_FAILURE.json').write_text(json.dumps({
                'status':'incomplete', 'arm':args.arm, 'seed':args.seed,
                'error_type':type(error).__name__, 'error':str(error),
                'no_complete_cell_adoption':True, 'wall_seconds':time.perf_counter()-started,
            },indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
