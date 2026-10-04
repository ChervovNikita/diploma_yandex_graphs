"""Disabled F4 target-only DDI entry point using the unchanged baseline data/fit driver."""
import argparse
import json
from pathlib import Path
import baseline_train_ddi as baseline

VARIANT = {
    'name':'F4_private_hop_target_only','members':4,'private_alpha':True,'storage':'aggregates',
    'embedding':'one_shared_learned_node_ID_matrix','input_dropout':'one_common_context',
    'affine':'shared_dense_weight_bias_plus_private_input_output_factors_and_bias_offsets',
    'predictor':'four_private_native_Hadamard_MLP_heads','target_loss':'mean_of_four_native_AUC_sums',
    'serving':'uniform_mean_raw_member_pair_scores','conditional_auxiliary':False,
    'counts_in_served_scores':False,'pretrained_or_resource_state':False,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    args = parser.parse_args()
    config, artifact_path = baseline.read_release_config(args.config.resolve())
    if (config.get('variant') != VARIANT or config['seeds'] != [0,1,2]
            or config['artifact_contract']['train_weight_present'] is not False):
        raise ValueError('Exactly F4 private-hop target-only, fresh seeds 0/1/2 and native AUC are required.')
    # Numerical imports stay behind the inherited disabled release gate.
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from torch_geometric.transforms import ToSparseTensor
    from ogb.linkproppred import Evaluator
    from f4_model import F4PrivateHopTargetModel
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Declared CUDA runtime is unavailable.')
    data, split_edge, num_nodes = baseline.load_train_valid(
        artifact_path,config['artifact_contract'],torch,Data,ToSparseTensor)
    data = data.to(device)
    output_root = Path(config['output_dir']).expanduser()
    if not output_root.is_absolute():output_root = args.config.resolve().parent/output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True,exist_ok=False)
    summaries = []
    for seed in config['seeds']:
        summaries.append(baseline.run_seed(
            seed,config,data,split_edge,num_nodes,device,output_root,torch,np,F4PrivateHopTargetModel,Evaluator))
    (output_root/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')


if __name__ == '__main__':
    main()
