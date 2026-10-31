"""Record the environment for new studies without opening datasets or CUDA contexts."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, platform, sys
import torch
import torch_geometric

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
report = {
    'python': sys.version,
    'platform': platform.platform(),
    'torch': torch.__version__,
    'torch_geometric': torch_geometric.__version__,
    'compiled_cuda': torch.version.cuda,
    'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions() if d.metadata['Name'])),
    'cuda_context_created': False,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ('python', 'torch', 'torch_geometric', 'compiled_cuda', 'cuda_context_created')}))
