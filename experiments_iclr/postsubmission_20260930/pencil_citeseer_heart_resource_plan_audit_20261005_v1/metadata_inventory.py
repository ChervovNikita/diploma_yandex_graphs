#!/usr/bin/env python3
"""Optional one-GPU stdlib metadata inventory; no model/data/module imports."""
from importlib import metadata
import json
import platform
import socket
import sys

REQUIRED = ('torch', 'numpy', 'torch-geometric', 'torch-sparse', 'torch-scatter',
            'transformers', 'tokenizers', 'huggingface-hub', 'safetensors',
            'wandb', 'rootutils', 'PyYAML', 'rich', 'scipy', 'scikit-learn',
            'ogb', 'tqdm', 'networkx', 'pandas', 'psutil', 'packaging')

if __name__ == '__main__':
    if socket.gethostname() != 'anogena-2-0':
        raise SystemExit('Inventory only the expressly authorized one-GPU host')
    rows = []
    for name in REQUIRED:
        try:
            item = metadata.distribution(name)
            rows.append(dict(distribution=name, version=item.version,
                             root=str(item.locate_file('')),
                             Requires_Python=item.metadata.get('Requires-Python'),
                             Requires_Dist=item.requires))
        except metadata.PackageNotFoundError:
            rows.append(dict(distribution=name, status='NOT_FOUND_IN_SELECTED_PYTHONPATH'))
    print(json.dumps(dict(host=socket.gethostname(), python=platform.python_version(),
                         executable=sys.executable, path=sys.path,
                         numerical_imports=False, data_reads=False, rows=rows), indent=2))
