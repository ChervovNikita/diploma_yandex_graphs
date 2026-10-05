#!/usr/bin/env python3
"""Exact native CPU imports only: no model/data constructors or CUDA calls."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata, import_module
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
HERE = PHASE / 'pencil_citeseer_cpu_import_qualification_20261005_v1'
NATIVE = PHASE / 'pencil_collab_paired_predictive_preparation_20261004_v3/native'
DEPENDENCY = PHASE / 'pencil_citeseer_missing_ancillary_resolver_execution_20261005_v1'
OVERLAY = PHASE / 'pencil_one_gpu_dependency_overlay_20261005_v1'
EXPECTED_PYTHONPATH = ':'.join((str(OVERLAY), str(PHASE / 'native_ncn_dependency_overlay_20261005_v1'),
                               str(REPO / '.venv/lib/python3.11/site-packages')))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert socket.gethostname() == 'anogena-2-0' and Path.cwd().resolve() == REPO
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ.get('PYTHONPATH') == EXPECTED_PYTHONPATH
    assert not (HERE / 'RESULT.json').exists() and not (HERE / 'FAILURE.json').exists()
    started = time.monotonic()
    result = dict(UTC=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
                  cwd=str(Path.cwd()), source_sha256=sha(Path(__file__)),
                  status='RUNNING_CPU_IMPORT_ONLY', packages_installed_this_step=False,
                  model_constructors_called=False, dataset_constructors_called=False,
                  TRAIN_or_VALID_inputs_opened=False, TEST_access=False,
                  scientific_fits=0, optimizer_updates=0,
                  explicit_CUDA_operator_calls=0, GPU_readiness=False,
                  network_weights_requested=False, wandb_initialized=False,
                  child_environment={key: os.environ.get(key) for key in (
                      'PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'PYTHONDONTWRITEBYTECODE',
                      'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'WANDB_MODE', 'OUTDATED_IGNORE')})
    try:
        binding = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
        for row in binding['files']:
            path = PHASE / row['phase_relative']
            assert path.resolve().is_relative_to(PHASE) and sha(path) == row['sha256']
        assert sha(DEPENDENCY / 'INSTALLED_FILE_INVENTORY.json') == '339dc97e7a1b5cb8280317c48e18b1749a4c7c065ce7d682043dd75b3dfd63bf'
        assert sha(DEPENDENCY / 'MISSING_WHEELS.lock') == '09694097759d457115ad983e38ff1cc4ae637aef5581e1f85fd3edda8d2bf922'
        inventory = json.loads((DEPENDENCY / 'INSTALLED_FILE_INVENTORY.json').read_text())
        assert Path(inventory['root']) == OVERLAY
        actual_files = {str(p.relative_to(OVERLAY)) for p in OVERLAY.rglob('*') if p.is_file()}
        assert actual_files == {row['path'] for row in inventory['files']}
        for row in inventory['files']:
            path = OVERLAY / row['path']
            assert not path.is_symlink() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        observed = json.loads((DEPENDENCY / 'RESOLVED_MISSING_WHEELS.json').read_text())
        original = json.loads((DEPENDENCY / 'PREINSTALL_SELECTED_METADATA.json').read_text())
        versions = {row['name']: row['version'] for row in observed['packages']}
        assert len(versions) == 22
        for name, version in versions.items():
            package = metadata.distribution(name)
            assert package.version == version and Path(package.locate_file('')).resolve() == OVERLAY
        for name, row in original.items():
            package = metadata.distribution(name)
            assert package.version == row['version'] and str(package.locate_file('')) == row['root']
            assert hashlib.sha256((package.read_text('METADATA') or '').encode()).hexdigest() == row['METADATA_sha256']
        result.update(installed_inventory_sha256=sha(DEPENDENCY / 'INSTALLED_FILE_INVENTORY.json'),
                      exact22versions=versions, original61_provider_metadata_unchanged=True,
                      native_source_commit=binding['native_commit'], native_source_file_count=20,
                      source_bindings_sha256=sha(HERE / 'SOURCE_BINDINGS.json'))

        import torch
        assert not torch.cuda.is_initialized()
        # CPU imports are the authorized qualification here. Sparse imports may
        # register extension libraries and inspect binary version metadata; no
        # model, sample, tensor workload or explicit CUDA operator is invoked.
        import numpy
        import torch_geometric
        import torch_sparse
        import torch_scatter
        for name in ('transformers', 'tokenizers', 'huggingface_hub', 'safetensors',
                     'wandb', 'rootutils', 'rich'):
            import_module(name)
        from transformers.models.bert.modeling_bert import BertEncoder, BertLayer
        from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
        assert isinstance(BertEncoder, type) and isinstance(BertLayer, type)
        assert callable(_prepare_4d_attention_mask_for_sdpa)

        sys.path.insert(0, str(NATIVE))
        old_cwd = Path.cwd()
        try:
            os.chdir(NATIVE)
            import run_lp
            import utils
            import datasets
            from datasets.dataset_map import ShaDowKHopSeqFromEdgesMapDataset
            from datasets.utils import get_unique_edges_with_mapping
            from models.transformers import lp_model, bert_lp
        finally:
            os.chdir(old_cwd)
        custody = []
        for row in binding['files']:
            if 'module' not in row:
                continue
            module = sys.modules.get(row['module'])
            assert module is not None
            path = Path(module.__file__).resolve()
            assert path == PHASE / row['phase_relative'] and sha(path) == row['sha256']
            custody.append(dict(module=row['module'], path=str(path), sha256=row['sha256']))
        assert len(custody) == 20
        assert callable(run_lp.train_loop) and callable(run_lp.evaluate_loop)
        assert callable(run_lp.build_loaders) and callable(run_lp.get_model)
        assert isinstance(bert_lp.BERTLP, type) and isinstance(ShaDowKHopSeqFromEdgesMapDataset, type)
        assert callable(get_unique_edges_with_mapping)
        assert not torch.cuda.is_initialized()
        assert metadata.version('transformers') == '4.46.2'
        symbol_module = sys.modules['transformers.models.bert.modeling_bert']
        mask_module = sys.modules['transformers.modeling_attn_mask_utils']
        assert Path(symbol_module.__file__).resolve().is_relative_to(OVERLAY)
        assert Path(mask_module.__file__).resolve().is_relative_to(OVERLAY)
        added_sources = []
        module_names = dict(transformers='transformers', tokenizers='tokenizers',
                            huggingface_hub='huggingface_hub', safetensors='safetensors',
                            wandb='wandb', rootutils='rootutils', rich='rich')
        for name in module_names.values():
            module = sys.modules[name]
            path = Path(module.__file__).resolve()
            assert path.is_relative_to(OVERLAY)
            added_sources.append(dict(module=name, path=str(path), sha256=sha(path)))
        result.update(status='PASS_EXACT_NATIVE_CPU_IMPORTS_ONLY',
                      native_module_custody=custody, added_package_import_source_pins=added_sources,
                      BertEncoder_module=BertEncoder.__module__, BertLayer_module=BertLayer.__module__,
                      SDPA_mask_symbol_module=_prepare_4d_attention_mask_for_sdpa.__module__,
                      transformer_bert_source_sha256=sha(Path(symbol_module.__file__)),
                      transformer_SDPA_mask_source_sha256=sha(Path(mask_module.__file__)),
                      torch_CUDA_initialized=False,
                      core_versions=dict(torch=str(torch.__version__), numpy=numpy.__version__,
                          torch_geometric=torch_geometric.__version__, torch_sparse=torch_sparse.__version__,
                          torch_scatter=torch_scatter.__version__),
                      inclusive_seconds=time.monotonic() - started)
        destination = HERE / 'RESULT.json'
    except BaseException as error:
        result.update(status='FAIL_CPU_IMPORT_ONLY_PRESERVED',
                      error=type(error).__name__ + ': ' + str(error),
                      traceback=traceback.format_exc(), inclusive_seconds=time.monotonic() - started)
        destination = HERE / 'FAILURE.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
