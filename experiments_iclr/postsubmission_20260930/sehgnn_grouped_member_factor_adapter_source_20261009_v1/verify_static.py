"""AST/source/namespace verification only; no Torch, native model or data imports."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

from grouped_member_factors import (AdapterConfig, AdapterContractError, channel_order,
                                    PINNED_MODEL_SHA256, SITES)


def main():
    here = Path(__file__).resolve().parent
    base = here.parent
    pinned = base/'imdb_role_isolated_public_schema_backbone_preparation_20261009_v1/public_sources/sehgnn_hgb_model.txt'
    source_path = here/'grouped_member_factors.py'
    source, native = source_path.read_text(), pinned.read_text()
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    assert hashlib.sha256(pinned.read_bytes()).hexdigest() == PINNED_MODEL_SHA256
    for path in here.glob('*.py'):
        compile(ast.parse(path.read_text()), str(path), 'exec')
    native_ast = ast.parse(native)
    native_classes = {node.name for node in native_ast.body if isinstance(node, ast.ClassDef)}
    assert native_classes == {'Transformer', 'LinearPerMetapath', 'SeHGNN'}
    assert "torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias.unsqueeze(0)" in native
    assert 'self.semantic_fusion(x, mask=None).transpose(1,2)' in native
    assert len(SITES) == 6 and not any('task_mlp' in site or 'embeding' in site for site in SITES)
    assert 'self._native_forward(self, x*r.unsqueeze(0))' in source
    assert 'self._native_forward(self, x*r)' in source
    assert 'memo = {id(parameter): parameter' in source
    assert 'member = copy.deepcopy(prototype, memo)' in source
    assert 'config.require_enabled()' in source
    assert 'input_factor.to(dtype=x.dtype)' in source
    assert 'output_factor.to(dtype=native_value.dtype)' in source
    # Non-numerical descriptor only: same string in both namespaces must survive.
    class Descriptor:
        feat_keys = ['M', 'MAM']
        label_feat_keys = ['MAM']
        num_channels = 3
    order = channel_order(Descriptor())
    assert [(c.namespace,c.key,c.row) for c in order] == [
        ('feature','M',0), ('feature','MAM',1), ('label','MAM',2)]
    try:
        AdapterConfig().require_enabled()
    except AdapterContractError:
        pass
    else:
        raise AssertionError('Default unexpectedly enabled')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    factors = 2*(37*512+37*512)+(512+128)*2+(512+512)+(37*512+512)
    assert factors == 97536
    result = {
        'schema':'SeHGNN-grouped-adapter-static-verification-v1',
        'status':'passed',
        'adapter_source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'pinned_model_sha256':PINNED_MODEL_SHA256,
        'checks':['Python AST compilation','exact pinned author source binding',
                  'native grouped orientation and semantic mask=None locator',
                  'six explicit projection sites excluding native task/embedding modules',
                  'actual native affine forward calls','slow-Parameter deepcopy memo',
                  'effective-factor AMP dtype casts','distinct feature/label namespace order',
                  'inactive default','symbolic factor count'],
        'symbolic_private_scalars_per_member':factors,
        'symbolic_private_scalars_M4':4*factors,
        'torch_numpy_or_model_imported':False,
        'scientific_data_or_server_actions':False,
        'qualification_limit':'AST/source/static descriptor arithmetic only. Not Torch execution, unit/native materiality, actual buffer ownership, preprocessing, Jacobians, optimizer, memory, native competence or source utility.'
    }
    (here/'STATIC_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','source_sha256':result['adapter_source_sha256'],
                      'private_scalars_per_member':factors,'model_imported':False}))


if __name__ == '__main__':
    main()
