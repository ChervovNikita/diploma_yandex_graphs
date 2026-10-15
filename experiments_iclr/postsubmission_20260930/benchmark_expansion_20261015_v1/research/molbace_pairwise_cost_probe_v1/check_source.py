"""Compact source/JSON checks; no data/model/tensor or remote execution."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
E=HERE.parents[1]


def check():
    contract=json.loads((HERE/'CONTRACT.json').read_text());root=contract['project_root']
    prefix=root+'/experiments_iclr/postsubmission_20260930/'
    def local(pin):
        assert pin['path'].startswith(prefix)
        p=E.parent/pin['path'][len(prefix):]
        assert hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256']
        return p
    for pin in contract['source_pins']:local(pin)
    definition=importlib.util.spec_from_file_location('_cost_frozen_producer',E/'research/molbace_pairwise_competence_v1/extension.py')
    producer=importlib.util.module_from_spec(definition);definition.loader.exec_module(producer)
    for row in contract['completed_CPU_lifecycles']:
        result=json.loads(local(row['receipt']).read_text())['result']
        assert result['planned_job']==producer.descriptor(root,101,row['arm'],0 if row['arm']=='native_common' else None)
        assert result['status']=='passed' and result['optimizer_calls']=={'attempts':[2],'completed':[2]}
    for p in HERE.glob('*.py'):
        tree=ast.parse(p.read_text())
        for n in tree.body:
            if isinstance(n,ast.Import):assert not any(a.name.startswith(('torch','numpy','ogb')) for a in n.names)
            if isinstance(n,ast.ImportFrom):assert not (n.module or '').startswith(('torch','numpy','ogb'))
    source=(HERE/'probe.py').read_text()
    assert source.count('adapter.train_epoch()')==1 and 'for index in (1,2):'in source
    assert 'torch.cuda.synchronize();before=time.perf_counter()'in source
    assert '[32]*37+[26]'in source and 'torch.save'not in source and 'Selector('not in source
    assert 'Popen('not in source and "['git'"not in source
    a=json.loads((HERE/'ADMISSION_TEMPLATE.json').read_text())
    assert a['approved']is False and a['resource_admission_approved']is False and a['maximum_wall_seconds']is None
    assert a['maximum_optimizer_transitions']==38 and a['maximum_full_VALID_calls']==2 and a['maximum_native_constructions']==1
    manifest=HERE/'MANIFEST.json'
    if manifest.exists():
        for row in json.loads(manifest.read_text())['files']:
            p=HERE/row['path'];assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    return {'schema':'molbace-pair-cost-source-checks-v1','status':'passed','checks':['source_and_actual_CPU_descriptor_joins','no_eager_scientific_import','one_native_epoch_two_VALID_no_selector_or_checkpoint','no_owner_queue_Git','closed_exact_bounds_and_manifest'],
        'scientific_fits_launched':0,'new_data_tensor_model_execution':False,'SSH_Git_GPU_operations':0,'TEST_accessed':False}


if __name__=='__main__':
    value=check();(HERE/'SOURCE_CHECKS.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');print(json.dumps(value,sort_keys=True))
