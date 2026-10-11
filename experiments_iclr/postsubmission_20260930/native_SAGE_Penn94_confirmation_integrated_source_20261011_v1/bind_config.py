"""Root-only metadata binding command; no model construction, fit or target scores.

This source is unexecuted. It copies CAL/CONFIRM ids only without loading y,
records actual provider files, and writes one fresh configuration. It admits no stage.
"""
import argparse
import importlib
from pathlib import Path
from contracts import route, scoped, read, write, receipt, describe, sha, require


def main():
    route()
    parser=argparse.ArgumentParser();parser.add_argument('--config-out',type=Path,required=True);args=parser.parse_args()
    here=scoped(Path(__file__).parent)
    seal=read(here/'SEAL.json')
    for name,digest in seal['files'].items():require(sha(here/name)==digest,'Integrated source changed')
    path=scoped(args.config_out);require(not path.exists() and path.parent.exists(),'Fresh CONFIG under existing root directory required')
    cfg=read(here/'CONFIG_TEMPLATE.json')
    data=read(receipt(cfg['input_custody_receipt']))
    require(data['complete'] is True and data['model_fits']==0 and data['prediction_or_TEST_metrics_read'] is False,
            'Source/role preparation receipt with no model outcomes required')
    for name,item in cfg['source_receipts'].items():receipt(item)
    for name in cfg['provider_modules']:
        provider=importlib.import_module(name)
        cfg['provider_modules'][name]=describe(Path(provider.__file__))
    cfg['provider_receipts']=list(cfg['provider_modules'].values())
    import numpy as np
    # Target payload y is never requested. Only source-bound public role membership.
    destination=scoped(path.parent/(path.stem+'_ids_only'));destination.mkdir(exist_ok=False)
    for name in ('cal','confirm'):
        with np.load(receipt(cfg['inputs'][name+'_truth']),allow_pickle=False) as a:
            require(set(a.files)=={'ids','y'},'Exact custodial proper-role export schema')
            ids=a['ids'].copy()
        require(ids.dtype==np.int64 and ids.shape==(cfg['role_counts'][name],)
                and len(np.unique(ids))==len(ids) and np.array_equal(ids,np.sort(ids)), 'Canonical membership-only ids')
        target=destination/(name+'_ids.npz');np.savez(target,ids=ids)
        cfg['inputs'][name+'_ids']=describe(target)
    cfg['actual_provider_versions']={name:importlib.import_module(name).__version__ for name in ('torch','numpy','torch_geometric','torch_sparse','scipy')}
    cfg['status']='BOUND_CONFIG_PENDING_ROOT_STAGE_ADMISSION'
    cfg['root_pending']=['Exact root stage admission and finite qualifier budget approval',
        'Actual complete-graph qualification/closed owner and measured full-study schedule',
        'Separate acquisition/CAL/one complete CONFIRM releases and closed owner receipts']
    write(path,cfg)
    print(__import__('json').dumps({'config':describe(path),'execution_admitted':False,'model_fits':0,'y_arrays_requested':False}))


if __name__=='__main__':main()
