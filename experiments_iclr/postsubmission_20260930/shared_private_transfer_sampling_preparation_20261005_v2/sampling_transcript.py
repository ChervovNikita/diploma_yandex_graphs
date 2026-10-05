"""Root-released finite TRAIN-only native sampling transcript; no model or optimizer."""
import argparse
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import socket
import sys
import time

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args()
    assert sha(args.release)==args.release_sha256
    release=json.loads(args.release.read_text())
    assert release['sampling_execution_enabled'] is True
    assert release['models_authorized'] is False and release['fits_authorized'] is False
    assert release['VALID_values_access'] is False and release['TEST_access'] is False
    assert release['program_sha256']==sha(__file__)
    assert release['seeds']==[20261005,0,1,2]
    cycles=release['cycles'];assert cycles and cycles==list(range(max(cycles)+1)) and max(cycles)<60
    assert release['outer_size']==64 and release['inner_size']==256 and release['members']==4
    assert platform.python_version()=='3.11.14'
    assert os.environ.get('PYTHONPATH','')==release['declared_PYTHONPATH'] and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    repo=Path(release['repository']);phase=repo/'experiments_iclr/postsubmission_20260930'
    assert Path.cwd().resolve()==repo and socket.gethostname()==release['expected_hostname']
    source=phase/release['training_source_name']
    assert sha(source/'SOURCE_MANIFEST.json')==release['source_manifest_sha256']
    for row in json.loads((source/'SOURCE_MANIFEST.json').read_text())['files']:
        assert sha(source/row['path'])==row['sha256'] and (source/row['path']).stat().st_size==row['bytes']
    geom=phase/'endpoint_episode_geometry_preparation_20261005_v1'
    expected={'episode_geometry.py':'f5562c94c8c90b999065e6b570f5a0e7c734f3f4f949274f98eeaf9859bf2321',
              'native_episode_cycle.py':'8cd5596459a389aeaf435dcf5ecec819ae94c19a734f24ecf27c96cb4b97c0b9'}
    for name,value in expected.items():assert sha(geom/name)==value
    out=Path(release['output_directory'])
    assert not out.exists() and out.resolve().is_relative_to(phase.resolve()) and out.parent.is_dir()
    custody=load('transcript_custody',source/'custody.py')
    geometry=load('transcript_geometry',geom/'episode_geometry.py')
    native=load('transcript_native_cycle',geom/'native_episode_cycle.py')
    assert custody.PHASE==phase
    versions={}
    providers={}
    for name in ('numpy','torch','torch_geometric','torch_sparse','torch_scatter'):
        module=importlib.import_module(name)
        versions[name]=str(module.__version__)
        providers[name]={'path':module.__file__,'sha256':sha(module.__file__)}
    import torch
    import numpy as np
    versions['CUDA']=torch.version.cuda
    assert versions==release['runtime_versions']
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    job={'available_manifest_relative':release['available_manifest_relative'],
         'available_manifest_sha256':release['available_manifest_sha256']}
    x,train,_,_,inputs=custody.load_inputs(torch,job,include_valid=False)
    pairs=train.tolist();assert len(pairs)==3870 and len(x)==3327
    train_hash=digest(pairs)
    out.mkdir()
    started=time.monotonic();transcripts=[]

    def rng():
        numpy=np.random.get_state()
        return {'python':digest(random.getstate()),'numpy':digest([numpy[0],numpy[1].tolist(),numpy[2],numpy[3],numpy[4]]),
                'torch_cpu':hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest()}

    for seed in release['seeds']:
        endpoint_streams=geometry.route_streams(seed,4)
        random_streams=geometry.route_streams(seed,4,control=True)
        for cycle_index in cycles:
            before=rng()
            negative,order,episodes=custody.make_pair(torch,geometry,native,train,len(x),seed,cycle_index,
                64,256,endpoint_streams,random_streams)
            assert rng()==before
            parts={'TRAIN_pairs':pairs,'native_negative_bank':negative.tolist(),'outer_permutation':order,
                   'endpoint_inner':[row[0]['inner'] for row in episodes],
                   'matched_random_inner':[row[1]['inner'] for row in episodes],
                   'outer_ids_and_endpoints':[{key:row[0][key] for key in ('outer_pos_ids','outer_neg_ids','outer_endpoints')} for row in episodes],
                   'support_indices':[{'removed':sorted(set(range(len(train)))-set(row[2])),'kept':row[2]} for row in episodes]}
            hashes={key:digest(value) for key,value in parts.items()}
            raw=canonical({'seed':seed,'cycle':cycle_index,'components':parts})
            name=f'seed{seed}_cycle{cycle_index:03d}.json.gz'
            with (out/name).open('xb') as stream:
                with gzip.GzipFile(filename='',fileobj=stream,mode='wb',mtime=0,compresslevel=9) as gz:gz.write(raw)
            row={'seed':seed,'cycle':cycle_index,'component_hashes':hashes,'canonical_sha256':hashlib.sha256(raw).hexdigest(),
                 'engineering_artifact':{'path':name,'bytes':(out/name).stat().st_size,'sha256':sha(out/name)},
                 'negative_bank_rows':len(negative),'outer_order_rows':len(order),'episodes_including_tail':len(episodes),
                 'last_outer_rows':len(episodes[-1][0]['outer_pos_ids']),'global_rng_unchanged':True}
            transcripts.append(row)
            print(json.dumps({'seed':seed,'cycle':cycle_index,'canonical_sha256':row['canonical_sha256']}),flush=True)
            assert time.monotonic()-started<release['soft_seconds']
    result={'status':'PASS','release_sha256':args.release_sha256,'program_sha256':sha(__file__),
            'source_manifest_sha256':release['source_manifest_sha256'],'source_custody_sha256':sha(source/'custody.py'),
            'geometry_sha256':expected,'runtime_versions':versions,'providers':providers,'python':sys.version,
            'hostname':socket.gethostname(),'declared_PYTHONPATH':release['declared_PYTHONPATH'],'sys_path':sys.path,'inputs':inputs,'TRAIN_pair_hash':train_hash,'transcripts':transcripts,
            'wall_seconds':time.monotonic()-started,'models_imported':False,'optimizers_created':False,
            'fits':0,'VALID_TEST_values_access':False,'CUDA_devices_visible':0}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':
    main()
