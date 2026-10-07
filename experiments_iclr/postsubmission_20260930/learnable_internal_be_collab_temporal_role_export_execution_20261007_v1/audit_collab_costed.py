"""Complete Collab-only temporal projection audit; no models or scoring."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
import zipfile

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
HERE=Path(__file__).resolve().parent
SOURCE=PHASE/'learnable_internal_be_contrastive_multitask_suite_20261007_v5'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['collab'],required=True)
    args=parser.parse_args();started=time.monotonic()
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()!=[GPU]:
        raise ValueError('Wrong authorized singleton')
    if Path.cwd().resolve()!=REPO or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not HERE.is_relative_to(PHASE):
        raise ValueError('Exact owned CPU audit context')
    job=json.loads((HERE/'jobs/collab.json').read_text())
    if job['data_export_authorized'] is not True or job['scientific_fit_authorized'] is not False or job['predictive_scoring_authorized'] is not False or job['TEST_access'] is not False:
        raise ValueError('Exact data-only root release')
    sys.path.insert(0,str(SOURCE))
    from runtime import verify_manifest,runtime_versions,bound,allocation
    allocation(cpu_check=True)
    source_sha=verify_manifest()
    if source_sha!=job['source_manifest_sha256'] or source_sha!='ecff016e9a68af1625293bf0166d1428bca20d7c2ac12b64b37914dda5c3af3b':
        raise ValueError('Exact reviewed temporal source')
    review=json.loads(bound(PHASE,job['source_review']).read_text())
    if review['approved'] is not True or review['source_manifest_sha256']!=source_sha:
        raise ValueError('Exact temporal source review')
    versions=runtime_versions()
    from data import load_projection
    import numpy as np
    import pandas as pd
    import torch
    path=HERE/'receipts/collab_OUTPUT_AUDIT_COSTED.json'
    if path.exists():raise ValueError('One audit attempt, no overwrite')
    output=HERE/'outputs/collab'
    binding={'path':str((output/'ROLE_MANIFEST.json').relative_to(PHASE)),'sha256':sha(output/'ROLE_MANIFEST.json')}
    train,valid,manifest=load_projection(PHASE,binding,'collab',True)
    plan=json.loads((SOURCE/'EXPORT_CUSTODY_PLAN.json').read_text())['collab']
    if manifest['source_custody']!=plan or manifest['source_manifest_sha256']!=source_sha:
        raise ValueError('Exact authentic export source custody')
    if manifest['VALID_pairs_used_to_filter_TRAIN_support'] is not False or manifest['task_target']!='official_temporal_event_prediction':
        raise ValueError('No future-pair filtering of authentic history')
    supplemental=json.loads((HERE/'SUPPLEMENTAL_ORIGINAL_ARRAY_DIGESTS.json').read_text())
    def digest(value):
        value=np.ascontiguousarray(value)
        h=hashlib.sha256(str((value.shape,value.dtype)).encode());h.update(memoryview(value).cast('B'))
        return h.hexdigest()
    def tensor_digest(value):return digest(value.detach().contiguous().numpy())
    fields={role:{key:{'shape':list(value.shape),'dtype':str(value.dtype),'bytes':value.numel()*value.element_size(),'tensor_sha256':tensor_digest(value)}
        for key,value in payload.items()} for role,payload in [('train',train),('valid',valid)]}
    actual={'TRAIN_edge':tensor_digest(train['positive']),'VALID_edge':tensor_digest(valid['positive']),
        'VALID_negative':tensor_digest(valid['negative']),'raw_features':tensor_digest(train['x']),
        'TRAIN_year':tensor_digest(train['positive_year']),'VALID_year':tensor_digest(valid['positive_year'])}
    if actual!=plan['expected_public_array_digests']:
        raise ValueError('Complete original projected arrays/year digests')
    archive=bound(PHASE,plan['archive'])
    origin=json.loads(bound(PHASE,plan['singleton_transfer_receipt']).read_text())
    if origin['GPU_uuid']!=GPU or origin['archive_sha256']!=sha(archive):
        raise ValueError('Authentic singleton archive transfer custody')
    opened=[]
    with zipfile.ZipFile(archive) as packed:
        if len(packed.namelist())!=len(set(packed.namelist())):raise ValueError('Repeated archive member')
        def read(name):
            if name not in plan['members']:raise ValueError('Nonallowlisted/TEST archive member')
            payload=packed.read(name)
            if hashlib.sha256(payload).hexdigest()!=plan['members'][name]:raise ValueError('Authentic member changed')
            opened.append(name);return payload
        original_train=torch.load(io.BytesIO(read('collab/split/time/train.pt')),map_location='cpu',weights_only=False)
        original_valid=torch.load(io.BytesIO(read('collab/split/time/valid.pt')),map_location='cpu',weights_only=False)
        x=pd.read_csv(io.BytesIO(gzip.decompress(read('collab/raw/node-feat.csv.gz'))),header=None).values.astype(np.float32)
        raw=np.loadtxt(io.BytesIO(gzip.decompress(read('collab/raw/edge.csv.gz'))),delimiter=',',dtype=np.int64)
    if set(opened)!=set(plan['members']) or len(opened)!=4:
        raise ValueError('Exact four allowed raw/TRAIN/VALID source members')
    if set(original_train)!={'edge','weight','year'} or set(original_valid)!={'edge','weight','year','edge_neg'}:
        raise ValueError('Exact original temporal split dictionaries')
    def array(value):return value.numpy() if isinstance(value,torch.Tensor) else value
    original_train={key:array(value) for key,value in original_train.items()}
    original_valid={key:array(value) for key,value in original_valid.items()}
    originals={'train':{'x':x,'positive':original_train['edge'],'positive_year':original_train['year']},
        'valid':{'positive':original_valid['edge'],'positive_year':original_valid['year'],'negative':original_valid['edge_neg']}}
    for role,payload in [('train',train),('valid',valid)]:
        for key,value in payload.items():
            projected=value.numpy();original=originals[role][key]
            if projected.shape!=original.shape or projected.dtype!=original.dtype or not np.array_equal(projected,original):
                raise ValueError('Complete ordered original field equality: '+role+'/'+key)
    raw_digest=digest(raw)
    if raw_digest!=supplemental['ordered_raw_graph_sha256']:
        raise ValueError('Original ordered raw topology array identity')
    weight_digests={}
    for role,original in [('TRAIN',original_train),('VALID',original_valid)]:
        expected=supplemental[role+'_weight'];weight=original['weight']
        weight_digests[role]=digest(weight)
        if list(weight.shape)!=expected['shape'] or str(weight.dtype)!=expected['dtype'] or weight_digests[role]!=expected['sha256']:
            raise ValueError('Original unused weight record custody')
    train_pairs=original_train['edge'].min(1)*235868+original_train['edge'].max(1)
    raw_pairs=raw.min(1)*235868+raw.max(1)
    valid_pairs=original_valid['edge'].min(1)*235868+original_valid['edge'].max(1)
    if raw.shape!=original_train['edge'].shape or not np.array_equal(np.sort(raw_pairs),np.sort(train_pairs)):
        raise ValueError('Complete duplicate-preserving raw/TRAIN correspondence')
    overlap=np.isin(valid_pairs,train_pairs)
    counts={'VALID_event_rows':int(overlap.sum()),'VALID_distinct_canonical_pairs':int(len(np.unique(valid_pairs[overlap])))}
    if manifest['historical_canonical_pair_overlap']!=counts:
        raise ValueError('Historical pair overlap counts changed')
    years={'TRAIN':{'count':len(train_pairs),'min':int(original_train['year'].min()),'max':int(original_train['year'].max()),'shape':list(original_train['year'].shape),'dtype':str(original_train['year'].dtype)},
        'VALID':{'count':len(valid_pairs),'min':int(original_valid['year'].min()),'max':int(original_valid['year'].max()),'shape':list(original_valid['year'].shape),'dtype':str(original_valid['year'].dtype)}}
    if years['TRAIN']['min']!=1963 or years['TRAIN']['max']!=2017 or years['VALID']['min']!=2018 or years['VALID']['max']!=2018:
        raise ValueError('Complete authentic temporal event roles')
    unique=int(len(np.unique(train_pairs)))
    canonical={'TRAIN_records':len(train_pairs),'TRAIN_unique_canonical_pairs':unique,'TRAIN_extra_repeated_canonical_records':len(train_pairs)-unique,
        'VALID_positive_records':len(valid_pairs),'VALID_unique_canonical_pairs':int(len(np.unique(valid_pairs))),'VALID_negative_records':len(valid['negative']),
        'historical_canonical_pair_overlap':counts,'TRAIN_event_rows_whose_pair_occurs_in_VALID':int(np.isin(train_pairs,valid_pairs).sum()),
        'event_year_roles':years,'complete_public_tensor_digests_matched':True,'complete_ordered_original_fields_equal':True,
        'complete_duplicate_preserving_raw_TRAIN_correspondence':True,'support_uses_only_authentic_past_TRAIN_records':True,
        'VALID_pairs_used_to_filter_TRAIN_support':False,'model_year_features_added':False}
    if torch.cuda.is_initialized():raise ValueError('CPU data audit initialized CUDA')
    usage=resource.getrusage(resource.RUSAGE_SELF)
    receipt={'schema':'internal-be-official-safe-role-output-audit-v2','task':'collab','passed':True,'source_manifest_sha256':source_sha,
        'source_review':job['source_review'],'role_manifest':binding,'role_fields':fields,'canonical_roles':canonical,
        'original_projected_array_digests':actual,'original_ordered_raw_graph_sha256':raw_digest,
        'original_UNUSED_weight_array_digests':weight_digests,
        'authentic_archive':plan['archive'],'authentic_singleton_transfer_receipt':plan['singleton_transfer_receipt'],
        'archive_members_opened':opened,'authentic_archive_and_member_custody_matched':True,'versions':versions,
        'inclusive_CPU_audit_seconds':time.monotonic()-started,'user_CPU_seconds':usage.ru_utime,'system_CPU_seconds':usage.ru_stime,
        'peak_RSS_bytes':int(usage.ru_maxrss)*1024,'input_blocks':usage.ru_inblock,'output_blocks':usage.ru_oublock,
        'auditor_program_sha256':sha(__file__),'source_data_custody_matched':True,'complete_schema_domain_guards_passed':True,
        'pre_deserialization_NPZ_inventory_checked':True,'CPU_only':True,'cuda_initialized':False,'model_constructions':0,
        'scientific_fits':0,'predictive_metrics_computed':False,'TEST_target_values_parsed':False,
        'TEST_features_or_labels_supplied_to_training':False,'binaries_server_only':True,'automatic_retry':False}
    path.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'task':'collab','role_manifest':binding,'canonical_roles':canonical,
        'audit_seconds':receipt['inclusive_CPU_audit_seconds'],'CPU_seconds':usage.ru_utime+usage.ru_stime,'peak_RSS_bytes':receipt['peak_RSS_bytes']}))


if __name__=='__main__':main()
