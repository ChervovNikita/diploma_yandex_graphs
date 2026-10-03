"""Focused loader custody/topology/feature and native selection fixtures; fabricated input only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import zipfile

packet = Path(__file__).resolve().parent
def load(name):
    spec = importlib.util.spec_from_file_location(name,packet/(name+'.py'))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value
inputs,driver = load('dblp_inputs'),load('train_dblp')
checks = []
def check(name, condition):
    assert condition,name; checks.append(name)

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory); archive = root/'fabricated.zip'
    # Two parallel raw same-type relations and a duplicate whose weights cancel
    # expose collapsing/endpoints-only/self/zero-sum support mistakes.
    with zipfile.ZipFile(archive,'w') as zipped:
        zipped.writestr('DBLP/node.dat','0\ta\t0\t1,2\n1\tb\t0\t3,4\n2\tc\t1\t8,9,10\n')
        zipped.writestr('DBLP/link.dat','0\t1\t0\t1\n0\t1\t0\t-1\n1\t0\t1\t1\n0\t2\t2\t1\n2\t0\t3\t1\n')
        zipped.writestr('DBLP/label.dat.test','synthetic forbidden sentinel')
    data = archive.read_bytes(); descriptor = dict(path=str(archive),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
    original_open = zipfile.ZipFile.open; opened = []
    def tracked(self,name,*args,**kwargs):
        assert name in ('DBLP/node.dat','DBLP/link.dat'),'Forbidden label member opened'
        opened.append(name); return original_open(self,name,*args,**kwargs)
    with patch.object(zipfile.ZipFile,'open',tracked):
        schema,attributes,edges = inputs.stream_schema(descriptor,dict(nodes='DBLP/node.dat',links='DBLP/link.dat'))
    check('only_node_link_members_opened',opened==['DBLP/node.dat','DBLP/link.dat'] and schema['label_members_opened']==[])
    check('native_feature2_target_attributes_other_identity_dimensions',schema['input_dims']=={'0':2,'1':1} and attributes==[[1.,2.],[3.,4.]])
    check('raw_relation_parallel_same_type_preserved',len(schema['relations'])==4 and edges[0] is not edges[1])
    check('native_duplicate_coalescing_retains_stored_zero_sum_support',schema['raw_link_records']==5 and schema['support_edges']==4
          and edges[0]=={(0,1):0.0} and schema['relations'][0]['duplicates_coalesced']==1)
    check('streamed_uncompressed_member_fingerprints',set(schema['member_sha256'])=={'nodes','links'} and all(len(s)==64 for s in schema['member_sha256'].values()))
    try:
        inputs.stream_schema(descriptor,dict(nodes='DBLP/label.dat.test',links='DBLP/link.dat'))
    except ValueError:
        check('forbidden_member_name_refused_before_open',True)
    else:
        raise AssertionError('Forbidden member accepted')
stop = driver.NativeEarlyStop()
check('first_post_update_checkpoint_eligible',stop.observe(1.0)==(True,False))
check('native_tie_replaces_and_resets',stop.observe(1.0)==(True,False) and stop.counter==0)
for _ in range(29):
    check_condition = stop.observe(2.0)==(False,False)
    assert check_condition
check('native_thirtieth_worse_epoch_stops',stop.observe(2.0)==(False,True))
check('descriptive_paired_uncertainty_retains_five_deltas',driver.paired_uncertainty([1.,2.,3.,4.,5.])['mean']==3.)
rows = [dict(seed=seed,arm=arm,status='selected',selection=dict(validation_NLL=1. if arm=='CP' else 2.))
        for seed in driver.SEEDS for arm in ('native_HGT','global_BE','CP')]
rows[-1]['status']='failed'
check('failed_fit_closes_subset_comparison',driver.paired_development(rows,('native_HGT','global_BE','CP'))['successful_subset_scored'] is False)
print(json.dumps(dict(status='PASS',checks=checks,fabricated_input_only=True,Torch_remote_GPU_real_data_or_label_payload_execution=False)))
