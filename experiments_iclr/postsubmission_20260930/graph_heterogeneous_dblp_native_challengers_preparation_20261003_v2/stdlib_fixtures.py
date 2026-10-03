"""Fabricated ZIP/topology checks only, no numerical runtime or original data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import zipfile
import native_inputs as subject
import train_native as driver


def main():
    packet = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('qualified_loader',packet.parent/'graph_heterogeneous_dblp_training_preparation_20261003_v2/dblp_inputs.py')
    loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
    checks = []
    def check(name,condition):
        assert condition,name; checks.append(name)
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory)/'fake.zip'
        with zipfile.ZipFile(archive,'w') as stream:
            stream.writestr('DBLP/node.dat','0\ta\t0\t1,2\n1\tp\t1\t3,4,5\n2\tt\t2\t6\n3\tv\t3\n')
            stream.writestr('DBLP/link.dat','0\t1\t0\t1\n1\t2\t1\t1\n1\t3\t2\t1\n1\t0\t3\t1\n2\t1\t4\t1\n3\t1\t5\t1\n')
            stream.writestr('DBLP/label.dat','MUST_NOT_OPEN_DEVELOPMENT\n')
            stream.writestr('DBLP/label.dat.test','MUST_NOT_OPEN_HELDOUT\n')
        desc = dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        opened = []; original = zipfile.ZipFile.open
        def spy(self,name,*args,**kwargs):
            opened.append(name); return original(self,name,*args,**kwargs)
        zipfile.ZipFile.open = spy
        try:
            schema,attributes,edges = subject.stream_all_attributes(loader,desc,dict(nodes='DBLP/node.dat',links='DBLP/link.dat'))
        finally:
            zipfile.ZipFile.open = original
        check('only_nodes_links_opened_across_all_feature_passes',opened==['DBLP/node.dat','DBLP/link.dat','DBLP/node.dat'])
        check('all_provided_attributes_retained',attributes=={'0':[[1.,2.]],'1':[[3.,4.,5.]],'2':[[6.]],'3':[]})
        records = subject.homogeneous_records(schema,edges)
        check('native_simple_full_relation_self_support',records['audit']['homogeneous_edges']==10 and records['num_etypes']==13)
        check('native_simple_self_types',records['etype'][-4:]==[6]*4)
        check('native_simple_no_synthetic_reverse_when_released',records['audit']['reverse_labels_synthetic']==0)
        check('native_simple_CSR_then_self_order',list(zip(records['src'],records['dst']))==[(0,1),(1,0),(1,2),(1,3),(2,1),(3,1),(0,0),(1,1),(2,2),(3,3)])
        bad = dict(edges); bad[0] = {(0,0):2.}
        # Attribute policy and finite checks are inside the actual loader; do not
        # write mirror assertions pretending to certify unseen numeric kernels.
    stopper = driver.NativeStopper('native_Simple_HGN')
    check('simple_first_post_update_eligible',stopper.observe(1,2.)==(True,False))
    check('simple_latest_tie_replaces_resets',stopper.observe(2,2.)==(True,False) and stopper.counter==0)
    for epoch in range(3,32): stopper.observe(epoch,3.)
    check('simple_native_patience30',stopper.observe(32,3.)==(False,True))
    stopper = driver.NativeStopper('native_SeHGNN')
    check('sehgnn_first_strict_tie_keeps_first',stopper.observe(1,2.)==(True,False) and stopper.observe(2,2.)==(False,False) and stopper.best_epoch==1)
    for epoch in range(3,52): stopper.observe(epoch,3.)
    check('sehgnn_native_51_nonimproving_epochs',stopper.observe(52,3.)==(False,True))
    check('native_topology_eval_order_without_TEST_membership',subject.native_evaluation_ids(6,dict(train_ids=[0,2],validation_ids=[1]))==[0,2,1,3,4,5])
    print(json.dumps(dict(status='PASS',checks=checks,fabricated_only=True,Torch_real_data_labels_remote_GPU_or_training_execution=False)))


if __name__=='__main__':
    main()
