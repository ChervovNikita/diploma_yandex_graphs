"""Future root CPU qualification of repaired actual HGEN/PyG on fabricated graphs.

No original arrays/labels, training driver, fitted score or GPU execution.
Requires Torch2.1.2, PyG2.6.1 and NumPy/SciPy in an isolated root environment.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True


def custody():
    packet = Path(__file__).resolve().parent
    manifest = (packet/'MANIFEST.json').read_bytes()
    assert hashlib.sha256(manifest).hexdigest()==json.loads((packet/'SEAL.json').read_text())['manifest_sha256']
    for row in json.loads(manifest)['payload']:
        raw = (packet/row['path']).read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],row['path']
    provenance = json.loads((packet/'PROVENANCE.json').read_text())
    for row in provenance['inputs']:
        raw = (packet.parent/row['path']).read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],row['path']
    return packet,provenance


def main():
    packet,provenance = custody()
    import numpy as np
    import torch
    from torch.nn import functional as F
    import torch_geometric
    from torch_geometric.data import Data,HeteroData
    from torch_geometric.transforms import AddMetaPaths
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    import hgen_adapter as subject
    import hgen_inputs as inputs
    assert torch.__version__.split('+')[0]=='2.1.2'
    assert torch_geometric.__version__=='2.6.1'
    torch.set_num_threads(1); torch.manual_seed(131)
    checks = []
    def close(name,a,b,tolerance=2e-9):
        torch.testing.assert_close(a,b,atol=tolerance,rtol=tolerance); checks.append(name)
    def check(name,value):
        assert value,name; checks.append(name)
    # Mechanical repair executes only the original354-line parseable prefix.
    # No model class or algorithm statement is rewritten in this namespace.
    path = packet.parent/provenance['HGEN_model_source']
    source = ''.join(path.read_text().splitlines(keepends=True)[:354]); ast.parse(source)
    namespace = {}; exec(compile(source,str(path)+'[orphan-tail-removed]','exec'),namespace)
    native_default = namespace['MultiGCN'](334,64,4,8,3,.1,2,3)
    port_default = subject.HGEN(334)
    check('actual_source_code_default_HGB334_capacity',sum(p.numel() for p in native_default.parameters())==
        sum(p.numel() for p in port_default.parameters())==312525)
    del native_default,port_default
    counts = {'0':6,'1':4,'2':3,'3':2}
    ap = [(0,0),(0,1),(1,0),(2,1),(2,2),(3,2),(4,3)]
    pt = [(0,0),(0,1),(1,1),(2,2),(3,0)]; pv = [(0,0),(1,0),(2,1),(3,1)]
    edges = {0:{p:1. for p in ap},1:{p:1. for p in pt},2:{p:1. for p in pv},
             3:{(v,u):1. for u,v in ap},4:{(v,u):1. for u,v in pt},5:{(v,u):1. for u,v in pv}}
    schema = dict(node_counts=counts,relations=[dict(raw_id=k,source=s,target=t) for k,(s,t) in inputs.EXPECTED_ENDPOINTS.items()])
    names = {'0':'author','1':'paper','2':'term','3':'conference'}
    hetero = HeteroData()
    for t,n in counts.items(): hetero[names[t]].num_nodes = n
    for row in schema['relations']:
        pairs = list(edges[row['raw_id']]); hetero[(names[row['source']],'to',names[row['target']])].edge_index = torch.tensor(pairs).T.contiguous()
    paths = [[('author','paper'),('paper','author')],
             [('author','paper'),('paper','term'),('term','paper'),('paper','author')],
             [('author','paper'),('paper','conference'),('conference','paper'),('paper','author')]]
    native = AddMetaPaths(metapaths=paths,drop_orig_edge_types=True,drop_unconnected_node_types=True)(hetero)
    supports = inputs.binary_metapaths(schema,edges)
    views,audit = inputs.normalized_views(supports,subject,dtype=torch.double)
    edge_indices = []
    for i,name in enumerate(subject.VIEW_NAMES):
        index = native['author',f'metapath_{i}','author'].edge_index
        edge_indices.append(index)
        actual = set(zip(index[0].tolist(),index[1].tolist()))
        coo = supports[name].tocoo()
        check('actual_PyG_AddMetaPaths_binary_support_'+name,actual==set(zip(coo.row.tolist(),coo.col.tolist())))
        normalized_index,weights = gcn_norm(index,None,6,False,True,'source_to_target',torch.double)
        dense = torch.zeros(6,6,dtype=torch.double)
        dense.index_put_((normalized_index[1],normalized_index[0]),weights,accumulate=True)
        close('actual_PyG_GCN_normalization_'+name,views[i].adjacency.to_dense(),dense)
    check('isolated_target_retained_with_native_unit_loop',all(float(v.adjacency.to_dense()[5,5])==1. for v in views))
    author = namespace['MultiGCN'](3,8,4,4,3,.25,2,3).double()
    port = subject.HGEN(3,hidden=8,attention_dim=4,members=3,dropout=.25,layers=2).double()
    port.load_state_dict(author.state_dict(),strict=True)
    check('one_to_one_author_parameter_names',set(dict(author.named_parameters()))==set(dict(port.named_parameters())))
    check('unused_parameters_included_in_capacity',sum(p.numel() for p in author.parameters())==sum(p.numel() for p in port.parameters()))
    train_ids = torch.tensor([0,1,2,3]); labels = torch.tensor([0,1,2,3])
    for mode in (False,True):
        author.train(mode); port.train(mode); author.zero_grad(); port.zero_grad()
        ax = torch.randn(6,3,dtype=torch.double,requires_grad=True); px = ax.detach().clone().requires_grad_()
        graphs = [Data(x=ax,edge_index=i) for i in edge_indices]
        torch.manual_seed(137); ao,ag,ae = author(graphs); actual_rng = torch.get_rng_state().clone()
        torch.manual_seed(137); po,pg,pe = port(px,views)
        close('actual_author_log_probabilities_'+str(mode),po,ao)
        close('actual_author_Gram_'+str(mode),pg,ag)
        close('actual_author_summed_view_embeddings_'+str(mode),pe,ae)
        check('unused_first_pass_dropout_draws_preserved_'+str(mode),torch.equal(actual_rng,torch.get_rng_state()))
        # The only settled training objective is the documented lambda_cov0.
        aloss = F.nll_loss(ao[train_ids],labels)+0.*torch.norm(ag,p=1)**2
        ploss,pnll,penalty = subject.objective(po,pg,train_ids,labels)
        close('code_default_lambda0_objective_'+str(mode),ploss,aloss)
        close('code_squared_Gram_penalty_returned_'+str(mode),penalty,torch.norm(ag,p=1)**2)
        aloss.backward(); ploss.backward()
        for name,value in author.named_parameters():
            other = dict(port.named_parameters())[name]
            if value.grad is None or other.grad is None:
                check('source_unused_gradient_'+name+'_'+str(mode),value.grad is None and other.grad is None)
            else:
                close('actual_author_parameter_gradient_'+name+'_'+str(mode),other.grad,value.grad)
        close('actual_author_input_gradient_'+str(mode),px.grad,ax.grad)
    # Degenerate source fusion is exposed, not smoothed using an invented value.
    fusion = subject.AttentionH(8,4,3,'code_literal').double()
    with torch.no_grad(): fusion.agg.weight.zero_(); fusion.agg.bias.zero_()
    try:
        fusion([torch.ones(2,8,dtype=torch.double) for _ in range(3)])
        raise AssertionError('Zero attention range was silently repaired')
    except ValueError as error:
        check('zero_range_fail_stop_no_epsilon', 'zero range' in str(error))
    try:
        subject.objective(torch.zeros(4,4),torch.eye(3),train_ids,labels,lambda_cov=.1)
        raise AssertionError('Unsettled regularized recipe admitted')
    except ValueError as error:
        check('nonzero_lambda_requires_separate_qualification','separate' in str(error))
    print(json.dumps(dict(status='PASS',checks=checks,actual_author_parse_repair='orphan model tail355-359 removed in memory only',
        source_profile='code_literal finite function,lambda_cov0',author_PyG_GCN_and_AddMetaPaths=True,
        synthetic_CPU_only=True,optimizer_steps_executed=0,original_data_labels_or_fitted_scores_read=False,
        training_driver_launched=False,GPU_execution=False)))


if __name__=='__main__': main()
