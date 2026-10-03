"""Actual-author model correspondence and fabricated CPU training only.

Requires root's Torch2.1.2, NumPy/SciPy and isolated CPU DGL1.1.3. No original
dataset, labels, published scores, main driver or GPU execution is permitted.
"""
import ast
import contextlib
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True


def custody():
    packet = Path(__file__).resolve().parent
    provenance = json.loads((packet/'PROVENANCE.json').read_text())
    for row in provenance['inputs']:
        path = packet.parent/row['path']; data = path.read_bytes()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],row['path']
    return packet,provenance


def actual_object(path,name,namespace,portability=False):
    source = path.read_text()
    if portability:
        needle = 'torch.FloatTensor([1e-12]).cuda()'
        assert source.count(needle)==1
        source = source.replace(needle,'torch.FloatTensor([1e-12])')
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name==name]
    assert len(matches)==1,name
    exec(compile(ast.Module(body=matches,type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def main():
    packet,provenance = custody()
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    import dgl
    from dgl import function as fn
    from dgl.nn.pytorch import edge_softmax
    from dgl.nn.pytorch.utils import Identity
    from dgl.utils import expand_as_pair
    import native_models as subject
    import native_inputs as data
    import train_native as training
    assert torch.__version__.split('+')[0]=='2.1.2'
    assert dgl.__version__=='1.1.3'
    torch.set_num_threads(1); torch.manual_seed(1729)
    checks = []
    def close(name,left,right,tolerance=2e-9):
        torch.testing.assert_close(left,right,atol=tolerance,rtol=tolerance); checks.append(name)
    def check(name,condition):
        assert condition,name; checks.append(name)
    source = packet.parent/provenance['reference_source_directory']
    namespace = dict(th=torch,torch=torch,nn=nn,fn=fn,edge_softmax=edge_softmax,
                     Identity=Identity,expand_as_pair=expand_as_pair,DGLError=dgl.DGLError)
    legacy = actual_object(source/'dgl_v043__python__dgl__nn__pytorch__conv__gatconv.py','GATConv',namespace)
    simple = actual_object(source/'hgb__NC__benchmark__methods__baseline__conv.py','myGATConv',namespace)
    wrapper_namespace = dict(torch=torch,nn=nn,GATConv=legacy,myGATConv=simple)
    actual_gat = actual_object(source/'hgb__NC__benchmark__methods__GNN__GNN.py','GAT',wrapper_namespace)
    actual_simple = actual_object(source/'hgb__NC__benchmark__methods__baseline__GNN.py','myGAT',wrapper_namespace,True)
    counts = {'0':6,'1':4,'2':3,'3':2}; shifts = {'0':0,'1':6,'2':10,'3':13}
    pairs = [(0,0),(0,1),(1,0),(2,1),(2,2),(3,2),(4,3)]
    pt = [(0,0),(0,1),(1,1),(2,2),(3,0)]
    pv = [(0,0),(1,0),(2,1),(3,1)]
    edges = {0:{p:1. for p in pairs},1:{p:1. for p in pt},2:{p:1. for p in pv},
             3:{(v,u):1. for u,v in pairs},4:{(v,u):1. for u,v in pt},5:{(v,u):1. for u,v in pv}}
    schema = dict(node_counts=counts,node_shifts=shifts,relations=[dict(raw_id=k,source=s,target=t) for k,(s,t) in data.EXPECTED_ENDPOINTS.items()])
    records = data.homogeneous_records(schema,edges)
    graph = data.materialize_homogeneous(records,subject,'cpu')
    author_graph = dgl.graph((graph.src,graph.dst),num_nodes=graph.nodes)
    dims = [3,4,2,2]
    # Models are compared after a one-to-one actual-author parameter transfer;
    # initialization draws of the independent port are not called reproduction.
    for name,is_simple,constructor in [('GAT',False,actual_gat),('Simple_HGN',True,actual_simple)]:
        args = (author_graph,3,13,dims,4,4,2,[2,2,1],F.elu,.25,.25,.05,True,.05) if is_simple else (author_graph,dims,4,4,2,[2,2,1],F.elu,.25,.25,.05,False)
        author = constructor(*args).double()
        port = subject.HGBGAT(dims,classes=4,simple=is_simple,hidden=4,heads=2,edge_dim=3,dropout=.25).double()
        port.load_state_dict(author.state_dict(),strict=True)
        for training_mode in (False,True):
            author.train(training_mode); port.train(training_mode)
            author.zero_grad(); port.zero_grad()
            af = [torch.randn(counts[str(i)],d,dtype=torch.double,requires_grad=True) for i,d in enumerate(dims)]
            pf = [x.detach().clone().requires_grad_() for x in af]
            torch.manual_seed(101)
            ao = author(af,graph.etype) if is_simple else author(af)
            author_rng = torch.get_rng_state().clone()
            torch.manual_seed(101); po = port(graph,pf); port_rng = torch.get_rng_state().clone()
            close(name+'_actual_logits_'+str(training_mode),po,ao)
            check(name+'_matched_dropout_draws_'+str(training_mode),torch.equal(author_rng,port_rng))
            objective = torch.arange(ao.numel(),dtype=torch.double).reshape_as(ao)/ao.numel()
            (ao*objective).sum().backward(); (po*objective).sum().backward()
            check(name+'_parameter_name_mapping',set(dict(author.named_parameters()))==set(dict(port.named_parameters())))
            for an,ap in author.named_parameters():
                pp = dict(port.named_parameters())[an]
                if ap.grad is None or pp.grad is None:
                    check(name+'_unused_gradient_'+an,ap.grad is None and pp.grad is None)
                else:
                    close(name+'_actual_gradient_'+an+'_'+str(training_mode),pp.grad,ap.grad)
            for i,(a,p) in enumerate(zip(af,pf)):
                close(name+'_actual_input_gradient_'+str(i)+'_'+str(training_mode),p.grad,a.grad)
        identity_args = (author_graph,3,13,list(counts.values()),4,4,2,[2,2,1],F.elu,0.,0.,.05,True,.05) if is_simple else (author_graph,list(counts.values()),4,4,2,[2,2,1],F.elu,0.,0.,.05,False)
        identity_author = constructor(*identity_args).double().eval()
        identity_port = subject.HGBGAT(list(counts.values()),simple=is_simple,hidden=4,heads=2,edge_dim=3,dropout=0.).double().eval()
        identity_port.load_state_dict(identity_author.state_dict(),strict=True)
        identity = [torch.eye(n,dtype=torch.double).to_sparse() for n in counts.values()]
        identity_actual = identity_author(identity,graph.etype) if is_simple else identity_author(identity)
        identity_optimized = identity_port(graph)
        close(name+'_actual_identity_projection_without_dense_identity',identity_optimized,identity_actual)
        identity_objective = torch.arange(identity_actual.numel(),dtype=torch.double).reshape_as(identity_actual)/identity_actual.numel()
        (identity_actual*identity_objective).sum().backward(); (identity_optimized*identity_objective).sum().backward()
        for i,(afc,pfc) in enumerate(zip(identity_author.fc_list,identity_port.fc_list)):
            close(name+'_actual_optimized_identity_weight_gradient_'+str(i),pfc.weight.grad,afc.weight.grad)
            close(name+'_actual_optimized_identity_bias_gradient_'+str(i),pfc.bias.grad,afc.bias.grad)
    _,attention = port.gat_layers[0](graph,torch.randn(graph.nodes,4,dtype=torch.double))
    check('Simple_HGN_attention_residual_is_detached',not attention.requires_grad)

    # Execute the preserved actual SeHGNN model, removing only its unused dense
    # branch import of torch_sparse. No model statements are replaced.
    sepath = source/'sehgnn__hgb__model.py'; tree = ast.parse(sepath.read_text())
    tree.body = [n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='torch_sparse')]
    se_namespace = {'SparseTensor':type('UnusedSparseTensorSentinel',(),{})}
    exec(compile(tree,str(sepath),'exec'),se_namespace)
    size = {'A':3,'AP':4,'APA':3,'APT':2,'APV':2}; label_keys = ['APA','APAPA','APTPA','APVPA']
    author = se_namespace['SeHGNN']('DBLP',4,8,4,size.keys(),label_keys,'A',.25,.25,0.,2,3,'none',True,data_size=size).double()
    with torch.no_grad():
        author.semantic_fusion.gamma.fill_(.37)
    port = subject.SeHGNN(size,label_keys,embed=4,hidden=8,dropout=.25,input_drop=.25).double()
    port.load_state_dict(author.state_dict(),strict=True)
    for mode in (False,True):
        author.train(mode); port.train(mode); author.zero_grad(); port.zero_grad()
        af = {k:torch.randn(6,v,dtype=torch.double,requires_grad=True) for k,v in size.items()}
        al = {k:torch.randn(6,4,dtype=torch.double,requires_grad=True) for k in label_keys}
        pf = {k:v.detach().clone().requires_grad_() for k,v in af.items()}
        pl = {k:v.detach().clone().requires_grad_() for k,v in al.items()}
        batch = torch.arange(6)
        torch.manual_seed(303); ao = author(batch,af,al); author_rng = torch.get_rng_state().clone()
        torch.manual_seed(303); po = port(batch,pf,pl)
        close('SeHGNN_actual_logits_'+str(mode),po,ao)
        check('SeHGNN_matched_dropout_draws_'+str(mode),torch.equal(author_rng,torch.get_rng_state()))
        target = torch.tensor([0,1,2,3,0,1]); F.cross_entropy(ao,target).backward(); F.cross_entropy(po,target).backward()
        for name,param in author.named_parameters():
            other = dict(port.named_parameters())[name]
            close('SeHGNN_actual_gradient_'+name+'_'+str(mode),other.grad,param.grad)
        for group,ag,pg in [('feature',af,pf),('label',al,pl)]:
            for key in ag:
                close('SeHGNN_actual_'+group+'_gradient_'+key+'_'+str(mode),pg[key].grad,ag[key].grad)

    # Native DGL feature propagation versus the optimized CPU SciPy port.
    attributes = {'0':np.arange(18,dtype=np.float32).reshape(6,3).tolist(),
                  '1':np.arange(16,dtype=np.float32).reshape(4,4).tolist(),
                  '2':np.arange(6,dtype=np.float32).reshape(3,2).tolist(),'3':[]}
    adjs = data.normalized_native_adjacencies(schema,edges)
    features = data.feature_channels(schema,attributes,adjs)
    dgl_edges = {}; raw = {}
    for key,adj in adjs.items():
        coo = adj.tocoo(); dgl_edges[(key[-1],key[-1]+'-'+key[0],key[0])] = (torch.tensor(coo.col),torch.tensor(coo.row))
    hg = dgl.heterograph(dgl_edges,num_nodes_dict={data.LETTERS[t]:n for t,n in counts.items()})
    for t,n in counts.items():
        key = data.LETTERS[t]
        raw[key] = torch.tensor(attributes[t],dtype=torch.float32) if attributes[t] else torch.eye(n)
        hg.nodes[key].data[key] = raw[key]
    function_namespace = dict(fn=fn,gc=gc,torch=torch)
    propagate = actual_object(source/'sehgnn__hgb__utils.py','hg_propagate_feat_dgl',function_namespace)
    with contextlib.redirect_stdout(io.StringIO()):
        propagated = propagate(hg,'A',2,3,[],False)
    check('SeHGNN_actual_DGL_feature_keys',set(propagated.nodes['A'].data)==set(features))
    for key,value in features.items():
        close('SeHGNN_actual_DGL_feature_mean_'+key,value,propagated.nodes['A'].data[key],2e-6)

    # Preserve actual author metapath extension code with a dense reference
    # operator. This qualifies path and diagonal mathematics, not PyG kernels.
    class DenseOperator:
        def __init__(self,value): self.value = value
        def clone(self): return DenseOperator(self.value.clone())
        def to(self,device): return DenseOperator(self.value.to(device))
        def matmul(self,other): return DenseOperator(self.value@other.value)
    dense_adjs = {k:DenseOperator(torch.from_numpy(v.toarray()).double()) for k,v in adjs.items()}
    sparse_propagate = actual_object(source/'sehgnn__hgb__utils.py','hg_propagate_sparse_pyg',dict(gc=gc,torch=torch))
    author_products = sparse_propagate(dense_adjs,'A',4,5,[],False,False,'cpu')
    products = data.label_products(adjs)
    check('SeHGNN_actual_label_path_keys',set(products)==set(author_products))
    for key,value in products.items():
        close('SeHGNN_actual_complete_label_product_'+key,torch.from_numpy(value.toarray()).double(),author_products[key].value,2e-6)
    split = dict(train_ids=[0,1,2,3],validation_ids=[4,5]); labels = [0,1,2,3]
    label_features = data.train_only_label_channels(products,6,split['train_ids'],labels)
    onehot = F.one_hot(torch.tensor(labels),4).double(); onehot = torch.cat([onehot,torch.zeros(2,4,dtype=torch.double)])
    for key,value in author_products.items():
        corrected = value.value.clone(); corrected.fill_diagonal_(0.)
        close('SeHGNN_complete_product_diagonal_removed_'+key,label_features[key].double(),corrected@onehot,2e-6)
    eval_ids = data.native_evaluation_ids(6,split)
    check('SeHGNN_topology_only_native_evaluation_composition',eval_ids==[0,1,2,3,4,5])

    # Meaningful serialization check: dropout and shuffled minibatches continue
    # identically for every runnable family after weights_only checkpoint load.
    optimizer_steps = 0
    for arm in training.ARMS:
        def build():
            return subject.SeHGNN({k:v.shape[1] for k,v in features.items()},label_features.keys(),embed=4,hidden=8,dropout=.25,input_drop=.25) if arm=='native_SeHGNN' else subject.HGBGAT(list(counts.values()),simple=arm=='native_Simple_HGN',hidden=4,heads=2,edge_dim=3,dropout=.25)
        training.seed_all(torch,131,torch.device('cpu')); model = build()
        optimizer = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
        loader = torch.utils.data.DataLoader(split['train_ids'],batch_size=10000,shuffle=True,drop_last=False) if arm=='native_SeHGNN' else None
        ids,y = torch.tensor(split['train_ids']),torch.tensor(labels)
        def update(m,o,l):
            nonlocal optimizer_steps
            result = training.train_epoch(torch,m,o,arm,graph,features,label_features,ids,y,l,None)
            optimizer_steps += 1; return result
        for _ in range(2): update(model,optimizer,loader)
        buffer = io.BytesIO()
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),RNG=training.global_state(torch,torch.device('cpu'))),buffer)
        expected = [update(model,optimizer,loader) for _ in range(3)]
        expected_state = {k:v.clone() for k,v in model.state_dict().items()}
        expected_rng = torch.get_rng_state().clone()
        restored = build(); ropt = torch.optim.Adam(restored.parameters(),lr=.001,weight_decay=0.)
        buffer.seek(0); saved = torch.load(buffer,weights_only=True)
        restored.load_state_dict(saved['model']); ropt.load_state_dict(saved['optimizer'])
        training.restore_global(torch,saved['RNG'],torch.device('cpu'))
        rloader = torch.utils.data.DataLoader(split['train_ids'],batch_size=10000,shuffle=True,drop_last=False) if arm=='native_SeHGNN' else None
        replay = [update(restored,ropt,rloader) for _ in range(3)]
        check(arm+'_safe_weights_only_loss_trajectory',expected==replay)
        for key,value in restored.state_dict().items():
            check(arm+'_safe_weights_only_state_'+key,torch.equal(value,expected_state[key]))
        check(arm+'_safe_weights_only_RNG',torch.equal(expected_rng,torch.get_rng_state()))
    stopper = training.NativeStopper('native_Simple_HGN')
    check('Simple_HGN_latest_validation_tie_replaces',stopper.observe(1,1.)[0] and stopper.observe(2,1.)[0])
    stopper = training.NativeStopper('native_SeHGNN')
    check('SeHGNN_strict_tie_does_not_replace',stopper.observe(1,1.)[0] and not stopper.observe(2,1.)[0])
    for epoch in range(3,52): stopper.observe(epoch,1.)
    check('SeHGNN_native_stops_after_51_nonimprovements',stopper.observe(52,1.)[1])
    print(json.dumps(dict(status='PASS',checks=checks,synthetic_CPU_only=True,optimizer_steps_executed=optimizer_steps,
        actual_author_models=['preserved DGL0.4.3 GATConv/HGB wrapper','preserved Simple-HGN conv/wrapper with epsilon CPU portability','preserved SeHGNN dense DBLP model'],
        feature_propagation_actual_DGL=True,label_path_reference='actual author extension function plus independent dense operator; native torch_sparse kernel not claimed',
        original_dataset_labels_or_published_scores_read=False,main_driver_launched=False,GPU_execution=False)))


if __name__=='__main__':
    main()
