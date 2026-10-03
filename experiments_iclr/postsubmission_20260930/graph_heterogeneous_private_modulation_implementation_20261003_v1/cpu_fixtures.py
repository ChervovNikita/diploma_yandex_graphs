"""Synthetic CPU correspondence/gradient fixtures. Never loads a released dataset.

Default includes preserved actual-author DGL model. --torch-only is explicitly
partial and cannot certify author-model correspondence. Run later with root's
Torch2.1.2 and qualified modern CPU DGL; no native training/optimizer is launched.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import types


def custody():
    packet = Path(__file__).resolve().parent
    provenance = json.loads((packet/'PROVENANCE.json').read_text())
    for row in provenance['inputs']:
        path = packet.parent/row['path']
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest()==row['sha256'] and len(data)==row['bytes'],row['path']
    return packet,provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--torch-only',action='store_true')
    args = parser.parse_args(argv)
    packet,provenance = custody()
    import torch
    from torch import nn
    from torch.nn import functional as F
    import hgt_private as subject
    assert torch.__version__.split('+')[0]=='2.1.2','Pinned Torch2.1.2 required'
    torch.set_num_threads(1)
    torch.manual_seed(314159)
    checks = []

    def close(name,left,right,atol=2e-10):
        torch.testing.assert_close(left,right,atol=atol,rtol=atol)
        checks.append(name)

    def check(name,condition):
        assert condition,name
        checks.append(name)

    # Parallel same-type raw relations, unequal degrees, isolated per-relation
    # rows, and all types having incoming relations expose normalization errors.
    long = lambda values:torch.tensor(values,dtype=torch.long)
    relations = [subject.Relation(2,'0','0',long([0,1,2]),long([1,2,3])),
                 subject.Relation(9,'0','0',long([1,2,3,0,2]),long([0,0,0,1,1])),
                 subject.Relation(3,'0','1',long([0,1,3]),long([0,1,2])),
                 subject.Relation(4,'1','0',long([0,1,2,1]),long([0,1,2,3])),
                 subject.Relation(5,'1','2',long([0,1,2]),long([0,1,1])),
                 subject.Relation(6,'2','1',long([0,1,0]),long([0,1,2]))]
    dgl_version,relation_names = None,None
    if not args.torch_only:
        import dgl
        dgl_version = dgl.__version__
        ordering_graph = dgl.heterograph({r.canonical:(r.src,r.dst) for r in relations},
                                         num_nodes_dict={'0':4,'1':3,'2':2},device='cpu')
        # Derive native parameter-row order from actual modern DGL, not a guess.
        relation_names = tuple(ordering_graph.etypes)
    graph = subject.HeteroGraph({'0':4,'1':3,'2':2},relations,relation_names)
    dims = {'0':3,'1':3,'2':2}
    features = {'0':torch.randn(4,3,dtype=torch.double),
                '1':torch.eye(3,dtype=torch.double),'2':torch.eye(2,dtype=torch.double)}
    original_features = {t:value.clone() for t,value in features.items()}
    ids,labels = long([0,2,3]),long([0,1,2])
    core = subject.NativeHGT(graph,dims,8,3,2,heads=2,use_norm=True).double().eval()
    factor_arms = subject.initialized_factors(8,6,2,generic_seed=2718,dtype=torch.double)
    arms = subject.matched_arms(core,factor_arms)
    arms = {mode:arm.double().eval() for mode,arm in arms.items()}
    logits,states = arms['cp'](graph,features,'0',return_states=True)
    unrestricted = arms['unrestricted'](graph,features,'0')
    close('CP_unrestricted_initial_member_logits_match',logits,unrestricted)
    close('fixed_mean_raw_logits',subject.mean_logits(logits),sum(logits)/4)
    close('ordinary_mean_own_member_CE',subject.mean_member_ce(logits,ids,labels),
          sum(F.cross_entropy(value[ids],labels) for value in logits)/4)
    check('distinct_private_layer_frames',all(states[m][layer][t].data_ptr()!=states[n][layer][t].data_ptr()
          for m in range(4) for n in range(m) for layer in range(3) for t in graph.ntypes))
    # A single-member perturbation propagates through the entire recurrence;
    # other members cannot acquire it through a shared hidden-state cache.
    changed = copy.deepcopy(arms['cp'])
    with torch.no_grad():
        changed.factors[0].a[0,0] += 0.37
    changed_logits,changed_states = changed(graph,features,'0',return_states=True)
    close('single_member_perturbation_does_not_overwrite_other_routes',changed_logits[1:],logits[1:])
    check('private_recurrence_changes_later_member_state',not torch.allclose(changed_states[0][2]['0'],states[0][2]['0']))
    single_member = copy.deepcopy(arms['cp'])
    F.cross_entropy(single_member(graph,features,'0')[0,ids],labels).backward()
    check('private_recurrence_gradients_do_not_enter_other_member_factors',all(
          torch.count_nonzero(layer.a.grad[1:])==0 and torch.count_nonzero(layer.b.grad[1:])==0
          and torch.count_nonzero(layer.c.grad[1:])==0 and torch.count_nonzero(layer.a.grad[0])>0
          for layer in single_member.factors))
    check('canonical_raw_relation_names_do_not_collapse_same_type_relations',
          len(graph.edge_dict)==6 and len([r for r in graph.relations if r.source==r.target])==2)
    check('shared_features_never_mutated',all(torch.equal(features[t],original_features[t]) for t in features))

    # Neighbor-normalized relation outputs average equally, irrespective of
    # degree: one 0-valued edge and three 10-valued edges ->5, not7.5.
    first = subject.relation_sum(torch.zeros(1,1,dtype=torch.double),
            torch.zeros(1,1,1,dtype=torch.double),long([0]),2)
    second = subject.relation_sum(torch.zeros(3,1,dtype=torch.double),
             torch.full((3,1,1),10.,dtype=torch.double),long([0,0,0]),2)
    close('relation_local_softmax_then_mean_including_zero_degree_rows',
          torch.stack([first,second]).mean(0),torch.tensor([[[5.]],[[0.]]],dtype=torch.double))

    # Independent dense block-diagonal affine reference, with noncommuting R,
    # nonzero V bias and h=0: input scale must not scale bias, output scale must.
    h = torch.tensor([[0.,0.,0.,0.],[1.,-2.,3.,0.5]],dtype=torch.double)
    a = torch.tensor([2.,-1.,0.5,3.],dtype=torch.double,requires_grad=True)
    weight = torch.arange(1,17,dtype=torch.double).reshape(4,4)/11
    bias = torch.tensor([0.7,-0.2,0.5,1.1],dtype=torch.double)
    transforms = torch.tensor([[[1.,2.],[3.,1.]],[[2.,-1.],[0.5,3.]]],dtype=torch.double)
    b = torch.tensor([1.,-1.,1.,-1.],dtype=torch.double,requires_grad=True)
    c = torch.tensor([-.03,.02],dtype=torch.double,requires_grad=True)
    q = torch.tensor([-1.,1.],dtype=torch.double,requires_grad=True)
    u = torch.tensor([1.,-1.,-1.,1.],dtype=torch.double,requires_grad=True)
    s = b[None,None,:]+c[:,None,None]*q[None,:,None]*u[None,None,:]
    transformed = torch.bmm(F.linear(h*a,weight,bias).view(-1,2,2).transpose(1,0),transforms).transpose(1,0).reshape(-1,4)
    block = torch.block_diag(*transforms)
    close('affine_site_head_order_and_shared_bias',transformed,(h*a)@weight.T@block+bias@block)
    scaled = transformed[None,None,:,:]*s[:,:,None,:]
    probe = torch.arange(1,scaled.numel()+1,dtype=torch.double).reshape_as(scaled)/13
    loss = (scaled.sin()*probe).sum()
    gradient_s = torch.autograd.grad(loss,s,retain_graph=True)[0]
    gc,gq,gu,gb = torch.autograd.grad(loss,(c,q,u,b))
    close('CP_site_gradient_c',gc,(gradient_s*q[None,:,None]*u).sum((1,2)))
    close('CP_site_gradient_q',gq,(gradient_s*c[:,None,None]*u).sum((0,2)))
    close('CP_site_gradient_u',gu,(gradient_s*c[:,None,None]*q[None,:,None]).sum((0,1)))
    close('CP_base_b_gradient_sums_relations',gb,gradient_s.sum((0,1)))
    check('moving_output_scale_before_relation_changes_affine_operator',
          not torch.allclose((F.linear(h*a,weight,bias)*s[0,0])@block,transformed*s[0,0]))

    # Sparse identity adapters match the dense identity computation.
    sparse = dict(features,**{'1':features['1'].to_sparse(),'2':features['2'].to_sparse()})
    close('sparse_identity_native_adapter_equivalence',core(graph,sparse,'0'),core(graph,features,'0'))
    # Persistent member streams replay dropout without changing caller RNG.
    stochastic = subject.PrivateHGT(copy.deepcopy(core)).train()
    streams = subject.MemberStreams([41,43,47,53])
    saved = streams.state_dict(); caller = torch.get_rng_state().clone()
    first = stochastic(graph,features,'0',streams)
    streams.load_state_dict(saved)
    close('independent_member_dropout_streams_replay',stochastic(graph,features,'0',streams),first,atol=0)
    check('dropout_caller_RNG_restored',torch.equal(caller,torch.get_rng_state()))
    check('factor_off_private_dropout_routes_distinct',not torch.equal(first[0],first[1]))
    check('CP_overhead_formula',subject.parameter_count(arms['cp'])-subject.parameter_count(arms['be'])==2*(4+6+8))

    if not args.torch_only:
        actual_path = packet.parent/provenance['actual_author_model']
        spec = importlib.util.spec_from_file_location('preserved_actual_HGB_HGT',actual_path)
        author = importlib.util.module_from_spec(spec); spec.loader.exec_module(author)

        def dgl_graph(current_features):
            value = dgl.heterograph({r.canonical:(r.src,r.dst) for r in graph.relations},
                                   num_nodes_dict=graph.node_counts,device='cpu')
            check('actual_DGL_node_and_relation_order',tuple(value.ntypes)==graph.ntypes
                  and tuple(value.etypes)==tuple(graph.edge_dict))
            value.node_dict = dict(graph.node_dict); value.edge_dict = dict(graph.edge_dict)
            for r in graph.relations:
                value.edges[r.canonical].data['id'] = torch.full_like(r.src,graph.edge_dict[r.name])
            for t in graph.ntypes:
                value.nodes[t].data['inp'] = current_features[t]
            return value

        def compare_gradients(name,model_a,model_b):
            rows_a,rows_b = dict(model_a.named_parameters()),dict(model_b.named_parameters())
            check(name+'_named_parameter_keys',set(rows_a)==set(rows_b))
            for key in rows_a:
                left,right = rows_a[key].grad,rows_b[key].grad
                assert (left is None)==(right is None),name+'_gradient_presence:'+key
                if left is not None:
                    # Aggregate this meaningful numerical check rather than
                    # generating a large list of implementation assertions.
                    torch.testing.assert_close(left,right,atol=3e-9,rtol=3e-9)
            checks.append(name+'_all_named_gradients')

        for use_norm in (False,True):
            torch_core = subject.NativeHGT(graph,dims,8,3,2,heads=2,use_norm=use_norm).double().eval()
            actual = author.HGT(dgl_graph(features),[dims[t] for t in graph.ntypes],8,3,2,2,use_norm).double().eval()
            actual.load_state_dict(torch_core.state_dict(),strict=True)
            feature_a = {t:value.clone().requires_grad_() for t,value in features.items()}
            feature_b = {t:value.clone().requires_grad_() for t,value in features.items()}
            output_a,output_b = torch_core(graph,feature_a,'0'),actual(dgl_graph(feature_b),'0')
            close('actual_author_factor_off_logits_norm'+str(use_norm),output_a,output_b)
            F.cross_entropy(output_a[ids],labels).backward(); F.cross_entropy(output_b[ids],labels).backward()
            compare_gradients('actual_author_factor_off_norm'+str(use_norm),torch_core,actual)
            for t in features:
                close('actual_author_input_gradient_norm'+str(use_norm)+':'+t,feature_a[t].grad,feature_b[t].grad)
            close('actual_author_sparse_adapter_norm'+str(use_norm),
                  torch_core(graph,sparse,'0'),actual(dgl_graph(sparse),'0'))

        # Reuse actual author forward/reduce/recurrence; only install the fixed
        # two value sites through hooks. No reimplementation of the DGL model.
        actual = author.HGT(dgl_graph(features),[dims[t] for t in graph.ntypes],8,3,2,2,True).double().eval()
        actual.load_state_dict(arms['cp'].core.state_dict(),strict=True)
        actual_factors = copy.deepcopy(arms['cp'].factors)
        active = [0]
        for index,layer in enumerate(actual.gcs):
            factors = actual_factors[index]
            for v in layer.v_linears:
                v.register_forward_pre_hook(lambda module,inputs,factors=factors:(inputs[0]*factors.a[active[0]],))
            original = layer.edge_attention
            def modulated(self,edges,original=original,factors=factors):
                result = original(edges)
                result['v'] = result['v']*factors.output(active[0],int(edges.data['id'][0])).view(1,self.n_heads,self.d_k)
                return result
            layer.edge_attention = types.MethodType(modulated,layer)
        value = dgl_graph(features)
        actual_members = []
        for member in range(4):
            active[0] = member
            with value.local_scope():
                actual_members.append(actual(value,'0'))
        actual_logits = torch.stack(actual_members)
        cp = arms['cp']; cp.zero_grad()
        torch_logits = cp(graph,features,'0')
        close('actual_author_CP_four_private_trajectories',torch_logits,actual_logits)
        objective_a,objective_b = subject.mean_member_ce(torch_logits,ids,labels),subject.mean_member_ce(actual_logits,ids,labels)
        close('actual_author_CP_mean_member_CE',objective_a,objective_b)
        objective_a.backward(); objective_b.backward()
        compare_gradients('actual_author_CP_core',cp.core,actual)
        compare_gradients('actual_author_CP_factors',cp.factors,actual_factors)
    print(json.dumps(dict(status='PASS',synthetic_CPU_only=True,torch=torch.__version__,dgl=dgl_version,
                         actual_author_correspondence=not args.torch_only,checks=checks,
                         canonical_raw_relation_to_parameter_row=graph.raw_relation_to_row,
                         no_data_labels_remote_GPU_or_training_execution=True)))


if __name__=='__main__':
    main()
