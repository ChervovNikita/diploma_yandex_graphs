"""Selected-row native feature/one-hop contexts; no full propagated cache."""
import hashlib
import json


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def tensor_pin(value):
    x=value.detach().cpu().contiguous()
    return {"shape":list(x.shape),"dtype":str(x.dtype),"bytes_sha256":hashlib.sha256(x.numpy().tobytes()).hexdigest()}


def build(torch,features,edge_index,selected_ids):
    x=features.detach().cpu().contiguous();edge=edge_index.detach().cpu().contiguous()
    ids=selected_ids.detach().cpu().contiguous();n=len(x)
    if x.ndim!=2 or x.shape[1]<1 or x.dtype not in(torch.float32,torch.float64)or not torch.isfinite(x).all():
        raise ValueError("Actual finite post-source-preprocessing native features required")
    if edge.dtype!=torch.long or edge.ndim!=2 or edge.shape[0]!=2 or((edge<0)|(edge>=n)).any():
        raise ValueError("Actual native directed edge_index records required")
    if ids.dtype!=torch.long or ids.ndim!=1 or not 0<len(ids)<=512 or len(torch.unique(ids))!=len(ids)or((ids<0)|(ids>=n)).any():
        raise ValueError("Authentic unique bounded source TRAIN node IDs required")
    source,target=edge[0],edge[1];nonself=source!=target
    src,dst=source[nonself],target[nonself]
    # Incoming row degrees of all nonself records + exactly one self per node.
    degree=torch.bincount(dst,minlength=n)+1
    row_of=torch.full((n,),-1,dtype=torch.long);row_of[ids]=torch.arange(len(ids))
    use=row_of[dst]>=0
    rows=torch.cat((row_of[dst[use]],torch.arange(len(ids))))
    columns=torch.cat((src[use],ids))
    destinations=torch.cat((dst[use],ids))
    values=(degree[destinations].to(x.dtype)*degree[columns].to(x.dtype)).rsqrt()
    matrix=torch.sparse_coo_tensor(torch.stack((rows,columns)),values,size=(len(ids),n),dtype=x.dtype)
    a=x[ids].clone();b=torch.sparse.mm(matrix,x)
    if not torch.isfinite(b).all():raise ValueError("Nonfinite selected native one-hop context")
    receipt={"schema":"selected-native-node-context-v1","source_features":tensor_pin(x),"source_edges":tensor_pin(edge),
        "selected_ids":ids.tolist(),"target_row_degrees":degree[ids].tolist(),"all_target_degrees":tensor_pin(degree),
        "removed_native_self_records":int((~nonself).sum()),"retained_nonself_records":len(src),
        "selected_nonself_records":int(use.sum()),"added_target_self_contributions":len(ids),
        "degree_rule":"incoming degree of all original nonself records plus one self per node; nonself duplicates retained",
        "operator":"selected rows of D^-1/2(A_nonself+I)D^-1/2; edge_index[target,source] incoming convention",
        "sparse_shape":[len(ids),n],"sparse_records":len(values),"coalescing_requested":False,
        "A":tensor_pin(a),"B":tensor_pin(b),"full_N_by_D_propagated_cache":False,
        "zero_A":int((a.norm(dim=-1)<=1e-12).sum()),"zero_B":int((b.norm(dim=-1)<=1e-12).sum())}
    receipt["sha256"]=digest(receipt)
    return a,b,receipt


def target_diagnostics(torch,a,b,labels,ids):
    compatible=labels[:,None].eq(labels[None,:])&~ids[:,None].eq(ids[None,:])
    support=compatible.sum(-1);eligible=support>0
    info={"objects":len(labels),"supported_anchors":int(eligible.sum()),"omitted_anchors":int((~eligible).sum()),
        "zero_A":int((a.norm(dim=-1)<=1e-12).sum()),"zero_B":int((b.norm(dim=-1)<=1e-12).sum())}
    if not eligible.any():return info
    targets=[]
    for name,value in(("A",a),("B",b)):
        z=torch.nn.functional.normalize(value,dim=-1,eps=1e-12)
        q=((z[eligible]@z.T)/.2).masked_fill(~compatible[eligible],-torch.inf).softmax(-1)
        uniform=compatible[eligible].to(q)/support[eligible,None]
        info[name+"_nonuniform_anchors_at_1e-8"]=int(((q-uniform).abs().max(-1).values>1e-8).sum())
        targets.append(q)
    difference=(targets[0]-targets[1]).abs()
    info.update(A_B_target_max_abs_difference=float(difference.max()),A_B_target_mean_abs_difference=float(difference.mean()),
        distinct_target_anchors_at_1e_8=int((difference.max(-1).values>1e-8).sum()))
    return info
