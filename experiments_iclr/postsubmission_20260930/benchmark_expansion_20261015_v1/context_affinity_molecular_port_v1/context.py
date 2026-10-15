"""Native per-field atom/bond frequency targets; canonical CPU construction."""
import hashlib
import json


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def native_vocabularies(session):
    native=session.model.native.gnn_node
    atom=tuple(int(layer.num_embeddings) for layer in native.atom_encoder.atom_embedding_list)
    bonds=[tuple(int(layer.num_embeddings) for layer in conv.bond_encoder.bond_embedding_list) for conv in native.convs]
    if len(atom)!=9 or not bonds or len(bonds[0])!=3 or any(v!=bonds[0] for v in bonds):
        raise ValueError("Exactly native nine atom and three bond categorical fields, identical bond vocabularies across source layers")
    return atom,bonds[0]


def _histograms(torch,values,owners,count,vocab):
    if values.ndim!=2 or values.shape[1]!=len(vocab) or values.dtype!=torch.long \
            or owners.dtype!=torch.long or owners.shape!=(len(values),):
        raise ValueError("Actual native categorical records and aligned graph owners required")
    total=torch.bincount(owners,minlength=count)
    fields=[]
    for column,size in enumerate(vocab):
        if type(size)is not int or size<1 or ((values[:,column]<0)|(values[:,column]>=size)).any():
            raise ValueError("Category outside pinned native embedding vocabulary")
        counts=torch.bincount(owners*size+values[:,column],minlength=count*size).reshape(count,size)
        fields.append(counts.to(torch.float32)/total.clamp_min(1)[:,None])
    return torch.cat(fields,-1),total


def build(torch,batch,atom_vocab,bond_vocab):
    """Use actual directed edge records, without changing native tensors."""
    x=batch.x.detach().cpu();edges=batch.edge_index.detach().cpu();attributes=batch.edge_attr.detach().cpu()
    owners=batch.batch.detach().cpu();count=int(batch.num_graphs)
    ids=batch.smoke_graph_id.detach().cpu().reshape(-1)
    if count<1 or ids.dtype!=torch.long or ids.shape!=(count,) or len(torch.unique(ids))!=count \
            or owners.dtype!=torch.long or len(owners)!=len(x) or ((owners<0)|(owners>=count)).any():
        raise ValueError("Actual packed native graph batch and unique source graph IDs required")
    if len(atom_vocab)!=9 or len(bond_vocab)!=3 or edges.dtype!=torch.long or edges.ndim!=2 \
            or edges.shape[0]!=2 or attributes.shape!=(edges.shape[1],len(bond_vocab)) \
            or ((edges<0)|(edges>=len(x))).any():
        raise ValueError("Full nine atom/three bond fields and original directed native edge records required")
    bond_owners=owners[edges[0]]
    if not torch.equal(bond_owners,owners[edges[1]]):raise ValueError("Native molecular edge crosses graph identities")
    a,atom_counts=_histograms(torch,x,owners,count,atom_vocab)
    b,bond_counts=_histograms(torch,attributes,bond_owners,count,bond_vocab)
    if (atom_counts==0).any():raise ValueError("Native source TRAIN molecules must contain atoms")
    receipt={"schema":"native-molecular-frequency-context-v1","graph_ids":ids.tolist(),
        "atom_vocab":list(atom_vocab),"bond_vocab":list(bond_vocab),"atom_counts":atom_counts.tolist(),
        "directed_bond_record_counts":bond_counts.tolist(),"zero_atom_contexts":int((a.norm(dim=-1)<=1e-12).sum()),
        "zero_bond_contexts":int((b.norm(dim=-1)<=1e-12).sum()),"A_shape":list(a.shape),"B_shape":list(b.shape),
        "A_sha256":hashlib.sha256(a.contiguous().numpy().tobytes()).hexdigest(),
        "B_sha256":hashlib.sha256(b.contiguous().numpy().tobytes()).hexdigest(),
        "encoding":"canonical CPU FP32 per-field category counts divided by graph atom/directed-bond count; empty bond fields zero",
        "native_features_edges_changed":False}
    receipt["sha256"]=canonical_hash(receipt)
    return a,b,receipt


def target_diagnostics(torch,a,b,labels,ids):
    """Passive factual target support; follows pinned core .2 donor law."""
    compatible=labels[:,None].eq(labels[None,:])&~ids[:,None].eq(ids[None,:])
    support=compatible.sum(-1);eligible=support>0
    info={"objects":len(labels),"supported_anchors":int(eligible.sum()),"omitted_anchors":int((~eligible).sum()),
        "zero_A":int((a.norm(dim=-1)<=1e-12).sum()),"zero_B":int((b.norm(dim=-1)<=1e-12).sum())}
    if not eligible.any():return info
    targets=[]
    for label,value in (("A",a),("B",b)):
        z=torch.nn.functional.normalize(value,dim=-1,eps=1e-12)
        q=((z[eligible]@z.T)/.2).masked_fill(~compatible[eligible],-torch.inf).softmax(-1)
        uniform=compatible[eligible].to(q)/support[eligible,None]
        info[label+"_nonuniform_anchors_at_1e-8"]=int(((q-uniform).abs().max(-1).values>1e-8).sum())
        targets.append(q)
    difference=(targets[0]-targets[1]).abs()
    info.update(A_B_target_max_abs_difference=float(difference.max()),A_B_target_mean_abs_difference=float(difference.mean()),
        distinct_target_anchors_at_1e_8=int((difference.max(-1).values>1e-8).sum()))
    return info
