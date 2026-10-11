"""Pinned reset-after GRU backend; injected Torch, live effective tensor list."""

def require_runtime(torch):
    if torch.__version__.split('+')[0]!='2.1.2':
        raise ValueError('Official Torch2.1.2 _VF.gru API is the sole bound implementation')

def pack_tour(torch,tour):
    """CPU-only immutable component packing; no randomness or CUDA token loop."""
    if tour['tokens'].device.type!='cpu' or tour['offsets'].device.type!='cpu':
        raise ValueError('Qualified CPU tour/offset payloads before packing')
    offsets=tour['offsets'].tolist();lengths=[b-a for a,b in zip(offsets[:-1],offsets[1:])]
    if len(lengths)!=tour['components'] or any(n<=0 for n in lengths):raise ValueError('Complete nonempty components')
    order=sorted(range(len(lengths)),key=lambda c:(-lengths[c],c));gather=[];batches=[]
    for time in range(max(lengths)):
        active=[c for c in order if time<lengths[c]];batches.append(len(active))
        gather.extend(offsets[c]+time for c in active)
    if sorted(gather)!=list(range(tour['length'])):raise ValueError('Packing must preserve every occurrence')
    index=torch.tensor(gather,dtype=torch.int64,device='cpu');inverse=torch.argsort(index)
    return dict(tokens=tour['tokens'],reverse_index=tour['reverse_index'],occurrences=tour['occurrences'],packed_tokens=tour['tokens'].index_select(0,index),reverse_packed_tokens=tour['tokens'].index_select(0,tour['reverse_index']).index_select(0,index),unpack_to_tokens=inverse,packing_index=index,batch_sizes=torch.tensor(batches,dtype=torch.int64,device='cpu'),components=tour['components'],length=tour['length'],offsets=tour['offsets'])

def to_device(packed,device):
    """batch_sizes stays CPU as required by the official packed ABI."""
    result=dict(packed)
    for name in ('tokens','reverse_index','occurrences','packed_tokens','reverse_packed_tokens','unpack_to_tokens','packing_index'):
        result[name]=packed[name].to(device)
    return result

def sequence(torch,inputs,weights,packed,training):
    require_runtime(torch)
    if inputs.ndim!=2 or len(inputs)!=packed['length'] or len(weights)!=4 or packed['batch_sizes'].device.type!='cpu':
        raise ValueError('Complete packed input, four live gate tensors, CPU batch_sizes')
    hidden=inputs.shape[1];initial=inputs.new_zeros((1,packed['components'],hidden))
    # Exact tensor-list overload from official nn.GRU.forward v2.1.2: no
    # Parameter copying/functional_call/no_grad flattening or derived cache.
    output,_=torch._VF.gru(inputs,packed['batch_sizes'],initial,weights,True,1,0.,training,False)
    return output.index_select(0,packed['unpack_to_tokens'])

def node_memory(torch,hidden,weights,packed,training):
    forward=sequence(torch,hidden.index_select(0,packed['packed_tokens']),weights,packed,training)
    reverse=sequence(torch,hidden.index_select(0,packed['reverse_packed_tokens']),weights,packed,training).index_select(0,packed['reverse_index'])
    paired=(forward+reverse)*.5
    return torch.zeros_like(hidden).index_add(0,packed['tokens'],paired)/packed['occurrences'].to(hidden.dtype)[:,None]
