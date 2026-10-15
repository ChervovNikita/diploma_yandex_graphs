"""Same-logit pair supervision on frozen MolBACE residual or native bodies."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

HERE=Path(__file__).resolve().parent
E=HERE.parents[1]
OFFSET=(0,100003,200003,300007)
ARMS=('common','structured','permuted','native_common')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def settings(arm,cohort_seed):
    require(arm in ARMS and cohort_seed in (101,202,303),'Frozen arm/cohort required')
    return {'arm':arm,'cohort_seed':cohort_seed,'pair_coefficient':.5,
        'METHOD_sha256':sha(HERE/'METHOD.json'),'roles_sha256':sha(HERE/'roles.py')}


def descriptor(root,cohort_seed,arm,body=None):
    """One in-memory descriptor; no campaign launcher or materialized job queue."""
    require(arm in ARMS and cohort_seed in (101,202,303),'Frozen MolBACE cohort/arm required')
    native=arm=='native_common'
    require((native and type(body)is int and 0<=body<4) or (not native and body is None),'Four independent native body identities required')
    job=json.loads((HERE/'BASE_CASE.json').read_text())
    cohort='molbace_pairwise_competence_v1/split-0/seed-'+str(cohort_seed)
    job.update(id=cohort+'/'+arm+('/body-'+str(body) if native else ''),cohort_id=cohort,
        procedure='native_single' if native else 'molbace_pair_'+arm+'_v1',
        base_procedure='native_single' if native else 'gnnm_boundary_native',
        body=body,seed=cohort_seed+(OFFSET[body] if native else 0),method_settings=settings(arm,cohort_seed))
    job['recipe']['_repository_root']=str(root)
    job['session_transform']={'path':str(Path(root)/'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/molbace_pairwise_competence_v1/extension.py'),
        'function':'transform','sha256':sha(__file__)}
    return job


def verify_sources(root):
    authority=json.loads((HERE/'SOURCES.json').read_text())
    for pin in authority['source_pins']+authority['data_role_pins']:
        path=Path(pin['path']).resolve()
        require(path.is_relative_to(root) and sha(path)==pin['sha256'],'Frozen source/data authority changed: '+str(path))
    require(not any(Path(p).exists() for p in authority['absent_role_overrides']),'TRAIN/VALID role override appeared')
    return authority


def table_from_view(torch,view,ids,atom_vocab,bond_vocab,variant,seed):
    """Sequential TRAIN input reads, no sampler advance and no model forward."""
    from torch_geometric.data import Batch
    context=load(E/'context_affinity_molecular_port_v1/context.py','_pair_native_histograms')
    roles=load(HERE/'roles.py','_molbace_pair_roles')
    require(getattr(view,'role',None)=='TRAIN' and len(view)==len(ids)==1210,'Complete native TRAIN1210 view required')
    histograms=[];graph_ids=[];labels=[];hist_receipts=[]
    for start in range(0,len(view),32):
        batch=Batch.from_data_list([view[i] for i in range(start,min(start+32,len(view)))])
        _,bond,receipt=context.build(torch,batch,atom_vocab,bond_vocab)
        require(tuple(batch.y.shape)==(batch.num_graphs,1),'Native single-task observed-label mask required')
        histograms.extend(bond.tolist());graph_ids.extend(receipt['graph_ids']);hist_receipts.append(receipt['B_sha256'])
        for y in batch.y[:,0].tolist():
            require(y!=y or y in (0.,1.),'Native TRAIN labels must be missing or binary')
            labels.append(None if y!=y else int(y))
    require(graph_ids==ids.tolist(),'Sequential TRAIN histogram identities differ from source role')
    table=roles.Table(graph_ids,histograms,labels,variant=variant,seed=seed)
    receipt={'source':'context_affinity_molecular_port_v1/context.py CPU FP32 B',
        'atom_vocab':list(atom_vocab),'bond_vocab':list(bond_vocab),'TRAIN_B_chunks_sha256':roles.digest(hist_receipts),
        'native_directed_bond_records':True,'same_label_inputs_in_quartiles':True,'TRAIN_inputs_only':True,
        'normalization':'Python float64 L2; exact zero vector remains zero; math.fsum cosine; no cutoff/clipping',
        'role_table':table.metadata()}
    return table,receipt


def objective(torch,own,logits,batch,table=None):
    """Return exactly own on an empty observed pair set, otherwise own+.5R."""
    require(logits.ndim==3 and logits.shape[0] in (1,4) and logits.shape[1:]==batch.y.shape and
        batch.y.shape[1]==1,'Same native binary minibatch logits required')
    ids=batch.smoke_graph_id.detach().cpu().reshape(-1).tolist()
    labels=[]
    for y in batch.y.detach().cpu().reshape(-1).tolist():
        require(y!=y or y in (0.,1.),'Native observed binary label mask required')
        labels.append(None if y!=y else int(y))
    require(len(set(ids))==len(ids),'Distinct source minibatch graphs required')
    if table is None:
        pairs=[(i,j) for i,y in enumerate(labels) if y==1 for j,z in enumerate(labels) if z==0]
        vectors=[(1.,)*logits.shape[0]]*len(pairs)
    else:pairs,vectors=table.batch(ids,labels)
    info={'positive_graphs':sum(y==1 for y in labels),'negative_graphs':sum(y==0 for y in labels),
        'missing_labels':sum(y is None for y in labels),'observed_pairs':len(pairs),'empty_pair_batch':not bool(pairs),
        'same_logits_no_extra_forward':True,'pair_denominator':'observed opposite-label pair count; never weight sum'}
    if not pairs:return own,dict(info,pair_loss=0.,empty_pair_uses_exact_native_F=True,route_pair_mass=[0.]*logits.shape[0])
    indices=torch.tensor(pairs,dtype=torch.long,device=logits.device)
    differences=logits[:,indices[:,0],0]-logits[:,indices[:,1],0]
    weight=torch.tensor(vectors,dtype=logits.dtype,device=logits.device).T
    require(weight.shape==differences.shape and torch.isfinite(weight).all().item(),'Complete route/pair weights required')
    pair_loss=(torch.nn.functional.softplus(-differences)*weight).mean(1).mean()
    return own+.5*pair_loss,dict(info,pair_loss=float(pair_loss.detach()),
        route_pair_mass=weight.detach().double().sum(1).cpu().tolist(),empty_pair_uses_exact_native_F=False)


def transform(session,job):
    arm=job.get('method_settings',{}).get('arm');cohort_seed=job.get('method_settings',{}).get('cohort_seed')
    require(arm in ARMS and cohort_seed in (101,202,303),'Frozen pairwise arm/cohort required')
    expected=descriptor(job['recipe']['_repository_root'],cohort_seed,arm,job.get('body'))
    require(job==expected or (job.get('device')=='cpu' and job==dict(expected,device='cpu')),
        'Only exact frozen descriptors or CPU engineering device override are allowed')
    root=Path(job['recipe']['_repository_root']).resolve();verify_sources(root)
    producer=load(E/'pilot_20261015_residual_only_private_bn_v1/session_extension.py','_pair_residual_private_bn_producer')
    require(session.config==producer.NATIVE_CONFIG and session.procedure==job['base_procedure'] and session.updates==0,
        'Fresh unchanged native MolBACE GIN session required')
    torch=session.torch;started=time.perf_counter();native=arm=='native_common'
    native_model=session.model if native else session.model.native
    native_parameters=tuple(native_model.parameters());optimizer=session.optimizer
    if not native:
        inherited=copy.deepcopy(job);inherited.update(procedure=producer.METHOD,method_settings=producer.settings_for())
        producer.transform(session,inherited)
    require(session.optimizer is optimizer and len(tuple(native_model.parameters()))==len(native_parameters) and
        all(a is b for a,b in zip(native_model.parameters(),native_parameters)),
        'Original native parameters and optimizer must be retained')
    table=None;role_receipt=None
    if not native:
        context=load(E/'context_affinity_molecular_port_v1/context.py','_pair_histogram_vocabularies')
        atom_vocab,bond_vocab=context.native_vocabularies(session)
        authority=json.loads((HERE/'METHOD.json').read_text())['native_vocabularies']
        require(list(atom_vocab)==authority['atom'] and list(bond_vocab)==authority['bond'],'Native vocabulary authority differs')
        table,role_receipt=table_from_view(torch,session.train_view,session.train_ids,atom_vocab,bond_vocab,arm,job['seed'])
    binding={'schema':'molbace-pairwise-competence-binding-v1','job_sha256':load(HERE/'roles.py','_pair_digest').digest(job),
        'method_settings':copy.deepcopy(job['method_settings']),'extension_sha256':sha(__file__),
        'SOURCES_sha256':sha(HERE/'SOURCES.json'),'BASE_CASE_sha256':sha(HERE/'BASE_CASE.json'),
        'role_receipt':role_receipt,'native_common_control':native,'TEST_accessed':False}
    counts={'updates':0,'empty_pair_batches':0,'observed_pairs':0,'positive_graphs':0,'negative_graphs':0,'missing_labels':0,
        'pair_support_histogram':{},'route_pair_mass':[0.]*(1 if native else 4)}
    original_metadata,original_state,original_load=session.metadata,session.state,session.load_state
    optimizer_contract=lambda: (tuple((id(g),tuple(id(p) for p in g['params']),{k:copy.deepcopy(v) for k,v in g.items() if k!='params'})
        for g in optimizer.param_groups),copy.deepcopy(optimizer.defaults))
    contract=optimizer_contract()

    def update(self):
        self.model.train();batch,positions,skipped=self._next_train_batch();self.last_train_positions=positions.clone()
        self.optimizer.zero_grad()
        if native:
            output=self.model(batch);own=self._native_loss(output,batch);logits=output.unsqueeze(0);member_losses=[float(own.detach())]
        else:
            routes=[self.model.run_member(m,batch) for m in range(4)]
            values=torch.stack([self._native_loss(r[0],batch) for r in routes]);own=values.mean()
            logits=torch.stack([r[1] for r in routes]);member_losses=values.detach().cpu().tolist()
        before=time.perf_counter();loss,pair=objective(torch,own,logits,batch,table)
        pair['host_pair_arithmetic_seconds']=time.perf_counter()-before
        require(torch.isfinite(loss).item(),'Finite same-logit pair objective required')
        loss.backward();self.optimizer.step()
        require(optimizer_contract()==contract,'Original live optimizer groups/options/defaults changed')
        if not native:self.model.assert_source_identity()
        self.updates+=1;counts['updates']+=1;counts['empty_pair_batches']+=int(pair['empty_pair_batch'])
        for key in ('observed_pairs','positive_graphs','negative_graphs','missing_labels'):counts[key]+=pair[key]
        key=str(pair['observed_pairs']);counts['pair_support_histogram'][key]=counts['pair_support_histogram'].get(key,0)+1
        for m,value in enumerate(pair['route_pair_mass']):counts['route_pair_mass'][m]+=value
        return {'loss':float(loss.detach()),'native_own_loss':float(own.detach()),'native_member_losses':member_losses,
            'pairwise':pair,'optimizer_updates':1,'update':self.updates,'native_skipped_batches':skipped,
            'native_batch_graphs':len(positions),'TRAIN_graph_ids':batch.smoke_graph_id.detach().cpu().reshape(-1).tolist(),
            'native_route_forwards':1 if native else 4,'work':{'optimizer_transitions':1,'reverse_collections':1,
                'pair_support':pair['observed_pairs'],'empty_pair_batch':int(pair['empty_pair_batch'])}}

    def metadata(self):
        value=original_metadata();value['pairwise_competence']={'binding':copy.deepcopy(binding),'support_counters':copy.deepcopy(counts),
            'construction_and_role_seconds':construction_seconds,'native_parameter_count':sum(p.numel() for p in native_parameters),
            'total_parameter_count':sum(p.numel() for p in self.model.parameters()),
            'training':'native BCE + .5 mean route pair softplus; one scalar backward for all native/private coordinates',
            'serving':'unchanged native raw logits; four-route mean for shared, ordinary own logit for native control',
            'time_scope':'host elapsed only; actual CUDA latency/memory must be measured separately','scientific_qualification':False}
        return value

    def state(self):
        value=original_state();value['pairwise_competence']={'binding':copy.deepcopy(binding),'support_counters':copy.deepcopy(counts)}
        return value

    def restore(self,value):
        saved=value.get('pairwise_competence',{})
        require(saved.get('binding')==binding,'Pair recipe/source/TRAIN role/rule/permutation restore authority differs')
        source=dict(value);source.pop('pairwise_competence');original_load(source)
        counts.clear();counts.update(copy.deepcopy(saved['support_counters']))
        require(counts['updates']==self.updates,'Pair support and original update clocks differ')
        # Genuine native optimizer restoration may replace dictionaries; bind
        # the restored live dictionary identities once, retaining parameters.
        nonlocal contract
        restored=optimizer_contract()
        require(tuple(v[1:] for v in restored[0])==tuple(v[1:] for v in contract[0]) and restored[1]==contract[1],
            'Restore changed original optimizer parameter identities/options/defaults')
        contract=restored

    construction_seconds=time.perf_counter()-started
    session.pairwise_table=table;session.pairwise_binding=copy.deepcopy(binding)
    session.train_update=types.MethodType(update,session);session.metadata=types.MethodType(metadata,session)
    session.state=types.MethodType(state,session);session.load_state=types.MethodType(restore,session)
    return session
