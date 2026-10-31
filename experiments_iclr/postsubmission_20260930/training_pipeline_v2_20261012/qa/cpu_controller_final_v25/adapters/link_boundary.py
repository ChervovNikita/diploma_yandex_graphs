"""Original two-boundary parameters adapted to pinned native NCN/NCNC links.

No added transforms: first GCN affine W/bias and terminal decoder affine map
are replaced. Native sparse GCN normalization/dropout/forward is called intact.
BE B replaces the existing GCN bias AFTER propagation. Shared members use
sequential own balanced BCE/M backwards and one native two-group Adam step.
"""
from __future__ import annotations
import ast
import copy
from contextlib import contextmanager
import random

from . import link as native_link
try:
    from ..engine import AdapterStepError
    from ..metrics import EpochMeans
except ImportError:
    from engine import AdapterStepError
    from metrics import EpochMeans

BOUNDARY_SOURCE = {"bytes":10977,"sha256":"07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303"}
EXTRA = {"boundary", "members", "source_optimizer_count", "train_minibatches_per_epoch",
         "is_original_gnnm", "dropout_seed", "shuffle_seed", "graphmodelsource"}
CONTRACT = {"map_replacements":["encoder.convs.0.lin_and_postpropagation_bias","decoder.lin.terminal_affine"],
    "added_affine_maps":0,"first_bias":"BE B AFTER unchanged native GCN propagation",
    "native_adjacency":"unchanged GCNConv/DropAdj/SparseTensor normalization,self-loops and mask",
    "initialization":"exact original BEBlock W Xavier; first R Rademacher; terminal R/S ones; B zero",
    "ownership":"one shared native encoder/decoder and one two-group Adam per fit",
    "training":"frozen shared support/queries/negatives; sequential member balanced BCE/M backwards; one Adam step",
    "serving":"mean raw binary logits; native one-body/independent controls use same raw score mean",
    "optimizer_departure":"native encoder/predictor two-group Adam replaces historical original AdamW decay0",
    "original_task_replication":False,"CAL":"absent; no SELECT reuse","source_epochs":100}


def base_config(config):
    return {k:copy.deepcopy(v) for k,v in config.items() if k not in EXTRA}


def validate_config(config, run):
    if not EXTRA <= set(config):raise ValueError("Explicit link boundary and compute contract required")
    native_link.validate_config(base_config(config),run)
    b=config["boundary"]
    if set(b)!={"kind","source"} or b["kind"] not in {"ordinary","original_boundary"}:
        raise ValueError("Declare ordinary or exact original boundary parameters")
    native_link._receipt(b["source"])
    if any(b["source"][k]!=BOUNDARY_SOURCE[k] for k in BOUNDARY_SOURCE):raise ValueError("Original BEBlock source changed")
    if config["graphmodelsource"]!=config["link_source"]:raise ValueError("Bind exact graph/decoder model source")
    m=config["members"]
    if type(m) is not int or m not in (1,4) or config["source_optimizer_count"]!=1:
        raise ValueError("One optimizer body with explicit one/four routes required")
    if m==4 and (b["kind"]!="original_boundary" or config["arm"]!="native_single" or run["body"]!=0):
        raise ValueError("Original shared4 is one body; genuine I4 is four member-equivalent paid fits")
    if config["is_original_gnnm"] is not (m==4):raise ValueError("Original GNNM reference tag must mean shared4")
    if type(config["train_minibatches_per_epoch"]) is not int or config["train_minibatches_per_epoch"]<1:
        raise ValueError("Bind positive exact floor TRAIN minibatch count")
    if config["training"]["tail_policy"]!="native_drop_tail":raise ValueError("Preserve native floor batching")
    if config["shuffle_seed"]!=config["initialization_seed"]:
        raise ValueError("Separate sampler/permutation stream starts at explicit native fit seed")
    if config["dropout_seed"]!=config["initialization_seed"]+3000007:
        raise ValueError("Explicit route dropout seed base required")
    return copy.deepcopy(config)


def validate_roster(study, run):
    validate_config(run["config"],run)
    clone=copy.deepcopy(study)
    for row in clone["runs"]:
        if row["kind"]=="model_fit":row["config"]=base_config(row["config"])
    actual=copy.deepcopy(run);actual["config"]=base_config(run["config"])
    native_link.validate_roster(clone,actual)
    group=run["config"].get("independence_group")
    if group:
        rows=[r for r in study["runs"] if r["kind"]=="model_fit" and r["config"].get("independence_group")==group]
        ignored={"initialization_seed","dropout_seed","shuffle_seed"}
        recipes=[{k:v for k,v in r["config"].items() if k not in ignored} for r in rows]
        if any(r!=recipes[0] for r in recipes[1:]):raise ValueError("Boundary I4 architecture/source/role/selector contracts differ")


def be_class(torch, path):
    """AST extract unchanged pinned BEBlock; no unrelated source imports."""
    raw=path.read_bytes()
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=="BEBlock"]
    if len(nodes)!=1:raise ValueError("Exactly one original BEBlock required")
    env={"torch":torch,"nn":torch.nn}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),env)
    return env["BEBlock"]


def install_boundaries(torch, model, block, members, device, routing):
    class InputMap(torch.nn.Module):
        def __init__(self, linear):
            super().__init__();self.block=block(linear.in_channels,linear.out_channels,True,device,members)
        def forward(self,x):
            r=routing.active
            if r is None:raise ValueError("Native boundary requires explicit member routing")
            b=self.block
            return b.W(x*b.R[r])*b.S[r]
    class FirstConv(torch.nn.Module):
        def __init__(self, conv):
            super().__init__();self.native=conv
            if conv.cached or conv.lin.bias is not None or conv.bias is None:
                raise ValueError("Only uncached native GCN affine W plus post-propagation bias is admitted")
            self.native.lin=InputMap(conv.lin);self.native.register_parameter("bias",None)
        def forward(self,x,adj):
            return self.native(x,adj)+self.native.lin.block.B[routing.active]
    class OutputMap(torch.nn.Module):
        def __init__(self,linear):
            super().__init__();self.block=block(linear.in_features,linear.out_features,False,device,members)
        def forward(self,x):return self.block(x,routing.active)
    first=model.encoder.convs[0]
    terminals=[(i,m) for i,m in enumerate(model.decoder.lin) if isinstance(m,torch.nn.Linear)]
    if not terminals or terminals[-1][0]!=len(model.decoder.lin)-1 or terminals[-1][1].out_features!=1:
        raise ValueError("Use the existing terminal native decoder affine map; no added head")
    model.encoder.convs[0]=FirstConv(first)
    i,last=terminals[-1];model.decoder.lin[i]=OutputMap(last)


class BoundaryAdapter(native_link.LinkAdapter):
    @contextmanager
    def member_scope(self, member, *, training):
        outer=native_link.rng_state(self.t,self.np)
        native_link.restore_rng(self.t,self.np,self.member_rng[member]);self.routing.active=member
        try:yield
        finally:
            if training:self.member_rng[member]=native_link.rng_state(self.t,self.np)
            self.routing.active=None;native_link.restore_rng(self.t,self.np,outer)

    def train_step(self,epoch):
        self.model.train();means=EpochMeans()
        counts={k:0 for k in ("optimizer_updates","optimizer_update_attempts","backward_calls","backward_attempts",
            "encoder_calls","encoder_call_attempts","decoder_query_passes","positive_records","negative_records",
            "sampler_calls","sampler_completed_calls","negative_records_drawn","unused_negative_records_drawn",
            "dropped_positive_tail","graph_rebuilds","support_input_records","graph_rebuild_attempts","supervised_examples",
            "native_decoder_call_attempts","native_decoder_completed_calls","native_decoder_query_rows",
            "member_positive_records","member_negative_records")}
        counts.update({k:0 if self._setup_charged else v for k,v in self.provenance["adapter_setup_operations"].items()})
        self._setup_charged=True;self._active_counts=counts
        n=len(self.train["positive"]);m=self.config["members"]
        try:
            with self._scope(training=True):
                counts["sampler_calls"]+=1
                negatives=self.sampler(self.graph["raw_edge_index"],len(self.graph["x"]));counts["sampler_completed_calls"]+=1
                if self.t.is_tensor(negatives) and negatives.ndim==2:counts["negative_records_drawn"]=negatives.shape[1]
                if negatives.dtype!=self.t.long or negatives.ndim!=2 or negatives.shape[0]!=2 or negatives.shape[1]<n:
                    raise ValueError("Native negative draw must cover every TRAIN record")
                if bool((negatives[0]==negatives[1]).any()) or int(negatives.min())<0 or int(negatives.max())>=len(self.graph["x"]):
                    raise ValueError("Native negative sampler returned self/outside endpoints")
                if native_link.intersects(self.t,native_link.pair_keys(self.t,negatives.T,len(self.graph["x"]),undirected=False),self.forbidden_directed):
                    raise ValueError("Native negative sampler returned forbidden TRAIN edges")
                for ids in self.utils.PermIterator(self.device,n,self.config["training"]["batch_size"],True):
                    keep=self.t.ones(n,device=self.device,dtype=self.t.bool);keep[ids]=False
                    counts["graph_rebuild_attempts"]+=1;graph=self._support(self.train["positive"][keep])
                    counts["graph_rebuilds"]+=1;counts["support_input_records"]+=int(keep.sum())
                    self.optimizer.zero_grad(set_to_none=True);posloss=negloss=0.
                    for member in range(m):
                        with self.member_scope(member,training=True):
                            counts["encoder_call_attempts"]+=1
                            h=self.model.encoder(self.graph["x"],graph);counts["encoder_calls"]+=1
                            pos=self.model.decoder.multidomainforward(h,graph,self.train["positive"][ids].T,cndropprobs=[]).flatten()
                            counts["decoder_query_passes"]+=1
                            neg=self.model.decoder.multidomainforward(h,graph,negatives[:,ids],cndropprobs=[]).flatten()
                            counts["decoder_query_passes"]+=1
                            if pos.shape!=(len(ids),) or neg.shape!=(len(ids),) or not bool(self.t.isfinite(pos).all() and self.t.isfinite(neg).all()):
                                raise ValueError("Incomplete/nonfinite native member logits")
                            p=-self.F.logsigmoid(pos).mean();q=-self.F.logsigmoid(-neg).mean()
                            posloss+=self._number(p)/m;negloss+=self._number(q)/m
                            counts["member_positive_records"]+=len(ids);counts["member_negative_records"]+=len(ids)
                            counts["backward_attempts"]+=1;((p+q)/m).backward();counts["backward_calls"]+=1
                            del h,pos,neg,p,q
                    self._finite_state(gradients=True)
                    counts["optimizer_update_attempts"]+=1;self.optimizer.step();counts["optimizer_updates"]+=1
                    self._finite_state(gradients=False)
                    counts["positive_records"]+=len(ids);counts["negative_records"]+=len(ids);counts["supervised_examples"]+=2*len(ids)
                    means.add({"objective":posloss+negloss,"ce":posloss+negloss,"regularization":0.,
                        "positive_ce":posloss,"negative_ce":negloss},samples=len(ids))
                counts["dropped_positive_tail"]=n-counts["positive_records"]
                counts["unused_negative_records_drawn"]=negatives.shape[1]-counts["negative_records"]
            if counts["optimizer_updates"]!=self.config["train_minibatches_per_epoch"]:raise ValueError("Exact floor minibatch law changed")
            result=means.finish();result.update(operation_counts=counts,positive_denominator=counts["positive_records"],
                negative_denominator=counts["negative_records"],loss_reduction="mean_members(mean_pos_logistic+mean_neg_logistic)",
                loss_timing="preupdate; private dropout; sequential BCE/M backward; one shared Adam step",
                tail_policy="native_drop_tail")
            self._epoch=epoch;return result
        except BaseException as error:
            counts["unused_negative_records_drawn"]=max(0,counts["negative_records_drawn"]-counts["negative_records"])
            raise AdapterStepError(str(error),operation_counts=counts) from error
        finally:self._active_counts=None;self.routing.active=None

    def capture_state(self):
        value=super().capture_state();value["member_rng"]=copy.deepcopy(self.member_rng);value["routing_active"]=self.routing.active
        return value
    def restore_state(self,state):
        if state["routing_active"] is not None or len(state["member_rng"])!=self.config["members"]:
            raise ValueError("Restore only complete boundary/member-stream boundaries")
        super().restore_state(state);self.member_rng=copy.deepcopy(state["member_rng"]);self.routing.active=None


def make_adapter(*,torch,project_root,study,run,device):
    config=validate_config(run["config"],run);validate_roster(study,run)
    clone=copy.deepcopy(study)
    for row in clone["runs"]:
        if row["kind"]=="model_fit":row["config"]=base_config(row["config"])
    native_run=copy.deepcopy(run);native_run["config"]=base_config(config)
    adapter=native_link.make_adapter(torch=torch,project_root=project_root,study=clone,run=native_run,device=device)
    if len(adapter.train["positive"])//config["training"]["batch_size"]!=config["train_minibatches_per_epoch"]:
        raise ValueError("TRAIN custody and declared floor minibatch count differ")
    class Routing:active=None
    adapter.routing=Routing();adapter.config=config;adapter.__class__=BoundaryAdapter
    outer=native_link.rng_state(torch,adapter.np)
    try:
        native_link.restore_rng(torch,adapter.np,adapter._rng)
        if config["boundary"]["kind"]=="original_boundary":
            source=native_link.bound_file(project_root,config["boundary"]["source"])
            install_boundaries(torch,adapter.model,be_class(torch,source),config["members"],adapter.device,adapter.routing)
        adapter._rng=native_link.rng_state(torch,adapter.np)
        adapter.member_rng=[]
        for member in range(config["members"]):
            seed=config["dropout_seed"]+1000003*member
            random.seed(seed);adapter.np.random.seed(seed);torch.manual_seed(seed)
            if adapter.device.type=="cuda":torch.cuda.manual_seed_all(seed)
            adapter.member_rng.append(native_link.rng_state(torch,adapter.np))
    finally:native_link.restore_rng(torch,adapter.np,outer)
    o=config["optimizer"]
    adapter.optimizer=torch.optim.Adam([{"params":adapter.model.encoder.parameters(),"lr":o["encoder_lr"]},
        {"params":adapter.model.decoder.parameters(),"lr":o["predictor_lr"]}],betas=tuple(o["betas"]),eps=o["eps"],
        weight_decay=o["weight_decay"],amsgrad=o["amsgrad"])
    adapter.provenance.update(config=config,boundary_contract=CONTRACT,members=config["members"],
        member_dropout_seeds=[config["dropout_seed"]+1000003*m for m in range(config["members"])],
        parameter_count=sum(p.numel() for p in adapter.model.parameters()),original_task_replication=False)
    # Evaluation retains the same native source body, traversed once per route;
    # link endpoints receive a single mean-logit scorer from a selected group.
    native_encoder=adapter.model.encoder.forward;native_decoder=adapter.model.decoder.forward
    def encoder(x,adj):
        if adapter.routing.active is not None:return native_encoder(x,adj)
        output=[]
        for member in range(config["members"]):
            with adapter.member_scope(member,training=False):output.append(native_encoder(x,adj))
        return torch.stack(output)
    def decoder(x,adj,queries,*args,**kwargs):
        if adapter.routing.active is not None:return native_decoder(x,adj,queries,*args,**kwargs)
        output=[]
        for member in range(config["members"]):
            with adapter.member_scope(member,training=False):output.append(native_decoder(x[member],adj,queries,*args,**kwargs))
        return torch.stack(output).mean(0)
    adapter.model.encoder.forward=encoder;adapter.model.decoder.forward=decoder
    return adapter
