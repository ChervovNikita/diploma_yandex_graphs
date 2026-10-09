"""Inactive genuine independent4: native bodies or untied six-site factors.

No numerical import, dataset loader, scheduler or launcher. Reuses the exact
native constructor, train/evaluate/snapshot and owned-state utilities. Source
supply correction for untied bodies is explicitly outside this control packet.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = ('plain_native', 'untied_same_six_factors')
SELECTION = 'per_body_strict_native_VALID_BCE_earliest_ties_200epochs_patience50'
COST_POLICY = 'all_four_full_body_constructors_own_fits_selected_state_and_pool_serving_charged'


class IndependenceError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise IndependenceError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_gate():
    seal = json.loads((HERE/'SEAL.json').read_text())
    require(seal['runtime_disabled'] is True and sha(HERE/'MANIFEST.json') == seal['manifest_sha256'], 'Exact inactive independent4 source seal')
    for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
        path = (HERE/row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed independent4 payload')
    sources = json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    for row in sources['files']:
        path = (HERE.parent/row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE.parent) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed exact independent4 dependency')
    return sources


@dataclass(frozen=True)
class Independent4Config:
    enabled: bool = False
    root_source_review_approved: bool = False
    source_seal_sha256: str | None = None
    native_qualification_binding: str | None = None
    committee_binding: str | None = None

    def require_enabled(self):
        require(self.enabled is True and self.root_source_review_approved is True, 'Inactive independent4: reviewed root release required')
        for value in (self.source_seal_sha256, self.native_qualification_binding, self.committee_binding):
            require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Literal source/native qualification/prospective committee SHA bindings')
        require(self.source_seal_sha256 == sha(HERE/'SEAL.json'), 'Exact independent4 reviewed source release')


@dataclass
class Body:
    model: object
    optimizer: object
    scalar: object
    rng: object
    identity: dict
    counters: dict


def extend_to_one_untied_factor_body(rt, adapter, prototype):
    """This successor explicitly admits M1 per scientific independent body.

    The old adapter's M1 contract remains qualification-only. Its unchanged
    implementation is reused here under this separately reviewed extension:
    four separately initialized prototypes, each with one private factor row.
    """
    one = adapter.install_member_bank(rt['model_module'], prototype, adapter.AdapterConfig(enabled=True, members=1))
    one.verify_ownership()
    require(len(one.members) == 1, 'One factor row per independently initialized native body')
    return one.members[0]


class Independent4:
    def __init__(self, rt, engine, adapter, bank_metadata, ctx, costs, prospective_spec, config=Independent4Config()):
        config.require_enabled()
        sources = source_gate()
        self.rt, self.engine, self.adapter, self.metadata = rt, engine, adapter, bank_metadata
        self.ctx, self.costs, self.config = ctx, costs, config
        self.spec = bank_metadata.plain_metadata(prospective_spec)
        require(digest(self.spec) == config.committee_binding, 'Exact prospective committee declaration, never a pool chosen from outcomes')
        require(set(self.spec) == {'variant','role_binding','full_view_binding','role_seed','body_seeds','body_order','selection','cost_policy'}, 'Complete fixed committee declaration')
        require(self.spec['variant'] in VARIANTS and self.spec['selection'] == SELECTION
                and self.spec['cost_policy'] == COST_POLICY and self.spec['body_order'] == [0,1,2,3], 'Fixed variant/own selection/member order/full cost policy')
        require(self.spec['role_seed'] == ctx.seed and type(ctx.seed) is int, 'Common frozen outer role; body seeds do not resplit roles')
        for key in ('role_binding','full_view_binding'):
            value = self.spec[key]
            require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Exact common role/factual view binding')
        seeds = self.spec['body_seeds']
        require(type(seeds) is list and len(seeds) == len(set(seeds)) == 4
                and all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds), 'Four explicitly frozen distinct full-body initialization/training seeds')
        torch = rt['torch']
        require(rt['device'].type == 'cuda' and rt['model_class'] is rt['model_module'].SeHGNN
                and rt['model_module'].torch is torch, 'Exact ordinary native CUDA runtime/class')
        for module, key in ((engine,'native_engine_sha256'), (bank_metadata,'bank_metadata_sha256'), (adapter,'adapter_source_sha256')):
            require(sha(module.__file__) == sources[key], 'Exact native/metadata/adapter module '+key)
        require(sha(rt['model_module'].__file__) == sources['native_model_sha256']
                and sha(rt['native'].__file__) == sources['native_helpers_sha256']
                and sha(engine.capture_rng.__code__.co_filename) == sources['native_state_sha256'], 'Exact model/native train/state helpers')
        require(len(ctx.feats) == 25 and len(ctx.label_feats) == 12 and 0 < ctx.train_count <= 10000
                and ctx.train_index.tolist() == list(ctx.data.train_ids) and ctx.valid_index.tolist() == list(ctx.data.valid_ids), 'Complete common native graph-derived channels/role rows')
        self.cache_stamp = self._cache_stamp()
        self.bodies = []
        self.started_bodies, self.selected_stamp = set(), None
        caller = engine.capture_rng(rt['numpy'], torch)
        try:
            for m, seed in enumerate(seeds):
                with costs.measure('independent_body_'+str(m)+'_seed_native_constructor_optional_factors_Adam_scaler', gpu=True):
                    rt['native'].set_random_seed(seed)  # AFTER common frozen role/cache preparation.
                    prototype = engine.make_model(rt, ctx, costs)
                    model = (prototype if self.spec['variant'] == 'plain_native'
                             else extend_to_one_untied_factor_body(rt, adapter, prototype))
                    require(type(model) is rt['model_module'].SeHGNN
                            and all(p.dtype == torch.float32 and p.device == rt['device'] for p in model.parameters()), 'Complete native FP32 body; no retyping/reset')
                    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0)
                    scalar = torch.cuda.amp.GradScaler()
                    require(scalar.is_enabled(), 'Separate enabled native AMP scaler per body')
                    identity = bank_metadata.plain_metadata(dict(schema='prospectively_owned_native_independent4_body_v1',
                        committee_binding=config.committee_binding, source_seal_sha256=config.source_seal_sha256,
                        native_qualification_binding=config.native_qualification_binding, specification=self.spec,
                        body=m, body_seed=seed, role_seed=ctx.seed))
                    self.bodies.append(Body(model,optimizer,scalar,engine.capture_rng(rt['numpy'],torch),identity,
                        dict(epochs_completed=0,TRAIN_member_forwards=0,TRAIN_backwards=0,actual_Adam_steps=0)))
                    prototype = model = optimizer = scalar = None
        finally:
            engine.restore_rng(rt['numpy'], torch, caller)
            require(engine.exact(torch, engine.capture_rng(rt['numpy'],torch), caller), 'Four independent constructions preserve caller streams')
        self.verify_independence()

    def _cache_stamp(self):
        return tuple((namespace,id(getattr(self.ctx,namespace)),tuple((k,id(v),int(v._version)) for k,v in getattr(self.ctx,namespace).items()))
                     for namespace in ('feats','label_feats'))

    def _state_stamp(self):
        return tuple(tuple((id(value),int(value._version)) for value in (*body.model.parameters(),*body.model.buffers())) for body in self.bodies)

    def verify_independence(self):
        seen_ids, seen_storage, optimizer_ids, scaler_ids, rng_ids, rng_storage = set(), set(), set(), set(), set(), set()
        torch = self.rt['torch']
        for body in self.bodies:
            parameters = tuple(body.model.parameters())
            buffers = tuple(body.model.buffers())
            values = (*parameters,*buffers)
            ids = {id(value) for value in values}
            storage = {(str(value.device),int(value.untyped_storage().data_ptr())) for value in values if value.numel()}
            require(not ids & seen_ids and not storage & seen_storage, 'Different bodies share no slow/fast parameter or registered buffer object/storage')
            actual = tuple(p for group in body.optimizer.param_groups for p in group['params'])
            require(len(actual) == len({id(p) for p in actual}) and {id(p) for p in actual} == {id(p) for p in parameters}, 'Each optimizer owns exactly its complete body once')
            require(all(group['lr'] == .001 and group['weight_decay'] == 0 for group in body.optimizer.param_groups), 'Native own Adam lr/zero decay')
            require(id(body.optimizer) not in optimizer_ids and id(body.scalar) not in scaler_ids, 'Optimizers/scalers genuinely distinct')
            state_tensors = (body.rng['torch_cpu'],*body.rng['torch_cuda'])
            current_rng_storage = {(str(value.device),int(value.untyped_storage().data_ptr())) for value in state_tensors}
            require(id(body.rng) not in rng_ids and not current_rng_storage & rng_storage, 'Owned RNG snapshots share no mutable Torch state storage')
            rng_ids.add(id(body.rng)); rng_storage |= current_rng_storage
            seen_ids |= ids; seen_storage |= storage; optimizer_ids.add(id(body.optimizer)); scaler_ids.add(id(body.scalar))
        require(len(self.bodies) == 4 and self._cache_stamp() == self.cache_stamp, 'Complete four-body group and immutable common caches')
        named = dict(self.bodies[0].model.named_parameters())
        factor_names = {f'{site}.{factor}' for site in self.adapter.SITES for factor in ('input_factor','output_factor')}
        fast = sum(p.numel() for name,p in named.items() if name in factor_names)
        total = sum(p.numel() for p in named.values())
        require(fast == (0 if self.spec['variant'] == 'plain_native' else 97536) and total-fast == 83659532, 'Complete native body and exact same-site factor capacity')
        return dict(bodies=4,shared_parameters=[],shared_registered_buffers=[],separate_optimizers=True,separate_scalers=True,
                    original_native_parameters_per_body=total-fast,fast_parameters_per_body=fast,total_parameters_per_body=total,
                    body_rng_seeds=self.spec['body_seeds'],unscaled_native_loss_per_body=True)

    def fit_body(self, m):
        """Native BCE/AMP/Adam; own VALID selector, unscaled independent loss."""
        torch, engine, body = self.rt['torch'], self.engine, self.bodies[m]
        require(m in range(4) and m not in self.started_bodies, 'Each prospectively fixed body is fit once in a fresh group')
        self.started_bodies.add(m)
        self.selected_stamp = None
        caller = engine.capture_rng(self.rt['numpy'],torch)
        best_epoch, best_loss, selected, history = -1, 1000000.0, None, []
        try:
            engine.restore_rng(self.rt['numpy'],torch,body.rng)
            for epoch in range(200):
                with self.costs.measure('independent_body_'+str(m)+'_native_pre_epoch_GC'):
                    gc.collect()
                with engine.observe_update(self.rt,body.model,body.optimizer,body.counters), self.costs.measure('independent_body_'+str(m)+'_native_own_TRAIN_epoch',gpu=True):
                    # Exact native helper scales loss only with this body's own
                    # GradScaler. There is no /4, pool loss or joint optimizer.
                    loss, accuracy = self.rt['native'].train(body.model,self.ctx.feats,self.ctx.label_feats,self.ctx.targets_cuda,
                        torch.nn.BCEWithLogitsLoss(),body.optimizer,self.ctx.train_loader,self.rt['native'].evaluator,scalar=body.scalar)
                    require(math.isfinite(loss), 'Finite complete native body TRAIN loss')
                with self.costs.measure('independent_body_'+str(m)+'_native_post_TRAIN_cleanup'):
                    torch.cuda.empty_cache()
                body.counters['epochs_completed'] += 1
                with torch.cuda.amp.autocast(enabled=False):
                    scores, logits = engine.evaluate(self.rt,body.model,self.ctx,body.counters,self.costs,'independent_body_'+str(m)+'_own_VALID_selection')
                improved = scores['VALID']['BCE'] < best_loss
                if improved:
                    best_epoch, best_loss = epoch, scores['VALID']['BCE']
                    with self.costs.measure('independent_body_'+str(m)+'_own_selected_CPU_state_copy',gpu=True):
                        selected = self.metadata.checkpoint_tree(torch,engine.snapshot(self.rt,body.model,body.optimizer,body.scalar,self.ctx,epoch,scores,logits,body.identity))
                history.append(dict(epoch=epoch,native_TRAIN_BCE=float(loss),native_TRAIN_F1=[float(v) for v in accuracy],scores=scores,selected=improved,counters=dict(body.counters)))
                if epoch-best_epoch > 50:
                    break
            body.rng = engine.capture_rng(self.rt['numpy'],torch)
            require(selected is not None, 'Owned body selected state exists')
            return dict(body=m,body_seed=self.spec['body_seeds'][m],selected_state=selected,selected_epoch=best_epoch,selected_VALID_BCE=best_loss,history=history)
        finally:
            engine.restore_rng(self.rt['numpy'],torch,caller)
            require(engine.exact(torch,engine.capture_rng(self.rt['numpy'],torch),caller), 'Independent own fit preserves caller streams')
            self.verify_independence()

    def fit(self):
        """Exactly the four predeclared bodies; no selection of a subset/pool."""
        group = dict(schema='owned_native_independent4_control_v1',committee_binding=self.config.committee_binding,
                    specification=self.spec,bodies=[self.fit_body(m) for m in self.spec['body_order']],
                    ownership=self.verify_independence(),source_supply_untied_correction=False)
        self.restore(group)
        return group

    def restore(self, group):
        """Restore each own checkpoint, never an arbitrary native-five subset."""
        torch, engine = self.rt['torch'], self.engine
        require(group['schema'] == 'owned_native_independent4_control_v1' and group['committee_binding'] == self.config.committee_binding
                and group['specification'] == self.spec and len(group['bodies']) == 4, 'Exact prospectively fixed committee/state identity')
        for m, row in enumerate(group['bodies']):
            body, saved = self.bodies[m], row['selected_state']
            require(row['body'] == m and row['body_seed'] == self.spec['body_seeds'][m]
                    and saved['identity'] == body.identity and saved['seed'] == self.ctx.seed, 'Owned own-selected body identity and common role')
            with self.costs.measure('independent_body_'+str(m)+'_own_selected_restore',gpu=True):
                body.model.load_state_dict(saved['model_state'],strict=True)
                body.optimizer.load_state_dict(saved['optimizer_state'])
                body.scalar.load_state_dict(saved['scaler_state'])
                body.rng = engine.cpu_tree(torch,saved['training_rng'])
                require(engine.exact(torch,engine.cpu_tree(torch,body.model.state_dict()),saved['model_state'])
                        and engine.exact(torch,engine.cpu_tree(torch,body.optimizer.state_dict()),saved['optimizer_state'])
                        and engine.exact(torch,body.scalar.state_dict(),saved['scaler_state'])
                        and engine.exact(torch,body.rng,saved['training_rng']), 'Exact model/buffer/factor/own Adam/scaler/RNG restoration')
        ownership = self.verify_independence()
        self.selected_stamp = self._state_stamp()
        return ownership

    def serve(self):
        """Native FP32 all-full known-role logits; mean of four per label."""
        torch, engine = self.rt['torch'], self.engine
        require(self.selected_stamp is not None and self._state_stamp() == self.selected_stamp, 'Serve only restored four own-selected states, never last-epoch states')
        caller = engine.capture_rng(self.rt['numpy'],torch)
        modes = tuple(tuple((module,module.training) for module in body.model.modules()) for body in self.bodies)
        body_outputs, pool, metrics = [], {}, {}
        try:
            with torch.cuda.amp.autocast(enabled=False):
                for m, body in enumerate(self.bodies):
                    scores, logits = engine.evaluate(self.rt,body.model,self.ctx,body.counters,self.costs,'independent_body_'+str(m)+'_factual_FP32_serving')
                    body_outputs.append(dict(scores=scores,logits=logits))
                with self.costs.measure('predeclared_independent4_full_input_probability_mean',gpu=True),torch.no_grad():
                    for role,index in [('TRAIN',self.ctx.train_index),('VALID',self.ctx.valid_index)]:
                        values = [row['logits'][role].float() for row in body_outputs]
                        probability = torch.stack([torch.sigmoid(value) for value in values],dim=0).mean(dim=0)
                        truth = self.ctx.targets[index].float()
                        observed = [torch.where(truth.bool(),torch.nn.functional.logsigmoid(value),torch.nn.functional.logsigmoid(-value)) for value in values]
                        bce = -(torch.logsumexp(torch.stack(observed),dim=0)-math.log(4)).mean()
                        micro,macro = self.rt['native'].evaluator(truth,(probability>.5).int())
                        require(math.isfinite(float(bce)) and torch.isfinite(probability).all().item(), 'Finite full-input per-label Bernoulli pool')
                        pool[role] = probability.detach().clone()
                        metrics[role] = dict(BCE=float(bce),micro_F1=float(micro),macro_F1=float(macro))
            return dict(body_outputs=body_outputs,full_input_probability_mean=pool,metrics=metrics,
                        pooling='axis0 over four own-selected independent bodies, separately for five labels',pool_selection=False)
        finally:
            for body_modes in modes:
                for module,training in body_modes:
                    module.training = training
            engine.restore_rng(self.rt['numpy'],torch,caller)
            require(engine.exact(torch,engine.capture_rng(self.rt['numpy'],torch),caller), 'Committee serving preserves caller streams')
            self.verify_independence()
            require(self._state_stamp() == self.selected_stamp, 'Factual FP32 serving preserves all own-selected parameters/buffers')


def reconstruct_and_serve(rt,engine,adapter,bank_metadata,ctx,costs,spec,group,config=Independent4Config()):
    """Fresh four native constructors, distinct Adam/scalers, own restore/pool."""
    config.require_enabled()
    caller = engine.capture_rng(rt['numpy'],rt['torch'])
    committee = None
    try:
        committee = Independent4(rt,engine,adapter,bank_metadata,ctx,costs,spec,config)
        ownership = committee.restore(group)
        return dict(ownership=ownership,served=committee.serve(),exact_owned_state_restore=True,output_bitwise_gate=False)
    finally:
        committee = None
        gc.collect()
        rt['torch'].cuda.empty_cache()
        engine.restore_rng(rt['numpy'],rt['torch'],caller)
        require(engine.exact(rt['torch'],engine.capture_rng(rt['numpy'],rt['torch']),caller), 'Fresh independent4 reconstruction preserves caller streams')
