"""Disabled four correction banks on one original native WikiCS trajectory."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ARMS = ('C4', 'S_joint4head', 'U4_sharedB', 'S_one_path')
CONDITIONS = dict(C4='label_only4_detached', S_joint4head='multihead_single4',
                  U4_sharedB='shared_backbone_untied_correctors4', S_one_path='one_path')
EXPECTED = dict(C4=(4,4,4,1), S_joint4head=(1,4,1,1), U4_sharedB=(4,4,4,4), S_one_path=(1,1,1,1))
SEEDS = (6101, 6203, 6307)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _failure_write(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def _dependencies():
    pins = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
    require(sha(HERE/'SEEDS.json') == pins['seed_table_sha256'], 'Exact prospective seed table')
    for row in pins['sources']:
        path = (PHASE / row['path']).resolve(strict=True)
        require(path.is_relative_to(PHASE) and sha(path) == row['sha256']
            and path.stat().st_size == row['bytes'], 'Frozen four-bank source binding changed')
    root = PHASE / pins['native_integration_directory']
    for row in json.loads((root / 'MANIFEST.json').read_text())['files']:
        path = root / row['path']
        require(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Changed native integration payload')
    spec = importlib.util.spec_from_file_location('_four_bank_frozen_native_engine', root / 'train.py')
    engine = importlib.util.module_from_spec(spec); sys.modules[spec.name] = engine; spec.loader.exec_module(engine)
    common, capture_module = engine._helpers()
    public, data, driver, core_module, public_root = common.dependencies()
    binding = json.loads((root / 'CONTROL_CORE_BINDING.json').read_text())['source_binding']
    controls = common.control_module(binding)
    protocol = json.loads((PHASE / pins['screen_protocol_path']).read_text())
    require(protocol['native_seeds'] == list(SEEDS) and protocol['arms'] == list(ARMS)
        and protocol['mask_and_corrector']['initializer_seed'].startswith('native_seed;')
        and protocol['mask_and_corrector']['mask_seed'] == 'native_seed+1900001', 'Exact frozen screen/seed contract')
    return pins, engine, common, capture_module, public, data, driver, core_module, public_root, controls, binding


def source_identity():
    pins = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
    return dict(program_sha256=sha(__file__), manifest_sha256=sha(HERE / 'MANIFEST.json'),
        integration_manifest_sha256=pins['native_integration_manifest_sha256'],
        protocol_sha256=pins['screen_protocol_sha256'], seeds_sha256=sha(HERE / 'SEEDS.json'))


def _readout(torch, prediction, truth):
    bank, pooled = prediction['member_logits'], prediction['served_probabilities']
    require(bank.shape[1:] == (5274, 10) and pooled.shape == (5274, 10), 'Full5274 development predictions')
    count = int((pooled.argmax(-1) == truth).sum())
    member_counts = [int((row.argmax(-1) == truth).sum()) for row in bank]
    lp = torch.nn.functional.log_softmax(bank, dim=-1)
    true_lp = lp.gather(-1, truth[None,:,None].expand(len(bank),-1,1)).squeeze(-1)
    pool_lp = torch.logsumexp(true_lp,0) - math.log(len(bank)); nll = -pool_lp.mean()
    brier = ((pooled - torch.nn.functional.one_hot(truth, 10))**2).sum(-1).mean()
    value = dict(correctcount=count, served_accuracy=count/5274, served_NLL=float(nll), Brier=float(brier),
        member_correctcount=member_counts, member_accuracy=[x/5274 for x in member_counts],
        member_NLL=[float(-row.mean()) for row in true_lp])
    require(all(math.isfinite(x) for x in [value['served_NLL'],value['Brier']]+value['member_NLL']), 'Finite readouts')
    return value


def make_session(*, train_data, seed, device='cpu', polynormer=None,
                 purpose='engineering_qualification', later_execution_authorized=False, only_arm=None):
    """Full-input engine; default refuses before imports or tensor access."""
    require(later_execution_authorized is True, 'Disabled four-bank source; separate root adoption required')
    require(type(seed) is int and seed in SEEDS and purpose in
        ('first_screen','engineering_qualification','selected_serving'), 'Fixed seed/purpose')
    require(only_arm is None or (purpose == 'selected_serving' and only_arm in ARMS), 'No arm dropping in the screen')
    pins, engine, common, capture_module, public, data, driver, core_module, public_root, controls, binding = _dependencies()
    # Provider imports precede native Session seeding exactly as in the original driver.
    from ogb.graphproppred import Evaluator as GraphEvaluator
    from ogb.linkproppred import Evaluator as LinkEvaluator
    native = public.Session('wikics', 'single', seed, device, polynormer)
    require(tuple(train_data['x'].shape) == (11701,300) and len(train_data['ids']) == 580
        and tuple(train_data['edge_index'].shape) == (2,442907), 'Complete frozen WikiCS TRAIN/graph inputs')
    config = native.config
    require(config['training']['epochs'] == 1100 and config['training']['local_epochs'] == 100
        and config['model']['hidden_channels']*config['model']['heads'] == 512, 'Unchanged native1100/100/512 recipe')
    selected_arms = (only_arm,) if only_arm else ARMS
    seeds = json.loads((HERE / 'SEEDS.json').read_text())['blocks'][str(seed)]
    context = common.context_identity(native, train_data); backbone = common.backbone_id(native)
    arguments = dict(nodes=11701,feature_width=512,classes=10,edge_index=train_data['edge_index'],
        train_ids=train_data['ids'],train_labels=train_data['y'],device=str(native.device),later_execution_authorized=True)
    banks = {}; before = common.native_rng_state(native)
    for arm in selected_arms:
        if arm == 'C4':
            banks[arm] = core_module.make_core(**arguments, initializer_seed=seeds[arm][0],
                mask_seed=seeds['mask_seed'],public_root=public_root)
        else:
            banks[arm] = controls.make_shared_controls(**arguments,mode=CONDITIONS[arm],
                initializer_seeds=tuple(seeds[arm]),mask_seed=seeds['mask_seed'],context_identity=context,
                backbone_ids=(backbone,),selection_policy=common.selection_policy(CONDITIONS[arm]),
                core_root=common.PHASE / common.CORE_NAME)
        require(banks[arm].members == EXPECTED[arm][0] and banks[arm].allowed_count == 580
            and banks[arm].query_count == 290, 'Correct full-population bank operator')
    common.require_same_native_rng(native,before,'all isolated four-bank constructors')
    native_ids = {id(p) for p in native.model.parameters()}; owned = set(); counts = {}
    for arm, bank in banks.items():
        ids = {id(p) for p in bank.parameters()}
        require(not ids & (native_ids | owned), 'No native/corrector or inter-bank learned sharing')
        optimizers = getattr(bank,'optimizers',[bank.optimizer])
        opt_ids = {id(p) for opt in optimizers for group in opt.param_groups for p in group['params']}
        require(ids == opt_ids and len(optimizers) == EXPECTED[arm][3], 'Exact separate corrector optimizer ownership')
        owned |= ids; counts[arm] = sum(p.numel() for p in bank.parameters())
    expected_counts = dict(C4=76328,S_joint4head=283648,U4_sharedB=283648,S_one_path=70912)
    require(all(counts[arm] == expected_counts[arm] for arm in banks), 'Frozen source-derived corrector counts')

    class FourBankSession:
        def __init__(self):
            self.native,self.banks,self.engine,self.common,self.data,self.driver = native,banks,engine,common,data,driver
            self.train_data,self.purpose,self.parameter_epoch = train_data,purpose,0
            self.capture = capture_module.NativeSingleCapture(native,11701,512,10)
            self.restored_logical_steps = None
            self.run = dict(seed=seed,source=source_identity(),data_context=context,backbone_id=backbone,
                native_recipe=config,native_core=native.core_provenance,native_source=native.native_provenance,
                arms=list(selected_arms),initializer_seeds=seeds,control_core_binding=binding,corrector_parameters=counts,
                purpose=purpose,second_old_capture_only=True,TEST_scoring=False,exact_resume_supported=False)
            self.work = dict(native_update_attempts=0,native_update_completions=0,all_bank_update_completions=0,
                common_Q_checks=0,native_RNG_isolation_checks=0,native_local_restorations=0,complete_VALID_events=0)

        def metadata(self):
            return engine._capture_metadata(native,self.parameter_epoch,common,logical_steps=self.restored_logical_steps)

        def draw_common_masks(self):
            require(self.purpose != 'selected_serving' and tuple(banks) == ARMS, 'Full four-bank training only')
            before = common.native_rng_state(native)
            masks = {arm:bank.draw_common_query_mask() for arm,bank in banks.items()}
            first = masks[ARMS[0]]
            require(all((x.query_positions,x.query_ids,x.draw_id,x.training_value_scale) ==
                (first.query_positions,first.query_ids,first.draw_id,first.training_value_scale) for x in masks.values()),
                'Identical fixed common Q schedule in every bank; no replacement/rewind')
            common.require_same_native_rng(native,before,'four private common masks')
            self.work['common_Q_checks'] += 1; self.work['native_RNG_isolation_checks'] += 1
            return masks

        def update_banks(self,H,base,masks,*,native_capture=None,verify_native_gradients=False):
            require(self.purpose != 'selected_serving' and tuple(banks) == ARMS, 'No selected-state training or dropped bank')
            require(tuple(masks) == ARMS and all(masks[arm] is bank._pending_mask for arm,bank in banks.items()),
                    'Every bank must receive its own pending mask before any update')
            first=masks[ARMS[0]]
            require(all((x.query_positions,x.query_ids,x.draw_id,x.training_value_scale) ==
                (first.query_positions,first.query_ids,first.draw_id,first.training_value_scale) for x in masks.values()),
                'Identical Q schedule is required even for the engineering executor')
            metadata = self.metadata() if native_capture is None else native_capture
            versions = tuple(p._version for p in native.model.parameters())
            before = common.native_rng_state(native)
            gradients = [None if p.grad is None else p.grad.detach().clone() for p in native.model.parameters()] if verify_native_gradients else None
            result = {arm:engine._train_core(bank,H,base,masks[arm],CONDITIONS[arm],metadata) for arm,bank in banks.items()}
            require(tuple(p._version for p in native.model.parameters()) == versions, 'No correction-native parameter update')
            if gradients is not None:
                require(all((p.grad is None and old is None) or (p.grad is not None and old is not None
                    and native.torch.equal(p.grad,old)) for p,old in zip(native.model.parameters(),gradients)),
                    'Observed native own-gradient bank unchanged by all correctors')
            common.require_same_native_rng(native,before,'all detached old-view-B bank updates')
            self.work['all_bank_update_completions'] += 1; self.work['native_RNG_isolation_checks'] += 1
            return result

        def train_step(self,batch,labels,*,epoch,masks=None,verify_native_gradients=False):
            require(self.purpose != 'selected_serving' and type(epoch) is int and 1 <= epoch <= 1100, 'Explicit TRAIN epoch')
            masks = self.draw_common_masks() if masks is None else masks
            metadata = self.metadata(); self.capture.begin('TRAIN',2); self.work['native_update_attempts'] += 1
            native_result = native.train_step(batch,labels)
            self.work['native_update_completions'] += 1
            H,base = self.capture.finish()
            result = self.update_banks(H,base,masks,native_capture=metadata,verify_native_gradients=verify_native_gradients)
            self.parameter_epoch = epoch
            return dict(native=native_result,banks=result,old_view_B_native_capture=metadata)

        def serve_banks(self,H,base,heldout_ids,*,native_capture=None):
            metadata = self.metadata() if native_capture is None else native_capture
            before = common.native_rng_state(native)
            result = {arm:engine._serve_core(bank,H,base,heldout_ids,CONDITIONS[arm],metadata) for arm,bank in banks.items()}
            common.require_same_native_rng(native,before,'all full-context bank serving')
            self.work['native_RNG_isolation_checks'] += 1
            return result

        def serve_ids(self,heldout_ids):
            native.model.eval()
            with native.torch.no_grad():
                batch = dict(x=train_data['x'].to(native.device),edge_index=train_data['edge_index'].to(native.device),
                             ids=heldout_ids.to(native.device))
                self.capture.begin('SERVE',1); native.forward(batch); H,base = self.capture.finish()
                return self.serve_banks(H,base,batch['ids'])

        def evaluate(self,valid):
            require(self.purpose == 'first_screen', 'Engineering/serving engine cannot read VALID truth or select')
            self.capture.begin('VALID',1); native_metric,native_per = driver.evaluate(native,train_data,valid)
            H,base = self.capture.finish(); ids=valid['ids'].to(native.device); truth=valid['y'].to(native.device)
            predictions = self.serve_banks(H,base,ids)
            stats = {arm:_readout(native.torch,prediction,truth) for arm,prediction in predictions.items()}
            count = int((base[ids].argmax(-1) == truth).sum())
            require(abs(native_metric-count/5274) < 2e-7, 'Original native correctcount selector')
            self.work['complete_VALID_events'] += 1
            return dict(native_correctcount=count,native_accuracy=native_metric,native_members=native_per,arms=stats)

        def snapshot(self,*,epoch,kind='engineering_snapshot',native_metric=None,stats=None,arm=None):
            require(kind in ('engineering_snapshot','native_own_local','arm_final_selected','native_own_best')
                and type(epoch) is int and 1 <= epoch <= 1100 and (arm is None or arm in banks), 'Coherent declared snapshot')
            require(all(bank._pending_mask is None and not bank._failed_update for bank in banks.values()), 'Complete bank transitions')
            require(kind != 'engineering_snapshot' or (native_metric is None and stats is None), 'Engineering checkpoint has no VALID metric')
            require(self.purpose in ('first_screen','engineering_qualification')
                and (self.purpose != 'engineering_qualification' or kind == 'engineering_snapshot'),
                'No scientific selection in engineering/serving sessions')
            chosen = (arm,) if arm else tuple(banks)
            state = driver.joint_snapshot(native,epoch,native_metric,None,self.run)
            state['checkpoint_kind'] = kind + '; one native and coherent correction banks'
            state['selector_performed'] = kind != 'engineering_snapshot'
            state['snapshot_purpose'] = 'engineering_qualification' if kind == 'engineering_snapshot' else 'first_screen'
            return native._cpu_tree(dict(schema='label-only-four-bank-coherent-state-v1',kind=kind,epoch=epoch,arm=arm,
                selector_performed=kind != 'engineering_snapshot',
                snapshot_purpose='engineering_qualification' if kind == 'engineering_snapshot' else 'first_screen',
                global_mode=bool(native.model.models[0].body._global),native_logical_steps=native.steps,native=state,
                banks={key:dict(parameters=engine._learned_corrector_state(banks[key]),
                    optimizer=banks[key].optimizer.state_dict(),mask_rng_state=banks[key].mask_generator.get_state(),
                    logical_steps=banks[key].steps,descriptor=banks[key].descriptor()) for key in chosen},
                stats=stats,run=self.run,exact_resume_supported=False))

        def restore_native_own_local(self,state):
            require(state['schema'] == 'label-only-four-bank-coherent-state-v1' and state['kind'] in
                ('native_own_local','engineering_snapshot') and state['global_mode'] is False
                and 1 <= state['epoch'] <= 100 and state['run'] == self.run and tuple(state['banks']) == ARMS,
                'One own-native-local epoch with all four coherent banks')
            before = common.native_rng_state(native); native_steps=native.steps
            live = {arm:(bank.mask_generator.get_state().clone(),bank.steps,copy.deepcopy(bank.counters)) for arm,bank in banks.items()}
            native.core['selection'].local_transition(native.model,native.optimizers,lambda filename:state['native'],ordinary_independent=False)
            for arm,bank in banks.items():
                engine._restore_learned_corrector(bank,state['banks'][arm],native)
                require(native.torch.equal(bank.mask_generator.get_state(),live[arm][0]) and bank.steps == live[arm][1]
                    and bank.counters == live[arm][2], 'Every mask stream and work counter remains live')
            common.require_same_native_rng(native,before,'one native/all-bank own-local restoration')
            require(native.steps == native_steps, 'Native physical logical-step count remains live')
            self.parameter_epoch=state['epoch']; self.work['native_local_restorations'] += 1
            return state['epoch']

        def close(self):
            self.capture.close()

    return FourBankSession()


def run_complete(*,train,valid,output,polynormer,seed=6101,device='cpu',later_execution_authorized=False):
    """One complete seed block/all four arms; no owner/campaign/aggregate opening."""
    require(later_execution_authorized is True, 'Disabled full screen; separate root adoption required')
    require(type(seed) is int and seed in SEEDS, 'Only the frozen three native blocks')
    started=time.monotonic(); engine=None
    output=Path(output); require(not output.exists(), 'Fresh once-only output; no resume/overwrite')
    output.mkdir(parents=True,exist_ok=False); session=None; trace=[]
    try:
        pins,engine,common,capture_module,public,data,driver,core,public_root,controls,binding = _dependencies()
        train_data,valid_data,origin=data.load_train_valid('wikics',train,valid)
        session=make_session(train_data=train_data,seed=seed,device=device,polynormer=polynormer,
            purpose='first_screen',later_execution_authorized=True)
        session.run['data']=origin; engine.json_write(output/'RUN.json',session.run)
        for arm in ARMS:(output/arm).mkdir()
        best={arm:-1 for arm in ARMS}; best_local=best_native=-1
        for epoch in range(1,1101):
            if epoch == 101:
                local=session.native.torch.load(output/'NATIVE_OWN_LOCAL_ALL_BANKS.pt',map_location=session.native.device,weights_only=False)
                session.restore_native_own_local(local)
            updates=0
            for batch,labels in session.data.train_batches('wikics',train_data,epoch,seed,session.native.device):
                update=session.train_step(batch,labels,epoch=epoch); updates+=1
            require(updates == 1, 'Exactly one complete original native update per epoch')
            stats=session.evaluate(valid_data); native_count=stats['native_correctcount']
            for arm in ARMS:
                if stats['arms'][arm]['correctcount'] > best[arm]:
                    best[arm]=stats['arms'][arm]['correctcount']
                    session.native.torch.save(session.snapshot(epoch=epoch,kind='arm_final_selected',arm=arm,
                        native_metric=stats['native_accuracy'],stats=stats['arms'][arm]),output/arm/'selected.pt')
            if native_count > best_native:
                best_native=native_count
                session.native.torch.save(session.snapshot(epoch=epoch,kind='native_own_best',native_metric=stats['native_accuracy']),
                                          output/'NATIVE_OWN_BEST_ALL_BANKS.pt')
            if epoch <= 100 and native_count > best_local:
                best_local=native_count
                session.native.torch.save(session.snapshot(epoch=epoch,kind='native_own_local',native_metric=stats['native_accuracy']),
                                          output/'NATIVE_OWN_LOCAL_ALL_BANKS.pt')
            trace.append(dict(epoch=epoch,readouts=stats,TRAIN=update,seconds=time.monotonic()-started))
            engine.json_write(output/'PRIVATE_VALID_TRACE.json',trace)
            engine.json_write(output/'PROGRESS.json',dict(complete=False,epoch=epoch,work=session.work,
                capture_counts=session.capture.counts,bank_counts={arm:bank.counters for arm,bank in session.banks.items()}))
        require(session.native.steps == 1100 and session.work['native_update_completions'] == 1100
            and session.work['all_bank_update_completions'] == 1100 and session.work['complete_VALID_events'] == 1100
            and session.work['common_Q_checks'] == 1100 and session.work['native_local_restorations'] == 1
            and session.capture.counts['TRAIN_head_captures'] == 2200 and session.capture.counts['VALID_head_captures'] == 1100,
            'Complete one-native trajectory work')
        records=[]
        for arm,bank in session.banks.items():
            predictions,heads,backwards,adam=EXPECTED[arm]
            require(bank.steps == bank.counters['mask_draws'] == 1100
                and bank.counters['training_route_forwards'] == bank.counters['serving_route_forwards'] == heads*1100
                and bank.counters['route_backwards'] == backwards*1100 and bank.counters['corrector_Adam_steps'] == adam*1100,
                'Complete required bank work: '+arm)
            record=dict(complete=True,arm=arm,epochs=1100,selected_sha256=sha(output/arm/'selected.pt'),
                descriptor=bank.descriptor(),native_trajectory_shared=True,ordinary_independent_GNN4=False)
            engine.json_write(output/arm/'COMPLETE.json',record); records.append(record)
        engine.json_write(output/'COMPLETE.json',dict(complete=True,run=session.run,required_bank_records=records,
            native_trajectories=1,epochs=1100,work=session.work,capture_counts=session.capture.counts,
            seconds=time.monotonic()-started,comparative_opening_authorized=False,whole_three_seed_family_gate_evaluated=False))
        return dict(output=str(output),seed=seed,complete=True,arms=list(ARMS),scores_read=False)
    except BaseException as error:
        write=engine.json_write if engine else _failure_write
        write(output/'FAILURE.json',dict(complete=False,error_type=type(error).__name__,error=str(error),
            complete_epochs=len(trace),work=session.work if session else None,
            bank_counts={arm:bank.counters for arm,bank in session.banks.items()} if session else None,
            seconds=time.monotonic()-started,automatic_retry=False,TEST_scoring=False))
        raise
    finally:
        if session is not None:session.close()


def reconstruct_for_serving(state,*,train_data,polynormer,device='cpu',later_execution_authorized=False):
    """Trusted complete selected/engineering state; no VALID labels or resume."""
    require(later_execution_authorized is True, 'Disabled coherent serving reconstruction')
    require(state['schema'] == 'label-only-four-bank-coherent-state-v1'
        and state['kind'] in ('arm_final_selected','engineering_snapshot') and state['run']['source'] == source_identity(),
        'Exact own source coherent final or discarded engineering checkpoint')
    arm=state['arm'] if state['kind'] == 'arm_final_selected' else None
    session=make_session(train_data=train_data,seed=state['run']['seed'],device=device,polynormer=polynormer,
        purpose='selected_serving',only_arm=arm,later_execution_authorized=True)
    require(session.run['data_context'] == state['run']['data_context']
        and session.run['native_core'] == state['run']['native_core']
        and session.run['native_source'] == state['run']['native_source']
        and session.run['native_recipe'] == state['run']['native_recipe'] and tuple(session.banks) == tuple(state['banks']),
        'Exact full TRAIN/graph/source/recipe and coherent bank inventory')
    native=session.native
    native.model.load_state_dict(state['native']['model'],strict=True); native.model.set_global(state['global_mode'])
    for optimizer,saved in zip(native.optimizers,state['native']['optimizers']):optimizer.load_state_dict(saved)
    native.streams=session.common.clone_streams(state['native']['streams'])
    for key,bank in session.banks.items():session.engine._restore_learned_corrector(bank,state['banks'][key],native)
    session.parameter_epoch=state['epoch']; session.restored_logical_steps=state['native_logical_steps']
    return session
