"""Source-only preparation. Import/CLI never load tensors or run qualification.

Future explicit callable: run_qualification(graph_name, output, device).
Only Squirrel17/Photo17 fresh prescribed warm + disposable engineering checks.
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SERVER_PHASE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
ROUTE = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SOURCE = 'graph_curvature_selector_source_preparation_20261004_v3'
SOURCE_MANIFEST = 'd53d6963b3c66cb59061c4f5be6a81108adf8e9792a3f3cf93daea56fe35ec70'
CONSTANTS = 'graph_curvature_selector_prospective_constants_root_20261004_v1/FROZEN_CONSTANTS_AND_SCREEN.json'
CONSTANTS_SHA = '633ea814bfc82c094ecb2d98699e77258f0f644d13cf17fa7f8d65691cf637dc'


def require(value, message):
    if not value:
        raise ValueError(message)


def read_verified(record):
    relative = Path(record['path'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Phase-relative descriptor required')
    path = PHASE/relative
    require(path.resolve() == path.absolute(), 'Symlink input is not bound')
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == record['sha256']
            and ('bytes' not in record or len(data) == record['bytes']), 'Bound bytes differ: '+str(path))
    return path, data


def verified_json(record):
    return json.loads(read_verified(record)[1])


def status():
    provenance = json.loads((HERE/'INPUT_PROVENANCE.json').read_text())
    return dict(status='SOURCE_ONLY_UNEXECUTED_RUNTIME_AND_OPERATORS_UNRESOLVED'
        if provenance['supported'] else 'UNRESOLVED_INPUT_PROVENANCE_REACQUISITION_REQUIRED',
        input_provenance_supported=provenance['supported'], qualification_executed=False,
        callable='runner.run_qualification', data_loader='runner.load_declared_inputs',
        graphs=['Squirrel17'], predictive_continuation=False,
        unresolved_dependencies=['actual NumPy/PyTorch/SciPy/PyG/operator runtime',
                                 'actual-device native AD/state/trial/cost qualification'])


def verify_runtime_location():
    require(PHASE == SERVER_PHASE and os.environ.get('GNNM_SSH_DESTINATION') == ROUTE,
            'Only the authorized anogena-2 project process may read bound data')
    rows = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        capture_output=True, text=True, check=True, timeout=10).stdout.splitlines()
    require([row.strip() for row in rows if row.strip()] == [UUID], 'Sole authorized UUID required')


def load_declared_inputs(graph_name, np, torch, device):
    """Callable loader: full features/edges + seed17 TRAIN/VALIDATION only."""
    require(status()['input_provenance_supported'], 'Unresolved acquisition provenance; reacquire first')
    verify_runtime_location()
    cells = json.loads((HERE/'INPUT_BINDINGS.json').read_text())['cells']
    cell = next(row for row in cells if row['graph'] == graph_name)
    manifest, role = verified_json(cell['graph_input']), verified_json(cell['role_freeze'])
    require(role['graph'] == graph_name == manifest['graph'] and role['seed'] == 17
        and role['source_split_index'] == 0 and role['role_derivation_version'] == 'derived_roles_v2'
        and role['labels_read'] is False and role['graph_input']['sha256'] == cell['graph_input']['sha256'],
        'Exact label-blind seed17 input/role identity required')
    x = np.load(read_verified(cell['features'])[0], allow_pickle=False)
    edges = np.load(read_verified(cell['edges'])[0], allow_pickle=False)
    n, f, c = (manifest[key] for key in ('num_nodes', 'num_features', 'num_classes'))
    require(x.dtype == np.float32 and x.shape == (n,f) and np.isfinite(x).all(), 'Full FP32 features differ')
    require(edges.dtype == np.int64 and edges.shape == (2,manifest['num_edges'])
        and ((edges >= 0) & (edges < n)).all() and (edges[0] < edges[1]).all()
        and np.unique(edges.T,axis=0).shape[0] == edges.shape[1], 'Canonical full edges differ')
    labels = {}
    for name in ('train', 'validation'):
        nodes = np.load(read_verified(cell['role_nodes'][name])[0], allow_pickle=False)
        require(nodes.dtype == np.int64 and nodes.ndim == 1 and len(nodes) == role['source_counts'][name]
            and ((nodes >= 0) & (nodes < n)).all() and len(np.unique(nodes)) == len(nodes), 'Role nodes differ')
        with np.load(read_verified(cell['source_labels'][name])[0], allow_pickle=False) as pack:
            require(set(pack.files) == {'nodes','labels'} and np.array_equal(pack['nodes'],nodes), 'Compact row identity differs')
            y = pack['labels'].copy()
            require(y.dtype == np.int64 and y.shape == nodes.shape and ((y >= 0) & (y < c)).all(), 'Compact labels differ')
        labels[name] = SimpleNamespace(nodes=torch.from_numpy(nodes.copy()).to(device),
                                       labels=torch.from_numpy(y).to(device))
    require(not np.intersect1d(labels['train'].nodes.cpu().numpy(),
                             labels['validation'].nodes.cpu().numpy()).size, 'Source roles overlap')
    return torch.from_numpy(x.copy()).to(device), torch.from_numpy(edges.copy()).to(device), labels, cell


def load_driver():
    manifest = verified_json(dict(path=SOURCE+'/MANIFEST.json',sha256=SOURCE_MANIFEST))
    rows = {row['path']:row for row in manifest['files']}
    for name in ('selector', 'driver'):
        filename = name+'.py'
        record = dict(rows[filename], path=SOURCE+'/'+filename)
        path, data = read_verified(record)
        module_name = name if name == 'selector' else 'exact_native_curvature_driver_v3'
        if module_name in sys.modules:
            module = sys.modules[module_name]
            require(getattr(module,'__executed_sha256__',None) == record['sha256']
                and getattr(module,'__executed_path__',None) == str(path), 'Unverified cached selector/driver')
        else:
            module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(module_name,path))
            sys.modules[module_name] = module
            try:
                exec(compile(data,str(path),'exec'),module.__dict__)
                module.__executed_sha256__, module.__executed_path__ = record['sha256'], str(path)
            except BaseException:
                del sys.modules[module_name]
                raise
    return module, sys.modules['selector']


class ExactStateDifference(ValueError):
    """Failed exact equality with compact path-aware details; never a tolerance."""
    def __init__(self, details):
        self.details = details
        super().__init__('Exact state differs at '+details['path']+': '+details['reason'])


def _exact_scalar(value):
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else '+Inf' if value > 0 else '-Inf'
    return value


def exact_equal(left, right, torch, path='$'):
    """The v1 exact predicate, with diagnostics only when it rejects a state."""
    if torch.is_tensor(left):
        detail = dict(path=path, expected_kind='tensor', actual_kind=type(right).__name__,
                      expected_dtype=str(left.dtype), expected_shape=list(left.shape),
                      expected_device=str(left.device))
        if not torch.is_tensor(right):
            raise ExactStateDifference(dict(detail, reason='actual value is not a tensor'))
        detail.update(actual_dtype=str(right.dtype), actual_shape=list(right.shape),
                      actual_device=str(right.device))
        if left.dtype != right.dtype:
            raise ExactStateDifference(dict(detail, reason='tensor dtype differs'))
        if left.shape != right.shape:
            raise ExactStateDifference(dict(detail, reason='tensor shape differs'))
        expected, actual = left.detach().cpu(), right.detach().cpu()
        if not torch.equal(expected, actual):
            # These summaries explain rejection; they never change its predicate.
            unequal = expected != actual
            first = torch.nonzero(unequal, as_tuple=False)[0]
            index = tuple(int(value) for value in first.tolist())
            delta = (expected.double()-actual.double()).abs()
            detail.update(reason='tensor values fail torch.equal',
                          different_elements=int(unequal.sum()),
                          first_different_index=list(index),
                          expected_value=_exact_scalar(expected[index].item()),
                          actual_value=_exact_scalar(actual[index].item()),
                          max_absolute_difference=_exact_scalar(float(delta.max())))
            raise ExactStateDifference(detail)
    elif isinstance(left, dict):
        if not (type(left) is type(right) and set(left) == set(right)):
            raise ExactStateDifference(dict(path=path, reason='state dictionary type or keys differ',
                expected_type=type(left).__name__,actual_type=type(right).__name__,
                expected_keys=[repr(key) for key in left],
                actual_keys=[repr(key) for key in right] if isinstance(right,dict) else None))
        for key in left:
            exact_equal(left[key],right[key],torch,path=path+'['+repr(key)+']')
    elif isinstance(left,(tuple,list)):
        if not (type(left) is type(right) and len(left) == len(right)):
            raise ExactStateDifference(dict(path=path, reason='state sequence type or length differs',
                expected_type=type(left).__name__,actual_type=type(right).__name__,
                expected_length=len(left),actual_length=len(right) if isinstance(right,(tuple,list)) else None))
        for index,(a,b) in enumerate(zip(left,right)):
            exact_equal(a,b,torch,path=path+'['+str(index)+']')
    else:
        if not left == right:
            raise ExactStateDifference(dict(path=path, reason='state scalar value differs',
                expected_type=type(left).__name__,actual_type=type(right).__name__,
                expected_value=_exact_scalar(left),actual_value=_exact_scalar(right)))


def load_install_witness():
    record = json.loads((HERE/'SOURCE_BINDINGS.json').read_text())['witness_source']
    path, data = read_verified(record)
    name = 'graph_curvature_install_witness_v3'
    if name in sys.modules:
        module = sys.modules[name]
        require(getattr(module,'__executed_sha256__',None) == record['sha256']
            and getattr(module,'__executed_path__',None) == str(path), 'Unverified witness module cache')
        return module
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name,path))
    sys.modules[name] = module
    try:
        exec(compile(data,str(path),'exec'),module.__dict__)
        module.__executed_sha256__, module.__executed_path__ = record['sha256'], str(path)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def tensor_storage(value, torch):
    if torch.is_tensor(value):
        return {value.data_ptr()} if value.numel() else set()
    if isinstance(value,dict):
        value = value.values()
    if isinstance(value,(tuple,list)) or type(value).__name__ == 'dict_values':
        return set().union(*(tensor_storage(item,torch) for item in value))
    return set()


class Costs:
    def __init__(self, torch, device, rows):
        self.torch, self.device, self.rows = torch, device, rows

    def call(self, name, function):
        torch, cuda = self.torch, self.device.startswith('cuda')
        if cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        wall, cpu = time.perf_counter(), time.process_time()
        row = dict(operation=name, completed=False)
        try:
            value = function()
            row['completed'] = True
            return value
        except Exception as error:
            row['error'] = type(error).__name__+': '+str(error)
            raise
        finally:
            if cuda:
                torch.cuda.synchronize(self.device)
            row.update(wall_seconds=time.perf_counter()-wall, process_cpu_seconds=time.process_time()-cpu,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    *(1 if sys.platform == 'darwin' else 1024),
                rss_scope='cumulative process high water, not phase allocation delta',
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if cuda else None,
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if cuda else None)
            self.rows.append(row)


def check_five_returns(outputs, receipt, native, checkpoint, graph, train, edges,
                       selector, sources, constants, torch, witness_module, recorder, save, out):
    """Collect all exact custody/reconstruction checks; never promote qualification."""
    adapter = sources['integration']
    source_diagnostics = witness_module.source_custody(outputs,receipt,native,checkpoint,
        selector,adapter,torch,exact_equal,tensor_storage,recorder)
    save('SOURCE_CUSTODY_DIAGNOSTICS.json',source_diagnostics)
    try:
        adapter, boundary, source = (sources[key] for key in ('integration','boundary','initializer'))
        backbone = native.specification['backbone']
        prototype = adapter.clone_boundary(copy.deepcopy(native),boundary,4)
        optimizer, _ = adapter.transport_optimizer(native,checkpoint['optimizer'],prototype)
        frozen = adapter.named_optimizer_snapshot(prototype,optimizer)
        before = adapter.cpu_copy(prototype.state_dict())
        modes = {name:module.training for name,module in prototype.named_modules()}
        k1 = adapter.clone_boundary(copy.deepcopy(native),boundary,1).eval()
        args = adapter.raw_arguments(graph,backbone)
        theta, logits_fn, _ = selector.bind_head_only(k1,backbone,args)
        S = source.symmetric_normalized_adjacency(graph.teacher_input.shape[0],edges,theta.dtype,theta.device)
        nodes = torch.arange(S.shape[0],dtype=torch.int64,device=theta.device)
        common, _ = source.initialize_four_routes(logits_fn,theta,S,nodes,train.nodes,train.labels,tangent_mode='common_only')
        center, bases = common[0], {}
        selected_spans = {value['pair'][0] for value in receipt['selection'].values() if value['pair'] is not None}
        if selected_spans:
            bank, gradient, _ = selector.projected_bank(logits_fn,center,S,nodes,train.nodes,train.labels,source)
            for span in selected_spans:
                if span == 'permuted':
                    permuted, _ = source.permute_topology_nodes(S,80017)
                    bank_span, _, _ = selector.projected_bank(logits_fn,center,permuted,nodes,train.nodes,train.labels,source)
                elif span == 'random':
                    generator = torch.Generator(device='cpu').manual_seed(70017)
                    bank_span = torch.randn((4,center.numel()),dtype=torch.float64,generator=generator).to(center.device)
                else:
                    bank_span = bank
                bases[span], _ = selector.ordered_basis(bank_span,gradient,constants,source.ALGEBRA_TOLERANCE)
                require(bases[span] is not None, 'Selected basis reconstruction failed')
        independent_diagnostics, expected_heads = witness_module.independent_reconstruction(
            outputs,receipt,common,bases,before,frozen,modes,args,logits_fn,selector,adapter,
            constants,torch,exact_equal,recorder.frozen)
        with (out/'INDEPENDENT_RECONSTRUCTION_HEADS.pt').open('xb') as stream:
            torch.save(dict(schema='graph-curvature-independent-heads-v1',heads=expected_heads),stream)
    except Exception as error:
        failure = dict(error_type=type(error).__name__,error=str(error))
        if isinstance(error,ExactStateDifference):
            failure['exact_state_difference'] = error.details
        independent_diagnostics = dict(schema='graph-curvature-independent-reconstruction-diagnostics-v1',
            diagnostic_only=True,qualification_promoted=False,passed=False,
            preparation_or_collection_error=failure,
            arms=[dict(arm=arm,passed=False,not_fully_checked=True,error=failure)
                  for arm in selector.ARMS],completed_arms=len(selector.ARMS))
    save('INDEPENDENT_RECONSTRUCTION_DIAGNOSTICS.json',independent_diagnostics)
    adapter.rng_restore(checkpoint['rng'])
    return dict(passed=source_diagnostics['passed'] and independent_diagnostics['passed'],
        diagnostic_completed=True,qualification_promoted=False,
        source_custody_passed=source_diagnostics['passed'],
        independent_reconstruction_passed=independent_diagnostics['passed'],
        source_custody_artifact='SOURCE_CUSTODY_DIAGNOSTICS.json',
        independent_reconstruction_artifact='INDEPENDENT_RECONSTRUCTION_DIAGNOSTICS.json',
        returned_arm_diagnostics_completed=5,selection_trials_repeated=0,
        original_reconstruction_exact_failures_remain_failures=True)


def live_factor_probe(initialization, graph, train, sources, torch):
    """One disposable dropout-off coupled-Adam map with actual native factors."""
    import torch.nn.functional as F
    adapter = sources['integration']
    model = copy.deepcopy(initialization['model']).eval()
    optimizer = adapter.restore_named_optimizer(model,
        adapter.named_optimizer_snapshot(initialization['model'],initialization['optimizer']))
    active = ('stem','head') if graph.teacher_backbone == 'polyformer_mono' else ('stem','global_head')
    adapter.rng_restore(initialization['rng'])
    try:
        optimizer.zero_grad(set_to_none=True)
        logits = model(*adapter.raw_arguments(graph,graph.teacher_backbone))
        loss = F.cross_entropy(logits[:,train.nodes].reshape(-1,graph.classes),train.labels.repeat(4))
        require(bool(torch.isfinite(loss)), 'Native live-factor trial loss is nonfinite')
        loss.backward()
        rows = []
        for name,p in model.named_parameters():
            require(p.grad is None or bool(torch.isfinite(p.grad).all()), 'Native live-factor gradient is nonfinite')
            if name in [prefix+'.'+suffix for prefix in active for suffix in ('R','S')]:
                require(p.requires_grad and p.grad is not None, 'Active native factor is detached/frozen')
                rows.append(dict(name=name,gradient_norm=float(p.grad.double().norm())))
        require(len(rows) == 4 and any(row['gradient_norm'] > 0 for row in rows), 'No active native factor gradient witness')
        optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Live-factor Adam result is nonfinite')
        state = adapter.named_optimizer_snapshot(model,optimizer)
        require(all(not torch.is_tensor(value) or bool(torch.isfinite(value).all())
                    for values in state['state'].values() for value in values.values()), 'Live-factor moments are nonfinite')
        return dict(passed=True,active_factors=rows,updates=1,own_member_loss_mean=True,
                    all_shared_private_parameters_live=True,disposable_map=True,predictive_continuation=False)
    finally:
        adapter.rng_restore(initialization['rng'])


def head_affinity(logits_fn, theta, constants, torch):
    direction = torch.arange(1,theta.numel()+1,dtype=theta.dtype,device=theta.device)
    direction = constants.radius*direction/direction.norm()
    with torch.no_grad():
        center = logits_fn(theta)
        mean = (logits_fn(theta+direction)+logits_fn(theta-direction))/2
    require(bool(torch.isfinite(mean).all()) and torch.allclose(mean,center,
        atol=constants.mean_logit_atol,rtol=constants.mean_logit_rtol), 'Actual final-head affine midpoint guard failed')
    return dict(passed=True,radius=constants.radius,max_mean_logit_error=float((mean-center).abs().max()))


def run_qualification(graph_name, output, device='cpu'):
    """Future explicit call only. Returns unresolved/failure; never continues arms."""
    pending = status()
    if not pending['input_provenance_supported']:
        return pending
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    require(graph_name == 'Squirrel' and device in ('cpu','cuda:0'), 'Witness diagnostic admits Squirrel17 only')
    verify_runtime_location()
    require(device != 'cpu' or os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU qualification must hide CUDA')
    require(device == 'cpu' or os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Frozen deterministic CUDA workspace required')
    out = Path(output).resolve()
    require(out.is_relative_to(PHASE) and out != PHASE, 'Output must be a new project-phase folder')
    out.mkdir(exist_ok=False)
    report = dict(graph=graph_name,seed=17,source_split_index=0,status='STARTED',
        fresh_warm=True,predictive_continuation=False,costs=[],checks={})
    def save(name,value):
        with (out/name).open('x') as stream:
            json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
            stream.write('\n')
    try:
        import numpy as np
        import torch
        torch.use_deterministic_algorithms(True)
        report['runtime'] = dict(python=sys.version,device=device,torch=torch.__version__,numpy=np.__version__,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cuda_runtime=torch.version.cuda,packages={name:package_version(name) for name in
                ('scipy','torch-geometric','torch-scatter','torch-sparse','pyg-lib')})
        costs = Costs(torch,device,report['costs'])
        driver, selector = load_driver()
        witness_module = load_install_witness()
        sources = driver.load_sources()
        native_api, adapter = sources['native_adapter'], sources['integration']
        constants = selector.FrozenConstants(**verified_json(dict(path=CONSTANTS,sha256=CONSTANTS_SHA))['constants'])
        x, edges, labels, binding = costs.call('load_full_inputs',lambda:load_declared_inputs(graph_name,np,torch,device))
        graph = costs.call('cold_native_preprocessing',lambda:native_api.prepare_graph(x,edges,
            driver.CELLS[graph_name],report['runtime'],dict(graph_input=binding['graph_input'])))
        input_fingerprints = costs.call('observe_input_preprocessing_fingerprints',lambda:
            witness_module.input_preprocessing_fingerprints(x,edges,labels,graph,binding))
        input_fingerprints['native_preprocessing_metadata'] = copy.deepcopy(graph.preprocessing)
        save('INPUT_PREPROCESSING_FINGERPRINTS.json',input_fingerprints)
        preprocessing_snapshot = costs.call('capture_actual_native_preprocessing_snapshot',lambda:
            witness_module.capture_actual_preprocessing(graph,out/'ACTUAL_PREPROCESSED_INPUT.pt',torch,input_fingerprints))
        save('PREPROCESSING_SNAPSHOT_RECEIPT.json',preprocessing_snapshot)
        report['preprocessing_snapshot'] = preprocessing_snapshot
        trace, transitions = [], []
        def warm_trace(row):
            trace.append(row)
            with (out/'WARM_TRACE.jsonl').open('a') as stream:
                stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        def transition(checkpoint):
            transitions.append(copy.deepcopy(checkpoint['metadata']))
            torch.save(checkpoint,out/'FRESH_PHOTO_LOCAL_TRANSITION.pt')
        spec = native_api.specification(driver.CELLS[graph_name],'single_author',0,17)
        native, optimizer, checkpoint = costs.call('fresh_prescribed_native_warm',lambda:
            adapter.warm_native(native_api,spec,graph,labels['train'],labels['validation'],warm_trace,transition))
        require(len(trace) == (50 if graph_name == 'Squirrel' else 250), 'Fresh exact warm update count differs')
        require(checkpoint['metadata']['fixed_last_warm'] and checkpoint['metadata']['warm_stage_epoch'] == 50,
                'Fixed-last warm endpoint required')
        require(graph_name == 'Squirrel' or len(transitions) == 1 and
            checkpoint['metadata']['local_model_and_adam_restored'] and not checkpoint['metadata']['native_local_rng_restored'],
            'Photo exact local-best model/Adam and post-all200 RNG handoff required')
        torch.save(checkpoint,out/'FRESH_NATIVE_WARM.pt')
        warm_fingerprints = costs.call('observe_fresh_warm_state_fingerprints',lambda:
            witness_module.fresh_warm_fingerprints(checkpoint,out/'FRESH_NATIVE_WARM.pt',torch,input_fingerprints))
        save('FRESH_WARM_FINGERPRINTS.json',warm_fingerprints)
        report['fresh_warm_fingerprint_summary'] = dict(
            artifact='FRESH_WARM_FINGERPRINTS.json',
            model=warm_fingerprints['logical_model']['sha256_logical_descriptor'],
            optimizer=warm_fingerprints['logical_optimizer']['sha256_logical_descriptor'],
            RNG=warm_fingerprints['logical_RNG']['sha256_logical_descriptor'],
            teacher_input=input_fingerprints['teacher_input']['sha256_raw_logical_bytes'],
            raw_features=input_fingerprints['raw_features']['sha256_raw_logical_bytes'],
            canonical_edges=input_fingerprints['canonical_edges']['sha256_raw_logical_bytes'],
            file_hash_is_distinct_from_logical_state=True,
            replay_failure_or_repeatability_claim=False)
        roundtrip = torch.load(out/'FRESH_NATIVE_WARM.pt',map_location='cpu')
        restored, restored_optimizer = costs.call('fresh_checkpoint_state_rng_restore',lambda:adapter.restore_native(native_api,roundtrip,device))
        exact_equal(checkpoint,adapter.native_checkpoint(restored,restored_optimizer,checkpoint['metadata']),torch,
                    path='serialized_fresh_warm_checkpoint')
        report['checks']['serialized_full_state_rng'] = dict(passed=True)
        del restored, restored_optimizer, roundtrip
        report['checks']['native_K4_Adam_correspondence'] = costs.call('actual_native_identity_frozen_factor_Adam',lambda:
            adapter.optimizer_equivalence_audit(native,checkpoint['optimizer'],sources['boundary'],graph,labels['train'],
                lambda name,value:save(name+'.json',value)))
        k1 = adapter.clone_boundary(copy.deepcopy(native),sources['boundary'],1).eval()
        k4 = adapter.clone_boundary(copy.deepcopy(native),sources['boundary'],4).eval()
        report['checks']['native_K1_K4_identity'] = costs.call('actual_native_full_graph_identity',lambda:
            adapter.identity_logits_audit(copy.deepcopy(native),k1,k4,graph))
        theta, logits_fn, _ = selector.bind_head_only(k1,driver.CELLS[graph_name],adapter.raw_arguments(graph,driver.CELLS[graph_name]))
        report['checks']['actual_final_head_affinity'] = costs.call('actual_final_head_affinity',lambda:
            head_affinity(logits_fn,theta,constants,torch))
        report['checks']['actual_head_AD'] = costs.call('actual_native_head_AD',lambda:
            sources['initializer'].qualify_gradient_interface(logits_fn,theta,labels['train'].nodes,labels['train'].labels,17))
        require(report['checks']['actual_head_AD']['passed'], 'Actual native AD interface is unresolved/failed')
        del k1,k4,theta,logits_fn
        adapter.rng_restore(checkpoint['rng'])
        recorder = witness_module.InstallWitnessRecorder(driver,selector,adapter)
        report.update(diagnostic_only=True,qualification_promoted=False,
                      witness_source_sha256=witness_module.__executed_sha256__)
        try:
            with recorder:
                outputs, receipt = costs.call('instrumented_selector_all_slots_and_trials',lambda:
                    driver.prepare_from_native_warm(native,optimizer,checkpoint['metadata'],checkpoint['rng'],graph_name,
                        17,0,graph,labels['train'],edges,constants,sources))
        finally:
            report['runtime_instrumentation'] = recorder.summary()
            save('WITNESS_CAPTURE_RECEIPT.json',recorder.summary())
            with (out/'INSTALL_WITNESSES.pt').open('xb') as stream:
                torch.save(recorder.archive(),stream)
        save('SELECTOR_RECEIPT.json',receipt)
        diagnostics = costs.call('all_five_exact_witness_and_reconstruction_diagnostics',lambda:
            check_five_returns(outputs,receipt,native,checkpoint,graph,labels['train'],edges,
                selector,sources,constants,torch,witness_module,recorder,save,out))
        report['checks']['all_five_returned_states'] = diagnostics
        adapter.rng_restore(checkpoint['rng'])
        donor_checks = []
        witness_module.collect_check(donor_checks,'native_donor_after_diagnostics',lambda:
            exact_equal(checkpoint,adapter.native_checkpoint(native,optimizer,checkpoint['metadata']),torch,
                        path='native_donor_after_diagnostics'))
        report['checks']['native_donor_after_diagnostics'] = dict(
            passed=all(row['passed'] for row in donor_checks),checks=donor_checks)
        if not diagnostics['source_custody_passed'] or not all(row['passed'] for row in donor_checks):
            final_status = 'FAILED_SOURCE_CUSTODY_DIAGNOSTIC_ONLY'
        elif not diagnostics['independent_reconstruction_passed']:
            final_status = 'FAILED_INDEPENDENT_RECONSTRUCTION_DIAGNOSTIC_ONLY'
        else:
            final_status = 'DIAGNOSTIC_COMPLETED_NO_QUALIFICATION_PROMOTION'
        report.update(status=final_status,qualification_promoted=False,
            warm_metadata=checkpoint['metadata'],candidate_slots=9,
            trial_maps_started=receipt['trial_maps_started'],
            input_binding=binding,returned_arms_saved=False,
            witness_snapshots_saved=True,live_factor_probe_executed=False,
            original_native_qualification_not_established=True)
    except Exception as error:
        if isinstance(error, ExactStateDifference):
            report['exact_state_difference'] = error.details
        report.update(status='UNRESOLVED_OR_FAILED_ACTUAL_NATIVE_QUALIFICATION',error=type(error).__name__+': '+str(error),
                      traceback=traceback.format_exc())
    report.update(total_wall_seconds=time.perf_counter()-started_wall,
        total_process_cpu_seconds=time.process_time()-started_cpu,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024))
    save('QUALIFICATION_REPORT.json',report)
    return report


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == '__main__':
    print(json.dumps(status(),sort_keys=True))
