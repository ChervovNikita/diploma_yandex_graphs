"""Full public WikiCS graph arithmetic diagnostic; no gradients or updates."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
REPO = ROOT.parents[2]
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''): h.update(chunk)
    return h.hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')

def bound(record):
    rel = Path(record['path'])
    if rel.is_absolute() or '..' in rel.parts: raise ValueError('Phase-relative binding required')
    path = (PHASE/rel).resolve(strict=True)
    if not path.is_relative_to(PHASE.resolve()) or sha(path) != record['sha256']: raise ValueError('Bound file changed')
    return path

def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); job = json.loads(args.job.read_text()); output = args.output.resolve()
    if job.get('root_execution_authorized') is not True or job.get('source_review_approved') is not True:
        raise ValueError('Independent root review and release required')
    if any(job.get(k) is not False for k in ('fits_authorized','VALID_values_access','TEST_access','retry')):
        raise ValueError('Discarded TRAIN-only diagnostic only')
    if (job['soft_seconds'],job['hard_seconds'],job['repetitions']) != (900,1200,4) or job['external_hard_bound_confirmed'] is not True:
        raise ValueError('Fixed diagnostic bounds required')
    if Path.cwd().resolve() != REPO.resolve() or socket.gethostname() != 'anogena-2-0' or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise ValueError('Exact allocation required')
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).split() != [GPU]:
        raise ValueError('Physical GPU changed')
    if sha(__file__) != job['program_sha256'] or sha(ROOT/'MANIFEST.json') != job['manifest_sha256']:
        raise ValueError('Diagnostic source changed')
    for row in json.loads((ROOT/'MANIFEST.json').read_text())['files']:
        if sha(ROOT/row['path']) != row['sha256']: raise ValueError('Sealed diagnostic file changed')
    for record in job['bindings']: bound(record)
    if not job['source_review_evidence']: raise ValueError('Root independent review evidence required')
    for record in job['source_review_evidence']: bound(record)
    if output.exists() or not output.is_relative_to(PHASE.resolve()) or str(output) != job['output_directory'] or not output.parent.is_dir():
        raise ValueError('Fresh exact output required')
    if str(Path(sys.executable).absolute()) != job['python_executable']: raise ValueError('Exact interpreter required')
    import numpy as np
    import torch
    import torch_geometric, torch_sparse, torch_scatter
    versions = {'torch':str(torch.__version__),'numpy':np.__version__,'PyG':torch_geometric.__version__,
        'torch_sparse':torch_sparse.__version__,'torch_scatter':torch_scatter.__version__,'CUDA':torch.version.cuda}
    if versions != job['runtime_versions'] or torch.cuda.device_count() != 1: raise ValueError('Pinned runtime differs')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    source = PHASE/job['trainer_relative']
    sys.path.insert(0,str(source))
    from vendor.native_polynormer import Polynormer
    from vendor.backbone_boundary_adapter import PolynormerBoundaryFamily, set_boundary_identity_
    helper_path = bound(job['qualifier_helpers'])
    spec = importlib.util.spec_from_file_location('sealed_diagnostic_helpers',helper_path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    authority = json.loads(bound(job['TRAIN_manifest']).read_text())
    data = torch.load(bound(authority['available']),map_location='cpu',weights_only=True)
    if set(data) != {'x','edge_index','train_ids','train_y'} or tuple(data['x'].shape) != (11701,300) or tuple(data['edge_index'].shape) != (2,442907):
        raise ValueError('Full TRAIN-only public feature/support artifact required')
    output.mkdir(); device = torch.device('cuda:0')
    saved_rng = (random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
    results = {'source_bindings':job['bindings'],'cases':[],'optimizer_updates':0,'backwards':0,'fits':0,
        'VALID_values_access':False,'TEST_access':False,'labels_used':False,'runtime':versions,
        'qualification_tolerances_unchanged':helper.TOLERANCES,'qualification_pass_claimed':False}
    fp32 = {}; torch.cuda.reset_peak_memory_stats()

    def deadline():
        if time.monotonic()-started > 900: raise TimeoutError('Fixed diagnostic soft bound')
        if torch.cuda.max_memory_reserved() > 24*1024**3: raise MemoryError('Fixed24GiB reserve bound')

    def compare(a,b):
        if a.shape != b.shape or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()): raise ValueError('Nonfinite diagnostic comparison')
        d = (a.double()-b.double()).abs(); rtol,atol = helper.TOLERANCES['logits']
        threshold = atol+rtol*b.double().abs()
        return {'exact':bool(torch.equal(a,b)),'max_abs':float(d.max()),'RMS':float(d.square().mean().sqrt()),
            'violations_at_original_tolerance':int((d>threshold).sum()),'elements':d.numel()}

    def snapshot(model): return {n:p.detach().cpu().clone() for n,p in model.named_parameters()}

    try:
        edge = data['edge_index'].to(device)
        for dtype_name,dtype in (('FP32',torch.float32),('FP64',torch.float64)):
            x = data['x'].to(device=device,dtype=dtype)
            for global_stage in (False,True):
                deadline(); stage = 'global' if global_stage else 'local'
                _,native_construct,_ = helper.helpers(False,device,torch,np,Polynormer,PolynormerBoundaryFamily)
                _,family_construct,_ = helper.helpers(True,device,torch,np,Polynormer,PolynormerBoundaryFamily)
                native,opt_n,_,_ = native_construct(17); family,opt_f,_,_ = family_construct(17)
                set_boundary_identity_(family); del opt_n,opt_f
                native.to(dtype=dtype); family.to(dtype=dtype)
                native._global = global_stage; family.core._global = global_stage
                native.eval(); family.eval(); nn = dict(native.named_parameters()); fn = dict(family.named_parameters())
                mapping = {'lin_in':'stem','pred_local':'local_head','pred_global':'global_head'}
                for key in mapping.values():
                    boundary = getattr(family,key)
                    if not bool((boundary.R==1).all() and (boundary.S==1).all()) or boundary.B.stride(0) < boundary.B.shape[1]:
                        raise ValueError('Neutral factors or disjoint private bias rows differ')
                for name,value in nn.items():
                    prefix,_,suffix = name.partition('.')
                    other = fn[mapping[prefix]+'.weight'] if prefix in mapping and suffix=='weight' else fn[mapping[prefix]+'.B'][0] if prefix in mapping else fn['core.'+name]
                    if not torch.equal(value,other): raise ValueError('Native parameter copy differs: '+name)
                    if prefix in mapping and suffix=='bias' and not all(torch.equal(value,row) for row in fn[mapping[prefix]+'.B']): raise ValueError('Private bias rows differ')
                if {p.data_ptr() for p in native.parameters()} & {p.data_ptr() for p in family.parameters()}: raise ValueError('Storage alias between models')
                before_n,before_f = snapshot(native),snapshot(family)
                captures_n,captures_f = {},{}
                active = ('lin_in','pred_global' if global_stage else 'pred_local')
                def capture(destination,key):
                    def hook(module,args,result): destination[key] = (args[0].detach().clone(),result.detach().clone())
                    return hook
                handles = [getattr(native,key).register_forward_hook(capture(captures_n,key)) for key in active]
                handles += [getattr(family,mapping[key]).register_forward_hook(capture(captures_f,key)) for key in active]
                with torch.no_grad():
                    first_n = native(x,edge); first_f = family.forward_member(x,edge,0)
                    for handle in handles: handle.remove()
                    repeated_n = [first_n]; repeated_f = [first_f]
                    for _ in range(3):
                        repeated_n.append(native(x,edge)); deadline()
                        repeated_f.append(family.forward_member(x,edge,0)); deadline()
                    case = {'dtype':dtype_name,'stage':stage,'exact_parameter_copy':True,'no_storage_alias':True,
                        'native_repeatability':[compare(z,first_n) for z in repeated_n[1:]],
                        'family_repeatability':[compare(z,first_f) for z in repeated_f[1:]],
                        'native_vs_family_paired':[compare(a,b) for a,b in zip(repeated_n,repeated_f)],
                        'other_neutral_routes':[compare(family.forward_member(x,edge,m),first_f) for m in range(1,4)],'boundaries':{}}
                    for key in active:
                        n_input,n_output = captures_n[key]; f_input,f_output = captures_f[key]
                        linear = getattr(native,key); boundary = getattr(family,mapping[key])
                        fused = torch.nn.functional.linear(n_input,linear.weight,linear.bias)
                        separate = torch.nn.functional.linear(n_input,linear.weight,None)+linear.bias
                        literal = boundary(n_input,0)
                        double_reference = torch.nn.functional.linear(n_input.double(),linear.weight.double(),linear.bias.double())
                        case['boundaries'][key] = {'native_vs_family_inputs':compare(n_input,f_input),
                            'captured_outputs':compare(n_output,f_output),'fixed_same_input_fused_vs_separate':compare(fused,separate),
                            'fixed_same_input_fused_vs_literal_wrapper':compare(fused,literal),
                            'separate_vs_literal_wrapper':compare(separate,literal),
                            'fused_vs_FP64_reference':compare(fused,double_reference),
                            'literal_wrapper_vs_FP64_reference':compare(literal,double_reference)}
                        del fused,separate,literal,double_reference
                    if dtype_name=='FP32': fp32[stage] = {'native':first_n.detach().cpu().double(),'family':first_f.detach().cpu().double()}
                    else:
                        case['FP32_vs_FP64_full_trajectory'] = {'native':compare(fp32[stage]['native'],first_n.cpu()),'family':compare(fp32[stage]['family'],first_f.cpu())}
                if any(not torch.equal(before_n[n],v.detach().cpu()) for n,v in native.named_parameters()) or any(not torch.equal(before_f[n],v.detach().cpu()) for n,v in family.named_parameters()):
                    raise ValueError('Diagnostic changed a parameter')
                case['parameters_unchanged'] = True; results['cases'].append(case)
                write(output/'PROGRESS.json',{'completed_cases':len(results['cases']),'total_cases':4,'optimizer_updates':0,'VALID_TEST_values_access':False})
                del native,family,nn,fn,before_n,before_f,captures_n,captures_f,repeated_n,repeated_f,first_n,first_f,n_input,n_output,f_input,f_output,linear,boundary,value,other
                torch.cuda.empty_cache(); deadline()
        results.update(diagnostic_complete=True,inclusive_seconds=time.monotonic()-started,
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(),
            job_sha256=sha(args.job),states_discarded=True,
            interpretation='Report repeatability baselines and fixed-input boundary arithmetic deltas. FP64 uses exact converted FP32 initial parameter values, not a new initialization. No tolerance change or method/runtime qualification claim.')
    except BaseException as error:
        write(output/'FAILURE.json',{'error':type(error).__name__+': '+str(error),'results':results,'optimizer_updates':0,'VALID_TEST_values_access':False,'retry':False}); raise
    finally:
        random.setstate(saved_rng[0]); np.random.set_state(saved_rng[1]); torch.set_rng_state(saved_rng[2]); torch.cuda.set_rng_state(saved_rng[3])
    results['caller_RNG_restored'] = random.getstate()==saved_rng[0] and torch.equal(torch.get_rng_state(),saved_rng[2]) and torch.equal(torch.cuda.get_rng_state(),saved_rng[3])
    numpy_state = np.random.get_state()
    results['caller_RNG_restored'] = results['caller_RNG_restored'] and numpy_state[0]==saved_rng[1][0] and numpy_state[2:]==saved_rng[1][2:] and np.array_equal(numpy_state[1],saved_rng[1][1])
    if not results['caller_RNG_restored']: raise ValueError('Diagnostic RNG restoration failed')
    write(output/'DIAGNOSTIC.json',results)

if __name__=='__main__': main()
