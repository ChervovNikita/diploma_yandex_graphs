"""Label-free full-graph native/identity forward repeatability diagnostic."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from qualification_entry_v3 import PHASE, REPO, LOGIN, UUID, sha, confined, bound, write

sys.dont_write_bytecode=True

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',required=True)
    p.add_argument('--supervisor',required=True)
    args=p.parse_args()
    if Path.cwd().resolve()!=REPO or os.environ.get('GNNM_PHASE_ROOT')!=str(PHASE) or os.environ.get('GNNM_SSH_DESTINATION')!=LOGIN:
        raise ValueError('Wrong repository/environment/route')
    import subprocess
    if subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()!=[UUID]:
        raise ValueError('Wrong allocation')
    request_path=confined(args.request);request=json.loads(request_path.read_text())
    if not request['root_admitted'] or request['full_fit_admitted'] or request['heldout_scoring_admitted']:
        raise ValueError('Only a report-ineligible label-free diagnostic is admitted')
    for row in request['protected_files']:bound(row)
    packet=confined(request['source_packet'])
    for row in json.loads(bound(request['source_manifest']).read_text())['payload']:
        path=packet/row['path']
        if not path.resolve().is_relative_to(packet) or sha(path)!=row['sha256']:
            raise ValueError('Immutable source packet mismatch')
    out=confined(request['output']);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    write(out/'START.json',dict(request_sha256=sha(request_path),labels_read=False,training_steps=0))
    try:
        if request['deterministic']:
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        sys.path.insert(0,str(packet/'prototype'))
        import correction_screen_driver as core
        import torch
        torch.use_deterministic_algorithms(request['deterministic'])
        np,torch,environment=core.runtime('cuda:0')
        environment=json.loads(json.dumps(environment,allow_nan=False))
        import modern_teacher_adapter as adapter
        from backbone_boundary_adapter import set_boundary_identity_
        role_path=bound(request['role_freeze']);roles=json.loads(role_path.read_text())
        core.verify_tree(role_path.parent,roles['payload'])
        manifest=json.loads(bound(roles['graph_input']).read_text())
        x,edges=core.canonical_graph(np,manifest)
        graph=adapter.prepare_graph(torch.from_numpy(x.copy()).to('cuda:0'),torch.from_numpy(edges.copy()).to('cuda:0'),
            'polynormer_r',environment,dict(graph_input=roles['graph_input'],implementation_sha256=request['implementation_sha256']))
        spec=adapter.specification('polynormer_r','single_author',0,17)
        native=adapter.TeacherFamily(spec).to('cuda:0').eval()
        shared=adapter.TeacherFamily(adapter.specification('polynormer_r','gnnm_boundary_4',0,17)).to('cuda:0').eval()
        set_boundary_identity_(shared.boundary)
        def difference(a,b):
            d=(a-b).abs();bad=d>(1e-6+1e-5*a.abs())
            return dict(max_absolute_difference=float(d.max()),elements=int(d.numel()),
                        above_frozen_identity_tolerance=int(bad.sum()),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
        rows=[]
        captured={}
        pairs=(('stem','lin_in'),('local_head','pred_local'),('global_head','pred_global'))
        hooks=[]
        for boundary,name in pairs:
            def capture(module,inputs,output,key=name):captured[key]=inputs[0].detach().clone()
            hooks.append(getattr(native.models[0],name).register_forward_hook(capture))
        with torch.no_grad():
            for stage in (False,True):
                native.set_global_stage(stage);shared.set_global_stage(stage)
                captured.clear()
                a0=native(graph)[0];a1=native(graph)[0];a2=native(graph)[0]
                b0=shared(graph);b1=shared(graph)
                isolated=[]
                for boundary,name in pairs:
                    if name not in captured:continue
                    layer=getattr(native.models[0],name);block=getattr(shared.boundary,boundary)
                    inputs=captured[name]
                    fused=torch.nn.functional.linear(inputs,layer.weight,layer.bias)
                    separate=torch.nn.functional.linear(inputs,layer.weight)+layer.bias
                    isolated.append(dict(boundary=boundary,
                        fused_vs_separate_bias=difference(fused,separate),
                        native_vs_identity_block=difference(fused,block(inputs,0))))
                rows.append(dict(global_stage=stage,native_repeat=[difference(a0,a1),difference(a0,a2)],
                    shared_repeat=[difference(b0[m],b1[m]) for m in range(4)],
                    native_vs_identity=[difference(a0,b0[m]) for m in range(4)],
                    member0_vs_identity_members=[difference(b0[0],b0[m]) for m in range(1,4)],
                    isolated_boundary_arithmetic=isolated))
        for hook in hooks:hook.remove()
        write(out/'REPEATABILITY.json',dict(environment=environment,checks=rows,
            graph_input=roles['graph_input'],role_freeze=request['role_freeze'],
            source_manifest=request['source_manifest'],request_sha256=sha(request_path),
            report_eligible=False,qualification_pass=False,labels_read=False,training_steps=0,
            tolerance_unchanged=True,reason='Diagnose native stochastic reduction versus wrapper arithmetic; no fit or gate promotion.'))
        write(out/'TERMINAL.json',dict(complete=True,seconds=time.monotonic()-started,labels_read=False,training_steps=0,
            report_eligible=False,qualification_pass=False,repeatability_sha256=sha(out/'REPEATABILITY.json')))
    except Exception as error:
        write(out/'FAILED_ATTEMPT.json',dict(error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            seconds=time.monotonic()-started,labels_read=False,training_steps=0,automatic_retry=False))
        raise

if __name__=='__main__':main()
