"""Synthetic Bernoulli serving algebra and checkpoint custody; no fitting."""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:raise ValueError('Exact authorized route required')
    root=Path(a.project_root).resolve();out=root/a.output
    if out.exists() or not out.resolve().is_relative_to(root):raise ValueError('Fresh confined engineering output required')
    os.environ['CUDA_VISIBLE_DEVICES']='';out.mkdir();sys.path.insert(0,str(Path(__file__).resolve().parent/'frozen_runtime'))
    import torch
    from adapters.sehgnn_endpoints import pooled_log_odds,selected_authority
    from source_utils import receipt
    try:
        g=torch.Generator().manual_seed(84901);z=torch.randn(4,12,5,generator=g,dtype=torch.float64);truth=((torch.arange(12)[:,None]+torch.arange(5)[None])%2).double();checks=[]
        for fusion in ('mean_probabilities','mean_logits'):
            observed=pooled_log_odds(torch,z,fusion);oracle=z.sigmoid().mean(0) if fusion=='mean_probabilities' else z.mean(0).sigmoid()
            if not torch.allclose(observed.sigmoid(),oracle,rtol=1e-12,atol=1e-12):raise ValueError('Bernoulli serving law differs from independent direct sigmoid oracle')
            loss=torch.nn.functional.binary_cross_entropy_with_logits(observed,truth);ref=-torch.where(truth.bool(),oracle.log(),torch.log1p(-oracle)).mean()
            if not torch.allclose(loss,ref,rtol=1e-12,atol=1e-12):raise ValueError('Marginal BCE differs from direct Bernoulli proper score')
            checks.append({'fusion':fusion,'independent_probability_and_BCE_algebra':'pass'})
        extreme=torch.stack((torch.full((2,5),-1000.),torch.full((2,5),1000.)))
        if not torch.isfinite(pooled_log_odds(torch,extreme,'mean_probabilities')).all():raise ValueError('Stable extreme Bernoulli aggregation required')
        folder=out/'synthetic-authority'/'study'/'fit';folder.mkdir(parents=True);cp=folder/'selected.pt';cp.write_bytes(b'synthetic metadata placeholder; never a real checkpoint');r=receipt(root,cp);authority={'path':'selected.pt','fit_id':'fit','step':900,**{k:r[k] for k in ('bytes','sha256')}};prefixes=[]
        for cap in (200,500):
            name='native'+str(cap);d=folder/'prefixes'/name;d.mkdir(parents=True);(d/'selected.pt').write_bytes(b'not model state');prefix={'prefix_id':name,'cap_step':cap,'close_on_selector_stop':False,'closed':True,'closure_step':cap,'closure_reason':'cap','selected':{'step':cap-1,'path':'prefixes/'+name+'/selected.pt','validation':{}}};(d/'certificate.json').write_text(json.dumps(prefix));prefixes.append(prefix)
        row={'state':'complete','contract':{'run_id':'fit','budget':{'maximum':1000}},'result':{'status':'complete','completed_steps':1000,'selected_checkpoint':authority,'selected':{'step':900},'prefixes':prefixes}}
        selected_authority(root,out/'synthetic-authority',{'study_id':'study'},row)
        invalid=[]
        for field,value in (('completed_steps',500),('status','failed')):
            x=copy.deepcopy(row);x['result'][field]=value
            try:selected_authority(root,out/'synthetic-authority',{'study_id':'study'},x)
            except ValueError:invalid.append(field)
            else:raise ValueError('Partial/failed fit promoted by endpoint')
        x=copy.deepcopy(row);x['result']['prefixes'][0]['selected']['step']=201
        try:selected_authority(root,out/'synthetic-authority',{'study_id':'study'},x)
        except ValueError:invalid.append('prefix_outside_cap')
        else:raise ValueError('Invalid prefix promoted')
        cp.write_bytes(b'changed metadata placeholder')
        try:selected_authority(root,out/'synthetic-authority',{'study_id':'study'},row)
        except ValueError:invalid.append('changed_selected_bytes')
        else:raise ValueError('Changed checkpoint bytes promoted')
        import shutil;shutil.rmtree(out/'synthetic-authority')
        if torch.cuda.is_initialized():raise ValueError('CPU algebra initialized CUDA')
        result={'status':'passed','serving':checks,'extreme_aggregation_finite':True,'custody_rejections':invalid,'model_constructors':0,'real_input_model_calls':0,'optimizer_updates':0,'fits':0,'predictive_scores':0,'CUDA':False,'GO':False};(out/'RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'passed','fusion_algebra_cases':2,'custody_negative_cases':4,'fits':0,'CUDA':False}))
    except BaseException as e:
        (out/'FAILURE.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc(),'fits':0,'GO':False},indent=2)+'\n');raise


if __name__=='__main__':main()
