"""Model-free public Twitch-DE binary roles and native paired-incidence custody."""
import argparse,hashlib,inspect,json,socket,subprocess
from pathlib import Path


def receipt(root,p):
 b=p.read_bytes();return {'path':str(p.relative_to(root)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def prepare(root,packet):
 import torch
 from torch_geometric.datasets import Twitch
 root=Path(root).resolve();packet=Path(packet).resolve();out=packet/'data'
 if out.exists():raise ValueError('Fresh role preparation;preserve prior attempt')
 out.mkdir();cache=packet/'public_provider_cache';dataset=Twitch(str(cache),name='DE');data=dataset.get(0);x=data.x.float().contiguous();y=data.y.long().flatten();edge=data.edge_index.cpu()
 if x.ndim!=2 or x.shape[1]!=128 or y.shape!=(len(x),) or sorted(y.unique().tolist())!=[0,1] or not torch.isfinite(x).all():raise ValueError('Public binary mature Twitch-DE provider geometry')
 # Label-blind native support transformation: paired undirected/no-self/unique;fixed lexicographic order.
 edge=edge[:,edge[0]!=edge[1]];keys=torch.cat((edge[0]*len(x)+edge[1],edge[1]*len(x)+edge[0])).unique(sorted=True);edge=torch.stack((keys//len(x),keys%len(x))).long()
 from adapters.nsd_sources import environment
 B=json.loads((packet/'SOURCE_BINDINGS.json').read_text());env=environment(torch,root,B);env.adapter.validate_native_graph(edge,len(x))
 torch.save({'x':x,'edge_index':edge},out/'GRAPH.pt');graph=receipt(root,out/'GRAPH.pt');splits=[];files=[graph]
 for split in range(10):
  roles={r:[] for r in ('train','select','cal','evaluate')}
  for cls in (0,1):
   ids=torch.where(y==cls)[0];g=torch.Generator().manual_seed(97091+1009*split+13*cls);order=ids[torch.randperm(len(ids),generator=g)];cut1=int(.4*len(ids));cut2=cut1+int(.2*len(ids));cut3=cut2+int(.2*len(ids))
   for r,z in zip(roles,(order[:cut1],order[cut1:cut2],order[cut2:cut3],order[cut3:])):roles[r].append(z)
  node={'graph':graph};end={};counts={}
  for role,chunks in roles.items():
   ids=torch.cat(chunks).sort().values;payload={'ids':ids,'y':y[ids]};p=out/(role+'_sp'+str(split)+'.pt');torch.save(payload,p);row=receipt(root,p);files.append(row);counts[role]=len(ids)
   if role in ('train','select'):node[role]=row
   else:
    end[role+'_targets']=row;ip=out/(role+'_ids_sp'+str(split)+'.pt');torch.save(ids,ip);end[role+'_ids']=receipt(root,ip);files.append(end[role+'_ids'])
  splits.append({'split_id':split,'node_inputs':node,'endpoint_inputs':end,'counts':counts})
 source=Path(inspect.getfile(Twitch));provider_files=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in cache.rglob('*') if p.is_file()]
 value={'schema':'NSD-new-public-binary-role-custody-v1','dataset':'twitch-de','model_graph':{'nodes':len(x),'features':128,'classes':2,'edges':edge.shape[1]},'splits':splits,'files':files,'provider':{'module':Twitch.__module__,'installed_file':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'public_url':Twitch.url,'raw_processed_files':provider_files},
  'graph_policy':'copy public mature binary task and128features;label-blind undirected coalesced no-self support,lexicographic ordered incidences;source reverse pairing and augmented block-degree SVD normalization unchanged','role_rule':'10fixed seed-stratified node masks40%TRAIN20%SELECT20%CALremainder evaluate;onepublicgraph,not10independentgraphs','prior_exposure':'development;wholeproject prior-exposure audit/root review pending;no protected confirmation claim','source_compatibility':'binary NSD AUROC/NLL and binarymargin liveaction semantics preserved;no multiclasslabelbinarization','model_constructors':0,'optimizer_objects':0,'scientific_fits':0,'CUDA':0,'GO':False}
 (packet/'CUSTODY.json').write_text(json.dumps(value,indent=2)+'\n');return value

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--packet',required=True);a=p.parse_args()
 if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:raise ValueError('Exact allocation identity')
 v=prepare(a.project_root,a.packet);print(json.dumps({'graph':v['model_graph'],'splits':len(v['splits']),'models':0,'optimizers':0,'CUDA':0,'fits':0,'GO':False}))
