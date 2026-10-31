"""Model-free public Twitch-DE binary roles and native paired-incidence custody."""
import argparse,csv,hashlib,io,json,socket,subprocess,zipfile
from pathlib import Path


def receipt(root,p):
 b=p.read_bytes();return {'path':str(p.relative_to(root)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def prepare(root,packet):
 import torch
 root=Path(root).resolve();packet=Path(packet).resolve();out=packet/'data'
 if out.exists():raise ValueError('Fresh role preparation;preserve prior attempt')
 out.mkdir();archive=packet/'official_acquisition/twitch.zip';origin=json.loads((packet/'official_acquisition/ORIGIN.json').read_text())
 if archive.stat().st_size!=origin['bytes'] or hashlib.sha256(archive.read_bytes()).hexdigest()!=origin['sha256']:raise ValueError('Authoritative exact SNAP archive byte authority')
 z=zipfile.ZipFile(archive);names=['twitch/DE/musae_DE.json','twitch/DE/musae_DE_edges.csv','twitch/DE/musae_DE_target.csv'];raw_files=[]
 rawdir=packet/'raw_DE';rawdir.mkdir()
 for name in names:
  p=rawdir/Path(name).name;p.write_bytes(z.read(name));raw_files.append(receipt(root,p))
 feature=json.loads((rawdir/'musae_DE.json').read_text());N=len(feature)
 targets=list(csv.DictReader(io.StringIO((rawdir/'musae_DE_target.csv').read_text())));records=list(csv.DictReader(io.StringIO((rawdir/'musae_DE_edges.csv').read_text())))
 if N!=9498 or len(targets)!=N or {int(k) for k in feature}!=set(range(N)) or {int(r['new_id']) for r in targets}!=set(range(N)):raise ValueError('Complete original DE new_id custody')
 F=max(int(i) for ids in feature.values() for i in ids)+1
 if F!=3170 or any(int(i)<0 for ids in feature.values() for i in ids):raise ValueError('Complete raw3170 attribute-ID scope')
 x=torch.zeros(N,F,dtype=torch.float32);y=torch.empty(N,dtype=torch.long)
 for key,ids in feature.items():x[int(key),torch.tensor(sorted(set(ids)),dtype=torch.long)]=1
 for r in targets:
  if r['mature'] not in ('True','False'):raise ValueError('Literal public mature binary labels')
  y[int(r['new_id'])]=int(r['mature']=='True')
 edge=torch.tensor([[int(r['from']),int(r['to'])] for r in records],dtype=torch.long).t().contiguous();raw_edge_records=edge.shape[1]
 if x.shape!=(9498,3170) or sorted(y.unique().tolist())!=[0,1] or int(edge.min())<0 or int(edge.max())>=N:raise ValueError('Fresh authoritative raw DE task geometry')
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

 value={'schema':'NSD-new-public-binary-role-custody-v1','dataset':'twitch-de-musae-raw','model_graph':{'nodes':len(x),'features':F,'classes':2,'edges':edge.shape[1]},'splits':splits,'files':files,'provider':{'origin':origin,'archive':receipt(root,archive),'raw_DE_files':raw_files,'raw_edge_records':raw_edge_records,'feature_rule':'binary indicator per original raw attribute-ID;duplicates idempotent;no learned transform,model or teacher','label_rule':'literal mature boolean by original new_id'},
  'graph_policy':'fresh authoritative SNAP/MUSAE public mature binary task and3170raw attribute indicators;label-blind undirected coalesced no-self support,lexicographic ordered incidences;source reverse pairing and augmented block-degree SVD normalization unchanged','role_rule':'10fixed seed-stratified node masks40%TRAIN20%SELECT20%CALremainder evaluate;onepublicgraph,not10independentgraphs','prior_exposure':'development;wholeproject prior-exposure audit/root review pending;no protected confirmation claim','source_compatibility':'binary NSD AUROC/NLL and binarymargin liveaction semantics preserved;no multiclasslabelbinarization;not unavailable128embedding/source-score replication','model_constructors':0,'optimizer_objects':0,'scientific_fits':0,'CUDA':0,'GO':False}
 (packet/'CUSTODY.json').write_text(json.dumps(value,indent=2)+'\n');return value

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--packet',required=True);a=p.parse_args()
 if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:raise ValueError('Exact allocation identity')
 v=prepare(a.project_root,a.packet);print(json.dumps({'graph':v['model_graph'],'splits':len(v['splits']),'models':0,'optimizers':0,'CUDA':0,'fits':0,'GO':False}))
