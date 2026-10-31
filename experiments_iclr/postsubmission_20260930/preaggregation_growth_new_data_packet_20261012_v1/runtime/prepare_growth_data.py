"""Fresh public HeaRT role custody and model-free TRAIN episode materialization."""
import argparse,hashlib,io,json,socket,subprocess,tarfile,time
from pathlib import Path

try:from adapters.native_label_families import bound,receipt
except ImportError:from .adapters.native_label_families import bound,receipt

ARCHIVE={'path':'citeseer_heart_official_acquisition_server_20261005_v1/withheld_inputs/HeaRT.tar.gz','bytes':880240878,'sha256':'7b7042476319a353bdb6c50b5f402b89b9006a2fde2d1258b7adcbd1d22629ba'}
PINS={'gnn_feature':{'bytes':49280235,'sha256':'14b7d9de27cf0775147737126899c15151e82bcd4dde67afa31f3650f2609944'},
 'train_pos.txt':{'bytes':36532,'sha256':'1e97ad3a73ecfeb69489aa6d92d02dc44c2925b4056a8e4cd4b548f1c0ebaa27'},
 'valid_pos.txt':{'bytes':2127,'sha256':'11ad6734a54824d84a3dfcede53bcd6b3051c012f4c44add04891858dacd968c'},
 'heart_valid_samples.npy':{'bytes':1816128,'sha256':'89cdc6fd98e059e2be200025174762c2ee82f86703152bb733271322b8fb3e65'}}
SEARCH=(99101,99103);COMPARE=(99201,99203,99207)


def positive(torch,raw,nodes):
 pairs=[];selfs=0
 for line in raw.decode().splitlines():
  cells=line.split('\t')
  if len(cells)!=2:raise ValueError('Native tab positive record format')
  a,b=map(int,cells)
  if min(a,b)<0 or max(a,b)>=nodes:raise ValueError('Native node range')
  if a==b:selfs+=1
  else:pairs.append([a,b])
 if len({tuple(sorted(p)) for p in pairs})!=len(pairs):raise ValueError('Duplicate positive undirected source facts')
 return torch.tensor(pairs,dtype=torch.long),selfs


def prepare(root,packet,*,include_endpoints=False):
 import numpy as np
 import torch
 from adapters.preaggregation_growth import sources
 root=Path(root).resolve();packet=Path(packet).resolve();out=packet/'data'
 if out.exists():raise ValueError('Retain prior source preparation output')
 out.mkdir();B=json.loads((packet/'SOURCE_BINDINGS.json').read_text());s=sources(root,B)
 base=root/'citeseer_heart_official_acquisition_server_20261005_v1/available/citeseer';raw={};source_records={}
 for name in ('gnn_feature','train_pos.txt'):
  row={'path':str((base/name).relative_to(root)),**PINS[name]};p=bound(root,row);raw[name]=p.read_bytes();source_records[name]=row
 feature=torch.load(io.BytesIO(raw['gnn_feature']),map_location='cpu',weights_only=True)
 if set(feature)!={'entity_embedding'} or feature['entity_embedding'].shape!=(3327,3703) or feature['entity_embedding'].dtype!=torch.float32 or not torch.isfinite(feature['entity_embedding']).all():raise ValueError('Exact official HeaRT source feature tensor')
 train,selfs=positive(torch,raw['train_pos.txt'],3327)
 if train.shape!=(3870,2) or selfs!=48:raise ValueError('Complete native TRAIN population')
 torch.save({'x':feature['entity_embedding'],'train_support_edge_index':torch.cat((train.t(),train.t().flip(0)),1)},out/'GRAPH.pt');torch.save({'positive':train},out/'TRAIN.pt')
 manifest={'schema':'independent-growth-public-HeaRT-role-custody-v1','dataset':'citeseer-heart','nodes':3327,'features':3703,'TRAIN_positives':3870,'TRAIN_removed_self_records':48,
  'link_inputs':{'graph':receipt(root,str((out/'GRAPH.pt').relative_to(root))),'train':receipt(root,str((out/'TRAIN.pt').relative_to(root)))},'official_inputs':source_records,
  'source_feature_recipe':'copy official entity_embedding float32 exactly;no normalization/label-derived scaling;official archive provenance retained,not a trained donor tensor',
  'prior_exposure':'development;previous TRAIN/VALID research exposure;no protected confirmation claim','model_constructors':0,'optimizer_objects':0,'CUDA':0,'scientific_fits':0,'GO':False,
  'endpoint_preparation':False,'calibration_draws':{},'future_train_episode_policy':'exact source native_cycle and endpoint/matched-random union mask;all61episodes incltail,outer64 inner256,no redraw/skip;own checkpointed route Python sampling streams'}
 for seed in (*SEARCH,*COMPARE):
  e=s.geometry.route_streams(seed,4);r=s.geometry.route_streams(seed,4,control=True)
  negative,order,episodes=s.custody.make_pair(torch,s.geometry,s.cycle,train,3327,seed,0,64,256,e,r)
  if len(order)!=3870 or len(episodes)!=61:raise ValueError('Complete calibration source cycle')
  path=out/('CALIBRATION_TRAIN_DRAW_s'+str(seed)+'.pt')
  torch.save({'negative':negative,'outer_order':order,'episodes':episodes,'seed':seed,'cycle_index':0,'GRAPH':manifest['link_inputs']['graph'],'TRAIN':manifest['link_inputs']['train'],'source_BINDINGS':{k:B[k] for k in ('geometry','cycle','custody_functions')}},path)
  manifest['calibration_draws'][str(seed)]=receipt(root,str(path.relative_to(root)))
  print(json.dumps({'seed':seed,'TRAIN_episode_cycle_materialized':61,'model_or_optimizer_work':0}),flush=True)
 if include_endpoints:
  from prepare_link_endpoints import shuffle_records
  held={}
  archive=bound(root,ARCHIVE)
  # Reauthenticate the public archive; unpack only the two necessary TEST query authorities.
  with tarfile.open(archive,'r:gz') as tar:
   for name in ('test_pos.txt','heart_test_samples.npy'):
    member=tar.getmember('dataset/citeseer/'+name)
    if not member.isfile() or member.size>64*1024**2:raise ValueError('Bound public TEST member size/type')
    held[name]=tar.extractfile(member).read();source_records[name]={'archive_member':member.name,'bytes':len(held[name]),'sha256':hashlib.sha256(held[name]).hexdigest()}
  validraw={}
  for name in ('valid_pos.txt','heart_valid_samples.npy'):
   row={'path':str((base/name).relative_to(root)),**PINS[name]};validraw[name]=bound(root,row).read_bytes();source_records[name]=row
  valid,_=positive(torch,validraw['valid_pos.txt'],3327);test,_=positive(torch,held['test_pos.txt'],3327)
  vpool=np.load(io.BytesIO(validraw['heart_valid_samples.npy']),allow_pickle=False);tpool=np.load(io.BytesIO(held['heart_test_samples.npy']),allow_pickle=False)
  if valid.shape!=(227,2) or vpool.shape!=(227,500,2) or tpool.shape!=(len(test),500,2) or vpool.dtype!=np.int64 or tpool.dtype!=np.int64:raise ValueError('Whole public perpositive500negative pools')
  for pos,pool in ((valid,vpool),(test,tpool)):
   if np.any((pool<0)|(pool>=3327)) or np.any(pool[:,:250,0]!=pos[:,0,None].numpy()) or np.any(pool[:,250:,1]!=pos[:,1,None].numpy()):raise ValueError('Source negative query association')
  sets=[{tuple(sorted(p)) for p in z.tolist()} for z in (train,valid,test)]
  if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):raise ValueError('Public positive roles overlap')
  torch.save({'positive':valid,'negative':torch.from_numpy(vpool)},out/'SELECT.pt');manifest['link_inputs']['select']=receipt(root,str((out/'SELECT.pt').relative_to(root)))
  # Prospectively fixed deterministic query-level half division; no model/score/label criterion.
  order=np.random.default_rng(97091).permutation(len(test));cut=(len(test)+1)//2;manifest['endpoint_inputs']={};manifest['endpoint_counts']={}
  for role,ids in (('cal',order[:cut]),('evaluate',order[cut:])):
   pos=test[torch.from_numpy(ids)];neg=torch.from_numpy(tpool[ids].copy())
   query,target,nonce=shuffle_records(torch,np,pos,neg.reshape(-1,2));target['negative_pool_width']=500
   # The public opaque shuffle's original_index suffices to reconstruct whole ranking groups.
   qp=out/(role.upper()+'_QUERIES.pt');tp=out/('SEALED_'+role.upper()+'_TARGETS.pt');torch.save(query,qp);torch.save(target,tp)
   manifest['endpoint_inputs'].update({role+'_queries':receipt(root,str(qp.relative_to(root))),role+'_targets':receipt(root,str(tp.relative_to(root)))})
   manifest['endpoint_counts'][role]={'positive_queries':len(pos),'negative_queries':len(pos)*500,'records':len(query['ids']),'nonce_sha256':nonce}
  manifest.update(endpoint_preparation=True,SELECT_complete_official_VALID227=True,public_TEST_preparation_only=True,
   CAL_evaluate_role_rule='fixed seed97091 permutation of official TEST positive rows;CAL ceiling half/evaluate remainder;retain each full500 associated negative queries;opaque unlabelled export queries,targets separate',
   heldout_guided_basis=False,official_archive=ARCHIVE,predictive_scores_computed=False)
 path=packet/'CUSTODY.json';path.write_text(json.dumps(manifest,indent=2)+'\n');return manifest

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--packet',required=True);p.add_argument('--include-endpoints',action='store_true');a=p.parse_args()
 if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:raise ValueError('Exact allocation identity before project source operation')
 value=prepare(a.project_root,a.packet,include_endpoints=a.include_endpoints);print(json.dumps({'prepared':True,'endpoint_preparation':value['endpoint_preparation'],'CUDA':0,'models':0,'optimizers':0,'fits':0,'GO':False}))
