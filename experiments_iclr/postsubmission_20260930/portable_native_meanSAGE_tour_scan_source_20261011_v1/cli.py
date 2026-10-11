"""Prepare public Physics, then acquire selected fixed banks and native exports."""
import argparse
from pathlib import Path
import time
from urllib.request import urlopen
from common import ARMS,BLOCKS,bundle,read,receipt,require,runtime_versions,write

def prepare(args):
    from prepare import DOWNLOAD_URL,prepare as build
    work=args.work.resolve();work.mkdir(parents=True,exist_ok=False)
    source=bundle();versions=runtime_versions();raw=args.raw.resolve() if args.raw else work/'public_physics.npz'
    if args.download:
        with urlopen(DOWNLOAD_URL,timeout=60) as response,raw.open('xb') as target:
            for value in iter(lambda:response.read(1048576),b''):target.write(value)
    require(raw.is_file(),'Supply the fixed public archive or request its pinned download')
    build(raw,work/'prepared')
    write(work/'PREPARATION.json',dict(complete=True,source_manifest=source,runtime=versions,custody=receipt(work/'prepared/CUSTODY.json'),training_verified=False))

def fit(args):
    from fit import fit as own_fit
    from runtime import Runtime
    source=bundle();preparation=read(args.work.resolve()/'PREPARATION.json')
    require(preparation['complete'] and preparation['source_manifest']==source,'Exact portable preparation source')
    from common import checked
    prepared=checked(preparation['custody']);output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    rt=None;rows=[];selected=0;started=time.monotonic()
    roster=[(b,a) for b in BLOCKS for a in ARMS] if args.all else [(args.block,args.arm)]
    result=dict(complete=False,terminal=False,records=rows,source_manifest=source,preparation=preparation['custody'],requested_roster=[dict(block=b,arm=a) for b,a in roster],scope='Full fixed family acquired afresh' if args.all else 'One fixed bank demonstration; no scientific family verdict',private_prior_states_reused=False,CAL_DEV_truth_read=False,retry=False)
    try:
        rt=Runtime(prepared,args.device,args.wall_seconds);result.update(graph=rt.graph,runtime=rt.versions_metadata,device=args.device)
        write(output/'RUN_BINDING.json',dict(source_manifest=source,preparation=preparation['custody'],graph=rt.graph,roster=result['requested_roster'],device=args.device,runtime=rt.versions_metadata))
        for block,arm in roster:
            rt.guard();roles=rt.roles(block,select=True);folder=output/f'block_{block}_{arm}';folder.mkdir()
            core,optimizers,audit=rt.construct(arm,block);row=dict(block=block,arm=arm,status='running',construction=audit,fits=[],exports={},actual_served_members=core['members'],origin='portable_fresh_fit');rows.append(row);finals=[]
            for index,(model,optimizer) in enumerate(zip(core['models'],optimizers)):
                streams=rt.streams(block,model.members,start_index=index if core['genuine_independent'] else 0)
                meta=dict(block=block,arm=arm,body=index,source_manifest=source,preparation=preparation['custody'],construction=audit,graph=rt.graph,device=args.device)
                record,final=own_fit(rt,model,optimizer,streams,roles,folder/f'fit_{index}',meta);row['fits'].append(record);finals.append(final);selected+=1
                del streams
            logits=rt.t.cat(finals,dim=0);require(logits.shape==(core['members'],34493,5),'Full selected serving membership')
            probability=logits.softmax(-1);pool=probability.mean(0);lp=logits.log_softmax(-1)
            for role in ('cal','dev'):
                ids=roles[role+'_ids_numpy'];index_tensor=rt.tensor(ids);path=folder/('selected_'+role+'.npz')
                rt.np.savez(path,ids=ids,member_log_probs=lp[:,index_tensor].cpu().numpy(),native_member_argmax=logits[:,index_tensor].argmax(-1).cpu().numpy(),native_pool_argmax=pool[index_tensor].argmax(-1).cpu().numpy(),native_pool_probabilities=pool[index_tensor].cpu().numpy());row['exports'][role]=receipt(path)
            row.update(status='exported_complete',optimizer_trajectories=len(row['fits']),physical_backbones=sum(f['physical_backbones'] for f in row['fits']),probability_pool_authority=True)
            write(folder/'BANK_RESULT.json',row);write(output/'PROGRESS.json',dict(closed_banks=sum(r['status']=='exported_complete' for r in rows),selected_trajectories=selected,requested_banks=len(roster)))
            print('Selected and exported block '+str(block)+' arm '+arm,flush=True)
            del core,optimizers,model,optimizer,final,finals,logits,probability,pool,lp
            if rt.device.type=='cuda':rt.t.cuda.empty_cache()
        require(len(rows)==len(roster) and all(r['status']=='exported_complete' for r in rows),'Every requested fixed bank closes')
        if args.all:require(selected==48 and sum(r['actual_served_members'] for r in rows)==102,'Full30bank/48fresh-trajectory/102output family')
        result.update(complete=True,terminal=True,banks=len(rows),optimizer_trajectories=selected,physical_bodies=sum(r['physical_backbones'] for r in rows),logical_member_view_outputs=sum(r['actual_served_members'] for r in rows),CAL_DEV_readout_complete=False)
    except BaseException as error:
        result.update(terminal=True,failure=dict(type=type(error).__name__,message=str(error)))
        if rows and rows[-1]['status']=='running':rows[-1]['status']='failed_retained'
        raise
    finally:
        result['seconds']=time.monotonic()-started
        if rt is not None:result['costs']=rt.costs()
        write(output/'SELECTED_EXPORTS.json',result)

def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare');p.add_argument('--work',type=Path,required=True);raw=p.add_mutually_exclusive_group(required=True);raw.add_argument('--raw',type=Path);raw.add_argument('--download',action='store_true')
    p=commands.add_parser('fit');p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=('cpu','cuda:0'),required=True);p.add_argument('--wall-seconds',type=int,default=43200)
    roster=p.add_mutually_exclusive_group();roster.add_argument('--arm',choices=ARMS,default='SW4');roster.add_argument('--all',action='store_true');p.add_argument('--block',type=int,choices=BLOCKS,default=10103)
    args=parser.parse_args()
    if args.command=='fit':require(args.wall_seconds>0,'Finite positive user wall limit')
    try:globals()[args.command](args)
    except BaseException as error:
        if args.command=='prepare' and args.work.exists():
            write(args.work/'FAILED_RETAINED.json',dict(complete=False,type=type(error).__name__,message=str(error),retry=False))
        raise

if __name__=='__main__':main()
