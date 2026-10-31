"""Approval-gated own source reference; does not extend author CLI dispatch."""
import argparse
import json
from pathlib import Path
import traceback

from prepare_node_data import verify_allocation
from source_utils import bound,receipt,load,native_runtime
from spec import fingerprint,confined


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('project-root','packet','approval','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();verify_allocation();root=Path(a.project_root).resolve();path=confined(root,a.packet);packet=json.loads(path.read_text());approval=json.loads(confined(root,a.approval).read_text());out=confined(root,a.output)
    if (packet.get('schema')!='native-SeHGNN200-patience50-source-recipe-reference-v1' or packet.get('GO') is not False
            or approval.get('approved_sehgnn_reference_packet_hash')!=fingerprint(packet) or approval.get('approval_source')!='explicit_user_go_ahead' or not approval.get('user_message_reference')
            or not approval.get('actual_input_resource_admission_passed') or a.device!='cuda:0' or out.exists()):raise ValueError('Fresh confined, actual-input admitted, explicitly approved exact reference packet required')
    for row in packet['runtime_files']+list(packet['raw_files'].values()):bound(root,row)
    if packet['seeds']!=[84301,84303,84307] or packet['fit_recipe']['maximum_complete_native_epochs']!=200 or packet['geometry']['parameters']!=83659532:raise ValueError('Exact ordered full-native source reference required')
    out.mkdir();rt,engine=native_runtime(root,packet['source'],a.device);t,np=rt['torch'],rt['numpy'];loader=load(root,packet['source']['modules']['role_loader'],'reference_role_loader');schema=json.loads(bound(root,packet['source']['metadata']['schema']).read_text());results=[]
    try:
        data=loader.load(confined(root,packet['raw_input_dir']),bound(root,packet['runs'][0]['role']),schema)
        # The full graph-only cache was newly paid and source-function qualified.
        # Build the exact original six row-normalized typed sparse relations;
        # native prepare_seed regenerates its TRAIN-only label channels.
        symbols=('M','D','A','K');adjs={}
        for row in data.relation_rows:
            h,k=row['raw_head_type'],row['raw_tail_type'];pairs=data.canonical_relation_pairs[row['raw_relation_id']];rows=t.LongTensor([i-data.offsets[h] for i,j in pairs]);cols=t.LongTensor([j-data.offsets[k] for i,j in pairs]);adj=rt['SparseTensor'](row=rows,col=cols,sparse_sizes=(data.counts[h],data.counts[k]));adj.storage._value=None;adj.storage._value=t.ones(adj.nnz())/adj.sum(dim=-1)[adj.storage.row()];adjs[symbols[h]+symbols[k]]=adj
        full=t.load(bound(root,packet['new_paid_graph_only_full_feature_cache']),map_location='cpu',weights_only=True);static=engine.StaticContext(data,full,adjs,{k:v.shape[-1] for k,v in full.items()},{k:list(v.shape) for k,v in full.items()});costs=engine.Costs(out,t,rt['device']);scalar=t.cuda.amp.GradScaler();identity={'packet_hash':fingerprint(packet),'runtime_and_source_files_exact':True,'current_independent_GO':approval['user_message_reference'],'historical_source_admission_inherited':False,'reference_not_published_score_replication':True}
        for row in packet['runs']:
            role=json.loads(bound(root,row['role']).read_text());results.append(engine.run_one(rt,static,role,row['seed'],out/('seed_'+str(row['seed'])),costs,scalar,identity,qualify=False))
        (out/'RESULT.json').write_text(json.dumps({'status':'complete','packet_hash':fingerprint(packet),'results':results,'TEST_access':False,'source_recipe_not_published_score_replication':True},indent=2,sort_keys=True)+'\n')
    except BaseException as e:
        (out/'FAILURE.json').write_text(json.dumps({'status':'failed','error':str(e),'traceback':traceback.format_exc(),'completed':results,'packet_hash':fingerprint(packet)},indent=2)+'\n');raise


if __name__=='__main__':main()
