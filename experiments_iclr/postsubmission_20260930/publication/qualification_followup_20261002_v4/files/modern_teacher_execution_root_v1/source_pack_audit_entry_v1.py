"""Independently verify provider rows and compact train/validation label packs.

The public raw label vector is decoded solely to re-extract the already frozen
source indices. No other label statistic, fit, score or final-pack read occurs.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from qualification_entry_v5 import PHASE,REPO,LOGIN,UUID,sha,confined,bound,write
sys.dont_write_bytecode=True

def descriptor(path):return dict(path=str(path),sha256=sha(path))
def require(value,message):
    if not value:raise ValueError(message)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True);parser.add_argument('--supervisor',required=True)
    args=parser.parse_args()
    require(Path.cwd().resolve()==REPO and os.environ.get('GNNM_PHASE_ROOT')==str(PHASE) and os.environ.get('GNNM_SSH_DESTINATION')==LOGIN,'Wrong repository/environment/route')
    import subprocess
    require(subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==[UUID],'Wrong allocation')
    request_path=confined(args.request);request=json.loads(request_path.read_text())
    require(request['root_admitted'] and not request['full_fit_admitted'] and not request['heldout_scoring_admitted'],'No training/scoring admitted')
    for item in request['protected_files']:bound(item)
    out=confined(request['output']);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    write(out/'START.json',dict(request_sha256=sha(request_path),training_steps=0,final_pack_labels_read=False))
    try:
        import numpy as np
        import scipy.sparse as sp
        pairs=[];checks=[]
        for case in request['graphs']:
            graph_path=bound(case['graph_input']);graph=json.loads(graph_path.read_text())
            pack_path=bound(case['label_pack_manifest']);packs=json.loads(pack_path.read_text())
            require(packs['graph']==case['graph_input'] and packs['raw_labels_dereferenced_by_provider'] and not packs['fits_or_scores_performed'],'Provider pack provenance differs')
            raw_path=bound(graph['release_provenance']['raw'])
            actual_x=np.load(bound(graph['features']),mmap_mode='r',allow_pickle=False)
            actual_edges=np.load(bound(graph['edges']),mmap_mode='r',allow_pickle=False)
            require(actual_x.shape==(graph['num_nodes'],graph['num_features']) and actual_edges.shape==(2,graph['num_edges']),'Dimensions differ')
            source_roles=[]
            for cell in packs['cells']:
                role_path=bound(cell['role_freeze']);roles=json.loads(role_path.read_text())
                require(roles['graph_input']==case['graph_input'] and roles['seed']==cell['seed'] and roles['source_split_index']==cell['source_split_index'] and roles['labels_read'] is False and cell['roles_frozen_before_label_extraction'],'Roles not frozen before provider labels')
                node_sets={}
                for role in ('train','validation'):
                    path=role_path.parent/(role+'_nodes.npy')
                    rel=path.relative_to(role_path.parent).as_posix()
                    require(any(r['path']==rel and r['sha256']==sha(path) for r in roles['payload']),'Role nodes not bound')
                    nodes=np.load(path,allow_pickle=False)
                    require(nodes.dtype==np.int64 and nodes.ndim==1 and len(nodes)>0 and len(np.unique(nodes))==len(nodes) and bool(((nodes>=0)&(nodes<graph['num_nodes'])).all()),'Source node identities invalid')
                    node_sets[role]=nodes
                require(not np.intersect1d(node_sets['train'],node_sets['validation']).size,'Source roles overlap')
                source_roles.append((cell,node_sets))
            with np.load(raw_path,allow_pickle=False) as raw:
                if graph['graph']=='Squirrel':
                    expected_x=np.asarray(raw['node_features'],dtype=np.float32)
                    edges=np.asarray(raw['edges'],dtype=np.int64)
                    if edges.shape[1]==2:edges=edges.T
                    require(edges.shape[0]==2,'Squirrel provider edge shape differs')
                    label_key='node_labels'
                else:
                    attributes=sp.csr_matrix((raw['attr_data'],raw['attr_indices'],raw['attr_indptr']),shape=tuple(raw['attr_shape']))
                    expected_x=attributes.toarray().astype(np.float32)
                    expected_x[expected_x>0]=1
                    adjacency=sp.csr_matrix((raw['adj_data'],raw['adj_indices'],raw['adj_indptr']),shape=tuple(raw['adj_shape'])).tocoo()
                    edges=np.stack((adjacency.row,adjacency.col)).astype(np.int64)
                    label_key='labels'
                require(np.array_equal(expected_x,actual_x),'Provider feature rows differ')
                nonself=edges[:,edges[0]!=edges[1]]
                unique=np.unique(np.sort(nonself,axis=0).T,axis=0).T
                require(np.array_equal(unique,actual_edges),'Provider edge rows differ')
                # Decompression decodes the public raw vector, but numerical
                # access below is restricted to predeclared source-node arrays.
                raw_labels=raw[label_key]
                require(raw_labels.shape==(graph['num_nodes'],),'Raw label shape differs')
                graph_checks=[]
                for cell,node_sets in source_roles:
                    role_checks={}
                    for role,nodes in node_sets.items():
                        record=cell['source_labels'][role]
                        with np.load(bound(record),allow_pickle=False) as pack:
                            require(set(pack.files)=={'nodes','labels'} and np.array_equal(pack['nodes'],nodes),'Source label node alignment differs')
                            labels=pack['labels'];expected=np.asarray(raw_labels[nodes],dtype=np.int64)
                            require(labels.dtype==np.int64 and labels.shape==nodes.shape and np.array_equal(labels,expected),'Source re-extraction differs')
                            require(bool(((labels>=0)&(labels<graph['num_classes'])).all()),'Source label range differs')
                        role_checks[role]=dict(nodes=len(nodes),nodes_sha256=sha(bound(cell['role_freeze']).parent/(role+'_nodes.npy')),pack=record,row_alignment_verified=True,independent_raw_source_reextraction_exact=True)
                    graph_checks.append(dict(seed=cell['seed'],role_freeze=cell['role_freeze'],roles=role_checks))
                del raw_labels
            row_path=out/(graph['graph']+'_PROVIDER_ROW_IDENTITY.json')
            extraction_path=out/(graph['graph']+'_SOURCE_EXTRACTION_AUDIT.json')
            write(row_path,dict(schema='modern-provider-row-identity-audit-v1',graph_input=case['graph_input'],raw_provider=graph['release_provenance']['raw'],full_feature_and_canonical_edge_rows_exact=True,node_order='provider row index; no row subsampling or permutation',dimensions={k:graph[k] for k in ('num_nodes','num_features','num_classes','num_edges')},labels_used_to_establish_rows=False))
            write(extraction_path,dict(schema='modern-independent-source-extraction-audit-v1',auditor_source=descriptor(Path(__file__)),provider_raw=graph['release_provenance']['raw'],label_pack_manifest=case['label_pack_manifest'],checks=graph_checks,source_roles=['train','validation'],public_raw_label_vector_decoded_for_source_reextraction=True,non_source_label_values_used=False,final_label_pack_files_opened=False,fits_or_scores_performed=False))
            for cell,_ in source_roles:
                pairs.append(dict(backbone=case['backbone'],seed=cell['seed'],role_freeze=cell['role_freeze'],graph_input=case['graph_input'],source_labels={k:cell['source_labels'][k] for k in ('train','validation')},independently_extracted_and_row_verified=True,provider_row_identity=descriptor(row_path),extraction_provenance=descriptor(extraction_path)))
            checks.append(dict(graph=graph['graph'],cases=len(graph_checks),provider_row_identity=descriptor(row_path),extraction_provenance=descriptor(extraction_path)))
        require(len(pairs)==6,'Expected six source pairs')
        write(out/'SOURCE_LABEL_BINDING.json',dict(schema='modern-source-label-bindings-v2',pairs=pairs,independent_audit=True,raw_label_access_scope='Decode published raw label vectors solely to re-extract frozen train/validation node indices. No other values used.',final_pool_labels_opened=False,fits_or_scores_performed=False))
        write(out/'TERMINAL.json',dict(complete=True,request_sha256=sha(request_path),seconds=time.monotonic()-started,source_binding=descriptor(out/'SOURCE_LABEL_BINDING.json'),checks=checks,training_steps=0,final_label_pack_files_opened=False,scientific_metrics_calculated=False))
    except Exception as error:
        write(out/'FAILED_ATTEMPT.json',dict(error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started,training_steps=0,automatic_retry=False))
        raise

if __name__=='__main__':main()
