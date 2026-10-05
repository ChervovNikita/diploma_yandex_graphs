"""Root-authorized numerical diagnostic only: split0 shared fold0, no fits/scores.

Authenticates complete original source/cohort/data via the existing custody
implementation, then isolates the first failing old probability-hull QP.
Never loads features/TRAIN/control/TEST labels or checkpoints. No retries of
scientific study, no metric functions, no head or calibrator construction.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback


def require(value,message):
    if not value:
        raise ValueError(message)


def record(path):
    p=Path(path)
    b=p.read_bytes()
    return {'path':str(p.resolve()),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def clean(value):
    if isinstance(value,dict):return {key:clean(v) for key,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    return value


def bind_source(source,manifest_hash,seal_hash):
    manifest=record(source/'MANIFEST.json');seal=record(source/'SEAL.json')
    require(manifest['sha256']==manifest_hash and seal['sha256']==seal_hash,'Exact source hashes required')
    m=json.loads((source/'MANIFEST.json').read_text())
    s=json.loads((source/'SEAL.json').read_text())
    require(s['manifest']['sha256']==manifest_hash and s['manifest']['bytes']==manifest['bytes'],
            'Source seal differs')
    for row in m['payload']:
        path=source/row['path']
        require('..' not in Path(row['path']).parts and not path.is_symlink(),'Source path differs')
        actual=record(path)
        require(actual['sha256']==row['sha256'] and actual['bytes']==row['bytes'],'Source file differs')
    return {'manifest':manifest,'seal':seal}


def old_face_witness(n,p,posterior,a,b):
    scale=max(float(n.np.abs(a).max()),float(n.np.abs(b).max()),1e-12)
    an,bn=a/scale,b/scale
    faces=[]
    for k in range(1,5):
        for free_tuple in itertools.combinations(range(4),k):
            free=n.np.array(free_tuple)
            fixed=n.np.array([j for j in range(4) if j not in free],dtype=int)
            rhs=bn[free].copy()
            if len(fixed):rhs-=n.LOWER*an[n.np.ix_(free,fixed)].sum(-1)
            matrix=n.np.zeros((k+1,k+1));matrix[:k,:k]=an[n.np.ix_(free,free)]
            matrix[:k,k]=matrix[k,:k]=1.
            target=n.np.zeros(k+1);target[:k]=rhs;target[k]=1-len(fixed)*n.LOWER
            solution=n.np.linalg.pinv(matrix,rcond=1e-12)@target
            w=n.np.full(4,n.LOWER);w[free]=solution[:k]
            g=an@w-bn;level=-solution[k]
            sum_error=abs(float(w.sum()-1))
            floor_error=max(0.,float(n.LOWER-w.min()))
            free_error=float(n.np.abs(g[free]-level).max())
            dual_error=(max(0.,float((level-g[fixed]).max())) if len(fixed) else 0.)
            finite=bool(n.np.isfinite(w).all())
            feasible=finite and floor_error<=1e-12 and sum_error<1e-12 and free_error<1e-10 and dual_error<=1e-10
            singular=n.np.linalg.svd(matrix,compute_uv=False)
            faces.append({'free_indices':list(free_tuple),'scaled_KKT_matrix':matrix.tolist(),
                          'scaled_KKT_target':target.tolist(),'KKT_singular_values':singular.tolist(),
                          'pinv_rank_at_rcond_1e12':int(n.np.sum(singular>1e-12*singular[0])),
                          'raw_weights':w.tolist(),'sum_error':sum_error,'floor_error':floor_error,
                          'free_stationarity_error':free_error,'inactive_dual_error':dual_error,
                          'old_face_feasible':feasible})
    return clean({'P':p.tolist(),'posterior':posterior.tolist(),'A':a.tolist(),'b':b.tolist(),
                  'existing_QP_scale':scale,'scaled_A':an.tolist(),'scaled_b':bn.tolist(),
                  'member_probability_singular_values':n.np.linalg.svd(p,compute_uv=False).tolist(),
                  'old_face_feasible_count':sum(f['old_face_feasible'] for f in faces),'faces':faces})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--seal-sha256',required=True)
    parser.add_argument('--phase',required=True)
    parser.add_argument('--closure-freeze',required=True)
    parser.add_argument('--closure-sha256',required=True)
    parser.add_argument('--protocol',default='amazon_polynormer_logits_graph_moment_retrospective_protocol_20261005_v2/PROTOCOL.json')
    parser.add_argument('--protocol-sha256',default='5d667f15640102995fa998ca6932c53c5b71554273c84d4938a331c0a4eae7ba')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    source,phase=Path(args.source).resolve(),Path(args.phase).resolve()
    output=Path(args.output).resolve()
    require(output.is_relative_to(phase),'Witness output must stay inside the root repository phase')
    output.mkdir(exist_ok=False)
    started=time.perf_counter()
    report={'status':'started','purpose':'numerical QP witness only','fits':0,'optimizer_updates':0,
            'metric_calls':0,'scored_fold_outcomes':0,'scientific_retry':False}
    try:
        bound=bind_source(source,args.manifest_sha256,args.seal_sha256)
        sys.dont_write_bytecode=True
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[name]='1'
        os.environ['CUDA_VISIBLE_DEVICES']=''
        require('custody' not in sys.modules and 'numerical' not in sys.modules,'Fresh diagnostic process required')
        sys.path.insert(0,str(source))
        import custody as c
        closure=c.record(phase,args.closure_freeze);protocol=c.record(phase,args.protocol)
        require(closure['sha256']==args.closure_sha256 and protocol['sha256']==args.protocol_sha256,
                'Root-bound closure/protocol differs')
        state=c.prepare(phase,closure,protocol)  # Full15 authentication before any numerical decode.
        import numerical as n
        import scipy
        n.setup()
        require(not n.torch.cuda.is_initialized(),'CPU-only diagnostic required')
        edge,val_mask=c.load_visible(state)
        ids,labels=c.load_validation(state,val_mask,0)
        family=next(f for f in state['registry']['families'] if f['split']==0 and f['family']=='gnnm_boundary_4')
        z,provenance=c.load_bank(state,family)
        a=n.arrays(z);t,degree,graph=n.graph(edge,a['native_class'])
        scored=n.folds(ids,0)[0];anchors=n.np.setdiff1d(ids,scored)
        anchor_y=labels[n.np.searchsorted(ids,anchors)]
        require(len(scored)==2041 and len(anchors)==4082,'Exact fold population differs')
        ctx=n.context(t,a['p'],n.pairwise(a['p']),anchors,anchor_y,scored)
        write(output/'BOUND_CONTEXT.json',{'source':bound,'closure':closure,'protocol':protocol,
                                          'original_inputs':state['payloads'],'bank':family,'provenance':provenance,
                                          'split':0,'outer_fold':0,'graph':graph,
                                          'anchor_ids_sha256':c.sha_object(anchors.tolist()),
                                          'scored_fold_ids_sha256':c.sha_object(scored.tolist()),
                                          'all_scored_labels_excluded_from_context':True,
                                          'environment':{'numpy':n.np.__version__,'torch':str(n.torch.__version__),
                                                         'scipy':scipy.__version__,'CPU_threads':1}})
        matrices=n.np.einsum('nmc,nkc->nmk',a['p'],a['p'])
        linear=n.np.einsum('nmc,nc->nm',a['p'],ctx['posterior'])
        bad_chunk=None
        witness=None
        for offset in range(0,n.N,2048):
            try:n.simplex_qp(matrices[offset:offset+2048],linear[offset:offset+2048],singular=True)
            except ValueError as error:
                bad_chunk=(offset,min(offset+2048,n.N),str(error))
                tb=error.__traceback__
                failed_locals=None
                while tb is not None:
                    frame=tb.tb_frame
                    if frame.f_code.co_name=='simplex_qp' and Path(frame.f_code.co_filename).resolve()==source/'numerical.py':
                        failed_locals=frame.f_locals
                    tb=tb.tb_next
                require(failed_locals is not None and 'chosen' in failed_locals,
                        'Unexpected QP failure before chosen-row trace hook')
                bad_rows=n.np.flatnonzero(~n.np.isfinite(failed_locals['chosen']).all(axis=1))
                require(len(bad_rows)>0,'Failure has no chosen NaN row; preserve without changing solver')
                local=int(bad_rows[0]);internal_offset=int(failed_locals['offset'])
                node=offset+internal_offset+local
                witness=old_face_witness(n,a['p'][node],ctx['posterior'][node],matrices[node],linear[node])
                witness.update({'native_node_id':node,'original_error':str(error),'bad_chunk':list(bad_chunk[:2]),
                                'source_trace_internal_offset':internal_offset,'source_trace_NaN_local_row':local,
                                'source_trace_scaled_A':failed_locals['an'][local].tolist(),
                                'source_trace_scaled_b':failed_locals['bn'][local].tolist(),
                                'source_trace_scale':float(failed_locals['scale'][local]),
                                'whole_chunk_error':bad_chunk[2],
                                'node_in_fusion_anchor_B':bool(node in set(anchors.tolist())),
                                'node_in_scored_fold_D':bool(node in set(scored.tolist())),
                                'native_node_true_label_not_exported':True,'source':bound})
                break
        require(witness is not None,'Original failure not reproduced; preserve diagnostic without retry')
        write(output/'WITNESS.json',witness)
        report.update(status='WITNESS_EXTRACTED_NO_FITS_OR_SCORES',witness=record(output/'WITNESS.json'),
                      native_node_id=witness['native_node_id'],old_face_feasible_count=witness['old_face_feasible_count'],
                      source=bound,source_helper_counters=n.COUNTERS)
        bind_source(source,args.manifest_sha256,args.seal_sha256)
        for row in state['payloads']:c.verify(phase,row)
    except Exception as error:
        report.update(status='DIAGNOSTIC_FAILED_NO_RETRY',error=str(error),traceback=traceback.format_exc())
    finally:
        raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['cost']={'wall_seconds':time.perf_counter()-started,
                        'peak_RSS_bytes':int(raw if sys.platform=='darwin' else raw*1024)}
        report['diagnostic_source']=record(Path(__file__))
        write(output/'DIAGNOSTIC_RESULT.json',report)
        print(json.dumps({'status':report['status'],'result':str(output/'DIAGNOSTIC_RESULT.json')}))
    return 0 if report['status']=='WITNESS_EXTRACTED_NO_FITS_OR_SCORES' else 1


if __name__=='__main__':raise SystemExit(main())
