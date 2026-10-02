"""Only after all primary correction SCORE_FREEZEs: paired modern teacher APS controls.

No fitting or selection is performed. Uses frozen FP32 primary probability/APS
tables. CPU64 logits recomputation is a declared numeric reference and secondary.
SOURCE ONLY: never imported or executed by the author.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode = True
import correction_screen_driver as core
import modern_custody as custody
from modern_teacher_driver import verify_cell, protocol


def report(args):
    started = time.perf_counter()
    import numpy as np
    study_record = core.descriptor(args.study_freeze)
    study = core.read_json(args.study_freeze)
    request_record = core.descriptor(args.pool_labels)
    request = core.read_json(args.pool_labels)
    core.require(study['schema'] == 'modern-teacher-study-selection-v2' and
        study['teacher_protocol'] == protocol() and study['source_selection_closed'] is True and
        study['complete_family_cells'] == 72, 'Source study selection must close first')
    core.check_implementation(study['implementation_sha256'])
    release = custody.verify_release(request['final_label_release'])
    core.require(request['study_freeze'] == study_record and
        release['teacher_study_selection_freeze'] == study_record and len(request['labels']) == 6,
        'One shared six-cell final-label release must bind the control report')
    expected = {(b,s) for b in protocol()['native'] for s in protocol()['seeds']}
    label_records = {(r['backbone'],r['seed']):r for r in request['labels']}
    core.require(set(label_records) == expected and len(label_records) == 6, 'Six compact pool packs required')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Exclusive receipt prevents post-report changes or repeated control reporting.
    core.write_json(Path(args.study_freeze).parent/'TEACHER_CONTROL_REPORT_STARTED.json',
        dict(study=study_record, pool_labels=request_record, output=str(out)), exclusive=True)
    try:
        rows = []
        for family_record in study['family_selections']:
            selection = core.read_json(core.verified(family_record['selection']))
            for selected in selection['selected_teachers']:
                freeze, cell_out = verify_cell(selected['freeze'])
                spec = selected['specification']
                key = (spec['backbone'],spec['seed'])
                labels_record = label_records[key]
                core.require(labels_record['role_freeze'] == selected['role_freeze'], 'Unpaired control labels')
                role_path = core.verified(selected['role_freeze'])
                roles, role_root = core.read_json(role_path), role_path.parent
                core.verify_tree(role_root, roles['payload'])
                pool = np.load(role_root/'pool_nodes.npy', allow_pickle=False)
                pack = core.np_load(np, labels_record['labels'])
                core.require(set(pack.files) == {'nodes','labels'} and np.array_equal(pack['nodes'],pool),
                    'Compact pool labels must match exact committed identities')
                labels = pack['labels']
                point = np.load(cell_out/'primary_point_probabilities.npy', allow_pickle=False)
                scores = np.load(cell_out/'primary_aps_scores.npy', allow_pickle=False)
                core.require(labels.dtype == np.int64 and labels.shape == pool.shape and
                    ((labels >= 0) & (labels < point.shape[1])).all(), 'Invalid pool labels')
                label_by_node = np.full(len(point),-1,dtype=np.int64)
                label_by_node[pool] = labels
                logits = np.load(core.verified(selected['saved_logits']),allow_pickle=False).astype(np.float64)
                mean_logits = logits.mean(0)
                exp = np.exp(mean_logits-mean_logits.max(1,keepdims=True))
                reference = exp/exp.sum(1,keepdims=True)
                member_exp = np.exp(logits-logits.max(2,keepdims=True))
                secondary = (member_exp/member_exp.sum(2,keepdims=True)).mean(0)
                allocations = []
                for j in range(20):
                    cal = np.load(role_root/f'calibration{j:02d}.npy',allow_pickle=False)
                    test = np.load(role_root/f'test{j:02d}.npy',allow_pickle=False)
                    core.require(np.array_equal(np.sort(np.concatenate((cal,test))),pool) and
                        not len(np.intersect1d(cal,test)), 'Wrong precommitted final allocation')
                    rank = math.ceil((len(cal)+1)*0.9)
                    values = np.sort(scores[cal,label_by_node[cal]],kind='stable')
                    threshold = float(values[rank-1]) if rank <= len(cal) else float('inf')
                    metrics = core.metrics(np,scores,scores,point,test,label_by_node[test],threshold,point)
                    allocations.append(dict(allocation=j, threshold=threshold if math.isfinite(threshold) else '+inf',
                        metrics=metrics, cpu64_primary_reference=dict(
                        accuracy=float((reference[test].argmax(1)==label_by_node[test]).mean()),
                        nll=float(-np.log(np.maximum(reference[test,label_by_node[test]],np.finfo(np.float64).tiny)).mean())),
                        fixed_secondary_mean_probability=dict(selected=False,
                        accuracy=float((secondary[test].argmax(1)==label_by_node[test]).mean()),
                        nll=float(-np.log(np.maximum(secondary[test,label_by_node[test]],np.finfo(np.float64).tiny)).mean()))))
                rows.append(dict(specification=spec, selected=selected, allocations=allocations,
                    primary='saved FP32 softmax(mean raw member logits) and fixed randomized APS',
                    secondary='same saved logits; no selection, no alternate correction fit'))
        core.verified(study_record); core.verified(request_record)
        core.check_implementation(study['implementation_sha256'])
        core.write_json(out/'TEACHER_CONTROL_REPORT.json',dict(study=study_record,cells=rows,
            report_scope='18 selected native family/graph/seed cells;20paired final allocations each',
            independent_allocations_are_not_independent_graphs=True,
            final_labels_read=True, no_refit=True, no_significance_claim=True,
            report_wall_seconds_before_serialization=time.perf_counter()-started),exclusive=True)
    except Exception as error:
        core.write_json(out/'FAILED_REPORT.json',dict(error_type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc(),automatic_retry=False),exclusive=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-freeze',required=True)
    parser.add_argument('--pool-labels',required=True)
    parser.add_argument('--output',required=True)
    report(parser.parse_args())
