"""Prepared full40 mixed-policy study; root source/freeze/release required."""
import argparse
import copy
import inspect
import json
import math
from pathlib import Path
import time
import traceback

import block_policy
import common as c

COMPATIBILITY = ('source', 'architecture', 'initial_state', 'graph_bytes', 'split_bytes',
                 'runtime', 'recipe', 'member_streams', 'loss_reduction', 'selector', 'checkpoint_replay')
FAMILY_ARMS = ('native_HGT', 'global_BE', 'shared_relation', 'CP', 'unrestricted', 'untied_HGT', 'wider_BE')


def reuse_guard(frozen, sources):
    """All ten eligible controls or none; complete-family custody is root-owned."""
    if frozen['baseline_reuse']['mode'] == 'fresh_all40':
        c.require(frozen['baseline_reuse'].get('audit') is None, 'Fresh full40 must not reuse baseline artifacts')
        return {}
    audit = json.loads(c.verify(frozen['baseline_reuse']['audit']).read_text())
    c.require(audit['schema'] == 'HGB_mixed_all10_baseline_reuse_audit_v1'
              and audit['root_observed'] is True and audit['successful_subset_reuse'] is False
              and audit['favorable_outcome_filtering'] is False,
              'Root-observed outcome-independent all-ten reuse audit required')
    c.require([r['dataset'] for r in audit['families']] == list(c.DATASETS), 'Both source families must close and be audited')
    family_rows = {}
    for family, authority in zip(audit['families'], frozen['source_freezes']):
        dataset = family['dataset']
        c.require(family['source_freeze_sha256'] == authority['record']['sha256'], 'Reuse family source freeze differs')
        receipt = json.loads(c.verify(family['complete_family_receipt']).read_text())
        rows = receipt['rows']
        c.require([(r['seed'], r['arm']) for r in rows] == [(s, a) for s in c.SEEDS for a in FAMILY_ARMS]
                  and all(r['status'] in ('selected', 'failed', 'resource_deferred', 'blocked') for r in rows),
                  'Every original family terminal must be present; no successful-subset closure')
        root_audit = json.loads(c.verify(family['closed_family_audit']).read_text())
        c.require(root_audit['root_observed'] is True and root_audit['all35_closed'] is True
                  and root_audit['all35_audited'] is True and root_audit['artifacts_preserved'] is True
                  and root_audit['source_freeze_sha256'] == authority['record']['sha256'],
                  'Complete original family closure/audit required before reuse')
        family_rows[dataset] = rows
    expected = [(d, s) for d in c.DATASETS for s in c.SEEDS]
    c.require([(r['dataset'], r['seed']) for r in audit['baselines']] == expected, 'Exactly all ten baseline blocks required')
    baselines = {}
    for row in audit['baselines']:
        key = row['dataset'], row['seed']
        c.require(set(row['compatibility']) == set(COMPATIBILITY)
                  and all(row['compatibility'][k] is True for k in COMPATIBILITY),
                  'Each baseline needs exact source/init/runtime/recipe/reduction/selector compatibility')
        selected = json.loads(c.verify(row['selection_receipt']).read_text())
        original = next(r for r in family_rows[key[0]] if r['seed'] == key[1] and r['arm'] == 'global_BE')
        c.require(selected['status'] == 'selected' and selected['arm'] == 'global_BE'
                  and selected['seed'] == key[1] and selected['selected_state_replay'] is True
                  and all(original.get(k) == v for k, v in selected.items()),
                  'Byte-bound baseline must match the complete family terminal')
        for name in ('selected_checkpoint', 'selected_logits', 'training_trace', 'runtime_receipt'):
            c.verify(row[name])
        runtime = json.loads(c.verify(row['runtime_receipt']).read_text())
        c.require(runtime['torch'] == '2.1.2+cu118' and runtime['device'] == 'cpu'
                  and runtime['CPU_threads'] == 1 and runtime['interop_threads'] == 1
                  and runtime['python_version'] == [3, 11, 14], 'Baseline native CPU runtime differs')
        baselines[key] = dict(row, original_selection=selected)
    return baselines


def optimizer_tensor_cost(torch, optimizer_state):
    """Logical unique-tensor payload of the immutable selected optimizer state."""
    seen, tensors = set(), []
    def visit(value):
        if isinstance(value, torch.Tensor):
            if id(value) not in seen:
                seen.add(id(value)); tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
    visit(optimizer_state)
    return dict(schema='selected_optimizer_tensor_payload_cost_v1',
        optimizer_tensor_payload_bytes=sum(t.numel()*t.element_size() for t in tensors),
        optimizer_tensor_count=len(tensors), source='immutable selected.pt optimizer field',
        accounting='logical payload bytes of unique tensor objects; Python/scalar metadata excluded',
        checkpoint_file_bytes_included=False, physical_storage_or_peak_memory_claim=False,
        model_parameters_or_gradients_included=False)


def fixed_final_endpoint(trace_record, updates):
    """Use the last immutable TRACE row, with no new inference or selection."""
    lines = c.verify(trace_record).read_text().splitlines()
    c.require(lines, 'Complete selected fit requires a final trace endpoint')
    last = json.loads(lines[-1])
    c.require(last['epoch'] == updates and all(math.isfinite(last[k]) for k in
        ('validation_NLL', 'validation_micro_F1', 'validation_macro_F1')), 'Final trace endpoint differs from completed fit')
    return dict(schema='fixed_final_trace_endpoint_v1', epoch=last['epoch'],
        **{k: last[k] for k in ('validation_NLL', 'validation_micro_F1', 'validation_macro_F1')},
        source='last row of byte-bound immutable TRACE.jsonl', new_prediction_or_selection=False,
        selected_checkpoint_changed=False, descriptive_only=True)


def selected_descriptives(torch, driver, implementation, logits, split, validation_labels, classes):
    """Read immutable selected logits; no new validation choice or checkpoint."""
    c.require(logits.device.type == 'cpu' and logits.dtype == torch.float32
              and logits.ndim == 3 and logits.shape[0] == 4, 'Saved native FP32 member logits required')
    ids = torch.tensor(split['validation_ids'], dtype=torch.long)
    labels = torch.tensor(validation_labels, dtype=torch.long)
    with torch.no_grad():
        own = implementation.mean_member_ce(logits, ids, labels)
        pool = torch.nn.functional.cross_entropy(logits.mean(0)[ids], labels)
        served = driver.metrics(torch, logits, ids, labels, classes)
        members = [driver.metrics(torch, logits[m:m+1], ids, labels, classes) for m in range(4)]
        wide_logits = logits.to(torch.float64)
        wide_own = implementation.mean_member_ce(wide_logits, ids, labels)
        wide_pool = torch.nn.functional.cross_entropy(wide_logits.mean(0)[ids], labels)
    return dict(schema='mixed_policy_fixed_selected_logit_descriptives_v1', served=served, members=members,
        mean_member_validation_CE=float(own), pooled_validation_CE=float(pool),
        own_minus_pool_D=float(own-pool), selected_member_logits_sha256=c.tensor_sha(logits),
        own_minus_pool_arithmetic=dict(dtype='FP32', aggregation='FP32 member logits.mean(0), then CE',
            served_predictor_arithmetic=True, unclamped=True,
            exact_nonnegativity_claim=False, note='Tiny negative gaps can arise from FP32 arithmetic'),
        FP64_algebraic_Jensen_reference=dict(mean_member_CE=float(wide_own), pooled_CE=float(wide_pool),
            own_minus_pool_gap=float(wide_own-wide_pool), dtype='FP64',
            aggregation='convert every saved member logit to FP64 before mean(0), then CE',
            served_predictor_arithmetic=False, selection_or_inference_changed=False, unclamped=True),
        raw_FP32_mean_logit_inference=True, diagnostic_only=True, member_checkpoint_selection=False,
        new_validation_selection=False, heldout=False, calibrated=False)


def complete_summary(rows, driver):
    expected = [(d, s, p) for d in c.DATASETS for s in c.SEEDS for p in c.POLICIES]
    c.require([(r['dataset'], r['seed'], r['policy']) for r in rows] == expected, 'Exactly40 ordered terminals required')
    eligible = all(r['status'] == 'selected' and r.get('selected_state_replay') is True
                   and r.get('checkpoint_bindings_verified') is True for r in rows)
    if not eligible:
        return dict(status='incomplete', all40_terminals=True, successful_subset_scored=False, comparison=None)
    values, final_values = {}, {}
    for dataset in c.DATASETS:
        values[dataset], final_values[dataset] = {}, {}
        for policy in c.POLICIES:
            selected = [next(r for r in rows if (r['dataset'], r['seed'], r['policy']) == (dataset, seed, policy))
                        for seed in c.SEEDS]
            values[dataset][policy] = dict(validation_NLL=[r['selection']['validation_NLL'] for r in selected],
                validation_macro_F1=[r['selection']['validation_macro_F1'] for r in selected])
            final_values[dataset][policy] = dict(validation_NLL=[r['fixed_final_endpoint']['validation_NLL'] for r in selected],
                validation_macro_F1=[r['fixed_final_endpoint']['validation_macro_F1'] for r in selected])
    deltas = {dataset: {control: [x-y for x, y in zip(values[dataset]['pool/own']['validation_NLL'],
               values[dataset][control]['validation_NLL'])] for control in ('own/own', 'pool/pool', 'own/pool')}
               for dataset in c.DATASETS}
    paired = {}
    for endpoint, collection in (('selected', values), ('fixed_final_trace', final_values)):
        paired[endpoint] = {}
        for dataset in c.DATASETS:
            paired[endpoint][dataset] = {}
            for control in ('own/own', 'pool/pool', 'own/pool'):
                paired[endpoint][dataset][control] = {}
                for metric in ('validation_NLL', 'validation_macro_F1'):
                    vector = [x-y for x, y in zip(collection[dataset]['pool/own'][metric], collection[dataset][control][metric])]
                    paired[endpoint][dataset][control][metric] = dict(
                        pool_own_minus_control=vector, **driver.paired_uncertainty(vector),
                        leave_one_block_out=[dict(omitted_seed=seed, remaining_mean=sum(vector[:i]+vector[i+1:])/4)
                                             for i, seed in enumerate(c.SEEDS)],
                        descriptive_stability_only=True, acceptance_threshold_added=False)
    return dict(status='complete_development_summary', all40_terminals=True, successful_subset_scored=False,
        paired_seed_order=list(c.SEEDS), values=values, pool_own_minus_controls_validation_NLL=deltas,
        fixed_final_trace_values=final_values, paired_uncertainty_and_leave_one_block_out=paired,
        practical_utility_decision='Root applies the bound prospective scientific design; no gate duplicated here',
        comparison_scope='complete paired development study; no heldout, significance or global novelty claim')


def reuse_case(torch, driver, implementation, model, baseline, frozen, split_record, schema):
    c.require(c.model_sha(model) == baseline['initial_model_sha256'], 'Reused baseline initial state differs')
    saved = torch.load(c.verify(baseline['selected_checkpoint']), map_location='cpu', weights_only=True)
    expected_names = [name for name, _ in model.named_parameters()]
    c.require(saved['optimizer_parameter_names'] == expected_names and saved['seed'] == baseline['seed']
              and saved['arm'] == 'global_BE', 'Reused checkpoint parameter/seed/arm identity differs')
    c.require(saved['bindings'] == baseline['original_checkpoint_bindings'], 'Root-audited checkpoint binding differs')
    bound = saved['bindings']
    c.require(bound['archive'] == frozen['archive'] and bound['development_labels'] == frozen['development_labels']
              and bound['split'] == split_record and bound['graph_schema'] == schema
              and bound['implementation_manifest_sha256'] == frozen['implementation_manifest_sha256'],
              'Reused checkpoint graph/split/implementation binding differs')
    c.require(saved['selection'] == baseline['original_selection']['selection'], 'Reused selected-state receipt differs')
    model.load_state_dict(saved['model'])
    # Previous complete-state replay is independently audited and byte-bound.
    # This source adds only descriptive computations on the saved selected logits.
    return dict(baseline['original_selection'], reused=True, checkpoint_bindings_verified=True,
                optimizer_state_cost=optimizer_tensor_cost(torch, saved['optimizer']),
                selected_checkpoint=baseline['selected_checkpoint'], selected_logits=baseline['selected_logits'],
                selection_receipt=baseline['selection_receipt'], training_trace=baseline['training_trace'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', required=True); parser.add_argument('--admission', required=True)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args(argv)
    c.require(Path(args.run_name).name == args.run_name and args.run_name not in ('', '.', '..'), 'Fresh simple run name required')
    provenance, frozen, sources, release, admission, original_bytes = c.admission(args.freeze, args.admission, 'training')
    c.require(release['run_name'] == args.run_name, 'Root run-name binding differs')
    baselines = reuse_guard(frozen, sources)  # Before any new fit: all ten or none.
    c.require(not list((c.PACKET/'runs').glob('*/STUDY_STARTED.json')), 'No automatic restart or replacement study')
    out = c.PACKET/'runs'/args.run_name
    out.mkdir(parents=True, exist_ok=False)
    c.write(out/'STUDY_STARTED.json', admission)
    rows, errors, started, driver = [], [], time.perf_counter(), None
    originals_preserved = False
    try:
        torch, runtime = c.runtime()
        c.write(out/'RUNTIME.json', dict(runtime, python_version=[3, 11, 14]))
        mods = c.modules(provenance)
        implementation, driver, onecycle = mods['implementation'], mods['driver'], mods['onecycle']
        _, custody = block_policy.fit_tree(inspect.getsource(driver.fit))
        c.write(out/'FIT_BODY_CUSTODY.json', custody)
        for dataset in c.DATASETS:
            graph, features, schema, development = None, None, None, None
            try:
                graph_freeze = sources[dataset]
                inputs, graph, features, schema, development = c.graph_inputs(mods, dataset, graph_freeze, implementation, torch)
                graph_out = out/dataset
                graph_out.mkdir(); c.write(graph_out/'GRAPH_SCHEMA.json', schema)
                classes = len(development['train_class_schema'])
                for split_record in graph_freeze['splits']:
                    seed = split_record['seed']
                    split, train_labels, validation_labels = inputs.verify_split(split_record['descriptor'], development, seed)
                    base = c.build(mods, dataset, graph, schema, classes, seed)
                    initial_sha = c.model_sha(base)
                    for policy in c.POLICIES:
                        case = graph_out/f'seed{seed}'/policy.replace('/', '__')
                        case.parent.mkdir(exist_ok=True)
                        model, case_started = copy.deepcopy(base), time.perf_counter()
                        try:
                            blocks = block_policy.partition(implementation, model)
                            bindings = dict(admission, dataset=dataset, model='global_BE', policy=policy,
                                archive=graph_freeze['archive'], development_labels=graph_freeze['development_labels'],
                                implementation_manifest_sha256=graph_freeze['implementation_manifest_sha256'],
                                graph_schema=schema, split=split_record, initial_model_sha256=initial_sha,
                                module_semantic_partition=blocks['receipt'], fit_body_custody=custody)
                            if policy == 'own/own' and baselines:
                                case.mkdir()
                                row = reuse_case(torch, driver, implementation, model, baselines[(dataset, seed)], graph_freeze, split_record, schema)
                                c.write(case/'REUSED_BASELINE.json', baselines[(dataset, seed)])
                            else:
                                fit = block_policy.fit_function(torch, implementation, driver, model, policy, blocks)
                                row = fit(torch, implementation, model, graph, features, split, train_labels, validation_labels,
                                    seed, policy, case, bindings, onecycle)
                                saved = torch.load(case/'selected.pt', map_location='cpu', weights_only=True)
                                c.require(saved['bindings'] == bindings and saved['optimizer_parameter_names'] == [n for n, _ in model.named_parameters()],
                                          'Selected checkpoint binding/name custody differs')
                                row.update(reused=False, checkpoint_bindings_verified=True,
                                    optimizer_state_cost=optimizer_tensor_cost(torch, saved['optimizer']),
                                    **{key: c.record(case/name) for key, name in
                                       (('selected_checkpoint', 'selected.pt'), ('selected_logits', 'selected_member_logits.pt'),
                                        ('selection_receipt', 'SELECTION.json'), ('training_trace', 'TRACE.jsonl'))})
                                del saved
                            logits = torch.load(c.verify(row['selected_logits']), map_location='cpu', weights_only=True)
                            descriptive = selected_descriptives(torch, driver, implementation, logits, split, validation_labels, classes)
                            c.require(abs(descriptive['served']['validation_NLL']-row['selection']['validation_NLL']) <= 1e-7,
                                      'Saved-logit served NLL differs from selected receipt')
                            c.write(case/'SELECTED_DESCRIPTIVES.json', descriptive)
                            endpoint = fixed_final_endpoint(row['training_trace'], row['updates'])
                            c.write(case/'FIXED_FINAL_AND_OPTIMIZER_DIAGNOSTICS.json',
                                    dict(fixed_final_endpoint=endpoint, optimizer_state_cost=row['optimizer_state_cost']))
                            row.update(selected_descriptives=c.record(case/'SELECTED_DESCRIPTIVES.json'),
                                fixed_final_endpoint=endpoint,
                                fixed_final_and_optimizer_diagnostics=c.record(case/'FIXED_FINAL_AND_OPTIMIZER_DIAGNOSTICS.json'),
                                initial_model_sha256=initial_sha)
                            del logits
                        except Exception as error:
                            case.mkdir(exist_ok=True)
                            resource_failure = isinstance(error, MemoryError) or 'out of memory' in str(error).lower() or "can't allocate memory" in str(error).lower()
                            row = dict(status='resource_deferred' if resource_failure else 'failed', error_type=type(error).__name__,
                                error_message=str(error), traceback=traceback.format_exc(), successful_subset_scored=False)
                            c.write(case/'POLICY_FAILURE.json', row)
                        finally:
                            del model
                        row.update(dataset=dataset, seed=seed, policy=policy, attempted=True,
                                   final_labels_closed=True, wall_seconds=time.perf_counter()-case_started, memory=c.memory())
                        rows.append(row)
                        c.write(case/'TERMINAL.json', row)
                    del base
            except Exception as error:
                errors.append(dict(dataset=dataset, error_type=type(error).__name__, error_message=str(error), traceback=traceback.format_exc()))
            finally:
                del graph, features, schema, development
    except Exception as error:
        errors.append(dict(error_type=type(error).__name__, error_message=str(error), traceback=traceback.format_exc()))
    finally:
        seen = {(r['dataset'], r['seed'], r['policy']): r for r in rows}
        rows = [seen.get((d, s, p), dict(dataset=d, seed=s, policy=p, status='blocked', attempted=False,
                 reason='study or graph prerequisite failed', final_labels_closed=True, successful_subset_scored=False))
                for d in c.DATASETS for s in c.SEEDS for p in c.POLICIES]
        try:
            c.preserved(provenance, frozen, sources, args.freeze, args.admission, original_bytes)
            if baselines:
                reuse_guard(frozen, sources)
            originals_preserved = True
        except Exception as error:
            errors.append(dict(error_type=type(error).__name__, error_message=str(error)))
        summary = complete_summary(rows, driver) if originals_preserved else None
        c.write(out/'STUDY.json', dict(schema='HGB_mixed_block_full40_training_v1', rows=rows, summary=summary,
            errors=errors, originals_preserved=originals_preserved, wall_seconds=time.perf_counter()-started,
            admission=admission, baseline_reuse_mode=frozen['baseline_reuse']['mode'], final_labels_closed=True,
            CP_gate_dependency=False, inference_changed=False, scientific_quality_claim=False))
    print(json.dumps(dict(output=str(out), summary=summary)))
    return 0 if summary and summary['status'] == 'complete_development_summary' else 1


if __name__ == '__main__':
    raise SystemExit(main())
