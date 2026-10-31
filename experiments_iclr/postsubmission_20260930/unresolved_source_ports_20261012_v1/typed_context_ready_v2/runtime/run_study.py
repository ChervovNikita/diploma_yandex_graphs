"""Execute one approved model-fit unit. Dataset labels stay in its adapter."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import socket
import sys
import time
import types

from registry import Registry
from spec import confined, load


def owner_identity():
    value = {"pid": os.getpid(), "hostname": socket.gethostname(), "started": time.time()}
    stat = Path("/proc") / str(os.getpid()) / "stat"
    if stat.exists():
        fields = stat.read_text().rsplit(")", 1)[1].split()
        value.update(start_ticks=int(fields[19]), pgid=int(fields[2]))
    return value


def require_approval(path, spec_hash):
    approval = json.loads(Path(path).read_text())
    approved = (approval.get("approved_spec_hash") == spec_hash or
                spec_hash in approval.get("approved_spec_hashes", []))
    if (not approved or approval.get("approval_source") != "explicit_user_go_ahead"
            or not approval.get("user_message_reference")):
        raise ValueError("This run list has not received the user's go-ahead")
    return approval


def require_phase_approval(root, spec, run, approval):
    """Direct child entrypoints cannot bypass the global release protocol."""
    if not spec["release"].get("global_plan_gates_required"):
        return
    from launcher import gate_ready
    from spec import fingerprint
    relative = approval.get("approved_plan_path")
    if not relative:
        raise ValueError("This study requires the approved distributed-plan binding")
    path = confined(root, relative)
    plan = json.loads(path.read_text())
    digest = fingerprint(plan)
    if digest != approval.get("approved_plan_hash"):
        raise ValueError("The distributed assignment plan changed after GO")
    if not any(row["spec_hash"] == fingerprint(spec) for row in plan["studies"]):
        raise ValueError("Study is outside the approved distributed plan")
    from retention_lifecycle import early_export_key,exporter_binding
    key=early_export_key(spec,run)
    exception=key in plan.get("label_free_export_run_keys",[]) if key else False
    if exception and exporter_binding(root,run)!=spec["release"]["checkpoint_retention_policy"]["audited_label_free_exports"][run["run_id"]]:
        raise ValueError("Early export source changed after approval")
    if not exception and not gate_ready(root, plan, digest, run["config"].get("stage", "search")):
        raise ValueError("The whole distributed phase has not closed")


def load_factory(root, entry):
    relative, separator, function = entry.partition(":")
    if not separator or not function.isidentifier():
        raise ValueError("Use project-relative adapter_file.py:factory")
    path = confined(root, relative)
    if not path.stem.isidentifier():
        raise ValueError("Adapter filenames must be Python identifiers")
    # Each frozen packet owns its relative imports. An anonymous module name
    # would make `.classifier` fall back to whichever `adapters` directory
    # happens to be on the launcher's sys.path, possibly another packet.
    package_root = path.parent
    components = [path.stem]
    while (package_root / "__init__.py").is_file():
        if not package_root.name.isidentifier() or package_root == Path(root).resolve():
            break
        components.insert(0, package_root.name)
        package_root = package_root.parent
    namespace = "gnnm_packet_" + hashlib.sha256(str(package_root).encode()).hexdigest()[:16]
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(package_root)]
        package.__package__ = namespace
        sys.modules[namespace] = package
    name = namespace + "." + ".".join(components)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return getattr(module, function)


def terminal_owner(owner):
    """PID reuse is distinguished from the original process by Linux start ticks."""
    if owner.get("hostname") != socket.gethostname() or "start_ticks" not in owner:
        raise ValueError("The previous owner's termination cannot be verified on this host")
    path = Path("/proc") / str(owner["pid"]) / "stat"
    if path.exists():
        fields = path.read_text().rsplit(")", 1)[1].split()
        if int(fields[19]) == owner["start_ticks"] and fields[0] != "Z":
            raise ValueError("The previous exact owner is still alive")
    return {"previous_owner_terminal": True, "previous_owner": owner,
            "verified_on": socket.gethostname(), "verified_at": time.time()}


def resolve_selected_config(spec, run, registry):
    """Bind a frozen TRAIN/SELECT choice, preserving its full search provenance."""
    reference = run["config"].get("selected_config_from")
    if reference is None:
        return run
    row = registry.run(spec["study_id"], reference)
    if row["state"] != "complete" or row["contract"]["kind"] != "selection":
        raise ValueError("Comparative training requires a closed search-selection endpoint")
    result = row["result"]
    choice = result["selected_configuration"]
    # Membership is checked against the approved search domain, rather than
    # trusting a generated file to add a new setting after search.
    domain = row["contract"]["config"]["configurations"]
    matching = [r for r in domain if r["config_id"] == choice["config_id"] and r["parameters"] == choice["parameters"]]
    if len(matching) != 1:
        raise ValueError("The selected configuration is outside the approved domain")
    actual = copy.deepcopy(run)
    actual["config"].update(choice["parameters"])
    actual["config"]["resolved_selection"] = {
        "selection_run_id": reference, "config_id": choice["config_id"],
        "ranking_source": result["ranking_source"], "choice_hash": result["choice_hash"]}
    return actual


def construct_adapter(factory, *, execution_context, **arguments):
    """Pass lifecycle context only when a frozen factory explicitly opts in."""
    if "execution_context" in inspect.signature(factory).parameters:
        return factory(execution_context=execution_context, **arguments)
    if paid_parent_ids(arguments["run"]):
        raise ValueError("Paid-parent continuation requires an explicit context-aware factory")
    return factory(**arguments)


def paid_parent_ids(run):
    lifecycle = run["config"].get("lifecycle")
    if lifecycle is None:
        return []
    if isinstance(lifecycle, dict) and lifecycle.get('schema') == 'paid-curvature-stage-v2':
        stage = run['config'].get('curvature_stage')
        if (set(lifecycle) != {'schema','parent_fit_ids','source_stage','named_Adam_reuse','source_RNG','returned_trial_states'}
                or stage not in ('trials','continuation')
                or lifecycle['source_stage'] != ('warm' if stage == 'trials' else 'trials')
                or lifecycle['named_Adam_reuse'] is not True
                or lifecycle['source_RNG'] != 'own_fixed_last_warm'
                or lifecycle['returned_trial_states'] != 'pretrial_only'
                or not isinstance(lifecycle['parent_fit_ids'],list) or len(lifecycle['parent_fit_ids']) != 1
                or not isinstance(lifecycle['parent_fit_ids'][0],str)):
            raise ValueError('Curvature requires own paid source, named Adam, original warm RNG and pretrial states')
        return lifecycle['parent_fit_ids']
    if isinstance(lifecycle, dict) and lifecycle.get('schema') == 'paid-native-capture-label-stage-v1':
        if (set(lifecycle) != {'schema','parent_fit_ids','new_label_optimizer','native_initialization','label_initialization'}
                or lifecycle['new_label_optimizer'] is not True
                or lifecycle['native_initialization'] != 'own_selected_native_frozen_capture'
                or lifecycle['label_initialization'] != 'fresh_source_maps_no_native_optimizer_reuse'
                or not isinstance(lifecycle['parent_fit_ids'],list) or len(lifecycle['parent_fit_ids']) != 1
                or not isinstance(lifecycle['parent_fit_ids'][0],str)):
            raise ValueError('Exact paid native/capture/fresh-label lifecycle required')
        return lifecycle['parent_fit_ids']
    if isinstance(lifecycle, dict) and lifecycle.get('schema') == 'paid-NSD-geometry-branch-v1':
        if (set(lifecycle) != {'schema','parent_fit_ids','new_optimizer','parameter_initialization'}
                or lifecycle['new_optimizer'] is not True
                or lifecycle['parameter_initialization'] != 'terminal100_slow_Adam_retained'
                or not isinstance(lifecycle['parent_fit_ids'],list) or len(lifecycle['parent_fit_ids']) != 1
                or not isinstance(lifecycle['parent_fit_ids'][0],str)):
            raise ValueError('Exact paid NSD100+400 source slow-Adam continuation required')
        return lifecycle['parent_fit_ids']
    if (not isinstance(lifecycle, dict) or set(lifecycle) != {
            "schema", "parent_fit_ids", "new_optimizer", "parameter_initialization"}
            or lifecycle.get("schema") != "paid-parent-continuation-v1"
            or lifecycle.get("new_optimizer") is not True
            or lifecycle.get("parameter_initialization") != "selected_parent_no_optimizer_reuse"
            or not isinstance(lifecycle.get("parent_fit_ids"), list)
            or len(lifecycle["parent_fit_ids"]) != 1
            or not isinstance(lifecycle["parent_fit_ids"][0], str)):
        raise ValueError("Unsupported paid-parent lifecycle")
    return lifecycle["parent_fit_ids"]


def require_curvature_paid_parent(parent, run):
    """Validate cohort/HPO custody before decoding any own paid source tensor."""
    c, p = run['config'], parent['config']
    expected = 'warm' if c.get('curvature_stage') == 'trials' else 'trials'
    if (c.get('curvature_stage') not in ('trials','continuation')
            or p.get('curvature_stage') != expected
            or any(parent.get(k) != run.get(k) for k in ('split','seed','body'))
            or any(p.get(k) != c.get(k) for k in ('stage','node_inputs','curvature_sources','curvature_constants',
                                                 'learning_rate','dropout','selected_config_from'))
            or parent['budget'] != {'unit':'optimizer_updates' if expected=='warm' else 'epochs',
                                    'policy':'full_budget','maximum':250 if expected=='warm' else 1}
            or parent['selection'].get('checkpoint0') is not False
            or (expected=='trials' and p.get('curvature_joint_selector') is not True)
            or (expected=='warm' and p.get('curvature_fixed_last') is not True)):
        raise ValueError('Curvature parent must be own complete fixed-last warm or five-arm pretrial block')
def require_growth_terminal20_parent(parent, run):
    """The post arm receives its own new terminal TRAIN-only warm authority."""
    a, b = parent["config"], run["config"]
    same_source = ("initialization_seed", "factor_seed", "learning_rate",
                   "link_inputs", "growth_sources", "calibration_draw")
    if (b.get("growth_stage") != "post" or a.get("growth_stage") != "warm"
            or a.get("growth_arm") != "ordinary_M1"
            or parent["budget"] != {"unit": "epochs", "policy": "full_budget", "maximum": 20}
            or run["budget"] != {"unit": "epochs", "policy": "full_budget", "maximum": 60}
            or parent["selection"].get("metric") != "training_completion"
            or parent["selection"].get("direction") != "max"
            or parent["selection"].get("ties") != "first"
            or parent["selection"].get("evaluation_interval") != 20
            or parent["selection"].get("checkpoint0") is not False
            or run["selection"].get("checkpoint0") is not False
            or a.get("stage") != b.get("stage")
            or any(parent.get(k) != run.get(k) for k in ("split", "seed", "body"))
            or any(k not in a or k not in b or a[k] != b[k] for k in same_source)):
        raise ValueError("Growth requires its newly paid exact terminal20 same-source TRAIN-only warm parent")


def require_nsd_geometry_parent(parent, run):
    """A paid100 source warm initializes slow Adam and preserves warm-best eligibility."""
    a,b = parent['config'],run['config']
    if (a.get('nsd_arm') != 'geometry_warm' or b.get('nsd_arm') != 'geometry_branch'
            or parent['budget'] != {'unit':'optimizer_updates','policy':'full_budget','maximum':100}
            or run['budget'] != {'unit':'optimizer_updates','policy':'full_budget','maximum':400}
            or parent['selection'].get('checkpoint0') is not False
            or run['selection'].get('checkpoint0') is not True
            or b.get('geometry_choice') not in {'identity','random_admissible','live_action_rank','classifier_response_rank'}
            or a.get('stage') != b.get('stage')
            or any(parent.get(k) != run.get(k) for k in ('split','seed','body'))
            or any(k not in a or k not in b or a[k] != b[k] for k in ('initialization_seed','learning_rate','dropout','node_inputs','nsd_sources'))):
        raise ValueError('NSD source continuation requires new paid100 warm,same roles/seed/LR/source,400post and paid warm-best eligibility')


def paid_parent_checkpoints(root, spec, run, registry, output_root):
    """Require completed, paid, directly declared selected-state ancestors."""
    identifiers = paid_parent_ids(run)
    model_parents = [identifier for identifier in run["parents"]
                     if registry.run(spec["study_id"], identifier)["contract"]["kind"] == "model_fit"]
    if (not isinstance(identifiers, list) or len(identifiers) != len(set(identifiers))
            or set(identifiers) != set(model_parents)):
        raise ValueError("Declare every and only direct paid model-fit ancestor")
    result = {}
    from engine import checkpoint_receipt
    for identifier in identifiers:
        parent = registry.run(spec["study_id"], identifier)
        if run['config'].get('curvature_stage') in ('trials','continuation'):
            require_curvature_paid_parent(resolve_selected_config(spec,parent['contract'],registry), run)
        value = parent["result"] or {}
        terminal_nsd = run['config'].get('lifecycle',{}).get('schema') == 'paid-NSD-geometry-branch-v1'
        if terminal_nsd:
            require_nsd_geometry_parent(parent['contract'],run)
        bound = value.get('final_training_checkpoint' if terminal_nsd else 'selected_checkpoint')
        if (parent["state"] != "complete" or value.get("status") != "complete"
                or value.get("completed_steps") != parent["contract"]["budget"]["maximum"]
                or not isinstance(bound, dict) or bound.get("fit_id") != identifier
                or bound.get("path") != ("final_training.pt" if terminal_nsd else "selected.pt") or type(bound.get("step")) is not int
                or not 1 <= bound["step"] <= parent["contract"]["budget"]["maximum"]
                or bound["step"] != (100 if terminal_nsd else value.get("selected", {}).get("step"))):
            raise ValueError("Paid source lacks a complete, positive-step selected checkpoint")
        if run["config"].get("growth_stage") == "post":
            require_growth_terminal20_parent(parent["contract"], run)
            if value.get("completed_steps") != 20 or bound["step"] != 20:
                raise ValueError("Growth requires complete terminal20 selected.pt authority")
        path = Path(output_root) / spec["study_id"] / identifier / ("final_training.pt" if terminal_nsd else "selected.pt")
        if not path.resolve().is_relative_to(Path(root).resolve()) or path.is_symlink():
            raise ValueError("Paid source checkpoint escapes the repository")
        observed = checkpoint_receipt(path, fit_id=identifier, step=bound["step"])
        if observed != bound:
            raise ValueError("Paid source checkpoint differs from completed byte authority")
        result[identifier] = {"checkpoint": path, "receipt": observed,
                              "contract": parent["contract"], "result": value}
    return result


def execute(project_root, study_path, run_id, registry_path, output_root, approval_path, device,
            recover=False):
    project_root = Path(project_root).resolve()
    spec, digest = load(study_path, project_root)
    approval = require_approval(approval_path, digest)
    rows = [row for row in spec["runs"] if row["run_id"] == run_id]
    if len(rows) != 1 or rows[0]["kind"] != "model_fit":
        raise ValueError("Run ID must identify one declared model fit")
    run = rows[0]
    require_phase_approval(project_root, spec, run, approval)
    from native_stopping import require_completion
    if run["budget"]["policy"] == "hgt_native_selector_stop":
        from hgt_stopping import validate_hgt_run
        validate_hgt_run(run)
    else:
        require_completion(run)
    registry = Registry(registry_path)
    registry.register(spec)
    if run["parents"] and not registry.required_complete(spec["study_id"], run["parents"]):
        registry.close()
        raise ValueError("Model-fit dependencies are not complete")
    run = resolve_selected_config(spec, run, registry)
    owner = owner_identity()
    admission = None
    if recover:
        prior = registry.run(spec["study_id"], run_id)
        admission = terminal_owner(prior["owner"] or {})
        retained = Path(output_root) / spec["study_id"] / run_id / "result.json"
        if not retained.is_file() or json.loads(retained.read_text()).get("status") != "failed":
            registry.close()
            raise ValueError("Recovery requires a retained interrupted engine result; inspect crashes manually")
        registry.recover(spec["study_id"], run_id, owner, admission)
    else:
        registry.claim(spec["study_id"], run_id, owner)
    tick = time.monotonic()
    try:
        import torch
        from engine import run_trajectory, resume_trajectory
        torch.set_num_threads(int(os.environ.get("GNNM_TORCH_THREADS", "2")))
        factory = load_factory(project_root, run["adapter"])
        context = {"registry": registry, "output_root": Path(output_root),
                   "paid_parent_checkpoints": paid_parent_checkpoints(
                       project_root, spec, run, registry, output_root)}
        adapter = construct_adapter(factory, execution_context=context, torch=torch,
                                    project_root=project_root, study=spec, run=run, device=device)
        engine_spec = {"fit_id": run_id, "budget_steps": run["budget"]["maximum"],
                       "budget_unit": run["budget"]["unit"],
                       "validation_interval": run["selection"]["evaluation_interval"],
                       "validate_initial": run["selection"]["checkpoint0"],
                       "prefixes": run["budget"].get("prefixes", []),
                       "recovery_interval": run["budget"].get("recovery_interval", 100),
                       "inherited_fit_ids": run["parents"],
                       "adapter_spec": {"study_hash": digest, "run": run,
                                        "data": spec["data"], "source": spec["source"]}}
        if run['config'].get('native_AMP_overflow_skip_allowed'):
            engine_spec['native_amp_cycle_policy']={'schema':'native-SeHGNN-AMP-complete-cycle-v1','optimizer_owner_attempts':1}
            if 'typed_context_source_contract' in run['config']:
                engine_spec['native_amp_cycle_policy']={'schema':'typed-context-AMP-complete-cycle-v1','optimizer_owner_attempts':1}
        if run["budget"]["policy"] in {"native_selector_stop", "hgt_native_selector_stop", "typed_context_native_selector_stop"}:
            engine_spec["completion_policy"] = run["budget"]["policy"]
        def guard(phase, step, result):
            if time.monotonic() - tick > spec["resources"]["wall_seconds"]:
                raise TimeoutError("Declared run wall budget exhausted")
            if str(device).startswith("cuda"):
                peak = int(torch.cuda.max_memory_allocated())
                ceiling = run["config"].get("maximum_cuda_bytes")
                if ceiling and peak > ceiling:
                    raise MemoryError("Declared run CUDA budget exhausted")
                if step % 100 == 0 or step == run["budget"]["maximum"]:
                    return {"peak_cuda_allocated_bytes": peak}
            return None
        destination = Path(output_root) / spec["study_id"] / run_id
        if admission:
            result = resume_trajectory(torch, adapter, engine_spec, destination,
                                       admission=admission, resource_check=guard)
        else:
            result = run_trajectory(torch, adapter, engine_spec, destination, resource_check=guard)
        from retention_lifecycle import close_fit
        result=close_fit(project_root,spec,run,destination,result)
        registry.finish(spec["study_id"], run_id, result, success=True)
        return result
    except BaseException as error:
        retained = Path(output_root) / spec["study_id"] / run_id / "result.json"
        failure = json.loads(retained.read_text()) if retained.exists() else {}
        failure.update({"failure_type": type(error).__name__, "failure": str(error),
                        "wall_seconds": time.monotonic() - tick, "owner": owner})
        registry.finish(spec["study_id"], run_id,
                        failure, success=False)
        raise
    finally:
        registry.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recover-terminal", action="store_true",
                        help="Resume only a retained failed trajectory whose exact owner is terminal")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    for value in (args.study, args.registry, args.output, args.approval):
        if not Path(value).resolve().is_relative_to(root):
            raise ValueError("Study, registry, outputs and approval must stay inside the repository")
    execute(root, args.study, args.run_id, args.registry, args.output, args.approval, args.device,
            recover=args.recover_terminal)


if __name__ == "__main__":
    main()
