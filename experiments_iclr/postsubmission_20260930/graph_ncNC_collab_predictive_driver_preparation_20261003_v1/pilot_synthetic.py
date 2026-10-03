"""Execution qualification on fabricated inputs only; no dataset reads."""
import copy
from pathlib import Path
from unittest.mock import patch
from pilot_common import require, atomic_json, utc
from pilot_model import make_native, make_factorized, seed_all, train_flag, named_parameters
from pilot_state import (rng_state, rng_digest, restore_rng, snapshot, restore_snapshot,
                         state_digest, write_journal, read_journal)
from pilot_data import tensor_sha
from pilot_evaluate import evaluator, hits50, strict_hits50, mean_native_scores, score_valid
from pilot_train import train_batch


def rejected(function):
    try:
        function()
    except (RuntimeError, ValueError):
        return
    raise AssertionError("Invalid synthetic input was accepted")


def run(context, mods, sampler, device, output, attempts):
    import random
    import json
    import numpy as np
    import torch
    checks = []
    def passed(name):
        checks.append(name)
        attempts.progress({"synthetic_checks_completed": len(checks)})

    attempts.phase("synthetic_bound_evaluator_and_frozen_selector")
    metric = evaluator(context)
    require(metric.name == "ogbl-collab" and metric.eval_metric == "hits@50" and metric.K == 50, "Evaluator attributes incomplete")
    negatives = torch.arange(100000, dtype=torch.float32)
    positives = torch.tensor([-1., 99950., 99951., 100001.]).repeat(15021)
    require(hits50(metric, positives, negatives) == .5, "Full100000-pool strict negative tie rule differs")
    for pos, neg in ((positives[:0], negatives), (positives[:-1], negatives), (positives, negatives[:-1]), (positives, negatives[:49]),
                     (positives[:, None], negatives), (positives, negatives[:, None]),
                     (positives.double(), negatives), (torch.full((60084,), float("nan")), negatives),
                     (positives, torch.full((100000,), float("inf")))):
        rejected(lambda p=pos, n=neg: hits50(metric, p, n))
    rows = [(torch.full((60084,), 10. if m == 0 else 0.), torch.ones(100000), {}) for m in range(4)]
    pooled = mean_native_scores(rows)
    require(hits50(metric, *pooled) == 1. and sum(hits50(metric, r[0], r[1]) for r in rows) / 4 == .25, "Serving must average logits before the metric")
    passed("bound_OGB_attributes_full100000_negative_pool_strict_ties_invalid_input_and_mean_before_metric")
    best = None
    for order in range(1, 102):
        best, replacement = mods["design"].select_validation_candidate(best, candidate_id=str(order), hits50=0., order=order)
        require(replacement == (order == 1) and best["order"] == 1, "First exact ties/epoch1 unconditional save differ")
    improved, replacement = mods["design"].select_validation_candidate(best, candidate_id="improve", hits50=.5, order=102)
    require(replacement and improved["order"] == 102, "Strict improvement selector differs")
    rejected(lambda: mods["design"].select_validation_candidate(improved, candidate_id="late", hits50=float("nan"), order=103))
    passed("first_ties_epoch1_and_strict_selector")

    attempts.phase("synthetic_native_capacity_and_fixed_Rademacher_initialization")
    for width, expected, active in ((64, 38147, 33922), (70, 44663, 39622)):
        model, optimizer = make_native(mods, 0, width, device)
        parameters = named_parameters(model)
        require(sum(p.numel() for _, p in parameters) == expected and sum(p.numel() for n, p in parameters if ".ptlin." not in n) == active, "Native total/active capacity differs")
        require(not optimizer.state and [g["lr"] for g in optimizer.param_groups] == [.0082, .0037], "Fresh native Adam differs")
        require(all(g["betas"] == (.9, .999) and g["eps"] == 1e-8 and g["weight_decay"] == 0 and not g["amsgrad"] for g in optimizer.param_groups), "Native Adam hyperparameters differ")
        del model, optimizer
    private, private_optimizer = make_factorized(mods, 0, device)
    initial = snapshot(private, private_optimizer)
    pooled_model, pooled_optimizer = make_factorized(mods, 0, device)
    require(state_digest(initial) == state_digest(snapshot(pooled_model, pooled_optimizer)), "Fresh scientific twins differ")
    require(sum(p.numel() for p in private.parameters()) == 43790 and sum(p.numel() for n, p in private.named_parameters() if ".ptlin." not in n) == 38793, "Factorized total/active capacity differs")
    generator = torch.Generator(device="cpu").manual_seed(mods["design"].factor_sign_seed(0))
    signs, includes_ptlin = [], False
    for name, parameter in sorted(private.named_parameters()):
        if name.endswith(".r") or name.endswith(".s"):
            expected = 2 * torch.randint(0, 2, parameter.shape, dtype=torch.int64, generator=generator) - 1
            require(torch.equal(parameter.detach().cpu(), expected.float()), "Fixed lexical CPU Rademacher initializer differs")
            signs.append(parameter.detach().cpu().reshape(4, -1))
            includes_ptlin |= ".ptlin." in name
    signatures = torch.cat(signs, dim=1)
    require(includes_ptlin and len({tensor_sha(row) for row in signatures}) == 4, "All four fixed sign members and unused ptlin factors required")
    seed_all(0)
    default_model = mods["prototype"].CompletionTwin(mods["prototype"].Recipe()).to(device)
    default_optimizer = mods["prototype"].native_optimizer(default_model)
    default = snapshot(default_model, default_optimizer)
    require(rng_digest(default["rng"]) == rng_digest(initial["rng"]), "Sign generator consumed training RNG")
    for name, value in default["models"][0].items():
        if not (name.endswith(".r") or name.endswith(".s")):
            require(torch.equal(value, initial["models"][0][name]), "Native shared W/bias/LN/beta defaults changed")
    del default_model, default_optimizer, default
    passed("native64_native70_F4_total_active_counts_Adam_defaults_fixed_all_member_signs_and_training_RNG")

    attempts.phase("synthetic_native_stream_and_actual_update_routes")
    feature_generator = torch.Generator(device="cpu").manual_seed(91234)
    pairs = torch.tensor([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,0),(0,1)], device=device)
    raw = pairs.T.repeat_interleave(2, dim=1)
    raw[:, 1::2] = pairs.T.flip(0)
    data = {"x": torch.randn(16, 128, generator=feature_generator).to(device), "pairs": pairs, "raw_edge_index": raw,
            "valid_positive": pairs, "valid_negative": torch.tensor([(i // 15, (i % 15) + 1) for i in range(50)], device=device)}
    end_rng, streams = [], []
    for mode, model, optimizer in (("private", private, private_optimizer), ("pooled_after_clamp", pooled_model, pooled_optimizer)):
        restore_snapshot(model, optimizer, initial)
        negatives = sampler(raw, len(data["x"]))
        iterator = mods["native_utils"].PermIterator(device, len(pairs), 4)
        stream = (tensor_sha(negatives), tensor_sha(iterator.idx))
        train_flag(model, True)
        for record_ids in iterator:
            train_batch(model, optimizer, data, mods, negatives.T, record_ids, mode=mode)
        streams.append(stream)
        end_rng.append(rng_digest(rng_state()))
        before = rng_digest(rng_state())
        pos, neg, receipt = score_valid(model, data, mods, mode=mode)
        require(len(pos) == len(pairs) and len(neg) == 50 and receipt["all_query_rows_complete"], "Actual F4 scoring omitted synthetic query tail")
        require(rng_digest(rng_state()) == before, "Actual F4 evaluation consumed training RNG")
    require(streams[0] == streams[1] and end_rng[0] == end_rng[1], "F4 modes changed native sampler/permutation/dropout RNG schedule")
    for width in (64, 70):
        model, optimizer = make_native(mods, 3, width, device)
        negatives = sampler(raw, len(data["x"]))
        record_ids = torch.tensor([0, 2, 5, 9], device=device)
        train_flag(model, True)
        train_batch(model, optimizer, data, mods, negatives.T, record_ids)
        before = rng_digest(rng_state())
        pos, neg, receipt = score_valid(model, data, mods)
        require(pos.shape == (len(pairs),) and neg.shape == (50,) and rng_digest(rng_state()) == before, "Native full synthetic TRAIN/VALID route differs")
        del model, optimizer
    iterator = mods["native_utils"].PermIterator(device, 1179052, 65536)
    batches = list(iterator)
    require(len(batches) == 17 and sum(len(b) for b in batches) == 1114112 and len(iterator.idx[17 * 65536:]) == 64940, "Native complete TRAIN tail policy differs")
    iterator = mods["native_utils"].PermIterator(device, 131079, 131072, False)
    batches = list(iterator)
    require([len(b) for b in batches] == [131072, 7] and torch.equal(torch.cat(batches), torch.arange(131079, device=device)), "Complete canonical evaluation tails differ")
    passed("actual_native64_native70_F4_both_routes_finite_gradients_Adam_VALID_and_paired_native_streams")
    del private, private_optimizer, pooled_model, pooled_optimizer, initial, data, pairs, raw, negatives, batches, iterator

    attempts.phase("synthetic_RNG_interleaving_and_atomic_resume_faults")
    def draw():
        return (random.random(), np.random.random(), torch.rand(5), torch.rand(5, device=device).cpu())
    def sequence(value, count):
        restore_rng(value)
        return [draw() for _ in range(count)], rng_state()
    starts = []
    for seed in (0, 5):
        seed_all(seed); starts.append(rng_state())
    isolated = [sequence(value, 3)[0] for value in starts]
    interleaved, current = [[], []], copy.deepcopy(starts)
    for _ in range(3):
        for member in range(2):
            drawn, current[member] = sequence(current[member], 1)
            interleaved[member].extend(drawn)
    require(state_digest(isolated) == state_digest(interleaved), "Per-fit interleaved RNG differs from isolated native schedule")
    sandbox = output / "synthetic_own_journal"
    sandbox.mkdir(mode=0o700)
    first = {"epoch": 0, "marker": "first", "rng": starts[0]}
    write_journal(sandbox, context, "synthetic_only", 0, first)
    second = {"epoch": 0, "marker": "second", "rng": starts[1]}
    write_journal(sandbox, context, "synthetic_only", 0, second)
    journal = json.loads((sandbox / "JOURNAL.json").read_text())
    require(journal["revision"] == 1, "Repeated-epoch final commits did not alternate revisions")
    with patch("pilot_state.atomic_json", side_effect=OSError("synthetic_injected_commit_fault")):
        try:
            write_journal(sandbox, context, "synthetic_only", 0, {"epoch": 1, "marker": "uncommitted"})
        except OSError:
            pass
        else:
            raise AssertionError("Injected journal commit failure was ignored")
    require(state_digest(read_journal(sandbox, context, "synthetic_only", 0)) == state_digest(second), "Failed commit destroyed the prior own state")
    slot = sandbox / journal["state_file"]["path"]
    original = slot.read_bytes()
    slot.write_bytes(original + b"corruption")
    with patch("torch.load", side_effect=AssertionError("untrusted_pickle_read")):
        rejected(lambda: read_journal(sandbox, context, "synthetic_only", 0))
    slot.write_bytes(original)
    rejected(lambda: read_journal(sandbox, context, "native_bank4", 0))
    passed("isolated_vs_interleaved_all_RNGs_revision_slots_precommit_failure_corruption_and_wrong_donor_rejection")

    attempts.phase("synthetic_actual_orchestrator_100epochs_101candidates_and_resume")
    _orchestration_checks(context, mods, metric, device, output, attempts)
    passed("actual_fit_orchestrator_all100_epochs_101_ordered_candidates_member0_reuse_and_failed_epoch_resume")
    return {"schema": "ncnc-pilot-synthetic-qualification-v1", "identity": context["identity"], "status": "PASS",
            "checks": checks, "check_count": len(checks), "fabricated_inputs_only": True,
            "data_files_opened": [], "test_file_opened": False, "scientific_state_donor": False,
            "project_metrics_computed": False, "UTC": utc()}


def _orchestration_checks(context, mods, metric, device, output, attempts):
    """Run the actual fit state machine with cheap fabricated numerical work."""
    import torch
    import random
    import numpy as np
    from pilot_fit import fit
    class QuietAttempts:
        def phase(self, *args, **kwargs):
            pass
        def progress(self, *args, **kwargs):
            pass
    sandbox_context = {**context, "identity": {**context["identity"], "synthetic_only": True}}
    root = output / "synthetic_orchestration"
    root.mkdir(mode=0o700)
    def make_fixture(_mods, seed, width, _device):
        require(width == 64, "Synthetic orchestration changed native width")
        seed_all(seed)
        encoder = torch.nn.Linear(1, 1)
        decoder = torch.nn.Linear(1, 1)
        encoder.register_buffer("fixture_seed", torch.tensor(seed))
        with torch.no_grad():
            for part in (encoder, decoder):
                part.weight.zero_(); part.bias.zero_()
        return (encoder, decoder), torch.optim.Adam([{"params": encoder.parameters(), "lr": 0.}, {"params": decoder.parameters(), "lr": 0.}])
    fail = {"enabled": False, "fired": False}
    def train_fixture(model, optimizer, data, _mods, sampler, **kwargs):
        random.random(); np.random.random(); torch.rand(1); torch.rand(1, device=device)
        optimizer.zero_grad(set_to_none=True)
        for _, parameter in named_parameters(model):
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        with torch.no_grad():
            model[0].weight.add_(1.)
        return {"full_batches": 17, "optimizer_steps": 17, "wall_seconds": 0.,
                "stream": {"negative_draw_sha256": rng_digest(rng_state()), "permutation_sha256": "fabricated",
                    "dropped_tail_sha256": "fabricated", "full_batches": 17, "supervised_records": 1114112,
                    "dropped_tail_records": 64940, "negative_rows_drawn": 2358104}}
    def score_fixture(model, data, _mods, **kwargs):
        member = int(model[0].fixture_seed) // 5
        epoch = int(model[0].weight.detach()[0, 0])
        if fail["enabled"] and not fail["fired"] and member == 1 and epoch == 2:
            fail["fired"] = True
            raise RuntimeError("synthetic_injected_epoch_failure")
        score = .5 if epoch == member + 1 else -.4
        return torch.full((4,), score), torch.zeros(50), {"positive_queries": 4, "negative_queries": 50,
            "wall_seconds": 0., "fixture_only": True}
    reference, resumed = root / "reference", root / "resumed"
    reference.mkdir(mode=0o700); resumed.mkdir(mode=0o700)
    with patch("pilot_fit.make_native", side_effect=make_fixture), patch("pilot_fit.train_epoch", side_effect=train_fixture), patch("pilot_fit.score_valid", side_effect=score_fixture), patch("pilot_fit.hits50", side_effect=strict_hits50), patch("builtins.print"):
        complete = fit(sandbox_context, mods, {}, None, metric, device, reference, "native_bank4", 0, False, QuietAttempts())
        fail["enabled"] = True
        try:
            fit(sandbox_context, mods, {}, None, metric, device, resumed, "native_bank4", 0, False, QuietAttempts())
        except RuntimeError as error:
            require(str(error) == "synthetic_injected_epoch_failure", "Unexpected simulated failure")
        else:
            raise AssertionError("Interrupted epoch was not injected")
        require(read_journal(resumed, sandbox_context, "native_bank4", 0)["epoch"] == 1, "Failure committed a partial epoch")
        fail["enabled"] = False
        after_resume = fit(sandbox_context, mods, {}, None, metric, device, resumed, "native_bank4", 0, True, QuietAttempts())
    a = read_journal(reference, sandbox_context, "native_bank4", 0)
    b = read_journal(resumed, sandbox_context, "native_bank4", 0)
    require(state_digest(a) == state_digest(b), "Actual fit resume changed model/Adam/RNG/selectors or ordered candidates")
    require(a["ensemble_best"]["order"] == 101 and len(a["ensemble_candidates"]) == 101, "Individual-best bank was not evaluated last and selected on its served score")
    require(a["fits"][0]["best"]["order"] == 1 and [r["best"]["order"] for r in a["fits"]] == [1,2,3,4], "Individual selectors or N64 member0 donor changed")
    require(complete["unique_fits"] == after_resume["unique_fits"] == 4 and complete["completed_optimizer_steps"] == 6800, "Orchestrator fit budget differs")
    native_selection = json_read(reference / "PRIVATE_SELECTION_native_single_64.json")
    require(native_selection["selection"] == a["fits"][0]["best"], "N64 did not reuse the exact individually selected member0 state")


def json_read(path):
    import json
    return json.loads(Path(path).read_text())
