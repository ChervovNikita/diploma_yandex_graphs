"""Hand-computed numerical fixtures; not training or scientific evidence."""
import json
from pathlib import Path

from paired_statistics import joint_confirmation, paired_summary, sign_pvalue


def main():
    assert sign_pvalue(20, 20) == 1 / 2**20
    assert sign_pvalue(15, 20) <= .05 < sign_pvalue(14, 20)
    assert sign_pvalue(0, 20) == 1
    baselines = {name: [0.] * 20 for name in ("native_M1", "base_GNNM", "native_I4")}
    a = [1.] * 15 + [-1.] * 5
    result = joint_confirmation(a, baselines)
    assert result["joint_intersection_union_p"] == sign_pvalue(15, 20)
    assert result["joint_repeat_win_claim_rejected_null"]
    baselines["native_I4"] = [2.] * 20
    result = joint_confirmation(a, baselines)
    assert result["joint_intersection_union_p"] == 1
    assert not result["joint_repeat_win_claim_rejected_null"]
    # An unresolved trial never acquires an invented score or disappears.
    baselines["native_I4"] = [0.] * 20
    result = joint_confirmation([None] + [1.] * 14 + [-1.] * 5, baselines,
                                unresolved=[True] + [False] * 19)
    assert result["unresolved_trials"] == 1
    assert not result["joint_repeat_win_claim_rejected_null"]
    assert all(row["wins"] == 14 and row["total_including_unresolved"] == 20 for row in result["components"].values())
    summary = paired_summary([0., 1., 2., 3.], [0., 0., 0., 0.], bootstrap_seed=103, draws=1000)
    assert summary["mean_difference"] == 1.5 and summary["wins"] == 3 and summary["ties"] == 1
    assert summary == paired_summary([0., 1., 2., 3.], [0., 0., 0., 0.], bootstrap_seed=103, draws=1000)
    try:
        paired_summary([1., float("nan")], [0., 0.], bootstrap_seed=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Finite-only deletion must be refused")
    return {"status": "passed", "scope": "hand-computed numerical fixtures only",
            "models_constructed": 0, "scientific_results_read": 0, "TEST_accessed": False}


if __name__ == "__main__":
    value = main()
    path = Path(__file__).with_name("SEMANTIC_CHECKS.json")
    raw = json.dumps(value, indent=2) + "\n"
    if path.exists() and path.read_text() != raw:
        raise ValueError("Preserve earlier check outcome")
    path.write_text(raw)
    print(json.dumps(value))
