"""Pure orchestration contracts; no numerical imports."""
from dataclasses import dataclass
import hashlib
import json

BLOCKS = ((0, 17), (1, 29), (2, 43))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fit_spec(kind, split, member=None):
    seeds = dict(BLOCKS)
    if split not in seeds:
        raise ValueError("Unplanned official split")
    if kind == "view_augmented_single" and member is None:
        return {"kind": kind, "split": split, "member": None, "seed": seeds[split]}
    if kind == "native_member" and type(member) is int and member in range(4):
        return {"kind": kind, "split": split, "member": member, "seed": seeds[split] + 1009 * member}
    raise ValueError("Reference kind/member not declared")


def pass_plan(kind):
    if kind == "view_augmented_single":
        return (("native", 1.0), ("equal", 0.5), ("different", 0.5))
    if kind == "native_member":
        return (("native", 1.0),)
    raise ValueError("Unknown reference")


def stage_at(update):
    if type(update) is not int or update not in range(1, 2701):
        raise ValueError("Actual updates 1..2700 required")
    return update > 200


@dataclass
class Selector:
    best: int = -1
    selected_update: int = 0

    def observe(self, correct, count, update):
        stage_at(update)
        if type(correct) is not int or not 0 <= correct <= count or count <= 0:
            raise ValueError("Invalid validation correct count")
        if correct > self.best:
            self.best, self.selected_update = correct, update
            return True
        return False


def independent_inventory(results):
    if len(results) != 4:
        raise ValueError("Exactly four physical native references required")
    by_member = {}
    common = None
    for result in results:
        binding = result["bindings"]
        spec = binding["fit"]
        if spec != fit_spec("native_member", spec["split"], spec["member"]):
            raise ValueError("Declared native member/seed required")
        member = spec["member"]
        if member in by_member:
            raise ValueError("Duplicate native member")
        context = {k: v for k, v in binding.items() if k != "fit"}
        if common is None:
            common = context
        elif common != context:
            raise ValueError("Reference input/source/protocol context differs")
        if result["completed_actual_updates"] != 2700:
            raise ValueError("Incomplete physical reference")
        by_member[member] = result
    return tuple(by_member[m] for m in range(4))
