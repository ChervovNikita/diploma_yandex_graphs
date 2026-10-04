"""One pooled bank clock, strict selector and balanced fixed role schedule."""
from dataclasses import dataclass
from itertools import combinations
from views import digest

LOCAL_UPDATES = 200
GLOBAL_UPDATES = 2500
TOTAL_UPDATES = LOCAL_UPDATES + GLOBAL_UPDATES
CONDITIONS = ("tied_persistent", "untied_persistent", "tied_shuffled", "tied_random_null")


def stage_at(update):
    if type(update) is not int or update not in range(1, TOTAL_UPDATES + 1):
        raise ValueError("Eligible updates are 1..2700; no epoch-zero candidate")
    return "global" if update > LOCAL_UPDATES else "local"


def assignment(condition, seed, update):
    stage = stage_at(update)
    if condition not in CONDITIONS:
        raise ValueError("Unknown comparison condition")
    if condition == "tied_shuffled":
        position = update - LOCAL_UPDATES if stage == "global" else update
        pair_index = (position - 1) // 2
        choices = tuple(combinations(range(4), 2))
        selected = choices[int(digest([seed, stage, pair_index, "roles"]), 16) % len(choices)]
        equal_members = set(selected) if position % 2 else set(range(4)).difference(selected)
    else:
        equal_members = {0, 1}
    prefix = "null_" if condition == "tied_random_null" else ""
    return tuple(prefix + ("equal" if m in equal_members else "different") for m in range(4))


def pass_plan(condition, seed, update):
    roles = assignment(condition, seed, update)
    return tuple((m, "native") for m in range(4)) + tuple((m, roles[m]) for m in range(4))


def schedule_identity(condition, seed):
    return {"condition": condition, "schedule_seed": seed,
            "algorithm": "persistent 0,1 equal; shuffled complementary hash-indexed 2-of-4 pairs per two updates",
            "stage_lengths": [LOCAL_UPDATES, GLOBAL_UPDATES],
            "assignments_sha256": digest([assignment(condition, seed, u) for u in range(1, TOTAL_UPDATES + 1)])}


@dataclass
class BankClock:
    actual_update: int = 0

    def advance(self):
        next_update = self.actual_update + 1
        stage_at(next_update)
        self.actual_update = next_update
        return next_update

    def record(self):
        return {"actual_update": self.actual_update,
                "actual_local_updates": min(self.actual_update, LOCAL_UPDATES),
                "actual_global_updates": max(0, self.actual_update - LOCAL_UPDATES)}

    def require_transition(self):
        if self.actual_update != LOCAL_UPDATES:
            raise ValueError("Transition uses actual 200-update clock, never an Adam step counter")


@dataclass
class Selector:
    best: int = -1
    selected_update: int = 0

    def observe(self, correct, count, update):
        stage_at(update)
        if type(correct) is not int or type(count) is not int or not 0 <= correct <= count or count <= 0:
            raise ValueError("Invalid fixed-cardinality validation correct count")
        if correct > self.best:
            self.best, self.selected_update = correct, update
            return True
        return False
