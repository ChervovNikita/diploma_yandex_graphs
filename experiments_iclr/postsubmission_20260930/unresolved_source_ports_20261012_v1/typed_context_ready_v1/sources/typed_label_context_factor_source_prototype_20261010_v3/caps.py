"""Closed execution capabilities. Importing this file performs no project reads."""
from dataclasses import dataclass


class DisabledPrototype(RuntimeError):
    pass


@dataclass(frozen=True)
class Caps:
    source_bound: bool = False
    model: bool = False
    data: bool = False
    runtime: bool = False
    scientific: bool = False
    root_review_sha256: str = ""

    def require(self, *names):
        sha = self.root_review_sha256
        if (len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
                or any(getattr(self, name) is not True for name in names)):
            raise DisabledPrototype("Root-reviewed successor capabilities required: " + ", ".join(names))


CLOSED = Caps()
