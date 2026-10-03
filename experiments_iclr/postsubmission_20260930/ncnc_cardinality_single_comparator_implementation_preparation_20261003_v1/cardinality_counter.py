"""Stateless SHA256 completion keys; no native RNG or label arguments.

SOURCE ONLY. D4 and its fixed-bank M4 control share this same key family.
The conceptual open uniform is (integer53 + .5) / 2**53. Return its log
directly so the upper midpoint cannot round to a closed FP64 endpoint.
"""
from dataclasses import dataclass
from hashlib import sha256
import math

DOMAIN = "ncnc-cardinality-single-v1"
DENOMINATOR = 2**53


def separated_seed(purpose):
    if purpose not in ("count-init", "diagnostic-records", "diagnostic-negatives"):
        raise ValueError("Unknown frozen seed domain")
    text = f"{DOMAIN}|seed=0|domain={purpose}"
    return int.from_bytes(sha256(text.encode("ascii")).digest()[:8], "little") % (2**31 - 1)


def integer53(text):
    # Algorithm binding: SHA256/ASCII, first8 little endian, low53 bits.
    return int.from_bytes(sha256(text.encode("ascii")).digest()[:8], "little") & (DENOMINATOR - 1)


def midpoint_log_uniform(value):
    if type(value) is not int or not 0 <= value < DENOMINATOR:
        raise ValueError("Uniform integer outside its fixed 53-bit range")
    if value < DENOMINATOR // 2:
        return math.log((value + .5) / DENOMINATOR)
    return math.log1p((value - DENOMINATOR + .5) / DENOMINATOR)


def canonical_pair(pair):
    if len(pair) != 2:
        raise ValueError("Expected one endpoint pair")
    a, b = (int(v) for v in pair)
    if a < 0 or b < 0:
        raise ValueError("Negative endpoint")
    return min(a, b), max(a, b)


@dataclass(frozen=True)
class CompletionKeys:
    mode: str
    epoch: int | None = None
    batch: int | None = None
    ordinal_start: int = 0

    def __post_init__(self):
        if self.mode == "eval":
            if self.epoch is not None or self.batch is not None or self.ordinal_start != 0:
                raise ValueError("Evaluation excludes epoch, batch and ordinal keys")
        elif self.mode == "train":
            if type(self.epoch) is not int or self.epoch < 1 or type(self.batch) is not int or self.batch < 1:
                raise ValueError("Training requires prospective positive epoch/batch IDs")
            if type(self.ordinal_start) is not int or self.ordinal_start < 0:
                raise ValueError("Invalid combined query ordinal")
        else:
            raise ValueError("Unknown completion key mode")

    def log_uniform(self, query, ordinal, draw, purpose, counterpart=None):
        if draw not in range(4) or purpose not in ("count", "slot"):
            raise ValueError("Frozen four-draw purpose required")
        if purpose == "slot" and counterpart is None or purpose == "count" and counterpart is not None:
            raise ValueError("Count/slot counterpart contract violated")
        if self.mode == "eval":
            i, j = canonical_pair(query)
            base = f"{DOMAIN}|seed=0|mode=eval|query={i},{j}"
        else:
            if type(ordinal) is not int or ordinal < 0:
                raise ValueError("Invalid supplied query ordinal")
            base = (f"{DOMAIN}|seed=0|mode=train|epoch={self.epoch}|batch={self.batch}"
                    f"|query={self.ordinal_start + ordinal}")
        base += f"|draw={draw}|purpose={purpose}"
        if counterpart is not None:
            a, b = canonical_pair(counterpart)
            base += f"|counterpart={a},{b}"
        return midpoint_log_uniform(integer53(base))


def diagnostic_record_key(record_id):
    if type(record_id) is not int or record_id < 0:
        raise ValueError("Invalid ascending TRAIN record ID")
    return sha256(f"{DOMAIN}|seed=0|domain=diagnostic-records|record={record_id}".encode("ascii")).digest()
