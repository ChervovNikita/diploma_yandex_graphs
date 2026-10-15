# Structured responsibility control v2

**Prepared, checked, unintegrated.** No training, native model/data imports,
optimizer transitions, or frozen-source/pilot changes.

V1's nonself endpoint weights have variance 1/2, while its purported matched
query control has variance 1. V2 corrects the query raw marginal:

| Raw law | Nonself variance | Self-loop variance |
| --- | ---: | ---: |
| Endpoint `c+(X_u+X_v)/2` | 1/2 | 1 |
| V1 query `c+Y` | 1 | 1 |
| V2 query: two independent halves for nonself, one draw for loop | 1/2 | 1 |

All raw means are c+1 in the ideal Exp(1) law. Duplicate semantic queries
reuse draws. Node fields are separate by relation and member. Endpoint raw
covariance is exactly `B Bᵀ` for the per-relation half-incidence matrix,
including loops and repeated rows.

**Raw matching does not match normalized marginals.** Dividing by each native
source group's current-batch mean changes the marginal law and generally
invalidates `E[normalized_weight]=1` on irregular incidence structures.
There is no unbiased-original-risk, posterior, novelty, or accuracy claim.

## Files

- [SPECIFICATION.md](SPECIFICATION.md): derivation, v1 confound, identity law,
  normalization counterexample, qualified local influence calculation, read scope.
- [sampler.py](sampler.py): pure-stdlib persistent raw sampler and small analytic
  covariance audits; separate v2 hash namespace.
- [objectives.py](objectives.py): Torch-injected native reduction and two
  explicit shared/private criteria; source parity and capture-signature gates.
- [checks.py](checks.py), [CHECKS.json](CHECKS.json): **11 passing CPU checks**,
  synthetic only. The suite accepts the complete native 131,072-row NCN batch
  and checks complete positive/negative group normalization.
- [MANIFEST.json](MANIFEST.json): source/read/delivery hashes and adoption status.

The sampler accepts at most 262,144 rows and 16 members. No source rows are
dropped or split for normalization. Only dense analytic covariance audits
are limited to 256 rows. These are explicit engineering bounds, not a native
runtime/memory qualification.

## Reproduce the bounded CPU checks

```sh
PYTHONPATH=/Users/alex/.cache/codex-research-molecular-tensor-tests/torch251 \
/Users/alex/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
"/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/benchmark_expansion_20261015_v1/structured_responsibility_control_v2/checks.py"
```

`sampler.py` needs only the Python standard library. The objective helper
requires the caller to inject Torch. The check writes `CHECKS.json` next to
its source. Root owns any future native integration, real-data qualification,
study freeze, launch, and publication.
