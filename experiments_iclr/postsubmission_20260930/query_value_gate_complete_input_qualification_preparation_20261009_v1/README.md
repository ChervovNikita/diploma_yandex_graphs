# Full-input query value gate qualification preparation

This is an unexecuted adaptation of the original staged-posterior `QUALIFY.py`
for the sealed five-arm source manifest
`67a6f621b23f77761a96954119bade6d3646021ebde4a0a89c6c74b76df8ced0`.
Root supplies the separate `--authorized` engineering release. The scientific
source and its disabled seal are unchanged.

The qualifier uses actual full WikiCS roles and seed6101, constructs every one of
the five banks, and spends four discarded common290-query updates within180s.
It serves all5274 development IDs without indexing development truths or calling
the development evaluator. Its only comparisons are engineering state/cache
reconstruction errors, with practical tolerance2e-6.

Before training it checks zero A/b and identity forwarding for all eight gate
instances, and verifies that the gate factory and identity checks consume no
ambient CPU/CUDA or live-mask RNG. After updates2/3/4 it checks nonzero actual A/b
gradients and parameter changes. The four U4 gates, parameters, optimizer owners
and actual Adam moment storage must be independent.

Observational gate hooks check exact zero-message preservation in actual training
and serving. An observational wrapper around the erased bank's value lookup
checks that every actual call receives no class identities and returns the
embedding row mean. Both observers return the original numerical outputs and
are removed in `finally`; no scientific source file is modified. Every bank's
actual no-anchor serving logits must be exactly zero and its served probability
must equal the frozen native fallback.

The original engineering sequence is retained: snapshot after update1, update2,
restore learned/gate/Adam state while retaining live masks and work, then updates3
and4. After update3 it saves and reloads its own engineering state and fixed
capture, constructs the cache-backed stage without a second native forward, and
checks serving reconstruction for every bank. Native versions, gradients, mode,
update count and RNG remain subject to the source's frozen-native guards.

The fresh output directory is
`query_value_gate_complete_input_qualification_execution_root_20261009_v1`.
`RESULT.json` preserves completed or partial update counters, timings, CPU time,
RSS and scoped CUDA peaks. The engineering capture/state files remain available.
There is no retry, scientific fit, development score, TEST access or quality claim.
Selected scientific endpoint reconstruction remains deferred until full15 closure.

Normal command, run by root on `anogena-2-0` from the original repository:

```sh
cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
PYTHONPATH=/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_ncn_dependency_overlay_20261005_v1:/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/lib/python3.11/site-packages CUDA_VISIBLE_DEVICES=GPU-44039938-fd82-41d2-fefd-de71514e2fac /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python -B /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/query_value_gate_complete_input_qualification_preparation_20261009_v1/QUALIFY.py --authorized
```

Preparation performed only stdlib syntax, source-hash and metadata checks.
No server, array, checkpoint, numerical module or scientific outcome was opened.
