# Exact residual-only + private-BN CPU lifecycle qualifier

Prepared only; **not executed**. The frozen pilot files and parent manifest are unchanged. This directory is a separate engineering artifact and creates no fourth scientific fit.

`qualify_cpu.py` reuses `portable_private_input_v2/qualify_native_cpu.py` for the retained CPU helper, plus the existing `runtime.adapters.create_adapter`, `run_fit.validate_job`/`OptimizerObserver`, RNG and checkpoint helpers. It reads the exact `JOB_seed101.json`, validates its original descriptor, then changes **only the execution device to CPU** in memory. Seed101 is the first fixed planned seed, chosen mechanically for qualification; all three scientific seeds remain required.

## Finite sequence

1. Verify authorized hostname/sole GPU UUID and hidden CUDA before project reads; verify the frozen parent manifest, exact job, helper/runtime bytes, source/data role pins and absence of split overrides.
2. Construct the exact candidate through the existing adapter. Use the first32 real TRAIN graphs. Compare all four initial eval routes bit-for-bit with the same unchanged native body, bypassing the residual while using cloned initial native BN state.
3. Run one no-gradient TRAIN forward through route2. Require only that route's ten BN counters to increment once, other route banks to remain identical, and canonical native buffer objects/values to remain unchanged outside the route. Restore diagnostic BN state and all RNG before training.
4. Perform the **original first optimizer update**, then whole official151-graph VALID. Save adapter/model/BN/optimizer/RNG state to an engineering-only checkpoint.
5. Construct a fresh exact candidate, restore, and require bit-exact whole-VALID outputs and IDs plus native parameter identity.
6. Restore the saved state/RNG again, perform the **original second update**, then whole VALID. Require the native update clock to equal2.

The same32 TRAIN graphs are reused for both engineering updates, as in the retained CPU helper. There is no full-epoch loop or selector call. The engineering checkpoint must never initialize a scientific fit.

## Work and limits

Maximum two native constructions and two optimizer transitions; three complete VALID calls. Expected successful work is **74 native forwards**: five initial parity forwards, one BN isolation forward, eight training-route forwards, and60 validation-route forwards. Actual construction attempts/completions, forward attempts/completions, optimizer attempts/completions, validation attempts/completions, wall/CPU time, peak RSS and checkpoint bytes are recorded even on failure. A failed attempt is preserved; no retry or alternative seed is attempted.

Successful completion qualifies this exact CPU lifecycle only. It establishes no accuracy gain, convergence, full-epoch/selector behavior, CUDA numerical behavior, throughput or transfer. Existing frozen controls remain separately produced evidence. No TEST view is created or scored.

## Root launch after review

`CONTRACT.json` gives exact argv and environment for the existing allocation virtualenv. Root must retain its normal finite owner and route guard. The output directory must be fresh. Source preparation here ran only AST/JSON/hash checks; `SOURCE_CHECKS.json` explicitly records that the qualifier was not run.
