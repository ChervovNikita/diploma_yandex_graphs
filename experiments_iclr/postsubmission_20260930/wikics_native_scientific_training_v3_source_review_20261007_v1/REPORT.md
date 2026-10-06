# Native scientific training v3 narrow source review

**Approved for root activation of the three fixed native1100 fits**, seeds17/29/43, with prescribed VALID selection, integrated finite checks, explicit retained-failure waiver and21600/22200-second limits. No diagnostic or TEST execution is approved here. No launch-blocking source/data/training issue was found.

Manifest `aae2dc35bde0339c38857923af1d202cc2e65eeab080c1adc8ee8f5f7d9407d3` and run `c1ab13f5abc08409e893156cbed64ecd87b828b79fb236fe61e9ad9f2a7a288a` were checked against every manifest file hash/size and Python AST. Native engine/vendor are byte-identical to v2. Native constructor, loss, Adam and dropout remain unchanged. Plan preserves local100/global1000, seeds and strict first maximum complete split0 VALID correct count spanning both stages. Transition still restores selected local model+Adam and retains live end-local dropout RNG. TEST payload remains excluded.

Actual epoch still uses the production reference=False path and checks finite loss/all active gradients plus inactive-gradient absence. V3 additionally rejects nonfinite updated parameters/serving probabilities. Local/global smoke receipts describe the existing first ordinary epoch in each stage; they add no optimizer update or diagnostic fit. Final selected replay records any difference without reselection. Prior failed parity checks remain failed through the exact bound engineering policy.

Nonblocking metadata limit: `SELECTED_REPLAY.source_and_checkpoint_hashes_verified=True` does not represent an expected hash comparison before checkpoint load. Source manifest is verified and checkpoint SHA is recorded in terminal FREEZE. Do not cite that field as stronger checkpoint certification. This does not affect the native recipe or selector.

Root must bind actual plan/waiver/review/source/data/output/caps and use existing direct child supervision. Remote acquisition data was not loaded in this review. No numerical imports, extra diagnostics, scientific training, remote launch or held-quality inspection was performed. Proceed with actual fixed scientific training under the user's engineering-priority amendment.
