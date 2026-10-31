# Independent model source review

Reviewed 1 October 2026. This is an engineering review, with no requested scientific verdict or novelty criterion. The reviewer read the complete model, analytic test, interface, README and immutable author model source. The reviewer did not import or execute models, run tests, acquire data, use SSH, compile PDFs or recalculate original scores. Writes are confined to this new review directory.

## Source conclusions

No concrete functional defect was identified in the inspected model implementation.

- **Original and identity controls:** `models.py:182–206` preserves frozen input/dropout/GELU, pre-normalized residuals and output projector ordering. The original path delegates its residuals to the frozen implementation. The custom path reproduces the default homogeneous SAGE neighbor/root sum, concatenation and FF order (`frozen_models.py:54–67`, `14–17`). Factor and bias-only initially reproduce the original function because interior factors start at one and private bias rows copy the original biases. The deliberately retained class axis is an interface adaptation when C=1 (`models.py:205–206`), so tensor shape is not identical to the frozen `.squeeze(1)` in that case.
- **Gather orientation:** `models.py:92–105` implements the stated column-vector map `D_s P_out W P_in D_r h + b_m`, with `P[i,index[i]]=1`. The input scale precedes its gather; the output gather precedes its scale. The bias is added afterwards. The asymmetric numeric fixture and folded matrix reference genuinely test this orientation (`test_analytic.py:34–49`, `90–104`).
- **Hidden-only scope:** only SAGE `lin_l`, `lin_r` and FF `linear_1`, `linear_2` are wrapped (`models.py:164–180`). Input/class projectors stay original BE blocks. FF1's 2h input is the concatenated hidden state/message, rather than the raw feature axis. Common affine LayerNorm and identity skips are deliberately unpermuted; this is a non-gauge composition, not a coherently transported network relabeling.
- **Biases and parameter sharing:** SAGE root remains bias-free; other eligible biases become private rows (`models.py:65–69`). Every context retains the original `weight` Parameter (`61`), so optimizer registration and gradient accumulation use one common matrix. Private context rows are explicit. The tests check a shared gradient against the mean of separately differentiated members and inactive private rows (`test_analytic.py:189–207`).
- **Initialization and buffers:** common CPU construction seeds reproduce the same TABM base; forked RNG preserves the caller's CPU RNG (`models.py:296–315`). A separate local CPU permutation generator creates fixed int64 buffers (`78–90`), which participate in `state_dict`. Identity permutations/factors safely omit constant tensors and gathers. The checkpoint restoration and alternate-permutation-seed tests cover those contracts.
- **Storage arithmetic:** for t=int(h×multiplier), the ordinary block count `2h²+3ht+4h+t`, context shared block count `2h²+3ht+2h`, private biases `K(2h+t)` and factors/indices `K(7h+2t)` agree with the source, including all affine norms and SAGE bias flags (`models.py:332–360`). Boundaries add shared dh+hC and private K(d+3h+2C); the final norm adds 2h. The protocol's published PF/S/U/H byte calculations agree with these counts. `storage_report` distinguishes registered tensor bytes from allocation-deduplicated storage and includes checkpointed indices (`363–404`).

## Qualification evidence and limits

Root separately admitted and ran the prescribed CPU fixtures. `coordinate_ensemble_root_v1/qualification_run01/RESULT.json` records return code 0, no real dataset read and no benchmark fit. `unittest_stderr.txt` records all 10 tests passing. The reviewed evidence SHA256s are recorded in the final review manifest.

The tests include an independent dense mean operator with duplicate edges and an isolated node, a folded effective matrix reference, and a direct comparison to an unmodified frozen SAGE residual. These are meaningful independent parity checks, rather than comparisons between two wrappers that share the same custom forward. They qualify the tested CPU/default homogeneous SAGE route. They do not qualify CUDA determinism, dataset/provider acquisition, benchmark optimizer fairness, deployment profiling or performance. Train-mode paired dropout parity is inferred from source order and RNG preservation rather than directly tested. A full folded-reference gradient test at non-unit interior factors is also absent; no corresponding source defect was found.

Calling `conv.propagate(..., x=(normalized, normalized), size=None)` (`models.py:195–197`) bypasses `SAGEConv.forward`. The explicit guard rejects projection, normalization, non-mean aggregation and disabled root weights (`161–163`); the source promises only homogeneous edge-index graphs. Root's passing CPU graph/residual fixtures establish that the tested installed PyG generated propagate signature and default aggregation route work. Other PyG versions or graph formats require separate qualification; their failure is not established by this review.

## Inspected model source hashes

| File | SHA256 |
|---|---|
| coordinate_ensemble_source_v1/models.py | a74a87dc26b7675a2d3fa0aaf0e7734b4786fa6c49411ef52f9bb21c0516dbc1 |
| coordinate_ensemble_source_v1/test_analytic.py | e479b3bf8f2ec127a7be4455f52af24302bbb19133f8e08bbe06e01e594e2499 |
| coordinate_ensemble_source_v1/protocol_interface.json | 128f694d45f9cebd86489b9ad77ba09deaf2b007c764db4a8323981d73902bde |
| coordinate_ensemble_source_v1/README.md | 32c35e6fb626fdb21f8168330c014f13a74bf216dd1ec31021907cefd0630c35 |
| propagation_cost_impl_v1/_pinned/frozen_models.py | 07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303 |

The runner/protocol review is finalized separately in `REVIEW.md`, against runner SHA256 `0b809a2edc4f05a9bd7ba4ea0193b4473a8965168c5c0fc6e9cdd21a2c8df679`. The retained runner snapshot was updated once to that final source after the author declared it stable. Intermediate findings are explicitly marked resolved in the final review.
