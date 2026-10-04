# Exploratory decision overlap on completed Collab TEST

The private predictor with four routes corrects more native model of width 64 misses than it introduces in each of the five seeds. This is a real difference in the already served hit decisions. Against the independent four-model ensemble, however, it introduces more misses than it corrects in four of five seeds. The decision diagnostic therefore supports a limited improvement over that single model of width 64 baseline and does not establish superiority to the stronger ensemble.

Private and independent ensemble positive served-score correlations are high (0.974–0.990). Their threshold decisions still differ in both directions. These pooled score correlations cannot identify internal member diversity, collapse, specialization or a causal training mechanism. The parameter-comparable native model of width 70 control also remains in all-pair results; the private predictor has a favorable decision-count contrast in only two of five paired seeds.

This is post hoc analysis of an already consumed TEST family on one graph/time split. Query decisions are dependent, and no significance or independent confirmation is claimed. The five seeds and all ten pairs per seed are retained in RESULT.json. No combined/oracle predictor was scored, no objective or hyperparameter changed, and original scores remain untouched.

## Frozen primary pair tables

Candidate is `factorized_private_4` throughout. Rescue means candidate hits a positive query the control misses; introduced miss means the converse. All proportions in RESULT.json use all 46,329 positive queries.

| Control | Seed | Rescues | Introduced misses | Positive Pearson | Negative Pearson |
|---|---:|---:|---:|---:|---:|
| native_single_64 | 0 | 1425 | 882 | 0.9507 | 0.8536 |
| native_single_64 | 1 | 826 | 568 | 0.9652 | 0.9371 |
| native_single_64 | 2 | 1139 | 960 | 0.9666 | 0.9216 |
| native_single_64 | 3 | 838 | 262 | 0.9588 | 0.8538 |
| native_single_64 | 4 | 1044 | 635 | 0.9798 | 0.9228 |
| independent_native_4 | 0 | 559 | 865 | 0.9782 | 0.8078 |
| independent_native_4 | 1 | 534 | 767 | 0.9854 | 0.9441 |
| independent_native_4 | 2 | 579 | 956 | 0.9796 | 0.9400 |
| independent_native_4 | 3 | 698 | 374 | 0.9739 | 0.8850 |
| independent_native_4 | 4 | 416 | 609 | 0.9902 | 0.9630 |
| factorized_pooled_after_clamp_4 | 0 | 878 | 844 | 0.9498 | 0.9418 |
| factorized_pooled_after_clamp_4 | 1 | 636 | 718 | 0.9729 | 0.9367 |
| factorized_pooled_after_clamp_4 | 2 | 784 | 1059 | 0.9653 | 0.9348 |
| factorized_pooled_after_clamp_4 | 3 | 1058 | 342 | 0.9531 | 0.8641 |
| factorized_pooled_after_clamp_4 | 4 | 753 | 628 | 0.9664 | 0.9583 |

## Execution

One CPU attempt on the authorized 18.77 project repository used the pinned Python 3.12 runtime, PyTorch 2.7.1, NumPy 1.26.4, two threads and hidden CUDA. Diagnostic wall time: 1.929 seconds; process CPU time: 2.125 seconds. All 25 saved-file/tensor digests, full finite FP32 shapes, inherited query-order references and original hit counts passed. No dataset files, checkpoints, labels, models, forwards or updates were opened/executed. Raw arrays remain on the server.
