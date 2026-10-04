# Bounded fabricated component check

Root separately authorized one CPU component check on 18.77 with CUDA hidden. This source preparation did not run it. `component_check.check_components` constructs six nodes, nine synthetic observed records and three query pairs. It performs no optimizer step or fit and accepts no dataset, checkpoint or output path.

The exact supplied helper bindings are:

| Argument | Existing source |
| --- | --- |
| `mods["native"], mods["native_utils"]` | Return values of bound `graph_ncNC_member_completion_qualification_preparation_20261003_v2/native_reference.native_modules()`; native model source SHA256 `5b6cfe26074c83d57900f589b37a4ef57fc7b8eb6dff3179fc52e3f6943aa5e3`, utils source SHA256 `29df3b9cf82ae5885ece0fac010b81383f1fb5391068649c88b7d2bc98ad86be` |
| `mods["design"]` | `graph_ncNC_collab_predictive_pilot_design_20261003_v1/design_spec.py` |
| `make_native` | `exact_cb_support_bucket_paired_predictive_preparation_20261004_v1/pilot_model.make_native`; its `seed_all` and unchanged N64/Adam constructor are already in that module |
| `graph_ops` | `graph_ncNC_member_completion_qualification_preparation_20261003_v2/graph_ops.py` |
| `adjacency_factory` | Existing `ncnc_cardinality_single_comparator_implementation_preparation_20261003_v1/cardinality_model.native_adjacency`; it constructs a `torch_sparse.SparseTensor` from the supplied fabricated graph only |
| `teacher_type` | `exact_cb_support_bucket_paired_predictive_preparation_20261004_v1/pattern_teacher.ObservationTeacher` |
| `loss_core` | `exact_cb_support_bucket_paired_predictive_preparation_20261004_v1/conditional_loss.py` |

`SOURCE_BINDINGS.json` pins these exact source bytes. A root caller authenticates them, uses the existing deferred native loader, imports these modules from their exact directories, and calls:

```python
result = component_check.check_components(
    mods, graph_ops, pilot_model.make_native,
    cardinality_model.native_adjacency,
    pattern_teacher.ObservationTeacher, conditional_loss)
```

Set CUDA visibility before any Torch/native import. Stop on a failed assertion; the function does not retry, lower tolerances or change the model. PyTorch's default FP32 `assert_close` tolerances are used. The two scalar parity calls reuse one encoded feature matrix per mode and restore the same CPU RNG before native versus adapter target decoding. Training parity enables the auxiliary trunk's autograd; this checks that adding the live head preserves native target/dropout work. Native-versus-portable common/left/right coordinates are checked directly on the fixture. The adapter uses native right-residual-before-left accumulation order.

One auxiliary-only differentiation checks finite nonzero gradients at the actual native encoder and xlin/xcnlin/xijlin/lin scorer tensors plus the new head. A main-only differentiation checks absence of gradients through native completion scores and the auxiliary head. `autograd.grad` does not populate parameter `.grad` or create Adam state. The result is a bounded component-correctness observation; it supplies no native-size, memory, training, validation or predictive qualification.
