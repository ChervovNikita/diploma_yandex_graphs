# Attribution and adaptation notice

Copyright 2019 PaddlePaddle Authors. Apache License, Version 2.0; the unmodified
license is included as `LICENSE`.

This packet derives its operations and schedule from PaddlePaddle/PGL commit
`6dbb47c4559352ea1b1e327ee0039c47095583af`, directory
`ogb_examples/nodeproppred/ogbn-arxiv/unimp_appnp_vnode_smooth`. Exact original
driver/model/helper/utility/license bytes and their preserved local paths are
bound in `SOURCE_BINDINGS.json`; originals remain unmodified.

Changed files: new current Torch implementation, explicit receiver ordering,
WikiCS feature/class dimensions, isolated seeded mask stream, VALID-only earliest
accuracy selection and selected-state replay, numerical/runtime release guards,
and fresh source documentation. No original notice or copyright is removed.

The float .1 finite pre-softmax edge mask is a prospectively chosen adaptation.
The original int64 attention-drop feed supplied with CLI .1 has unverified
legacy acceptance/coercion. Neither this repair nor the current Torch initializer,
optimizer/RNG/reduction execution is claimed identical to the legacy Paddle
runtime. Source custody/static parsing does not establish numerical reproduction,
benchmark competence or scientific superiority.
