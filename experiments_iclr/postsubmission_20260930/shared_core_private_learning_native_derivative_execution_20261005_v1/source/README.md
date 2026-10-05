# Source-only derivative qualification packet

Start with [REPORT.md](REPORT.md). `qualify_train_only_sgd.py` is unexecuted source for one discarded full-shape CPU SGD operator qualification. `PARAMETER_PARTITION.json` pins 47 native parameter names/shapes. `JOB_TEMPLATE.json` is disabled and contains no authenticated input/runtime/host/seed binding.

`sources/` contains reference-only public sparse operator source; `SOURCE_RETRIEVAL_RECEIPTS.json` records retrieval and explicitly disclaims installed-runtime equivalence. `STATIC_VERIFICATION.json` and `SOURCE_MANIFEST.json` record stdlib source checks and file hashes.

No execution or scientific fit is admitted by these files. Root review is the next action.
