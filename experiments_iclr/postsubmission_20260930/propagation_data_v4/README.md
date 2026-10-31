# Products read-only export recovery, version4

The first official acquisition downloaded and processed the complete products graph, then refused a noncontiguous tensor at export fingerprinting. The version3 fresh download later lost its network connection. Both failures remain unchanged.

This version reads only the exact products provider tree and preserved public archive under `propagation_data_v1/products_authorized1gpu_run01`. Before provider reload it verifies the original completed failure chain, exact preserved archive hash, pins the observed installed provider source, and hashes the whole provider tree opaquely. Provider download, processing and interactive update methods are temporarily refused. The original whole-tree file inventory must be identical afterward.

The original official provider reloads its existing processed data, internally materializing all public targets. The exporter selects target values only at official training IDs, preserves the complete graph/features/official split order and copies physical tensor layout to contiguous C order. Original strides and copy flags are disclosed. No validation/test target values are indexed, summarized or scored. No models or GPU context are created.

New outputs stay in a fresh version4 directory. This is an export recovery, not a fresh download or resumed original process. The source schema and train-target-only export contract match earlier versions. Exact file/tensor replay and completed supervision audit remain necessary before scientific fits.

Use the standard corrected allocation environment and run_authorized_v2 supervisor from exact repo cwd, with `--task products --authorize-public-download ogbn-products`; outputs and child `--supervisor` must be new and match the current supervised invocation. Never connect to the forbidden allocation.
