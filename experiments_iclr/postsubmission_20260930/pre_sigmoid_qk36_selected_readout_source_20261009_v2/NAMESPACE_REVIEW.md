# Verified origin namespace, preserved receipt bytes

Root identified a real V1 blocker: allocation receipt paths begin `/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/...`; GPU77 paths begin `/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/...`. Resolving a foreign receipt string against the serving filesystem cannot establish its original identity and rejects legitimate byte-preserved staging.

V2 separates the recorded origin from the local artifact lookup. A recorded path must exactly equal `PurePosixPath(original_route['phase']) / expected_relative_artifact`. The expected suffix is generated from the fixed plan/cell key or bound original release path, never selected from the receipt. Only then does `g.inside(expected_relative_artifact)` confine and read the local copy. SHA256/size and original source/cell/runtime/role/exit checks still establish its content. The mapping cannot choose a different seed, operator, family, cost or release.

| Receipt field | Exact original-route suffix | Local file used after namespace validation |
|---|---|---|
| `handle.output` | Frozen output root / route / scientific cell key | Same suffix under serving phase |
| `handle.release_path` | Frozen root release binding path | Exact hash-bound staged original release |
| `handle.cost` | Frozen output root / route / costs / original indexed cell filename | Exact original exit-bound worker cost |

All original JSON bytes remain unchanged, including owner contexts inside selected/own snapshots and original provider/module paths. Their hashes are checked directly. The fresh root namespace receipt binds all three original route roots and the selected serving route. This source creates no fake original directory, mount, namespace or symlink. Its mapping rejects an unexpected original canonical prefix; a discrepancy needs actual origin evidence and a separately reviewed mapping change, not a rewritten receipt.

Review of reused validators: `verify_union` uses plan/receipt fields and has no filesystem resolution. `work_receipt` accepts an explicit cell path, reads its endpoint and hashes its completion, and reads the local immutable routing seal; original `owner_context.runtime` remains compared to the original route catalog. The new caller supplies the staged cell path and unchanged original route metadata. No callback or code changes to either validator are needed.

Static verification exercises all nine source/serving route combinations using only strings from the frozen route catalog. Wrong origin paths and traversal suffixes are rejected. It also AST-checks that neither reused validator calls `resolve`. No original running output, real receipt, selected state, array, metric or numeric provider was accessed to construct these metadata checks.
