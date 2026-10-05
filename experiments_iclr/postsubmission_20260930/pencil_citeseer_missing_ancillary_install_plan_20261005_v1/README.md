# PENCIL missing ancillary dependencies: ordinary install plan

## Precise change

The actual one-GPU inventory is Python3.11.14 and reports seven missing direct ancillary packages. Pin them to retained versions: transformers4.46.2, tokenizers0.20.3, huggingface-hub0.36.2, safetensors0.8.0, wandb0.18.7, rootutils1.0.7 and rich14.0.0. `requirements.in` contains exactly these seven, with no optional extras. Rich14.0.0 is the retained77 version rather than the broad author requirements14.1.0; this is a declared version choice, not an established incompatibility. Torch2.1.2 is not rejected merely because the author used2.5.1: the later native import/operator qualification decides compatibility.

The observed fourteen existing packages, including NumPy1.26.4/Torch2.1.2+cu118/PyG2.7.0/sparse+scatterpt21cu118, remain exact. Their constraints are in `protected_constraints.txt`. Retain existing scipy1.14.1, sklearn1.5.2, networkx3.6.1, pandas2.2.3, psutil7.2.2 and packaging26.2; the77 versions do not override these. No broad author requirements installation, optional HuggingFace datasets package, Torch upgrade, NumPy2 or original environment mutation is needed.

## Transitive closure

Retained77 resolver metadata supports the exact absent-version preferences hf-xet1.6.0, protobuf5.29.6, docker-pycreds0.4.0, GitPython3.2.0, gitdb4.0.12, smmap5.0.3, python-dotenv1.2.4, regex2026.9.29, sentry-sdk2.71.0 and setproctitle1.3.8. **Use these only when the package is absent.** Existing compatible providers should keep their selected versions. The input inventory did not enumerate all these transitive packages, so their absence is not claimed.

The saved requirements show these direct relations:

| Parent | Required dependency closure |
| --- | --- |
| Transformers4.46.2 | filelock; hub>=.23.2,<1; NumPy>=1.17; packaging>=20; yaml>=5.1; regex!=2019.12.17; requests; safetensors>=.4.1; tokenizers>=.20,<.21; tqdm>=4.27 |
| Hub.36.2 | filelock; fsspec>=2023.5; hf-xet>=1.1.3,<2 on x86_64; packaging>=20.9; yaml>=5.1; requests; tqdm>=4.42.1; typing-extensions>=3.7.4.3 |
| Tokenizers.20.3 | hub>=.16.4,<1 |
| Wandb.18.7 | click!=8,>=7.1; docker-pycreds>=.4; GitPython!=3.1.29,>=1; platformdirs; protobuf>=3.19,<6 excluding4.21/5.28 for Python3.11/Linux; psutil>=5; yaml; requests>=2,<3; sentry-sdk>=2; setproctitle; setuptools; typing-extensions>=4.4,<5 for Python<3.12 |
| Rootutils1.0.7 | python-dotenv>=.20 |
| GitPython3.2.0 → gitdb4.0.12 | gitdb>=4.0.1,<5 → smmap>=3,<6 |
| Docker-pycreds.4 / sentry-sdk2.71 | six>=1.4 / urllib3>=1.26.11 and certifi |

The Python3.11 typing-extensions marker is active even though it was inactive in the77 Python3.12 resolver. It must be checked by the fresh resolver. All retained ancillary Python minimums observed in saved metadata admit3.11; cp312 **wheel files** for tokenizers/regex/setproctitle cannot be reused. Pure-Python and abi3 metadata may guide version choice, but all actual Linux3.11 wheels must resolve anew.

Rich was already installed on77, so its transitive wheel versions were not retained in that resolver. Fresh pip resolution will supply any missing Rich dependencies and their metadata; do not invent prior version provenance for them. The resulting wheel set is frozen by URL/SHA/version before installation.

## Normal two-stage pip workflow

Use the qualified private one-GPU interpreter with the same sparse overlay and repository base site-packages PYTHONPATH. Root should first run `prepare_constraints.py`: this reads **only stdlib installed metadata**, keeps the selected versions of every already-installed distribution, applies exact direct pins and retained absent-transitive preferences, and saves `BEFORE_METADATA.json` plus `RESOLVER_CONSTRAINTS.txt` in a fresh phase directory. This expands the partial fourteen-package receipt without model/data imports.

Run ordinary `pip install --dry-run` **without `--target`** so installed selected-path dependencies count as satisfied. Save its complete native Python3.11/Linux report and exit status. `--only-binary=:all:` prevents source builds. Do not request extras, --upgrade or --ignore-installed. The resolver may download metadata/wheels and use normal standard caches. It makes no installation. A genuine dependency conflict can lead to a narrowly reviewed ancillary successor; a version difference alone is not failure.

After a successful resolver receipt, `freeze_report.py` validates the report, requires every exact direct pin, rejects every proposed existing-package/core replacement, and writes a SHA-locked missing-wheel requirements file. Root reviews the actual report and lock. Then ordinary `pip install --target FRESH_REPO_OVERLAY --no-deps --require-hashes` installs that already-resolved closure. The `--no-deps` here avoids a second resolver blindly copying every already-present package into the overlay; it follows successful full dependency resolution against the unchanged base. There is no installation into the original venv.

Do not run the first resolver with `--target`: pip then generally plans dependencies for the target independently of installed providers and can copy NumPy and many unrelated existing packages. Direct target installation of all broad author requirements is unnecessary.

## Exact command argument plan

`PLAN.json` supplies argument arrays and repo-owned paths; `COMMANDS.txt` gives ordinary command forms. The proposed resolver directory and overlay are fresh paths, not created serverside here:

- Resolver evidence: `experiments_iclr/postsubmission_20260930/pencil_citeseer_missing_ancillary_resolver_execution_20261005_v1`.
- Overlay: `experiments_iclr/postsubmission_20260930/pencil_one_gpu_dependency_overlay_20261005_v1`.

After installation prepend only the new overlay to the same old PYTHONPATH. Verify complete installed overlay file inventory, actual installed distribution versions, imported source locations, and unchanged base/core versions. Then run the native PENCIL import qualification before any resource probe. Installation alone establishes no numerical/operator readiness. This plan introduces no mount changes, filesystem namespaces, special cache restrictions, sudo, dataset accesses or training.

## Provenance

`INPUT_BINDINGS.json` binds the exact newly acquired one-GPU inventory and existing77 resolver/PyPI metadata. `RETAINED_77_PACKAGE_METADATA.json` is a compact reuse of version/requirement metadata, not a copied binary overlay or a readiness claim. Both helper scripts were only AST-parsed locally. No network, server connection, pip invocation, installation, model/numerical import or dataset read occurred while preparing this packet.
