# Molecular grouping preparation v6

V5 passed its19 synthetic fixtures, then failed the full official-training export at an unnecessary second molecule-sanitization pass on a Murcko scaffold. No bundle or probe was produced. V5 and the unlaunched CPUv3 packet remain sealed; the failed exporter supervisor is preserved here.

Molecule preparation remains exactly v5: explicit parser `sanitize=False, removeHs=True`, permissive property cache, the pinned Cookbook partial flags and standard ADJUSTHS. Scaffold preparation now uniformly follows the primary `GetScaffoldForMol` active branch: `MurckoDecompose`, `ClearComputedProps`, `UpdatePropertyCache`, `GetSymmSSSR`. Only the cache argument is `strict=False` for the already documented metal-valence inputs. The primary C++ topology operation provides its aromatic heteroatom H treatment. No second molecule sanitization, per-atom rewrite, exception singleton or graph deletion exists.

The fixture gate retains all19 cases and adds seven fused heterocycles, including indole, carbazole and benzimidazole with N-substituted partners, plus quinoline. Every strictly parseable fixture must match standard RDKit molecule/scaffold keys and round-trip under this uniform procedure. Every actual train molecule must match the original frozen OGB atom order/atomic numbers and inverse-edge connectivity; all model features remain untouched. Any failure aborts the entire export.

## Root runtime gates

With the separately pinned RDKit2025.3.6 runtime, run `verify_chemical_groups.py --report engineering/CHEMICAL_GROUPS_run01.json`, then the unchanged mapping and synthetic autograd gates. Re-run `make_train_bundle.py` against the exact existing frozen training split, mapping and tensor cache with a new v6 output path. If it succeeds, run the unchanged real128-graph autograd gate and prospective freeze. Root owns every remote launch. No CPU packet is rebased or launched until the actual complete bundle passes those gates.

`source/` retains byte-identical numerical and GINE sources; probe recipe, labels, grouped allocation and selector stay unchanged. Molecule/scaffold groups are structural identity only, not a chemical-validity or provider-era exact chemistry claim. The exporter still materializes the existing all-target torch storage and can internally materialize public heldout CSV fields; it selects and exports only official-training targets and training-row SMILES. This does not certify test-blind field storage. No heldout score is read.

Primary evidence: pinned `MurckoScaffold_RELEASE_2025_03_6.py.txt` and `ChemTransforms_RELEASE_2025_03_6.cpp.txt`, together with the earlier pinned parser/ADJUSTHS sources. Exact failed preparations and diagnostics remain hash-bound. Source checks establish code identity only; the actual full training export and topology gates are required.
