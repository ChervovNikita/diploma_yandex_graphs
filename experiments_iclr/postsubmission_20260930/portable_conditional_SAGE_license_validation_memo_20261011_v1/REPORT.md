# Conditional SAGE source redistribution requirements

The sealed portable package has complete local source bindings but does not yet have a complete release license binding. The only retained grant verified here is the PyG 2.6.1 MIT license. The pinned GNNBenchmark tree confirms that an upstream LICENSE exists and provides its exact blob locator; the license text is not retained in the inspected carrier. Resolve the four source release items below before describing the package as cleared for public redistribution.

## Source release actions

| Item | Concrete evidence and issue | Required release action |
| --- | --- | --- |
| Package and local donor code | The package NOTICE assigns no new license. Its local owner, helper, route, factor, adapter and readout bindings establish file identity, with no copyright holder or release grant. | Identify the holder for locally authored code and record the chosen public release license and copyright notice. Confirm that each local donor is covered or has its own retained grant. Apply this to a new release version, preserving the sealed version. |
| Native wrapper snapshot | `models.py` and `run_common.py` are exact local copies at head `47c4af6228c52aa450ab8316efe5c7cfc33119a4`. SOURCE.json contains hashes, with no license or repository grant. | Retain the repository LICENSE and applicable notices at that head, or obtain a holder declaration covering these two files. Do not infer MIT terms from their PyG imports. |
| GNNBenchmark source derivation | The author Torch port is bound to semantic readings of the pinned TensorFlow source and helper. Saved tree `1e72912a0810cdf27ae54fd589a3b43358a2b161` has `LICENSE`, blob `7f42f634ba028940c8823b63650fe54b96d1017a`, 1124 bytes, and `truncated=false`. The text is absent from the retained source carrier. | Retain that exact LICENSE text, verify its Git blob and size, and determine its notice and modification requirements for the source translation. Include the applicable full grant and attribution in the next release. The mathematics and paper citation alone do not decide the code license. |
| PyG preparation source binding | The preparation code expressly retains the pinned reader semantics from PyG commit `76ff9c2ce18c8cebf52122b57e2aeadce9793d10`. Retained MIT text is tied to tag 2.6.1, while the required runtime is 2.7.0. | Bind retained license text to the provider commit for the preparation source lineage. Retain the exact runtime license with the installation receipt if the release supplies runtime artifacts. Carry the applicable copyright and permission notices for any source portions covered by the grant. |

The GNNBenchmark license retrieval locator is `https://api.github.com/repos/shchur/gnn-benchmark/git/blobs/7f42f634ba028940c8823b63650fe54b96d1017a`. This locator comes from the saved authoritative tree. No new retrieval was made. The missing local text is a evidence gap, not a claim that the upstream repository has no license.

## Retained PyG MIT grant

The saved license receipt binds `https://raw.githubusercontent.com/pyg-team/pytorch_geometric/2.6.1/LICENSE` to 1067 bytes and SHA256 `89cfb6edc309735916a0c1189dccf761add4af6062ddb34b5a26f60d3efadfea`. It grants use, copying, modification, publication and distribution subject to inclusion of the copyright and permission notices in all copies or substantial portions. Its full warranty disclaimer is preserved with the grant in PYG_2_6_1_LICENSE.txt.

This verifies the recorded tag 2.6.1 grant. It does not establish an exact-version grant for tag 2.7.0, the provider source commit, the native wrapper repository, or the GNNBenchmark source.

## Installed dependencies and dataset artifacts

The source package installs NumPy 1.26.4, SciPy 1.11.4, Torch 2.1.2 with the required CUDA 11.8 build, PyG 2.7.0, torch-scatter 2.1.2 and torch-sparse 0.6.18. Whole distributions are not bundled. Their package grants and bundled third-party notices should accompany an installation or distribution receipt when runtime artifacts are supplied; the current source bindings are not a wheel or container redistribution audit.

The raw Physics archive is identified by commit `1e72912a0810cdf27ae54fd589a3b43358a2b161`, Git blob `8d8bc12e29aa67f6753a712080eae81079bd7d01`, SHA256 `e4d68468eba5fb8f6b280ffd406bd53e6e4295ba1dd2967d6fe99b978e4b3375` and size 16374005 bytes. It is downloaded or supplied by the caller; neither it nor prepared arrays are included in the source package. A public URL and a code license do not establish dataset redistribution terms. Resolve the dataset terms before a later release includes raw or prepared dataset artifacts.

## CLI check and package preservation

The completed command `python3 portable_conditional_SAGE_source_20261011_v1/cli.py --help` exited 0 and exposed the prepare, acquire, export and score commands. This checks CLI help and the top-level stdlib import boundary. It does not check numerical imports, dependency installation, a TRAIN update, checkpoint restoration, export or scoring. CLI_HELP_CHECK.json records the exact observation and its limits.

All 29 manifest-bound package files retain their recorded sizes and SHA256 hashes. The package manifest remains `d52b185077d81692472a0bedc002df864c1c7a34f6de55b4e398126c2ce80781`. The separate memo preserves the sealed package. EVIDENCE.json binds the local license, source and metadata files used for these conclusions.

## One runtime validation

PORTABILITY_CHECK.md specifies one fresh-environment E4 update, checkpoint round trip and label-free forward using the portable binding. Root executes it. A successful result would resolve the most immediate import, extension, E4 transport and checkpoint integration risk. It would leave the native author reference, complete roster, calibration and scorer integration unverified.
