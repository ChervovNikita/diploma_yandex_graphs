# Resume the existing release with small plaintext commands

This local helper pins frozen coordinator v2, reuses its existing proof directory
and remote backend deployment, and resumes only candidate SHA256
`065b2574147b257ab7aa47163eaaa475f7b8a0f6df1f4a6378b4a43f30d8c4f4`.
The allocation delivery already succeeded. The server51 transfer failed during
a large SSH multiplexing command; its old partial incoming file is preserved.

The only transport change is server51 plaintext heredoc blocks of at most 8,000
bytes, with complete guarded SSH commands capped at 12,000 bytes. Each staging
attempt creates a new unique incoming file and performs the same hash check,
atomic no-overwrite installation and plaintext readback as v2. Allocation uses
the frozen v2 SCP implementation. No backend restaging, route modification,
owner operation or marker regeneration is included.

Before running, the dispatcher inspects current remote delivery state and the
old partial attempt. The helper requires the existing candidate and completed
validation receipt unchanged. Frozen `Coordinator.release` performs its normal
fresh verification, reuses that validated proof, and delivers identical bytes.
An identical allocation marker is accepted without replacement. Cross-host
delivery remains sequential; preserve receipts and inspect after any failure.

```sh
python3 -B resume_release.py \
  --source-sha256 "$RESUME_SOURCE_SHA" \
  --confirm-release-sha256 065b2574147b257ab7aa47163eaaa475f7b8a0f6df1f4a6378b4a43f30d8c4f4
```

The local test reconstructs exact text from generated bounded commands and
checks that retries use distinct temporary paths and the original v2 proof.
It makes no remote calls or numerical fits.
