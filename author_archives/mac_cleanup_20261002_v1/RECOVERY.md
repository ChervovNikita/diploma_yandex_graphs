# Recover the offloaded Desktop evidence

The private archive location, byte count and SHA256 are in CLEANUP_RECEIPT.json. The archive contains the exact EVIDENCE_MANIFEST.json, recovery source, this note and 7,342 SHA256-named objects representing every original file. No private key or passphrase is included.

Copy the archive from the authorized server using your existing SSH credentials. Verify its SHA256 against the local receipt. Recovery requires temporary space for the unique objects plus the reconstructed evidence tree, approximately 14.4 GB.

```sh
python3 -B recover_evidence.py desktop_evidence.objects.tar.gz NEW_EVIDENCE_DIRECTORY
```

Choose a new directory under an existing canonical parent. The recovery program verifies the manifest and every object, reconstructs all 98,611 files and empty directories, and restores stored file permissions and modification times. Existing destinations are refused. Nothing from the archive is executed. Original ownership is replaced by the restoring user's ownership.

The Overleaf folder and ZIP remain local and were not part of the removed evidence tree. Historical evidence is not required to compile that submission.
