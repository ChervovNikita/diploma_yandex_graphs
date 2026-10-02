# Provenance and read accounting

Created on 2 October 2026 in the local project. Acquisition used public HTTPS with Python standard-library `urllib.request`, not a dedicated web-search tool. Retrieval logs include UTC timestamps, returned URLs, HTTP failures, bytes and SHA-256. No remote datasets or model artifacts were requested.

Google discovery responses were successful HTTP responses containing redirect shells. Bing responses supplied no result or irrelevant broad-token matches. They are retained as unusable discovery evidence and do not certify a literature absence. Known-title direct arXiv metadata provided the authoritative title, authors and current v2 identities. The early guessed v3 requests failed with 404 and were superseded by exact metadata-linked v2 URLs.

Local arXiv HTML was converted to text with the saved standard-library `extract_primary.py`. Math uses the primary's `alttext`; paragraph IDs and ancestor IDs are retained. An initial ad hoc depth parser mishandled HTML void elements, so its intermediate extracted files were replaced before use by the saved DOM parser. An attempted BeautifulSoup import failed because the library was absent; nothing was installed. Extracted text is a navigation aid. The primary HTML bytes and their retrieval hashes remain the evidence.

The paper-linked NTKGP repository was resolved through GitHub's public API, then all author-source bytes were retrieved using the immutable commit path. The notebook, plots and other repository paths were not executed. Only the paths/lines in `READ_SCOPES.json` count as source reads. Source was neither imported nor used to train.

The report reuses the bound root analysis instead of presenting its first-order cancellation, Hessian or Jensen discussion as new theory. Prior paper conclusions keep their inherited scoped/full/unavailable status. The scoped conclusion for PENCIL is a current competence reference, not a reproduced score. The v16 memory was not edited.

Two new primary papers received scoped method reads; zero full-paper reads occurred. No third paper was acquired. Source and artifact hash verification is file verification only. No scientific result, measured cost, supported AD backend, current device identity, remote repository identity or original-score recalculation is asserted.
