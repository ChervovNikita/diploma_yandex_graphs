# Restored authorization: MacLink connection state

6 October 2026, Europe/Moscow. Current user authorization restores access to shmelev@192.168.18.77. Historical withdrawn-access notes and earlier receipts remain preserved. The seven-GPU allocation is permitted solely for MacLink forwarding.

The saved paired local controller was absent. I reopened the existing MacLink daily connection, preserving pairing/settings and filtering sensitive startup output before printing or logging. Its SSH forwarding relay started successfully and the local controller remains alive on127.0.0.1:8443. The paired other Mac has not connected. One bounded direct18.77 TCP22 probe timed out after5 seconds without authentication. This does not establish that18.77 is down or that its SSH/credentials fail: target hostname/repository/GPU/process metadata remain unverified in this attempt.

The missing external step is to run the existing listener on the already paired other Mac with its required network/VPN route, using QUICKSTART_RELAY.md:

```sh
cd "$HOME/Downloads/reverse_maclink_daily"
maclink-env/bin/python maclink.py listen --keep-awake
```

Keep that Terminal and lid open. The current controller is already waiting; no pairing code, reset, setup, settings change or password guess is needed.

The expected target from prior authenticated receipts is peptide, with project Git root/cwd /disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git and GPU UUIDs GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998 and GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced. These are historical expectations, not fresh observations. Once the paired Mac connects, the saved secure Expect route can read its previously designated credential file internally and issue only the authorized target identity/aggregate-GPU/previously-owned-handle checks.

No target password value was read here. No private-key bytes were printed/copied/logged and no pairing value was retained. No seven-GPU command/query/filesystem exploration or target filesystem/GPU query was performed. No fit, source/runtime change, stop/restart/cleanup, host setting, sudo, mount guard or process signal was issued. The controller remains running under the recorded local PID and driver session72073. Its sanitized live log is excluded from the seal because it continues to grow; the fixed connection receipt and source bindings are sealed.
