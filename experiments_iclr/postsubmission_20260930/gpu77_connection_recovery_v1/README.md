# 18.77 connection recovery

The local direct SSH probe timed out. The linked Mac name did not resolve on this network. The existing MacLink controller is connected, but its SSH transport uses the prohibited seven-GPU account. No remote command was sent through it.

`relay-onegpu.json` is a prospective replacement using the authorized one-GPU login at both endpoints and a distinct relay port. Its first startup was refused by strict host-key verification. The prospective configuration now pins the unique preexisting trusted ED25519 entry used by the working authorized one-GPU endpoint. It contains no private key or password. It preserves the identity-file references; the linked Mac must have authentication for the authorized login.

A direct/VPN route to the linked Mac or a MacLink connection through an authorized relay is needed before 18.77 can be inspected. The existing controller has not been modified.

Once connected, verify the saved project Git root `/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning` and the actual two GPU identities before writes. Sync an exact published Git commit into an isolated project branch. Keep its run registry and environment separate from the active one-GPU study. Assign different experiments to the two devices.

## Current state

The replacement controller is now running through the authorized one-GPU relay on remote port28444 and local port8444. A wrapper quotes the SSH known-hosts option because the allowed project path contains a space; strict host-key checking remains enabled. Its private TLS state is in `.ml77` and is excluded from the publication inventory. The other Mac has not paired with this replacement. No18.77 password was attempted and no18.77 commands ran.

On the other Mac, use `relay-onegpu.json` with `maclink.py client --relay --relay-config <path/to/relay-onegpu.json> --config <new-private-client-config.json> --pair --keep-awake`. Keep the existing client credential untouched by specifying a new config. Root must provide a fresh five-minute pairing code when ready.

## October 2 update

The user explicitly authorized the existing seven-GPU account solely as the MacLink forwarding relay and designated the linked Mac Desktop SSH password file. That route authenticated to18.77. Two A10080GB physical GPU UUIDs are verified. Existing activity is present; the user explicitly permits our jobs alongside it. No unrelated job was stopped. The parent project workspace has no Git metadata, so an exact checkout of the published research branch was created in its postsubmission_git subfolder. Existing parent files are preserved. No seven-GPU filesystem or compute command is permitted or used. The earlier failed direct/authentication attempts and unused replacement relay are retained as history.
