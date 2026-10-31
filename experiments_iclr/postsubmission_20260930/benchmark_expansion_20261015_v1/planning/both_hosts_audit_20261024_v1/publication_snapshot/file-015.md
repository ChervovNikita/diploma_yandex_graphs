# Server51 reporting recovery after natural drain

This one-use CPU waiter preserves the currently running server51 GPU1 training.
It polls its pinned queue and original owner every 30 seconds. It cannot signal
or kill a process and cannot change frozen assignments or scientific settings.

Only the exact two expected BUDDY reporting failures may pass the drain gate.
After the existing owner exits and its existing tmux pane is dead, the waiter
runs the reviewed CPU checkpoint audit and reporting-recovery activation once.
Unknown failures, changed ownership, missing panes, or unsuccessful audits stop
without a training restart. The normal recovery owner retains capacity waiting,
two-job concurrency, checkpoints and all original attempt records. A 24-hour
time limit prevents an unbounded dormant waiter. Commands and receipts remain
in the server51 recovery evidence directory.
