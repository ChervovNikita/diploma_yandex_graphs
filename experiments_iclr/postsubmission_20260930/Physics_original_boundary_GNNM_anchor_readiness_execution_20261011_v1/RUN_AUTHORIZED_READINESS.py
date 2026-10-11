import pathlib,subprocess,shlex
remote=pathlib.Path(__file__).with_name('REMOTE_ACTION_DEPLOY_REPAIR_V3.py').read_text()
raise SystemExit(subprocess.call(['ssh','-tt','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-p','2222','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(remote)]))
