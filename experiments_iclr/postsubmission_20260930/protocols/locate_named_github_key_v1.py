"""Inspect only filenames for the user-named Git credential."""
from inspect_git_for_publication_v2 import REPO, LOGIN, UUID, run
import json
import os
from pathlib import Path

if Path.cwd().resolve() != REPO or os.environ.get('GNNM_SSH_DESTINATION') != LOGIN:
    raise ValueError('Wrong repository or calling route')
if run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']).splitlines() != [UUID]:
    raise ValueError('Wrong allocation')

matches = []
for directory, dirs, files in os.walk(REPO, followlinks=False):
    dirs[:] = [name for name in dirs if not Path(directory, name).is_symlink()]
    for name in files:
        if name in ('the_github', 'the_github.txt'):
            matches.append(str(Path(directory, name)))
# These exact files are the named credential, not general home-directory searches.
for candidate in (Path('/home/jovyan/.ssh/the_github'), REPO.parent/'the_github', REPO.parent/'the_github.txt'):
    if candidate.is_file():
        matches.append(str(candidate))
print(json.dumps({'named_key_paths_only': sorted(set(matches)), 'key_contents_read': False}, indent=2))
