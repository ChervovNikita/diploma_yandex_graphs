"""Strict sampled RSS from the same Linux stat snapshot as process identity.

Retains the owned-session cap semantics. Linux may release a task's mm before
its visible state reaches Z; a valid R/S task with zero RSS is accepted.
"""
import os
from pathlib import Path


def parse_stat(raw, expected_pid, page_size):
    opening, closing = raw.find('('), raw.rfind(')')
    if opening <= 0 or closing <= opening or int(raw[:opening].strip()) != expected_pid:
        raise RuntimeError('Malformed or different proc stat PID/comm.')
    fields = raw[closing+2:].split()
    if len(fields) < 22 or len(fields[0]) != 1 or not fields[0].isalpha():
        raise RuntimeError('Malformed proc stat fields/state.')
    group, session, start, rss_pages = (int(fields[index]) for index in (2,3,19,21))
    if min(group,session,start,rss_pages) < 0 or page_size <= 0:
        raise RuntimeError('Invalid proc stat identity/RSS field.')
    return dict(pid=expected_pid,start_ticks=start,group=group,session=session,state=fields[0],
                RSS_bytes=rss_pages*page_size)


def members_of_session(session):
    members = []
    page_size = os.sysconf('SC_PAGE_SIZE')
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            row = parse_stat((path/'stat').read_text(),int(path.name),page_size)
        except FileNotFoundError:
            continue
        # Every retained identity/RSS pair came from a single stat snapshot.
        # Malformed snapshots fail above; they are never silently omitted.
        if row['session'] == session:
            members.append(row)
    return members
