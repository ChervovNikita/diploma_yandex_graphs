"""Proposed replacement sampling helpers; no automatic execution or signaling.

Linux /proc/<pid>/stat field24 is rss in pages. Use the same text snapshot
for state, birth, session, process group and RSS. A task that has released
its mm may still expose state R/S and rss0 before zombie state appears.
"""
import os
from pathlib import Path


def parse_proc_stat(raw, pid, page_size):
    if type(pid) is not int or pid <= 0 or type(page_size) is not int or page_size <= 0:
        raise RuntimeError('Invalid process identity/page size')
    closing = raw.rfind(')')
    opening = raw.find('(')
    if opening <= 0 or closing <= opening or int(raw[:opening].strip()) != pid:
        raise RuntimeError('Malformed or mismatched /proc stat identity')
    fields = raw[closing + 2:].split()
    if len(fields) < 22 or fields[0] not in {'R','S','D','Z','T','t','W','X','x','K','P','I'}:
        raise RuntimeError('Malformed /proc stat fields/state')
    group, session, start_ticks, rss_pages = (int(fields[index]) for index in (2,3,19,21))
    if group < 0 or session < 0 or start_ticks < 0 or rss_pages < 0:
        raise RuntimeError('Negative /proc stat identity or RSS')
    return dict(pid=pid,start_ticks=start_ticks,session=session,group=group,
                state=fields[0],RSS_bytes=rss_pages*page_size)


def process_identity_with_rss(pid):
    try:
        raw = (Path('/proc')/str(pid)/'stat').read_text()
    except FileNotFoundError:
        return None
    return parse_proc_stat(raw,pid,os.sysconf('SC_PAGE_SIZE'))


def members_of_session(pid):
    members=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        item=process_identity_with_rss(int(path.name))
        if item is not None and item['session']==pid:
            members.append(item)
    return members
