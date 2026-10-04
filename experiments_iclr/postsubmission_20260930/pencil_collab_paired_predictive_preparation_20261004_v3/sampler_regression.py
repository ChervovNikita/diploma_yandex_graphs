"""Narrow stdlib regression of the actual v3 RSS sampler; no live proc access."""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def stat_text(pid=101, state='R', group=100, session=100, birth=55, pages=0):
    fields = ['0'] * 22
    fields[0], fields[2], fields[3] = state, str(group), str(session)
    fields[19], fields[21] = str(birth), str(pages)
    return str(pid) + ' (loader (name with spaces)) ' + ' '.join(fields)


def check():
    tree = ast.parse((HERE / 'supervise.py').read_text())
    names = ('parse_proc_stat_rss', 'process_identity_with_rss', 'members_of_session')
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == 3
    texts, reads = {}, []

    class FakePath:
        def __init__(self, value): self.value = str(value)
        @property
        def name(self): return self.value.rsplit('/', 1)[-1]
        def __truediv__(self, name): return FakePath(self.value + '/' + str(name))
        def iterdir(self):
            assert self.value == '/proc'
            return [FakePath('/proc/' + name) for name in ('101', '102', '103', '104', 'self')]
        def read_text(self):
            assert self.value.endswith('/stat'), 'Sampler must not read separate status/RSS data'
            reads.append(self.value)
            if self.value not in texts: raise FileNotFoundError(self.value)
            return texts[self.value]

    class FakeOS:
        @staticmethod
        def sysconf(name):
            assert name == 'SC_PAGE_SIZE'
            return 4096

    namespace = dict(Path=FakePath, os=FakeOS, require=require)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<actual v3 sampler>', 'exec'), namespace)
    parser = namespace['parse_proc_stat_rss']
    accepted = []
    for state, pages in (('R', 0), ('S', 0), ('S', 17), ('Z', 0)):
        row = parser(stat_text(state=state, pages=pages), 101, 4096)
        assert row == dict(pid=101, start_ticks=55, session=100, group=100, state=state, RSS_bytes=pages*4096)
        accepted.append(dict(state=state, rss_pages=pages, RSS_bytes=row['RSS_bytes']))
    rejected = []
    for label, raw, pid, page_size in (
        ('negative_RSS', stat_text(pages=-1), 101, 4096),
        ('mismatched_PID', stat_text(), 102, 4096),
        ('truncated_fields', '101 (loader) R', 101, 4096),
        ('invalid_page_size', stat_text(), 101, 0),
    ):
        try: parser(raw, pid, page_size)
        except (RuntimeError, ValueError): rejected.append(label)
        else: raise AssertionError(label + ' accepted')

    texts.update({
        '/proc/101/stat': stat_text(),
        '/proc/102/stat': stat_text(pid=102, state='S', pages=4, group=200),
        '/proc/103/stat': stat_text(pid=103, session=200, group=200, pages=8),
    })
    members = namespace['members_of_session'](100)
    assert [row['pid'] for row in members] == [101, 102]
    assert members[0]['state'] == 'R' and members[0]['RSS_bytes'] == 0
    assert members[1]['group'] == 200 and members[1]['RSS_bytes'] == 4*4096
    assert reads == ['/proc/%d/stat' % pid for pid in (101, 102, 103, 104)]
    assert any(row['group'] != 100 for row in members)

    return dict(status='PASS_NARROW_SAME_STAT_RSS_REGRESSION', accepted_snapshots=accepted,
                malformed_or_mismatched_rejected=rejected, same_session_filter=True,
                changed_owned_group_retained_for_strict_guard=True, disappeared_task_omitted=True,
                exactly_one_stat_read_per_listed_task=True, status_reads=0,
                real_proc_access=False, processes_created=False, signals_sent=False,
                numerical_or_native_imports=False)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2, sort_keys=True))
