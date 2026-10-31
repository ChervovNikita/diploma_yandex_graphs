"""One transactional fit registry; model fits and cheap endpoints stay distinct."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

try:
    from .spec import canonical, fingerprint
except ImportError:
    from spec import canonical, fingerprint


def utc():
    return datetime.now(timezone.utc).isoformat()


class Registry:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS studies (
            id TEXT PRIMARY KEY, spec_hash TEXT NOT NULL, spec TEXT NOT NULL,
            created TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS runs (
            study TEXT NOT NULL REFERENCES studies(id), id TEXT NOT NULL,
            kind TEXT NOT NULL, method TEXT NOT NULL, split TEXT NOT NULL,
            seed INTEGER NOT NULL, body INTEGER NOT NULL, contract TEXT NOT NULL,
            state TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL,
            owner TEXT, result TEXT, PRIMARY KEY(study,id));
          CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY, study TEXT NOT NULL, run TEXT NOT NULL,
            time TEXT NOT NULL, type TEXT NOT NULL, payload TEXT NOT NULL,
            FOREIGN KEY(study,run) REFERENCES runs(study,id));
        """)
        self.db.commit()

    def close(self):
        self.db.close()

    def register(self, spec):
        """Repeated registration is safe only for identical prospective bytes."""
        body, digest, now = canonical(spec), fingerprint(spec), utc()
        with self.db:
            old = self.db.execute("SELECT spec_hash FROM studies WHERE id=?", (spec["study_id"],)).fetchone()
            if old:
                if old[0] != digest:
                    raise ValueError("Study contract changed; declare a new study")
                return digest
            self.db.execute("INSERT INTO studies VALUES (?,?,?,?)", (spec["study_id"], digest, body, now))
            for run in spec["runs"]:
                self.db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    spec["study_id"], run["run_id"], run["kind"], run["method"], str(run["split"]),
                    run["seed"], run["body"], canonical(run), "registered", now, now, None, None,
                ))
        return digest

    def claim(self, study, run_id, owner):
        """A stale-looking record is never permission for a duplicate launch."""
        if not owner:
            raise ValueError("Bind a real process/owner identity")
        now = utc()
        with self.db:
            cursor = self.db.execute("UPDATE runs SET state='running',owner=?,updated=? "
                                     "WHERE study=? AND id=? AND state='registered'",
                                     (canonical(owner), now, study, run_id))
            if cursor.rowcount != 1:
                raise ValueError("Run already claimed, terminal or absent")
            self._event(study, run_id, "started", owner)

    def _event(self, study, run_id, kind, payload):
        self.db.execute("INSERT INTO events(study,run,time,type,payload) VALUES (?,?,?,?,?)",
                        (study, run_id, utc(), kind, canonical(payload)))

    def event(self, study, run_id, kind, payload):
        with self.db:
            self._event(study, run_id, kind, payload)

    def finish(self, study, run_id, result, *, success):
        state = "complete" if success else "failed"
        with self.db:
            cursor = self.db.execute("UPDATE runs SET state=?,result=?,updated=? "
                                     "WHERE study=? AND id=? AND state='running'",
                                     (state, canonical(result), utc(), study, run_id))
            if cursor.rowcount != 1:
                raise ValueError("Only the claimed running record may finish")
            self._event(study, run_id, state, result)

    def release_ready(self, study):
        rows = self.db.execute("SELECT id,state FROM runs WHERE study=?", (study,)).fetchall()
        return bool(rows) and all(state == "complete" for _, state in rows)

    def summary(self, study):
        rows = self.db.execute("SELECT kind,state,COUNT(*) FROM runs WHERE study=? GROUP BY kind,state",
                               (study,)).fetchall()
        return {"study_id": study, "whole_roster_ready": self.release_ready(study),
                "counts": [{"kind": kind, "state": state, "records": count}
                           for kind, state, count in rows]}

    def export(self, study, path):
        spec = self.db.execute("SELECT spec FROM studies WHERE id=?", (study,)).fetchone()
        if not spec:
            raise ValueError("Unknown study")
        rows = self.db.execute("SELECT id,kind,state,owner,result FROM runs WHERE study=? ORDER BY id",
                               (study,)).fetchall()
        value = {"spec": json.loads(spec[0]), **self.summary(study), "runs": [
            {"run_id": a, "kind": b, "state": c,
             "owner": json.loads(d) if d else None, "result": json.loads(e) if e else None}
            for a, b, c, d, e in rows]}
        Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        return value
