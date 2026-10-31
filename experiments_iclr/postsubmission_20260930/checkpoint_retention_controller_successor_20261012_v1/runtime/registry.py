"""One transactional fit registry; model fits and cheap endpoints stay distinct."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

try:
    from .spec import canonical, fingerprint, validate
except ImportError:
    from spec import canonical, fingerprint, validate


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
        validate(spec)
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

    def run(self, study, run_id):
        row = self.db.execute("SELECT contract,state,owner,result FROM runs WHERE study=? AND id=?",
                              (study, run_id)).fetchone()
        if row is None:
            raise ValueError("Unknown run")
        contract, state, owner, result = row
        return {"contract": json.loads(contract), "state": state,
                "owner": json.loads(owner) if owner else None,
                "result": json.loads(result) if result else None}

    def states(self):
        """One small query for scheduler admission; do not deserialize fit evidence."""
        return {study + "/" + identifier: state for study, identifier, state in
                self.db.execute("SELECT study,id,state FROM runs")}

    def recover(self, study, run_id, new_owner, admission):
        """Custody transition only after the caller verifies exact owner termination."""
        if not admission.get("previous_owner_terminal") or not admission.get("previous_owner"):
            raise ValueError("Recovery requires a verified terminal owner")
        with self.db:
            row = self.run(study, run_id)
            if row["state"] not in {"running", "failed"} or row["owner"] != admission["previous_owner"]:
                raise ValueError("Recovery does not identify the registered interrupted owner")
            self.db.execute("UPDATE runs SET state='running',owner=?,updated=? WHERE study=? AND id=?",
                            (canonical(new_owner), utc(), study, run_id))
            self._event(study, run_id, "recovered", {"owner": new_owner, "admission": admission})

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
            if success and result.get("retention_authority"):
                self._event(study,run_id,"selected_serving_authority_closed",{
                    "retention_authority":result["retention_authority"],"serving_checkpoint":result["selected_serving_checkpoint"]})

    def consumer_records(self, study, run_id):
        """Derive direct/transitive leases from the immutable registered DAG."""
        raw=self.db.execute("SELECT spec FROM studies WHERE id=?",(study,)).fetchone()
        if raw is None:raise ValueError("Unknown study for retention leases")
        spec=json.loads(raw[0]);runs={r["run_id"]:r for r in spec["runs"]}
        if run_id not in runs:raise ValueError("Unknown fit for retention leases")
        children={k:[] for k in runs}
        for r in runs.values():
            for p in r["parents"]:children[p].append(r["run_id"])
        seen=set();todo=list(children[run_id])
        while todo:
            k=todo.pop()
            if k in seen:continue
            seen.add(k);todo.extend(children[k])
        states=dict(self.db.execute("SELECT id,state FROM runs WHERE study=?",(study,)))
        return {"direct_ids":children[run_id],"transitive_ids":sorted(seen),
            "contracts":{k:runs[k] for k in seen},"states":{k:states[k] for k in seen}}

    def release_ready(self, study):
        rows = self.db.execute("SELECT id,state FROM runs WHERE study=?", (study,)).fetchall()
        return bool(rows) and all(state == "complete" for _, state in rows)

    def all_terminal(self, study):
        rows = self.db.execute("SELECT state FROM runs WHERE study=?", (study,)).fetchall()
        return bool(rows) and all(state in {"complete", "failed"} for (state,) in rows)

    def required_complete(self, study, run_ids):
        rows = dict(self.db.execute("SELECT id,state FROM runs WHERE study=?", (study,)).fetchall())
        return bool(run_ids) and all(rows.get(identifier) == "complete" for identifier in run_ids)

    def required_terminal(self, study, run_ids):
        rows = dict(self.db.execute("SELECT id,state FROM runs WHERE study=?", (study,)).fetchall())
        return bool(run_ids) and all(rows.get(identifier) in {"complete", "failed"} for identifier in run_ids)

    def summary(self, study):
        rows = self.db.execute("SELECT kind,state,COUNT(*) FROM runs WHERE study=? GROUP BY kind,state",
                               (study,)).fetchall()
        return {"study_id": study, "whole_roster_ready": self.release_ready(study),
                "whole_roster_terminal": self.all_terminal(study),
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
