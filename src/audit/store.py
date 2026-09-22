from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_AUDIT_FIELDS = {
    "case_id",
    "timestamp",
    "event_type",
    "actor",
    "payload",
}


class AuditStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_audit_case ON audit_events(case_id)")

    def log(
        self,
        case_id: str,
        event_type: str,
        payload: dict[str, Any],
        actor: str = "system",
    ) -> int:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO audit_events(case_id,timestamp,event_type,actor,payload) VALUES(?,?,?,?,?)",
                (case_id, timestamp, event_type, actor, json.dumps(payload, default=str)),
            )
            return int(cur.lastrowid)

    def events(self, case_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as con:
            if case_id:
                rows = con.execute(
                    "SELECT * FROM audit_events WHERE case_id=? ORDER BY id", (case_id,)
                ).fetchall()
            else:
                rows = con.execute("SELECT * FROM audit_events ORDER BY id").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            result.append(item)
        return result

    def clear(self, case_id: str | None = None) -> None:
        with self._connect() as con:
            if case_id:
                con.execute("DELETE FROM audit_events WHERE case_id=?", (case_id,))
            else:
                con.execute("DELETE FROM audit_events")

    def export_json(self, path: str | Path, case_id: str | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.events(case_id), indent=2), encoding="utf-8")
        return path


def audit_completeness(events: list[dict[str, Any]]) -> float:
    if not events:
        return 0.0
    complete = 0
    for event in events:
        if REQUIRED_AUDIT_FIELDS.issubset(event.keys()):
            complete += 1
    return complete / len(events)
