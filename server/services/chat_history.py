"""Simple SQLite-backed chat history per session."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src import config
from src.utils import logger
from src.utils.datetime_utils import utc_isoformat


class ChatHistoryStore:
    """Persist chat messages into per-session SQLite databases."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        base_path = Path(base_dir) if base_dir else Path(config.save_dir) / "sessions"
        base_path.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_path

    def _db_path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.db"

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_name TEXT,
                tool_input TEXT,
                tool_output TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()

    def _connect(self, session_id: str) -> sqlite3.Connection:
        path = self._db_path(session_id)
        conn = sqlite3.connect(path)
        self._ensure_schema(conn)
        return conn

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        tool_name: str | None = None,
        tool_input: dict[str, Any] | None = None,
        tool_output: dict[str, Any] | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        conn = self._connect(session_id)
        with conn:
            conn.execute(
                "INSERT INTO messages (role, content, tool_name, tool_input, tool_output, metadata, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    role,
                    content,
                    tool_name,
                    json.dumps(tool_input or {}, ensure_ascii=False),
                    json.dumps(tool_output, ensure_ascii=False) if tool_output is not None else None,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    utc_isoformat(),
                ),
            )
        conn.close()

    def load(self, session_id: str) -> list[dict[str, Any]]:
        path = self._db_path(session_id)
        if not path.exists():
            return []

        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            self._ensure_schema(conn)
            rows = conn.execute(
                "SELECT role, content, tool_name, tool_input, tool_output, metadata FROM messages ORDER BY id"
            ).fetchall()
            history: list[dict[str, Any]] = []
            for row in rows:
                history.append(
                    {
                        "role": row["role"],
                        "content": row["content"],
                        "tool_name": row["tool_name"],
                        "tool_input": json.loads(row["tool_input"] or "{}"),
                        "tool_output": row["tool_output"],
                        "metadata": json.loads(row["metadata"] or "{}"),
                    }
                )
            return history
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load chat history for %s: %s", session_id, exc)
            return []
        finally:
            conn.close()
