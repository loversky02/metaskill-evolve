"""SQLite-backed branch DAG.

The search over skills is a tree/DAG of branches; children commit to a SQLite
DAG (arXiv:2607.05297, Alg. 1). Each node stores its utility ``U(s)`` and a full
skill+meta snapshot, so any branch can be restored and extended later. Nodes are
typed ``task`` (fast loop) or ``meta`` (slow loop).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id  INTEGER,
    kind       TEXT NOT NULL DEFAULT 'task',
    iter       INTEGER NOT NULL DEFAULT 0,
    utility    REAL NOT NULL DEFAULT 0.0,
    tag        TEXT,
    snapshot   TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(parent_id) REFERENCES nodes(id)
);
"""


@dataclass
class Node:
    id: int
    parent_id: Optional[int]
    kind: str
    iter: int
    utility: float
    tag: Optional[str]
    snapshot: Dict[str, str]


class DAG:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def add(self, *, parent_id: Optional[int] = None, kind: str = "task",
            iter: int = 0, utility: float = 0.0, tag: Optional[str] = None,
            snapshot: Optional[Dict[str, str]] = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO nodes(parent_id,kind,iter,utility,tag,snapshot) "
            "VALUES(?,?,?,?,?,?)",
            (parent_id, kind, iter, utility, tag, json.dumps(snapshot or {})),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get(self, node_id: int) -> Node:
        r = self.conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if r is None:
            raise KeyError(node_id)
        return self._row(r)

    def children(self, node_id: int) -> List[Node]:
        rs = self.conn.execute(
            "SELECT * FROM nodes WHERE parent_id=? ORDER BY id", (node_id,)
        ).fetchall()
        return [self._row(r) for r in rs]

    def best(self, n: int = 1, kind: str = "task") -> List[Node]:
        """Top-n nodes of a kind by utility (ties broken by insertion order)."""
        rs = self.conn.execute(
            "SELECT * FROM nodes WHERE kind=? ORDER BY utility DESC, id ASC LIMIT ?",
            (kind, n),
        ).fetchall()
        return [self._row(r) for r in rs]

    def frontier(self, weights, kind: str = "task") -> List[Tuple[Node, float]]:
        """Top-K_F branches paired with decaying weights eta (Eq. 4)."""
        top = self.best(len(weights), kind=kind)
        return list(zip(top, weights))

    @staticmethod
    def _row(r: sqlite3.Row) -> Node:
        return Node(
            id=r["id"], parent_id=r["parent_id"], kind=r["kind"], iter=r["iter"],
            utility=r["utility"], tag=r["tag"], snapshot=json.loads(r["snapshot"]),
        )
