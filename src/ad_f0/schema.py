from __future__ import annotations
import sqlite3


def table_names(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}


def require_tables(conn, names):
    have = table_names(conn)
    missing = [x for x in names if x not in have]
    if missing:
        raise RuntimeError(f'Missing required ChEMBL tables: {missing}')


def pick_column(conn, table: str, candidates: list[str], required=True):
    have = columns(conn, table)
    for c in candidates:
        if c in have:
            return c
    if required:
        raise RuntimeError(f'{table}: none of columns {candidates} found; have={sorted(have)}')
    return None
